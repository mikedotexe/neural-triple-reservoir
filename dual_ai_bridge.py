#!/usr/bin/env python3
"""
dual_ai_bridge.py — Two local LLMs feed a shared ANE triple reservoir.

Ollama and MLX-LM each get their own state handle on one compiled
Core ML reservoir model. Resonance emerges naturally from shared
dynamical weights — similar inputs trace nearby trajectories through
the same attractor landscape.

Usage:
    # Export a 16-input reservoir first
    python dual_ai_bridge.py --export reservoir_dual.mlpackage

    # Run with both AIs (Core ML / ANE)
    python dual_ai_bridge.py --model reservoir_dual.mlpackage \\
        --ollama-model llama3.2 \\
        --mlx-model mlx-community/Llama-3.2-1B-Instruct-4bit \\
        --prompt "What does it feel like to think?"

    # NumPy-only mode (no Core ML needed)
    python dual_ai_bridge.py \\
        --ollama-model llama3.2 \\
        --prompt "Describe the color blue"

    # Single source
    python dual_ai_bridge.py --ollama-only --ollama-model llama3.2
    python dual_ai_bridge.py --mlx-only --mlx-model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from triple_reservoir_coreml import ReservoirConfig, TripleReservoir


# ---------------------------------------------------------------------------
# Text → reservoir input projection
# ---------------------------------------------------------------------------

class TextProjection:
    """Frozen random projection: sliding byte window → bounded reservoir input.

    Each tick, the accumulated text so far is encoded as a fixed-width
    input vector.  The projection matrix is frozen random — consistent
    with how the reservoir itself uses frozen random recurrent weights.
    """

    def __init__(self, input_dim: int = 32, window: int = 64, seed: int = 42):
        self.input_dim = input_dim
        self.window = window
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((window, input_dim)) / np.sqrt(window)).astype(
            np.float32
        )

    def __call__(self, text: str) -> np.ndarray:
        """Project the last `window` bytes of text into a bounded input vector."""
        raw = text.encode("utf-8", errors="replace")[-self.window :]
        vec = np.zeros(self.window, dtype=np.float32)
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            vec[-len(arr) :] = arr / 127.5 - 1.0
        return np.tanh(vec @ self.W).reshape(1, self.input_dim).astype(np.float32)


# ---------------------------------------------------------------------------
# AI sources
# ---------------------------------------------------------------------------

class OllamaSource:
    """Stream tokens from a local Ollama model via REST API."""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.accumulated = ""

    def stream(self, prompt: str) -> Iterator[str]:
        self.accumulated = ""
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": True}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tok = data.get("response", "")
                if tok:
                    self.accumulated += tok
                    yield tok
                if data.get("done"):
                    break


class MLXSource:
    """Stream tokens from a local MLX-LM model."""

    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self.accumulated = ""

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from mlx_lm import load
            except ImportError:
                raise RuntimeError(
                    "mlx-lm required for MLX source: pip install mlx-lm"
                )
            self._model, self._tokenizer = load(self.model_name)

    def stream(self, prompt: str) -> Iterator[str]:
        self._ensure_loaded()
        from mlx_lm import stream_generate

        self.accumulated = ""
        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
        ):
            # Handle different mlx-lm versions
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if text:
                self.accumulated += text
                yield text


# ---------------------------------------------------------------------------
# Reservoir bridge — one model, multiple named state handles
# ---------------------------------------------------------------------------

class ReservoirBridge:
    """Wraps one reservoir (Core ML, MLX, or NumPy) with named state handles."""

    def __init__(
        self,
        mlpackage: Optional[Path] = None,
        config: Optional[ReservoirConfig] = None,
        compute_units: str = "cpu_and_ne",
        use_mlx: bool = False,
    ):
        self.config = config or ReservoirConfig(input_dim=16)
        self.use_coreml = mlpackage is not None
        self.use_mlx = (not self.use_coreml) and use_mlx
        self.states: dict = {}
        self.outputs: dict[str, list[float]] = {}
        self._tick_count = 0

        if self.use_coreml:
            import coremltools as ct

            cu_map = {
                "all": ct.ComputeUnit.ALL,
                "cpu_only": ct.ComputeUnit.CPU_ONLY,
                "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
                "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            }
            self._coreml = ct.models.MLModel(
                str(mlpackage), compute_units=cu_map[compute_units]
            )
        elif self.use_mlx:
            import mlx.core as mx
            from mlx_reservoir import MLXTripleReservoir

            self._np_model = TripleReservoir(self.config)
            self._train_synthetic()
            self._mlx_model = MLXTripleReservoir(self._np_model)
            self._mx = mx
        else:
            self._np_model = TripleReservoir(self.config)
            self._train_synthetic()

    def _train_synthetic(self):
        """Fit readout on synthetic data so the model is valid."""
        from triple_reservoir_coreml import CANONICAL_SEED
        rng = np.random.default_rng(CANONICAL_SEED)
        d = self.config.input_dim
        x = np.tanh(rng.standard_normal((3000, d))).astype(np.float32)
        y = np.zeros((3000, self.config.output_dim), dtype=np.float32)
        latent = 0.0
        for t in range(3000):
            latent = 0.9 * latent + 0.3 * float(x[t, : min(4, d)].sum())
            y[t, 0] = latent
        self._np_model.fit_readout(x, y)

    def make_state(self, name: str):
        """Create a fresh named state handle."""
        if self.use_coreml:
            self.states[name] = self._coreml.make_state()
        elif self.use_mlx:
            self.states[name] = self._mlx_model.zero_state()
        else:
            self.states[name] = self._np_model.zero_state()
        self.outputs[name] = []

    def tick(self, name: str, x: np.ndarray) -> float:
        """Feed one input vector into the named state, return scalar output."""
        x = np.asarray(x).reshape(1, -1)
        if self.use_coreml:
            out = self._coreml.predict(
                {"x": x.astype(np.float16)}, state=self.states[name]
            )
            val = float(np.asarray(out["y"]).ravel()[0])
        elif self.use_mlx:
            mx = self._mx
            x_mx = mx.array(x.astype(np.float32))
            y, _, new_state = self._mlx_model.step(x_mx, self.states[name])
            self.states[name] = new_state
            self._tick_count += 1
            if self._tick_count % 64 == 0:
                mx.eval(*new_state)
            val = float(y.item())
        else:
            y, _, new_state = self._np_model.step_numpy(
                x.astype(np.float32), self.states[name]
            )
            self.states[name] = new_state
            val = float(y.ravel()[0])
        self.outputs[name].append(val)
        return val

    def read_state(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (h1, h2, h3) numpy arrays for a named handle."""
        if self.use_coreml:
            raise NotImplementedError("CoreML state readback not supported yet")
        if self.use_mlx:
            return self._mlx_model.state_to_numpy(self.states[name])
        h1, h2, h3 = self.states[name]
        return h1.copy(), h2.copy(), h3.copy()

    def set_state(self, name: str, h1: np.ndarray, h2: np.ndarray, h3: np.ndarray):
        """Restore hidden state for a named handle from numpy arrays."""
        if self.use_coreml:
            raise NotImplementedError("CoreML state injection not supported yet")
        if self.use_mlx:
            self.states[name] = self._mlx_model.state_from_numpy((h1, h2, h3))
            return
        self.states[name] = (
            h1.astype(np.float32),
            h2.astype(np.float32),
            h3.astype(np.float32),
        )

    def h_norms(self, name: str) -> tuple[float, float, float]:
        """Return L2 norms of (h1, h2, h3) for a named handle."""
        if self.use_coreml:
            return (0.0, 0.0, 0.0)  # Not accessible from CoreML
        if self.use_mlx:
            mx = self._mx
            h1, h2, h3 = self.states[name]
            return (
                float(mx.sqrt(mx.sum(h1 * h1)).item()),
                float(mx.sqrt(mx.sum(h2 * h2)).item()),
                float(mx.sqrt(mx.sum(h3 * h3)).item()),
            )
        h1, h2, h3 = self.states[name]
        return (
            float(np.linalg.norm(h1)),
            float(np.linalg.norm(h2)),
            float(np.linalg.norm(h3)),
        )

    def get_layer_states(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return numpy arrays of (h1, h2, h3) for per-layer metrics."""
        if self.use_mlx:
            h1, h2, h3 = self.states[name]
            return (
                np.array(h1).ravel(),
                np.array(h2).ravel(),
                np.array(h3).ravel(),
            )
        h1, h2, h3 = self.states[name]
        return (
            np.asarray(h1).ravel(),
            np.asarray(h2).ravel(),
            np.asarray(h3).ravel(),
        )

    def has_handle(self, name: str) -> bool:
        """Check if a named handle exists."""
        return name in self.states

    def destroy_handle(self, name: str):
        """Remove a named handle."""
        self.states.pop(name, None)
        self.outputs.pop(name, None)


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


def export_model(
    path: Path,
    input_dim: int = 16,
    n_nodes: int = 192,
    compute_units: str = "cpu_and_ne",
):
    """Build, train (synthetic), and export a reservoir .mlpackage."""
    cfg = ReservoirConfig(input_dim=input_dim, n_nodes=n_nodes)
    model = TripleReservoir(cfg)

    rng = np.random.default_rng(99)
    x = np.tanh(rng.standard_normal((3000, input_dim))).astype(np.float32)
    y = np.zeros((3000, 1), dtype=np.float32)
    latent = 0.0
    for t in range(3000):
        latent = 0.9 * latent + 0.3 * float(x[t, : min(4, input_dim)].sum())
        y[t, 0] = latent
    mse = model.fit_readout(x, y)
    print(f"train_mse={mse:.8f}")
    model.export_coreml(path, compute_units=compute_units)
    print(f"exported {path}")


# ---------------------------------------------------------------------------
# Feed one source through the reservoir
# ---------------------------------------------------------------------------


def feed_source(
    name: str,
    source,
    prompt: str,
    proj: TextProjection,
    bridge: ReservoirBridge,
) -> int:
    """Run one AI, feeding each token into the reservoir.  Returns tick count."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")

    ticks = 0
    for tok in source.stream(prompt):
        sys.stdout.write(tok)
        sys.stdout.flush()
        x = proj(source.accumulated)
        bridge.tick(name, x)
        ticks += 1

    print(f"\n  [{ticks} ticks]\n")
    return ticks


# ---------------------------------------------------------------------------
# Coupled generation — bidirectional LLM ↔ reservoir
# ---------------------------------------------------------------------------


def coupled_feed_source(
    name: str,
    prompt: str,
    llm_model,
    tokenizer,
    mlx_reservoir,
    embedding_proj,
    processor,
    bridge: ReservoirBridge,
    max_tokens: int = 512,
    temp: float = 0.7,
) -> int:
    """Run MLX LLM with bidirectional reservoir coupling.

    At each token step:
      1. LLM generates a token (logits modulated by reservoir state)
      2. Token embedding extracted from LLM (zero-copy, same Metal memory)
      3. Embedding projected to reservoir input
      4. Reservoir ticked — state evolves
      5. Processor updated — influences next token's logits

    Returns tick count.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    print(f"\n{'=' * 60}")
    print(f"  {name} (coupled)")
    print(f"{'=' * 60}\n")

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    sampler = make_sampler(temp=temp)

    # State lives as mx.array — stays on Metal throughout
    state = mlx_reservoir.zero_state()

    ticks = 0
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    for token, logprobs in generate_step(
        prompt_tokens,
        llm_model,
        max_tokens=max_tokens,
        logits_processors=[processor],
        sampler=sampler,
    ):
        # Streaming output
        detokenizer.add_token(token)
        sys.stdout.write(detokenizer.last_segment)
        sys.stdout.flush()

        # Extract embedding for this token (zero-copy on UMA)
        embedding = llm_model.model.embed_tokens(
            mx.array([[token]])
        )  # (1, 1, hidden_dim)
        embedding = embedding.reshape(1, -1)  # (1, hidden_dim)

        # Project to reservoir input space
        r_input = embedding_proj.project(embedding)  # (1, input_dim)

        # Tick reservoir — all mx.array, same Metal command stream
        y, z, state = mlx_reservoir.step(r_input, state)

        # Prevent graph accumulation
        if ticks % 64 == 0:
            mx.eval(*state)

        val = float(y.item())
        bridge.outputs.setdefault(name, []).append(val)

        # Update processor for next token's logit modulation
        processor.update(val)
        ticks += 1

    # Store final state in bridge for summary/persistence
    if bridge.use_mlx and ticks > 0:
        bridge.states[name] = state

    print(f"\n  [{ticks} ticks, coupling={processor.coupling_strength}]\n")
    return ticks


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(bridge: ReservoirBridge, names: list[str]):
    print(f"{'=' * 60}")
    print("  reservoir summary")
    print(f"{'=' * 60}")

    for name in names:
        outs = bridge.outputs.get(name, [])
        if outs:
            print(f"  {name:8s}  {len(outs):>4d} ticks  final={outs[-1]:+.6f}")

    if len(names) == 2:
        a_outs = bridge.outputs.get(names[0], [])
        b_outs = bridge.outputs.get(names[1], [])
        n = min(len(a_outs), len(b_outs))
        if n > 0:
            div = abs(a_outs[-1] - b_outs[-1])
            print(f"\n  divergence (final): {div:.6f}")
        if n >= 10:
            a = np.array(a_outs[:n])
            b = np.array(b_outs[:n])
            if a.std() > 1e-8 and b.std() > 1e-8:
                corr = float(np.corrcoef(a, b)[0, 1])
                print(f"  correlation:        {corr:+.4f}")
            # Trajectory distance
            rmsd = float(np.sqrt(np.mean((a - b) ** 2)))
            print(f"  trajectory RMSD:    {rmsd:.6f}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Dual-AI bridge: Ollama + MLX → ANE triple reservoir"
    )
    ap.add_argument("--export", type=Path, help="Export .mlpackage and exit")
    ap.add_argument(
        "--model", type=Path, help="Path to .mlpackage (omit for NumPy mode)"
    )
    ap.add_argument("--input-dim", type=int, default=16)
    ap.add_argument("--n-nodes", type=int, default=192)
    ap.add_argument(
        "--compute-units",
        default="cpu_and_ne",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
    )
    ap.add_argument("--ollama-model", default="llama3.2")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument(
        "--mlx-model", default="mlx-community/Llama-3.2-1B-Instruct-4bit"
    )
    ap.add_argument("--mlx-max-tokens", type=int, default=512)
    ap.add_argument("--prompt", default="What does it feel like to think?")
    ap.add_argument("--ollama-prompt", default=None)
    ap.add_argument("--mlx-prompt", default=None)
    ap.add_argument("--ollama-only", action="store_true")
    ap.add_argument("--mlx-only", action="store_true")
    ap.add_argument(
        "--window", type=int, default=64, help="Byte window for text projection"
    )
    ap.add_argument(
        "--use-mlx", action="store_true",
        help="Use MLX backend (Metal/GPU) instead of NumPy for reservoir",
    )
    ap.add_argument(
        "--coupled", action="store_true",
        help="Bidirectional mode: reservoir state modulates LLM generation",
    )
    ap.add_argument(
        "--coupling-strength", type=float, default=0.1,
        help="How strongly reservoir modulates logits (default: 0.1)",
    )
    ap.add_argument(
        "--temp", type=float, default=0.7,
        help="LLM sampling temperature (default: 0.7)",
    )
    args = ap.parse_args()

    # --- Export mode ---
    if args.export:
        export_model(args.export, args.input_dim, args.n_nodes, args.compute_units)
        return

    # --- Coupled mode: bidirectional LLM ↔ reservoir ---
    if args.coupled:
        from mlx_lm import load as mlx_load
        from mlx_reservoir import (
            EmbeddingProjection,
            MLXTripleReservoir,
            ReservoirLogitProcessor,
        )

        cfg = ReservoirConfig(input_dim=args.input_dim, n_nodes=args.n_nodes)
        bridge = ReservoirBridge(config=cfg, use_mlx=True)
        bridge.make_state("coupled")

        print(f"loading MLX model: {args.mlx_model}")
        llm_model, tokenizer = mlx_load(args.mlx_model)

        # Get embedding dimension from the loaded model
        embed_dim = llm_model.args.hidden_size
        print(f"reservoir: MLX (Metal) | input_dim={args.input_dim} | n_nodes={args.n_nodes}")
        print(f"coupling:  strength={args.coupling_strength} | embed_dim={embed_dim} | temp={args.temp}")

        # Build MLX reservoir and projection
        mlx_res = bridge._mlx_model
        embed_proj = EmbeddingProjection(embed_dim=embed_dim, input_dim=args.input_dim)
        processor = ReservoirLogitProcessor(coupling_strength=args.coupling_strength)

        prompt = args.mlx_prompt or args.prompt
        try:
            coupled_feed_source(
                "coupled", prompt, llm_model, tokenizer,
                mlx_res, embed_proj, processor, bridge,
                max_tokens=args.mlx_max_tokens, temp=args.temp,
            )
        except Exception as e:
            print(f"\n  [coupled generation failed: {e}]")
            import traceback
            traceback.print_exc()

        print_summary(bridge, ["coupled"])
        return

    # --- Standard mode: one-directional AI → reservoir ---
    cfg = ReservoirConfig(input_dim=args.input_dim, n_nodes=args.n_nodes)
    bridge = ReservoirBridge(
        mlpackage=args.model, config=cfg, compute_units=args.compute_units,
        use_mlx=args.use_mlx,
    )
    mode = "Core ML (ANE)" if bridge.use_coreml else ("MLX (Metal)" if bridge.use_mlx else "NumPy")
    print(f"reservoir: {mode} | input_dim={args.input_dim} | n_nodes={args.n_nodes}")

    proj = TextProjection(input_dim=args.input_dim, window=args.window)

    run_ollama = not args.mlx_only
    run_mlx = not args.ollama_only

    names: list[str] = []

    # --- Feed Ollama ---
    if run_ollama:
        bridge.make_state("ollama")
        names.append("ollama")
        ollama = OllamaSource(model=args.ollama_model, base_url=args.ollama_url)
        p = args.ollama_prompt or args.prompt
        try:
            feed_source("ollama", ollama, p, proj, bridge)
        except Exception as e:
            print(f"\n  [ollama failed: {e}]")

    # --- Feed MLX ---
    if run_mlx:
        bridge.make_state("mlx")
        names.append("mlx")
        mlx = MLXSource(
            model_name=args.mlx_model, max_tokens=args.mlx_max_tokens
        )
        p = args.mlx_prompt or args.prompt
        try:
            feed_source("mlx", mlx, p, proj, bridge)
        except Exception as e:
            print(f"\n  [mlx failed: {e}]")

    # --- Summary ---
    if names:
        print_summary(bridge, names)


if __name__ == "__main__":
    main()
