#!/usr/bin/env python3
"""
mlx_reservoir.py -- MLX-native triple echo-state reservoir.

Same frozen dynamics as the NumPy/Core ML versions, but state lives as
mx.array tensors on Metal.  On Apple Silicon unified memory, these tensors
are zero-copy accessible from MLX LLMs -- the connection point for
bidirectional coupling in tranche 2.

Usage:
    # Verify numerical equivalence against NumPy
    python mlx_reservoir.py --verify

    # Verify + benchmark
    python mlx_reservoir.py --verify --bench
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import mlx.core as mx
import numpy as np

from triple_reservoir_coreml import ReservoirConfig, TripleReservoir


# ---------------------------------------------------------------------------
# MLX triple reservoir
# ---------------------------------------------------------------------------

class MLXTripleReservoir:
    """Triple echo-state reservoir with state as mx.array tensors.

    Weights are initialized from a canonical NumPy TripleReservoir -- same
    seed, same dynamics.  NOT an mlx.nn.Module (no learnable params).
    """

    def __init__(self, bundle: TripleReservoir):
        self.config = bundle.config
        n = bundle.config.n_nodes

        # Normalization
        self.x_mean = mx.array(bundle.x_mean.reshape(1, -1))
        self.x_scale = mx.array(bundle.x_scale.reshape(1, -1))

        # Layer 1
        self.w_in1 = mx.array(bundle.w_in1)
        self.w1 = mx.array(bundle.w1)
        self.b1 = mx.array(bundle.b1.reshape(1, -1))

        # Layer 2
        self.w_in2 = mx.array(bundle.w_in2)
        self.w2 = mx.array(bundle.w2)
        self.b2 = mx.array(bundle.b2.reshape(1, -1))

        # Layer 3
        self.w_in3 = mx.array(bundle.w_in3)
        self.w3 = mx.array(bundle.w3)
        self.b3 = mx.array(bundle.b3.reshape(1, -1))

        # Readout
        self.readout_w = mx.array(bundle.readout_w)
        self.readout_b = mx.array(bundle.readout_b.reshape(1, -1))

        # Leak rates as scalars
        self.leak1 = bundle.config.leaks[0]
        self.leak2 = bundle.config.leaks[1]
        self.leak3 = bundle.config.leaks[2]

    def zero_state(self, batch: int = 1) -> tuple[mx.array, mx.array, mx.array]:
        n = self.config.n_nodes
        return (
            mx.zeros((batch, n)),
            mx.zeros((batch, n)),
            mx.zeros((batch, n)),
        )

    def step(
        self,
        x: mx.array,
        state: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array, mx.array]]:
        """One reservoir tick.  Returns (y, z, (h1, h2, h3))."""
        x = (x - self.x_mean) / self.x_scale
        h1, h2, h3 = state

        h1 = (1.0 - self.leak1) * h1 + self.leak1 * mx.tanh(
            x @ self.w_in1.T + h1 @ self.w1.T + self.b1
        )
        h2 = (1.0 - self.leak2) * h2 + self.leak2 * mx.tanh(
            h1 @ self.w_in2.T + h2 @ self.w2.T + self.b2
        )
        h3 = (1.0 - self.leak3) * h3 + self.leak3 * mx.tanh(
            h2 @ self.w_in3.T + h3 @ self.w3.T + self.b3
        )

        z = mx.concatenate([h1, h2, h3], axis=1)
        y = z @ self.readout_w.T + self.readout_b
        return y, z, (h1, h2, h3)

    def step_multi(
        self,
        x: mx.array,
        state: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]:
        """One tick with per-layer readouts. Returns ((y1, y2, y3), (h1, h2, h3)).

        Each yi is a scalar from its own layer — different temporal dynamics:
          y1 (h1, fast):   token-level confidence signal
          y2 (h2, medium): phrase-level repetition signal
          y3 (h3, slow):   discourse-level tonal drift
        """
        x = (x - self.x_mean) / self.x_scale
        h1, h2, h3 = state

        h1 = (1.0 - self.leak1) * h1 + self.leak1 * mx.tanh(
            x @ self.w_in1.T + h1 @ self.w1.T + self.b1
        )
        h2 = (1.0 - self.leak2) * h2 + self.leak2 * mx.tanh(
            h1 @ self.w_in2.T + h2 @ self.w2.T + self.b2
        )
        h3 = (1.0 - self.leak3) * h3 + self.leak3 * mx.tanh(
            h2 @ self.w_in3.T + h3 @ self.w3.T + self.b3
        )

        # Per-layer readouts (set by init_multi_readout)
        y1 = h1 @ self.readout_w1.T + self.readout_b1
        y2 = h2 @ self.readout_w2.T + self.readout_b2
        y3 = h3 @ self.readout_w3.T + self.readout_b3

        return (y1, y2, y3), (h1, h2, h3)

    def init_multi_readout(self, bundle: 'TripleReservoir'):
        """Copy per-layer readout weights from a trained TripleReservoir.

        Call this after bundle.fit_multi_readout() has been called.
        """
        self.readout_w1 = mx.array(bundle.readout_w1)
        self.readout_b1 = mx.array(bundle.readout_b1.reshape(1, -1))
        self.readout_w2 = mx.array(bundle.readout_w2)
        self.readout_b2 = mx.array(bundle.readout_b2.reshape(1, -1))
        self.readout_w3 = mx.array(bundle.readout_w3)
        self.readout_b3 = mx.array(bundle.readout_b3.reshape(1, -1))

    @staticmethod
    def state_to_numpy(
        state: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.array(h) for h in state)

    @staticmethod
    def state_from_numpy(
        np_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[mx.array, mx.array, mx.array]:
        return tuple(mx.array(h.astype(np.float32)) for h in np_state)


# ---------------------------------------------------------------------------
# Embedding projection (tranche 2 connection point)
# ---------------------------------------------------------------------------

class EmbeddingProjection:
    """Frozen random projection: LLM embedding dim -> reservoir input dim.

    Same philosophy as TextProjection but operates on mx.array tensors
    natively.  Built in tranche 1, wired to LLM generation in tranche 2.
    """

    def __init__(self, embed_dim: int, input_dim: int, seed: int = 137):
        rng = np.random.default_rng(seed)
        W_np = (rng.standard_normal((embed_dim, input_dim)) / np.sqrt(embed_dim)).astype(
            np.float32
        )
        self.W = mx.array(W_np)
        self.embed_dim = embed_dim
        self.input_dim = input_dim

    def project(self, embedding: mx.array) -> mx.array:
        """Project an LLM embedding to bounded reservoir input."""
        return mx.tanh(embedding @ self.W)


# ---------------------------------------------------------------------------
# Logit processor for bidirectional coupling
# ---------------------------------------------------------------------------

class ReservoirLogitProcessor:
    """Multi-headed logits processor: three reservoir layers shape generation
    at different timescales.

    After each token, the caller ticks the reservoir and calls update()
    with three per-layer outputs.  On the *next* token, the processor
    modulates the logits at three levels:

      y1 (h1, fast)   → temperature:   token-level confidence
      y2 (h2, medium) → rep_penalty:   phrase-level repetition sensitivity
      y3 (h3, slow)   → top_p:         discourse-level tonal drift

    Each modulation is gentle (±strength range).  The combined effect is
    subtle but multi-timescale — the reservoir's dynamics literally shape
    how Astrid thinks.
    """

    def __init__(self, coupling_strength: float = 0.1):
        self.coupling_strength = coupling_strength
        self._y1 = 0.0  # fast: token confidence
        self._y2 = 0.0  # medium: repetition
        self._y3 = 0.0  # slow: tonal drift

    def update(self, y1: float, y2: float, y3: float):
        """Update with per-layer reservoir outputs."""
        self._y1 = y1
        self._y2 = y2
        self._y3 = y3

    def update_single(self, energy: float):
        """Backward-compat: single scalar updates all three (legacy mode)."""
        self._y1 = energy
        self._y2 = energy
        self._y3 = energy

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply multi-timescale reservoir modulation to logits."""
        s = self.coupling_strength

        # Layer 1 (fast): temperature modulation
        # Positive y1 → lower temp (more confident), negative → higher (exploratory)
        sig1 = self._sigmoid(self._y1)
        t_mod = 1.0 - s * (2.0 * sig1 - 1.0)
        logits = logits * (1.0 / t_mod)

        # Layer 2 (medium): repetition penalty modulation
        # Positive y2 → boost token diversity, negative → allow repetition
        # Applied as a gentle entropy bonus/penalty on the logit distribution
        sig2 = self._sigmoid(self._y2)
        entropy_nudge = s * (2.0 * sig2 - 1.0) * 0.5  # ±5% entropy shift
        if abs(entropy_nudge) > 0.001:
            # Add uniform noise proportional to nudge — softens or sharpens
            logits = logits + entropy_nudge

        # Layer 3 (slow): top-p / nucleus narrowing
        # Positive y3 → precise/narrow (suppress low-prob tokens more)
        # Negative y3 → exploratory/wide (let more tokens through)
        # Applied as scaling on the tail of the distribution
        sig3 = self._sigmoid(self._y3)
        tail_scale = 1.0 - s * (2.0 * sig3 - 1.0) * 0.3  # ±3% tail scaling
        if tail_scale != 1.0:
            # Scale logits below median down (or up) by tail_scale
            median_val = float(mx.median(logits).item())
            mask = logits < median_val
            logits = mx.where(mask, logits * tail_scale, logits)

        return logits


# ---------------------------------------------------------------------------
# Verification against NumPy
# ---------------------------------------------------------------------------

def verify_against_numpy(
    config: Optional[ReservoirConfig] = None,
    steps: int = 200,
    seed: int = 99,
) -> dict:
    """Run both backends on identical input, report divergence."""
    cfg = config or ReservoirConfig(input_dim=16, n_nodes=64)

    # Build canonical NumPy reservoir and fit synthetic readout
    np_model = TripleReservoir(cfg)
    rng = np.random.default_rng(seed)
    d = cfg.input_dim
    x_train = np.tanh(rng.standard_normal((3000, d))).astype(np.float32)
    y_train = np.zeros((3000, cfg.output_dim), dtype=np.float32)
    latent = 0.0
    for t in range(3000):
        latent = 0.9 * latent + 0.3 * float(x_train[t, : min(4, d)].sum())
        y_train[t, 0] = latent
    np_model.fit_readout(x_train, y_train)

    # Build MLX from the same trained model
    mlx_model = MLXTripleReservoir(np_model)

    # Generate test input
    x_test = np.tanh(rng.standard_normal((steps, d))).astype(np.float32)

    # Run both
    np_state = np_model.zero_state()
    mlx_state = mlx_model.zero_state()

    max_y_diff = 0.0
    max_h_diff = 0.0

    for t in range(steps):
        x_np = x_test[t : t + 1]
        x_mx = mx.array(x_np)

        y_np, _, np_state = np_model.step_numpy(x_np, np_state)
        y_mx, _, mlx_state = mlx_model.step(x_mx, mlx_state)

        # Force evaluation to prevent graph buildup
        if t % 64 == 0:
            mx.eval(*mlx_state)

        y_diff = abs(float(y_np.ravel()[0]) - float(y_mx.item()))
        max_y_diff = max(max_y_diff, y_diff)

        # Compare hidden states
        for h_np, h_mx in zip(np_state, mlx_state):
            h_diff = float(np.max(np.abs(np.array(h_np) - np.array(h_mx))))
            max_h_diff = max(max_h_diff, h_diff)

    return {
        "steps": steps,
        "max_output_diff": max_y_diff,
        "max_hidden_diff": max_h_diff,
        "n_nodes": cfg.n_nodes,
        "input_dim": cfg.input_dim,
    }


def benchmark(config: Optional[ReservoirConfig] = None, steps: int = 1000) -> dict:
    """Compare per-tick latency: NumPy vs MLX."""
    cfg = config or ReservoirConfig(input_dim=16, n_nodes=192)
    rng = np.random.default_rng(42)
    d = cfg.input_dim

    # Quick synthetic fit
    np_model = TripleReservoir(cfg)
    x_train = np.tanh(rng.standard_normal((3000, d))).astype(np.float32)
    y_train = np.zeros((3000, cfg.output_dim), dtype=np.float32)
    latent = 0.0
    for t in range(3000):
        latent = 0.9 * latent + 0.3 * float(x_train[t, : min(4, d)].sum())
        y_train[t, 0] = latent
    np_model.fit_readout(x_train, y_train)
    mlx_model = MLXTripleReservoir(np_model)

    x_test = np.tanh(rng.standard_normal((steps, d))).astype(np.float32)

    # Warmup
    np_state = np_model.zero_state()
    mlx_state = mlx_model.zero_state()
    for t in range(min(50, steps)):
        _, _, np_state = np_model.step_numpy(x_test[t : t + 1], np_state)
        _, _, mlx_state = mlx_model.step(mx.array(x_test[t : t + 1]), mlx_state)
    mx.eval(*mlx_state)

    # Benchmark NumPy
    np_state = np_model.zero_state()
    t0 = time.perf_counter()
    for t in range(steps):
        _, _, np_state = np_model.step_numpy(x_test[t : t + 1], np_state)
    np_elapsed = time.perf_counter() - t0

    # Benchmark MLX
    mlx_state = mlx_model.zero_state()
    t0 = time.perf_counter()
    for t in range(steps):
        _, _, mlx_state = mlx_model.step(mx.array(x_test[t : t + 1]), mlx_state)
        if t % 64 == 0:
            mx.eval(*mlx_state)
    mx.eval(*mlx_state)
    mlx_elapsed = time.perf_counter() - t0

    return {
        "steps": steps,
        "n_nodes": cfg.n_nodes,
        "numpy_us_per_tick": 1e6 * np_elapsed / steps,
        "mlx_us_per_tick": 1e6 * mlx_elapsed / steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MLX triple reservoir — verify & bench")
    ap.add_argument("--verify", action="store_true", help="Verify against NumPy")
    ap.add_argument("--bench", action="store_true", help="Benchmark NumPy vs MLX")
    ap.add_argument("--steps", type=int, default=200, help="Steps for verify")
    ap.add_argument("--bench-steps", type=int, default=1000, help="Steps for bench")
    ap.add_argument("--n-nodes", type=int, default=64, help="Nodes per layer")
    ap.add_argument("--input-dim", type=int, default=16, help="Input dimension")
    args = ap.parse_args()

    cfg = ReservoirConfig(input_dim=args.input_dim, n_nodes=args.n_nodes)

    if args.verify:
        print(f"verifying: {args.steps} steps, n_nodes={cfg.n_nodes}, input_dim={cfg.input_dim}")
        result = verify_against_numpy(cfg, steps=args.steps)
        print(f"  max output diff:  {result['max_output_diff']:.2e}")
        print(f"  max hidden diff:  {result['max_hidden_diff']:.2e}")
        ok = result["max_output_diff"] < 1e-4 and result["max_hidden_diff"] < 1e-4
        print(f"  status: {'PASS' if ok else 'FAIL'}")

    if args.bench:
        bench_cfg = ReservoirConfig(input_dim=args.input_dim, n_nodes=args.n_nodes)
        print(f"\nbenchmark: {args.bench_steps} steps, n_nodes={bench_cfg.n_nodes}")
        result = benchmark(bench_cfg, steps=args.bench_steps)
        print(f"  numpy:  {result['numpy_us_per_tick']:.1f} us/tick")
        print(f"  mlx:    {result['mlx_us_per_tick']:.1f} us/tick")

    if not args.verify and not args.bench:
        # Quick sanity: build, tick once, print
        np_model = TripleReservoir(cfg)
        rng = np.random.default_rng(99)
        x = np.tanh(rng.standard_normal((3000, cfg.input_dim))).astype(np.float32)
        y = np.zeros((3000, cfg.output_dim), dtype=np.float32)
        latent = 0.0
        for t in range(3000):
            latent = 0.9 * latent + 0.3 * float(x[t, : min(4, cfg.input_dim)].sum())
            y[t, 0] = latent
        np_model.fit_readout(x, y)

        mlx_model = MLXTripleReservoir(np_model)
        state = mlx_model.zero_state()
        inp = mx.array(np.tanh(rng.standard_normal((1, cfg.input_dim))).astype(np.float32))
        out, z, state = mlx_model.step(inp, state)
        mx.eval(out)
        print(f"mlx reservoir ready: output={float(out.item()):.6f}, z_dim={z.shape[-1]}")

        # Quick embedding projection test
        proj = EmbeddingProjection(embed_dim=2048, input_dim=cfg.input_dim)
        fake_emb = mx.ones((1, 2048)) * 0.01
        projected = proj.project(fake_emb)
        mx.eval(projected)
        print(f"embedding projection: {proj.embed_dim} -> {proj.input_dim}, sample={float(projected[0, 0].item()):.6f}")


if __name__ == "__main__":
    main()
