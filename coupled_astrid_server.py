#!/usr/bin/env python3
"""
coupled_astrid_server.py -- Drop-in replacement for mlx_lm.server on port 8090.

Loads an MLX model (default: Qwen3-8B-4bit) via mlx_lm.load(), runs
bidirectional coupled generation where the triple reservoir's dynamical state
modulates Astrid's logits at every token, and each token's embedding feeds
back into the reservoir. OpenAI-compatible /v1/chat/completions endpoint.

The bridge doesn't know or care that coupling is happening — it sends the
same JSON request and gets the same JSON response. The reservoir's influence
is woven into the generation process itself.

Usage:
    python coupled_astrid_server.py [--port 8090] [--coupling-strength 0.1]

Replaces:
    python -m mlx_lm.server --model mlx-community/gemma-4-12B-it-5bit --port 8090
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from coupled_http_gateway import AiohttpGateway, ModelRuntimeCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [coupled-astrid] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coupled-astrid")

# Reservoir service for state sync
RESERVOIR_WS_URL = "ws://127.0.0.1:7881"

GENERATION_STOP_SPECIALS = (
    "<turn|>",
    "<|turn>",
    "<channel|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<end_of_turn>",
    "<|eot_id|>",
    "</s>",
)
GENERATION_SKIP_SPECIALS = (
    "<|channel>",
    "<|think|>",
    "<|tool_call>",
    "<|tool>",
    "<|tool_response>",
    "<|image>",
    "<|audio>",
    "<|video|>",
    "<bos>",
    "<pad>",
    "<unk>",
)
GENERATED_TEXT_ARTIFACTS = (
    "<start_of_turn>",
    "<end_of_turn>",
    "<think>",
    "</think>",
    "/no_think",
    "<|im_start|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<eos>",
    "<bos>",
    "<pad>",
    "<unk>",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _runtime_identity() -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": sys.executable,
        "has_mmap_stats": bool(hasattr(mx, "last_mmap_load_stats")),
    }
    try:
        runtime["mlx_core_path"] = inspect.getfile(mx)
    except Exception as exc:  # pragma: no cover - defensive
        runtime["mlx_core_path_error"] = str(exc)
    try:
        import mlx_lm

        runtime["mlx_lm_path"] = inspect.getfile(mlx_lm)
    except Exception as exc:  # pragma: no cover - defensive
        runtime["mlx_lm_path_error"] = str(exc)
    return runtime


def _single_token_id(tokenizer, text: str) -> int | None:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return None
    if len(ids) != 1:
        return None
    return int(ids[0])


def _tokenizer_id_set(tokenizer, attr: str) -> set[int]:
    try:
        value = getattr(tokenizer, attr)
    except Exception:
        return set()
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    try:
        return {int(item) for item in value if item is not None}
    except TypeError:
        return set()


def _build_generation_token_policy(tokenizer) -> tuple[set[int], set[int]]:
    stop_ids: set[int] = set()
    skip_ids: set[int] = set()

    stop_ids.update(_tokenizer_id_set(tokenizer, "eos_token_ids"))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.add(int(eos_id))

    for special in GENERATION_STOP_SPECIALS:
        token_id = _single_token_id(tokenizer, special)
        if token_id is not None:
            stop_ids.add(token_id)

    skip_ids.update(_tokenizer_id_set(tokenizer, "all_special_ids"))
    for special in GENERATION_SKIP_SPECIALS:
        token_id = _single_token_id(tokenizer, special)
        if token_id is not None:
            skip_ids.add(token_id)

    skip_ids.difference_update(stop_ids)
    return stop_ids, skip_ids


def _token_to_int(token) -> int:
    try:
        return int(token.item())
    except Exception:
        pass
    try:
        return int(token)
    except Exception:
        return int(np.array(token).item())


def _clean_generated_text(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.I)
    cleaned = re.sub(
        r"\b(?:thought|analysis|final)\s*<channel\|>",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"<\|channel>\s*(?:thought|analysis|final)?\s*",
        "",
        cleaned,
        flags=re.I,
    )
    for artifact in GENERATED_TEXT_ARTIFACTS:
        cleaned = cleaned.replace(artifact, "")
    lines = [
        line
        for line in cleaned.splitlines()
        if line.strip().lower() not in {"thought", "analysis", "final"}
    ]
    return "\n".join(lines).strip()


def _merge_count_map(target: dict[str, int], source: dict[str, object] | None) -> None:
    for key, value in dict(source or {}).items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)


def _gemma4_unified_config_override(model_name: str) -> dict[str, object] | None:
    """Map Gemma 4 unified checkpoints onto mlx-lm's text Gemma 4 class.

    mlx-lm 0.31.x ships `mlx_lm.models.gemma4`, but MLX Community Gemma 4 12B
    checkpoints currently declare `model_type=gemma4_unified` because their
    config includes text, vision, and audio metadata. The coupled Astrid server
    only uses the language model and token embeddings, so the text class is the
    appropriate narrow canary target.
    """
    try:
        config_path = Path(model_name).expanduser() / "config.json"
        if not config_path.exists():
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=model_name, filename="config.json")
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.info("model config probe skipped for %s: %s", model_name, exc)
        return None

    model_type = str(config.get("model_type") or "")
    if not model_type.startswith("gemma4_unified"):
        return None

    _install_gemma4_unified_sanitize_patch()
    log.info(
        "model config override: %s declares %s; loading language lane via mlx_lm.models.gemma4",
        model_name,
        model_type,
    )
    return {"model_type": "gemma4"}


def _install_gemma4_unified_sanitize_patch() -> None:
    """Drop unified-modal tensors that mlx-lm's Gemma 4 text class does not use."""
    try:
        from mlx_lm.models import gemma4
    except Exception as exc:
        log.info("gemma4 unified sanitize patch unavailable: %s", exc)
        return

    if getattr(gemma4.Model, "_astrid_gemma4_unified_patch", False):
        return

    original_sanitize = gemma4.Model.sanitize

    def sanitize_language_only(self, weights):
        filtered = {
            key: value
            for key, value in weights.items()
            if not key.startswith(("vision_embedder.", "audio_embedder."))
        }
        return original_sanitize(self, filtered)

    gemma4.Model.sanitize = sanitize_language_only
    gemma4.Model._astrid_gemma4_unified_patch = True


def _load_mlx_runtime(
    model_name: str,
    *,
    memory_map_requested: bool,
) -> tuple[object, object, float, dict[str, object]]:
    from mlx_lm import load as mlx_load

    runtime = _runtime_identity()
    load_audit: dict[str, object] = {
        "memory_map_requested": bool(memory_map_requested),
        "memory_map_effective": bool(
            memory_map_requested and runtime.get("has_mmap_stats")
        ),
        "can_report_mapped_vs_copied": bool(runtime.get("has_mmap_stats")),
        "mapped_bytes": 0,
        "copied_bytes": 0,
        "fallback_reasons": {},
        "fallback_reason_bytes": {},
        "mx_load_calls": 0,
        "mx_load_wall_seconds": 0.0,
        "model_config_override": None,
    }
    model_config_override = _gemma4_unified_config_override(model_name)
    load_audit["model_config_override"] = model_config_override
    original_mx_load = mx.load

    def audited_mx_load(file, *args, **kwargs):
        if load_audit["memory_map_effective"] and isinstance(
            file, (str, os.PathLike, Path)
        ):
            kwargs.setdefault("memory_map", True)
        call_start = time.perf_counter()
        result = original_mx_load(file, *args, **kwargs)
        load_audit["mx_load_calls"] = int(load_audit["mx_load_calls"]) + 1
        load_audit["mx_load_wall_seconds"] = float(
            load_audit["mx_load_wall_seconds"]
        ) + (time.perf_counter() - call_start)
        if runtime.get("has_mmap_stats"):
            try:
                stats = mx.last_mmap_load_stats(clear=True)
            except Exception as exc:  # pragma: no cover - defensive
                load_audit["last_mmap_load_stats_error"] = str(exc)
                stats = None
            if stats:
                load_audit["mapped_bytes"] = int(load_audit["mapped_bytes"]) + int(
                    stats.get("mapped_bytes") or 0
                )
                load_audit["copied_bytes"] = int(load_audit["copied_bytes"]) + int(
                    stats.get("copied_bytes") or 0
                )
                _merge_count_map(
                    load_audit["fallback_reasons"], stats.get("fallback_reasons")
                )
                _merge_count_map(
                    load_audit["fallback_reason_bytes"],
                    stats.get("fallback_reason_bytes"),
                )
        return result

    mx.load = audited_mx_load
    try:
        if runtime.get("has_mmap_stats"):
            try:
                mx.last_mmap_load_stats(clear=True)
            except Exception:
                pass
        load_start = time.perf_counter()
        if model_config_override is None:
            model, tokenizer = mlx_load(model_name)
        else:
            model, tokenizer = mlx_load(
                model_name,
                model_config=model_config_override,
            )
        load_seconds = time.perf_counter() - load_start
    finally:
        mx.load = original_mx_load

    total_bytes = int(load_audit["mapped_bytes"]) + int(load_audit["copied_bytes"])
    load_audit["load_seconds"] = load_seconds
    load_audit["mapped_ratio_pct"] = (
        100.0 * int(load_audit["mapped_bytes"]) / total_bytes if total_bytes else 0.0
    )
    return model, tokenizer, load_seconds, {"runtime": runtime, "load": load_audit}


def _new_request_audit(
    *,
    handle_name: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    runtime_audit: dict[str, object],
) -> dict[str, object]:
    started_at = time.time()
    return {
        "request_id": f"{int(started_at * 1000)}-{handle_name}",
        "timestamp": round(started_at, 3),
        "request": {
            "handle_name": handle_name,
            "messages_count": len(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        "runtime": dict(runtime_audit.get("runtime", {}) or {}),
        "load": dict(runtime_audit.get("load", {}) or {}),
        "generation": {
            "first_token_s": None,
            "steady_tok_s": None,
            "total_turn_s": None,
            "generated_tokens": 0,
            "decode_tok_s": None,
            "host_sync_points": {"total": 0, "by_kind": {}},
            "host_sync_s": 0.0,
            "host_sync_breakdown_s": {},
        },
        "reservoir": {
            "pull_bytes": 0,
            "push_bytes": 0,
            "serialization_s": 0.0,
            "roundtrip_s": 0.0,
        },
    }


def _record_host_sync(
    audit: dict[str, object] | None,
    kind: str,
    elapsed_s: float,
    *,
    count: int = 1,
) -> None:
    if audit is None:
        return
    generation = dict(audit.get("generation", {}) or {})
    points = dict(generation.get("host_sync_points", {}) or {})
    by_kind = dict(points.get("by_kind", {}) or {})
    by_kind[kind] = int(by_kind.get(kind, 0)) + count
    points["by_kind"] = by_kind
    points["total"] = int(points.get("total", 0)) + count
    generation["host_sync_points"] = points
    generation["host_sync_s"] = float(generation.get("host_sync_s", 0.0)) + float(
        elapsed_s
    )
    breakdown = dict(generation.get("host_sync_breakdown_s", {}) or {})
    breakdown[kind] = float(breakdown.get(kind, 0.0)) + float(elapsed_s)
    generation["host_sync_breakdown_s"] = breakdown
    audit["generation"] = generation


def _timed_float_item(
    value: object,
    audit: dict[str, object] | None,
    kind: str,
) -> float:
    sync_start = time.perf_counter()
    out = float(value.item())
    _record_host_sync(audit, kind, time.perf_counter() - sync_start)
    return out


def _timed_state_to_numpy(
    reservoir,
    state,
    audit: dict[str, object] | None,
    *,
    kind: str = "state_to_numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sync_start = time.perf_counter()
    arrays = reservoir.state_to_numpy(state)
    _record_host_sync(audit, kind, time.perf_counter() - sync_start, count=len(arrays))
    return arrays


def _timed_state_from_numpy(
    reservoir,
    np_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    audit: dict[str, object] | None,
    *,
    kind: str = "state_from_numpy",
):
    sync_start = time.perf_counter()
    state = reservoir.state_from_numpy(np_state)
    _record_host_sync(
        audit, kind, time.perf_counter() - sync_start, count=len(np_state)
    )
    return state


def _timed_np_array(
    value: object,
    audit: dict[str, object] | None,
    *,
    kind: str = "np_array",
) -> np.ndarray:
    sync_start = time.perf_counter()
    arr = np.array(value)
    _record_host_sync(audit, kind, time.perf_counter() - sync_start)
    return arr


class CoupledAstridServer:
    """OpenAI-compatible server with bidirectional reservoir coupling."""

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-4-12B-it-5bit",
        coupling_strength: float = 0.15,  # 0.1→0.15: AGC adjusts dynamically (0.02-0.30), but higher baseline means perturbations are felt sooner
        input_dim: int = 32,
        n_nodes: int = 192,
        model_memory_map: bool = False,
        audit_dir: str | None = None,
        wide_coupling_strength: float = 0.0,
        wide_coupling_rank: int = 16,
        wide_bias_cap: float = 4.0,
        wide_pressure_floor: float = 0.25,
    ):
        from mlx_reservoir import EmbeddingProjection, MLXTripleReservoir
        from triple_reservoir_coreml import build_canonical_reservoir

        # Load LLM
        log.info("loading model: %s", model_name)
        (
            self.model,
            self.tokenizer,
            self.load_seconds,
            self.runtime_audit,
        ) = _load_mlx_runtime(
            model_name,
            memory_map_requested=model_memory_map,
        )
        runtime = dict(self.runtime_audit.get("runtime", {}) or {})
        load = dict(self.runtime_audit.get("load", {}) or {})
        log.info(
            "runtime: python=%s mlx=%s mmap_stats=%s memory_map_requested=%s mapped=%.2fMiB copied=%.2fMiB load=%.2fs",
            runtime.get("python"),
            runtime.get("mlx_core_path"),
            runtime.get("has_mmap_stats"),
            load.get("memory_map_requested"),
            int(load.get("mapped_bytes") or 0) / (1024**2),
            int(load.get("copied_bytes") or 0) / (1024**2),
            float(load.get("load_seconds") or 0.0),
        )

        # Get hidden size — varies by model architecture
        # Gemma-3: text_config dict; Llama/Mistral: model.args.hidden_size
        if hasattr(self.model.args, "hidden_size"):
            self.embed_dim = self.model.args.hidden_size
        elif hasattr(self.model.args, "text_config") and isinstance(self.model.args.text_config, dict):
            self.embed_dim = self.model.args.text_config["hidden_size"]
        else:
            raise ValueError("cannot determine hidden_size from model args")

        # Find embed_tokens — Gemma-3: language_model.model.embed_tokens
        # Llama/Mistral: model.model.embed_tokens
        if hasattr(self.model, "language_model"):
            self._embed_tokens = self.model.language_model.model.embed_tokens
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            self._embed_tokens = self.model.model.embed_tokens
        else:
            raise ValueError("cannot find embed_tokens layer in model")

        log.info("model loaded: hidden_size=%d", self.embed_dim)

        # Build generation token policy. Stop tokens end the response before
        # detokenization; skip tokens are non-content channel/tool sentinels
        # that should not tick the coupled reservoir.
        self._stop_token_ids, self._skip_token_ids = _build_generation_token_policy(
            self.tokenizer
        )
        log.info(
            "generation token policy: stop=%s skip=%s",
            sorted(self._stop_token_ids),
            sorted(self._skip_token_ids),
        )

        # Build reservoir from canonical seed — must match reservoir_service
        np_model, cfg = build_canonical_reservoir(input_dim=input_dim, n_nodes=n_nodes)

        self.reservoir = MLXTripleReservoir(np_model)
        self.reservoir.init_multi_readout(np_model)
        self.state = self.reservoir.zero_state()  # fallback if service unavailable
        self.n_nodes = n_nodes
        self.tick_count = 0

        # Embedding projection: LLM hidden_size → reservoir input_dim
        self.embed_proj = EmbeddingProjection(
            embed_dim=self.embed_dim, input_dim=input_dim
        )
        self.coupling_strength = coupling_strength
        self.input_dim = input_dim
        self._multi_head = True  # use per-layer readouts

        # Wide coupling (y4): embedding-tied low-rank vocab-bias matrices, built
        # once at load. Default-off (ceiling 0) → not built, server unchanged.
        self.wide_coupling_strength = max(0.0, wide_coupling_strength)
        self.wide_bias_cap = wide_bias_cap
        self.wide_pressure_floor = wide_pressure_floor
        self._wide_P = None
        self._wide_V = None
        if self.wide_coupling_strength > 0.0:
            try:
                import wide_coupling as _wc

                m_args = getattr(self.model, "args", None)
                vocab = None
                if m_args is not None:
                    if getattr(m_args, "vocab_size", None):
                        vocab = int(m_args.vocab_size)
                    elif isinstance(getattr(m_args, "text_config", None), dict):
                        vocab = int(m_args.text_config.get("vocab_size") or 0) or None
                if vocab is None:
                    vocab = int(getattr(self.tokenizer, "vocab_size", 262144))
                self._wide_P, self._wide_V, _info = _wc.build_wide_coupling_matrices(
                    self._embed_tokens, vocab, k=int(wide_coupling_rank)
                )
                log.info(
                    "wide coupling built: k=%d ceiling=%.3f cap=%.2f floor=%.2f vocab=%d",
                    int(wide_coupling_rank), self.wide_coupling_strength,
                    self.wide_bias_cap, self.wide_pressure_floor, vocab,
                )
            except Exception as exc:
                log.warning("wide coupling build failed: %s; wide channel disabled", exc)
                self._wide_P = self._wide_V = None
        configured_audit_dir = audit_dir or os.environ.get("COUPLED_ASTRID_AUDIT_DIR")
        self._audit_dir = Path(configured_audit_dir).expanduser() if configured_audit_dir else None
        self._audit_metrics_path = (
            self._audit_dir / "coupled_request_metrics.jsonl"
            if self._audit_dir is not None
            else None
        )

        # --- Coupling journal: persistent record of how the reservoir
        # modulates Astrid's generation. Each entry records the dynamical
        # state before/after generation and the modulation applied.
        # On restart, the last entry calibrates the coupling so the first
        # generation isn't flying blind. ---
        self._journal_path = Path("state/coupling_journal.json")
        self._journal: list[dict] = []
        self._y_variance_window: list[tuple[float, float, float]] = []
        self._initial_coupling = coupling_strength  # remember CLI value for logging
        self._restore_coupling_state()

        log.info(
            "reservoir: MLX, %d nodes×3, input=%dD, coupling=%.2f, embed=%d→%d, multi_head=%s",
            n_nodes, input_dim, coupling_strength, self.embed_dim, input_dim, self._multi_head,
        )

        # Pull initial state from service
        self._reservoir_health_snapshot: dict[str, object] = {
            "status": "unknown",
            "operation": "startup",
            "last_sync_unix_ms": None,
        }
        self._pull_state()

    def _set_reservoir_health(
        self,
        *,
        connected: bool,
        operation: str,
        detail: str | None = None,
    ) -> None:
        snapshot: dict[str, object] = {
            "status": "connected" if connected else "degraded",
            "operation": operation,
            "last_sync_unix_ms": int(time.time() * 1000),
        }
        if detail:
            snapshot["detail"] = detail
        self._reservoir_health_snapshot = snapshot

    def health_snapshot(self) -> dict[str, object]:
        """Return cached reservoir health without performing network or MLX work."""
        return dict(self._reservoir_health_snapshot)

    def _create_handle(self, handle_name: str = "astrid") -> bool:
        """Create a named handle on the reservoir service.

        Called automatically when pull/push discovers the handle is missing
        (e.g. after a reservoir service restart).
        """
        try:
            import websockets.sync.client as ws_sync
            with ws_sync.connect(RESERVOIR_WS_URL, open_timeout=2) as ws:
                ws.send(json.dumps({
                    "type": "create_handle",
                    "name": handle_name,
                    "entity": handle_name,
                }))
                response_text = ws.recv(timeout=5)
                r = json.loads(response_text)
                if r.get("type") == "error":
                    log.warning("create_handle failed: %s", r.get("message"))
                    return False
                log.info("create_handle: '%s' created (ticks=%s)", handle_name, r.get("ticks"))
                return True
        except Exception as e:
            log.warning("create_handle failed for '%s': %s", handle_name, e)
            return False

    def _pull_state(
        self,
        handle_name: str = "astrid",
        audit: dict[str, object] | None = None,
        _bootstrap_retry: bool = True,
    ) -> bool:
        """Check out the full handle state from the reservoir service.

        Returns True if state was successfully loaded. On failure, keeps
        the current local state (zero or previous generation).
        """
        import base64

        try:
            import websockets.sync.client as ws_sync

            serialization_start = time.perf_counter()
            request_text = json.dumps({"type": "pull_state", "name": handle_name})
            rpc_start = time.monotonic()
            with ws_sync.connect(RESERVOIR_WS_URL, open_timeout=2) as ws:
                ws.send(request_text)
                response_text = ws.recv(timeout=5)
            roundtrip_s = time.perf_counter() - (serialization_start + (time.perf_counter() - time.perf_counter()))
            rpc_elapsed = time.monotonic() - rpc_start
            if rpc_elapsed > 1.0:
                log.warning("pull_state: slow RPC (%.1fs)", rpc_elapsed)
            roundtrip_s = rpc_elapsed

            r = json.loads(response_text)
            if audit is not None:
                reservoir = dict(audit.get("reservoir", {}) or {})
                reservoir["pull_bytes"] = int(reservoir.get("pull_bytes", 0)) + len(
                    response_text.encode("utf-8")
                )
                reservoir["roundtrip_s"] = float(reservoir.get("roundtrip_s", 0.0)) + roundtrip_s
                audit["reservoir"] = reservoir
            if r.get("type") == "error":
                msg = r.get("message", "")
                if "not found" in msg and _bootstrap_retry:
                    log.info("pull_state: '%s' handle missing — creating", handle_name)
                    if self._create_handle(handle_name):
                        return self._pull_state(handle_name, audit, _bootstrap_retry=False)
                log.info("pull_state: no '%s' handle yet — using local state", handle_name)
                self._set_reservoir_health(
                    connected=False,
                    operation="pull_state",
                    detail=msg or "reservoir handle unavailable",
                )
                if audit is not None:
                    reservoir = dict(audit.get("reservoir", {}) or {})
                    reservoir["serialization_s"] = float(
                        reservoir.get("serialization_s", 0.0)
                    ) + (time.perf_counter() - serialization_start)
                    audit["reservoir"] = reservoir
                return False
            n = r.get("n_nodes", self.n_nodes)
            h1 = np.frombuffer(base64.b64decode(r["h1"]), dtype=np.float32).reshape(1, n)
            h2 = np.frombuffer(base64.b64decode(r["h2"]), dtype=np.float32).reshape(1, n)
            h3 = np.frombuffer(base64.b64decode(r["h3"]), dtype=np.float32).reshape(1, n)
            self.state = _timed_state_from_numpy(self.reservoir, (h1, h2, h3), audit)
            if audit is not None:
                reservoir = dict(audit.get("reservoir", {}) or {})
                reservoir["serialization_s"] = float(
                    reservoir.get("serialization_s", 0.0)
                ) + (time.perf_counter() - serialization_start)
                audit["reservoir"] = reservoir
            self.tick_count = r.get("tick_count", 0)
            norms = [float(np.linalg.norm(h)) for h in (h1, h2, h3)]
            log.info(
                "pull_state: checked out %s (ticks=%d, h_norms=[%.3f, %.3f, %.3f])",
                handle_name, self.tick_count, *norms,
            )
            self._set_reservoir_health(connected=True, operation="pull_state")
            return True
        except Exception as e:
            log.warning("pull_state failed for '%s': %s — using local state", handle_name, e)
            self._set_reservoir_health(
                connected=False,
                operation="pull_state",
                detail=str(e),
            )
            return False

    @staticmethod
    def _compact_preview(text: str, limit: int = 120) -> str:
        """Whitespace-collapse and truncate text for compact provenance."""
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _text_digest(text: str) -> str:
        """Small stable digest for provenance without storing full text."""
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]

    def _build_push_meta(
        self,
        *,
        handle_name: str,
        messages: list[dict],
        prompt_text: str,
        response_text: str,
        ticks: int,
        elapsed: float,
        temperature: float,
        max_tokens: int,
        coupling_used: float,
        y1: float,
        y2: float,
        y3: float,
        h_norms_before: list[float],
        h_norms_after: list[float],
    ) -> dict:
        """Build a compact narrative digest for Astrid's coupled check-in."""
        tok_per_sec = ticks / elapsed if elapsed > 0 else 0.0
        return {
            "source": "coupled_astrid_server",
            "operation": "coupled_generation_checkin",
            "handle_name": handle_name,
            "source_timestamp": round(time.time(), 3),
            "generated_tokens": int(ticks),
            "elapsed_s": round(elapsed, 3),
            "tok_per_s": round(tok_per_sec, 2),
            "temperature": round(float(temperature), 3),
            "max_tokens": int(max_tokens),
            "messages_count": len(messages),
            "prompt_chars": len(prompt_text),
            "response_chars": len(response_text),
            "prompt_preview": self._compact_preview(prompt_text),
            "response_preview": self._compact_preview(response_text),
            "prompt_sha256_12": self._text_digest(prompt_text),
            "response_sha256_12": self._text_digest(response_text),
            "coupling_strength": round(float(coupling_used), 4),
            "reservoir_readout": {
                "y1_final": round(float(y1), 6),
                "y2_final": round(float(y2), 6),
                "y3_final": round(float(y3), 6),
            },
            "h_norms_before": [round(float(n), 4) for n in h_norms_before],
            "h_norms_after": [round(float(n), 4) for n in h_norms_after],
        }

    def _push_state(
        self,
        ticks: int = 0,
        last_r_input: "mx.array | None" = None,
        handle_name: str = "astrid",
        meta: dict | None = None,
        audit: dict[str, object] | None = None,
    ):
        """Check in the evolved state back to the reservoir service.

        After generation, the state carries the full dynamical imprint of
        what was said — hundreds of token embeddings worth of trajectory.
        ticks: number of tokens generated (credited to the handle).
        last_r_input: final projected embedding — becomes the rehearsal afterimage.
        """
        import base64
        try:
            import websockets.sync.client as ws_sync

            serialization_start = time.perf_counter()
            h1, h2, h3 = _timed_state_to_numpy(self.reservoir, self.state, audit)
            h1_b64 = base64.b64encode(h1.astype(np.float32).tobytes()).decode()
            h2_b64 = base64.b64encode(h2.astype(np.float32).tobytes()).decode()
            h3_b64 = base64.b64encode(h3.astype(np.float32).tobytes()).decode()
            msg = {
                "type": "push_state",
                "name": handle_name,
                "h1": h1_b64, "h2": h2_b64, "h3": h3_b64,
                "tick_delta": ticks,
            }
            if last_r_input is not None:
                msg["last_input"] = _timed_np_array(last_r_input, audit).ravel().tolist()
            if meta:
                msg["meta"] = meta
            request_text = json.dumps(msg)
            serialization_s = time.perf_counter() - serialization_start
            rpc_start = time.monotonic()
            with ws_sync.connect(RESERVOIR_WS_URL, open_timeout=2) as ws:
                ws.send(request_text)
                response_text = ws.recv(timeout=5)
            roundtrip_s = time.monotonic() - rpc_start
            if roundtrip_s > 1.0:
                log.warning("push_state: slow RPC (%.1fs)", roundtrip_s)
            ack_parse_start = time.perf_counter()
            r = json.loads(response_text)
            serialization_s += time.perf_counter() - ack_parse_start
            if audit is not None:
                reservoir = dict(audit.get("reservoir", {}) or {})
                reservoir["push_bytes"] = int(reservoir.get("push_bytes", 0)) + len(
                    request_text.encode("utf-8")
                )
                reservoir["serialization_s"] = float(
                    reservoir.get("serialization_s", 0.0)
                ) + serialization_s
                reservoir["roundtrip_s"] = float(reservoir.get("roundtrip_s", 0.0)) + roundtrip_s
                audit["reservoir"] = reservoir
            if r.get("ok"):
                norms = r.get("h_norms", [0, 0, 0])
                log.info("push_state: checked in %s (+%d ticks, h_norms=[%.3f, %.3f, %.3f])", handle_name, ticks, *norms)
                self._set_reservoir_health(connected=True, operation="push_state")
            elif r.get("type") == "error" and "not found" in r.get("message", ""):
                log.info("push_state: '%s' handle missing — creating", handle_name)
                if self._create_handle(handle_name):
                    log.info("push_state: handle created, state will sync on next pull")
                self._set_reservoir_health(
                    connected=False,
                    operation="push_state",
                    detail=r.get("message", "reservoir handle unavailable"),
                )
            else:
                log.warning("push_state: %s", r.get("message", "unknown error"))
                self._set_reservoir_health(
                    connected=False,
                    operation="push_state",
                    detail=r.get("message", "unknown reservoir error"),
                )
        except Exception as e:
            log.warning("push_state failed for '%s': %s", handle_name, e)
            self._set_reservoir_health(
                connected=False,
                operation="push_state",
                detail=str(e),
            )

    def _restore_coupling_state(self):
        """Restore coupling journal from previous session.

        The journal records the trajectory of how the reservoir has been
        modulating Astrid's generation — the membrane between symbolic
        cognition and dynamical systems. On restart, the last entry
        provides calibration: what were y1/y2/y3 doing? What coupling
        strength was effective? This prevents the first post-restart
        generation from flying blind.
        """
        try:
            if self._journal_path.exists():
                with open(self._journal_path) as f:
                    data = json.load(f)
                self._journal = data.get("entries", [])[-50:]  # keep last 50
                if self._journal:
                    last = self._journal[-1]
                    self.tick_count = last.get("tick_count", 0)
                    # Restore adapted coupling strength if it drifted
                    if "coupling_strength" in last:
                        self.coupling_strength = last["coupling_strength"]
                    # Warm-start the AGC variance window from journal history
                    for entry in self._journal:
                        y = (entry.get("y1_final", 0), entry.get("y2_final", 0), entry.get("y3_final", 0))
                        self._y_variance_window.append(y)
                    log.info(
                        "restored coupling journal: %d entries, last y=[%.4f, %.4f, %.4f], "
                        "ticks=%d, coupling=%.3f, agc_window=%d",
                        len(self._journal),
                        last.get("y1_final", 0), last.get("y2_final", 0),
                        last.get("y3_final", 0),
                        self.tick_count, self.coupling_strength,
                        len(self._y_variance_window),
                    )
        except Exception as e:
            log.warning("coupling journal restore failed: %s", e)

    def _save_coupling_state(self):
        """Persist coupling journal to disk."""
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "entries": self._journal[-50:],  # rolling window
            }
            with open(self._journal_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning("coupling journal save failed: %s", e)

    def _write_request_audit(self, audit: dict[str, object]) -> None:
        if self._audit_metrics_path is None:
            return
        try:
            self._audit_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("request audit write failed: %s", exc)

    def _log_generation(self, handle_name: str, ticks: int, elapsed: float, y1: float, y2: float, y3: float,
                        h_norms_before: list[float], h_norms_after: list[float]):
        """Record a generation in the coupling journal.

        This is the persistent record of the membrane between language
        and dynamics. Each entry captures: how the reservoir was modulating
        the LLM, how many tokens flowed through, and how the dynamical
        state evolved as a result.
        """
        entry = {
            "timestamp": time.time(),
            "handle_name": handle_name,
            "tick_count": self.tick_count,
            "tokens": ticks,
            "elapsed_s": round(elapsed, 2),
            "coupling_strength": self.coupling_strength,
            "y1_final": round(y1, 6),
            "y2_final": round(y2, 6),
            "y3_final": round(y3, 6),
            "h_norms_before": [round(n, 4) for n in h_norms_before],
            "h_norms_after": [round(n, 4) for n in h_norms_after],
        }
        self._journal.append(entry)
        # Keep rolling window
        if len(self._journal) > 50:
            self._journal = self._journal[-50:]

        # --- Adaptive coupling (AGC) ---
        # Track y-value variance over a sliding window. If the reservoir
        # is quiet (low variance), amplify the coupling so it has more
        # voice. If it's dominating (high variance), attenuate so the
        # LLM retains its own agency. 5% nudge per generation — takes
        # ~20 generations to double, slow enough to be stable.
        self._y_variance_window.append((y1, y2, y3))
        if len(self._y_variance_window) > 30:
            self._y_variance_window = self._y_variance_window[-30:]

        if len(self._y_variance_window) >= 10:
            recent = self._y_variance_window[-20:]
            total_var = sum(
                np.var([e[i] for e in recent]) for i in range(3)
            )
            old_strength = self.coupling_strength
            if total_var < 0.01:
                # Reservoir is quiet — amplify its voice
                self.coupling_strength = min(0.3, self.coupling_strength * 1.05)
            elif total_var > 1.0:
                # Reservoir is dominating — let the LLM breathe
                self.coupling_strength = max(0.02, self.coupling_strength * 0.95)
            if abs(self.coupling_strength - old_strength) > 0.001:
                log.info(
                    "AGC: coupling %.3f → %.3f (y_var=%.4f)",
                    old_strength, self.coupling_strength, total_var,
                )

        # Save after every generation
        self._save_coupling_state()

    def _format_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI message format to a chat-formatted prompt string."""
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Model template doesn't accept enable_thinking (e.g. Gemma)
                try:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: manual ChatML formatting
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def generate_coupled(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 512,
        handle_name: str = "astrid",
        aperture: float = 1.0,
    ) -> str:
        """Run coupled generation with unified reservoir state.

        Before: pull full h1/h2/h3 from the reservoir service (the state
        reflects feeder ticks, cross-feed from minime, rehearsal decay).
        During: each token embedding ticks the reservoir, multi-headed
        outputs modulate logits at three timescales.
        After: push the evolved state back — it now carries the full
        dynamical imprint of what was said.
        """
        from mlx_lm.generate import generate_step, generation_stream
        from mlx_lm.sample_utils import make_sampler
        from mlx_reservoir import ReservoirLogitProcessor

        request_start = time.perf_counter()
        request_audit = _new_request_audit(
            handle_name=handle_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            runtime_audit=self.runtime_audit,
        )

        # CHECK OUT: pull state from service before generation
        self._pull_state(handle_name, audit=request_audit)

        # Capture pre-generation h-norms for the coupling journal
        h_norms_before = [0.0, 0.0, 0.0]
        try:
            h1_np, h2_np, h3_np = _timed_state_to_numpy(
                self.reservoir,
                self.state,
                request_audit,
            )
            h_norms_before = [float(np.linalg.norm(h)) for h in (h1_np, h2_np, h3_np)]
        except Exception:
            pass

        prompt_text = self._format_prompt(messages)
        request_info = dict(request_audit.get("request", {}) or {})
        request_info["prompt_sha256_12"] = self._text_digest(prompt_text)
        request_info["prompt_chars"] = len(prompt_text)
        request_audit["request"] = request_info
        prompt_tokens = mx.array(self.tokenizer.encode(prompt_text))

        sampler = make_sampler(temp=temperature)
        processor = ReservoirLogitProcessor(
            coupling_strength=self.coupling_strength,
            sync_observer=lambda kind, elapsed: _record_host_sync(
                request_audit, kind, elapsed
            ),
            wide_strength=self.wide_coupling_strength,
            wide_cap=self.wide_bias_cap,
            wide_pressure_floor=self.wide_pressure_floor,
            wide_P=self._wide_P,
            wide_V=self._wide_V,
        )
        # Astrid's per-request aperture fraction within the operator ceiling.
        processor.aperture = max(0.0, min(1.0, float(aperture)))

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        ticks = 0
        last_r_input = None
        decode_start = time.monotonic()
        first_token_seconds = None

        for token, _logprobs in generate_step(
            prompt_tokens,
            self.model,
            max_tokens=max_tokens,
            logits_processors=[processor],
            sampler=sampler,
        ):
            now = time.monotonic()
            token_id = _token_to_int(token)
            if token_id in self._stop_token_ids:
                break
            if token_id in self._skip_token_ids:
                continue
            if first_token_seconds is None:
                first_token_seconds = now - decode_start
            detokenizer.add_token(token_id)

            # Reservoir coupling: extract token embedding, project to reservoir
            # input, tick the reservoir.  All ops run on generation_stream to
            # stay serialized with generate_step's own Metal commands.
            with mx.stream(generation_stream):
                embedding = self._embed_tokens(mx.array([[token_id]]))
                embedding = embedding.reshape(1, -1)
                r_input = self.embed_proj.project(embedding)
                last_r_input = r_input

                if self._multi_head:
                    # Multi-headed: three per-layer readouts
                    (y1, y2, y3), self.state = self.reservoir.step_multi(r_input, self.state)
                    if ticks % 64 == 0:
                        mx.eval(*self.state)
                    processor.update(
                        _timed_float_item(y1, request_audit, "scalar_item"),
                        _timed_float_item(y2, request_audit, "scalar_item"),
                        _timed_float_item(y3, request_audit, "scalar_item"),
                    )
                    if processor._wide_V is not None:
                        # y4 (wide): pass the reservoir state (h1|h2|h3) for the
                        # embedding-tied vocab bias. Lazy concat — only evaluated
                        # inside the processor when the wide channel is active.
                        processor.update_state(mx.concatenate(self.state, axis=1))
                else:
                    # Legacy single-headed
                    y, _z, self.state = self.reservoir.step(r_input, self.state)
                    if ticks % 64 == 0:
                        mx.eval(*self.state)
                    processor.update_single(
                        _timed_float_item(y, request_audit, "scalar_item")
                    )

            ticks += 1

        detokenizer.finalize()
        text = _clean_generated_text(detokenizer.text)
        request_info = dict(request_audit.get("request", {}) or {})
        request_info["response_sha256_12"] = self._text_digest(text)
        request_info["response_chars"] = len(text)
        request_audit["request"] = request_info

        self.tick_count += ticks
        decode_elapsed = time.monotonic() - decode_start
        tok_per_sec = ticks / decode_elapsed if decode_elapsed > 0 else 0
        steady_tok_s = None
        if (
            first_token_seconds is not None
            and ticks > 1
            and decode_elapsed > first_token_seconds
        ):
            steady_tok_s = (ticks - 1) / (decode_elapsed - first_token_seconds)

        log.info(
            "generated %d tokens on %s in %.1fs (%.1f tok/s), reservoir ticks=%d, "
            "y1=%.4f y2=%.4f y3=%.4f",
            ticks, handle_name, decode_elapsed, tok_per_sec, self.tick_count,
            processor._y1, processor._y2, processor._y3,
        )

        # Capture post-generation h-norms
        h_norms_after = [0.0, 0.0, 0.0]
        try:
            h1_np, h2_np, h3_np = _timed_state_to_numpy(
                self.reservoir,
                self.state,
                request_audit,
            )
            h_norms_after = [float(np.linalg.norm(h)) for h in (h1_np, h2_np, h3_np)]
        except Exception:
            pass

        coupling_used = self.coupling_strength

        # Log to coupling journal — the persistent record of how language
        # and dynamics interact through this membrane
        self._log_generation(
            handle_name=handle_name,
            ticks=ticks, elapsed=decode_elapsed,
            y1=processor._y1, y2=processor._y2, y3=processor._y3,
            h_norms_before=h_norms_before, h_norms_after=h_norms_after,
        )

        # CHECK IN: push evolved state back to the service
        self._push_state(
            ticks=ticks,
            last_r_input=last_r_input,
            handle_name=handle_name,
            meta=self._build_push_meta(
                handle_name=handle_name,
                messages=messages,
                prompt_text=prompt_text,
                response_text=text,
                ticks=ticks,
                elapsed=decode_elapsed,
                temperature=temperature,
                max_tokens=max_tokens,
                coupling_used=coupling_used,
                y1=processor._y1,
                y2=processor._y2,
                y3=processor._y3,
                h_norms_before=h_norms_before,
                h_norms_after=h_norms_after,
            ),
            audit=request_audit,
        )

        generation = dict(request_audit.get("generation", {}) or {})
        generation["generated_tokens"] = int(ticks)
        generation["first_token_s"] = first_token_seconds
        generation["steady_tok_s"] = steady_tok_s
        generation["decode_tok_s"] = tok_per_sec
        generation["total_turn_s"] = time.perf_counter() - request_start
        request_audit["generation"] = generation
        self._write_request_audit(request_audit)

        return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Coupled Astrid generation server (drop-in for mlx_lm.server)"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument(
        "--model", default="mlx-community/gemma-4-12B-it-5bit",
        help="MLX model to load",
    )
    ap.add_argument(
        "--coupling-strength", type=float, default=0.1,
        help="Reservoir coupling strength (0.0=off, 0.1=gentle, 0.3=strong)",
    )
    ap.add_argument("--input-dim", type=int, default=32)
    ap.add_argument("--n-nodes", type=int, default=192)
    ap.add_argument(
        "--model-memory-map",
        action="store_true",
        default=_env_flag("COUPLED_ASTRID_MODEL_MEMORY_MAP", False),
        help="Request explicit memory-mapped model loading when supported.",
    )
    ap.add_argument(
        "--audit-dir",
        default=os.environ.get("COUPLED_ASTRID_AUDIT_DIR"),
        help="Directory for per-request JSONL audit metrics. Disabled unless set.",
    )
    ap.add_argument(
        "--wide-coupling-strength", type=float,
        default=float(os.environ.get("COUPLED_ASTRID_WIDE_STRENGTH", "0.0") or 0.0),
        help="Operator CEILING for the y4 wide (logit-space aperture) channel. "
             "0.0 = OFF (default). Astrid's SET_APERTURE scales within this.",
    )
    ap.add_argument("--wide-coupling-rank", type=int, default=16,
                    help="Low-rank k for the wide channel (8/16/32).")
    ap.add_argument("--wide-bias-cap", type=float, default=4.0,
                    help="Hard clamp on the per-token wide bias (logits).")
    ap.add_argument("--wide-pressure-floor", type=float, default=0.25,
                    help="Min aperture fraction at rest; opens to 1.0 under λ₁ pressure.")
    ap.add_argument(
        "--generation-queue-capacity",
        type=int,
        default=int(os.environ.get("COUPLED_ASTRID_QUEUE_CAPACITY", "1")),
        help="Number of requests allowed to wait behind the active generation.",
    )
    ap.add_argument(
        "--readiness-stale-seconds",
        type=float,
        default=float(os.environ.get("COUPLED_ASTRID_READINESS_STALE_SECONDS", "5")),
        help="Idle worker heartbeat age after which /readyz reports stale.",
    )
    args = ap.parse_args()

    runtime = ModelRuntimeCoordinator(
        queue_capacity=args.generation_queue_capacity,
        stale_after_s=args.readiness_stale_seconds,
    )
    gateway = AiohttpGateway(runtime, args.host, args.port)
    try:
        # Publish liveness before loading the model. The HTTP event loop stays
        # in its own thread; all MLX construction and generation remains here.
        gateway.start()
    except Exception:
        log.exception("unable to start HTTP gateway")
        return 1

    stop_event = threading.Event()

    def request_stop(signum, _frame):
        log.info("received signal %s; draining active generation", signum)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    server: CoupledAstridServer | None = None
    exit_code = 0
    try:
        server = CoupledAstridServer(
            model_name=args.model,
            coupling_strength=args.coupling_strength,
            input_dim=args.input_dim,
            n_nodes=args.n_nodes,
            model_memory_map=bool(args.model_memory_map),
            audit_dir=args.audit_dir,
            wide_coupling_strength=args.wide_coupling_strength,
            wide_coupling_rank=args.wide_coupling_rank,
            wide_bias_cap=args.wide_bias_cap,
            wide_pressure_floor=args.wide_pressure_floor,
        )
        runtime.attach_server(server)
        log.info(
            "model ready on %s:%d (coupled, strength=%.2f)",
            args.host,
            gateway.bound_port or args.port,
            server.coupling_strength,
        )
        runtime.run_worker(stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    except BaseException as exc:
        runtime.mark_failed(exc)
        log.exception("coupled model worker failed")
        exit_code = 1
    finally:
        if runtime.phase != "failed":
            runtime.mark_stopping()
        if server is not None:
            log.info("shutting down — saving coupling journal")
            server._save_coupling_state()
        try:
            gateway.stop()
        except Exception:
            log.exception("HTTP gateway shutdown failed")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
