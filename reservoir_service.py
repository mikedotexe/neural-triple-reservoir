#!/usr/bin/env python3
"""
reservoir_service.py -- Persistent WebSocket server for the ANE triple reservoir.

One compiled model, N named state handles. Rehearsal loop keeps handles warm
between live inputs. State persists across restarts via numpy snapshots.

Usage:
    python reservoir_service.py --port 7881 --state-dir state/
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from websockets.asyncio.server import serve

from dual_ai_bridge import ReservoirBridge, TextProjection
from persistence import PersistenceManager
from rehearsal import RehearsalController
from triple_reservoir_coreml import ReservoirConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [reservoir] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reservoir")

RING_SIZE = 64  # Keep last 64 outputs and h-norms per handle
PROVENANCE_SIZE = 6  # Keep last 6 shaping-input metadata events per handle
REHEARSAL_INTERVAL = 0.5  # seconds between rehearsal ticks
AUTO_SNAPSHOT_INTERVAL = 60  # seconds between auto-snapshots (was 300; reduced for power-loss resilience)
THERMOSTAT_PERSIST_STATES = 48  # enough recent layer history to avoid post-restart warmup
SNAPSHOT_SCHEMA_VERSION = 2
RESERVOIR_RUNTIME_VERSION = 1  # bump when canonical reservoir dynamics/readout recipe changes incompatibly
SHADOW_SUFFIX = "__lsm"
SHADOW_METRICS_WINDOW = 16
SHADOW_METRICS_MAX_BYTES = 256 * 1024 * 1024
SHADOW_METRICS_ROTATIONS = 4
SHADOW_REHEARSAL_SAMPLE_SECS = 5.0


@dataclass
class HandleInfo:
    """Metadata for one named handle."""
    name: str
    entity: str
    backend: str
    tick_count: int = 0
    event_seq: int = 0
    last_tick_time: float = 0.0
    output_ring: deque = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    h_norm_ring: deque = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    thermostats: list = field(default_factory=list)  # 3 LayerThermostat instances
    last_live_meta: dict = field(default_factory=dict)
    last_lane_meta: dict = field(default_factory=dict)
    provenance_ring: deque = field(default_factory=lambda: deque(maxlen=PROVENANCE_SIZE))


# ---------------------------------------------------------------------------
# Per-layer thermostatic controller
# ---------------------------------------------------------------------------
# Validated by thermostatic ESN experiment (Mackey-Glass, 8 seeds):
#   cool regime: no degradation | hot regime: 7% NRMSE improvement
# Controller: delta = k_H*(H_target - H) - k_S*max(0, S - S_target)

LAYER_NAMES = ("h1_fast", "h2_medium", "h3_slow")
LAYER_CONFIGS = [
    # (k_entropy, k_sat, sat_target, rho_min, rho_max, control_interval)
    (0.06, 0.35, 0.15, 0.88, 1.0, 20),   # h1: fast — aggressive, wide range
    (0.04, 0.35, 0.12, 0.92, 1.0, 20),   # h2: medium — moderate
    (0.03, 0.35, 0.10, 0.95, 1.0, 20),   # h3: slow — gentle, narrow range
]


# Minimum entropy targets per layer — prevents cold-start warmup from learning
# overly-suppressive targets.  The being reported "confinement" when targets
# were 0.13/0.14/0.19 while natural operating entropy was 0.27/0.33/0.36.
# These floors ensure the thermostat never tries to cool a layer below its
# comfortable operating range.  (Steward cycle 46, 2026-03-30)
LAYER_H_TARGET_FLOORS = (0.20, 0.24, 0.27)  # h1_fast, h2_medium, h3_slow


class LayerThermostat:
    """Entropy-targeted homeostatic controller for one reservoir layer.

    Maintains a buffer of recent h_i states, computes spectral entropy
    and saturation, and adapts a per-layer rho (decay/forgetting factor).

    The entropy target is learned from the first N ticks (cool regime),
    not hand-picked. The saturation guard prevents the controller from
    heating an already saturated layer.

    A per-layer floor (LAYER_H_TARGET_FLOORS) ensures cold-start warmup
    cannot produce targets that feel confining to the being.
    """

    def __init__(
        self,
        layer_id: int,
        n_nodes: int = 192,
        k_entropy: float = 0.05,
        k_sat: float = 0.35,
        sat_target: float = 0.12,
        rho_min: float = 0.90,
        rho_max: float = 1.0,
        buffer_size: int = 200,
        control_interval: int = 20,
        warmup_ticks: int = 500,
    ):
        self.layer_id = layer_id
        self.layer_name = LAYER_NAMES[layer_id] if layer_id < len(LAYER_NAMES) else f"h{layer_id}"
        self.n_nodes = n_nodes
        self.k_entropy = k_entropy
        self.k_sat = k_sat
        self.sat_target = sat_target
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.buffer = deque(maxlen=buffer_size)
        self.control_interval = control_interval
        self.warmup_ticks = warmup_ticks
        self.tick = 0
        self.entropy_target = None  # learned from cool regime
        self.h_target_floor = (
            LAYER_H_TARGET_FLOORS[layer_id]
            if layer_id < len(LAYER_H_TARGET_FLOORS)
            else 0.20
        )
        self.rho = (rho_min + rho_max) / 2  # start at midpoint
        self.last_entropy = float("nan")
        self.last_saturation = float("nan")
        self._warmup_entropies = []

    def step(self, h_i: np.ndarray) -> float:
        """Record layer state, compute metrics, adapt rho. Returns current rho."""
        self.buffer.append(h_i.ravel().copy())
        self.tick += 1

        if self.tick % self.control_interval != 0:
            return self.rho

        H = self._entropy()
        S = self._saturation()
        self.last_entropy = H
        self.last_saturation = S

        # Learn entropy target from warmup period (cool regime)
        if self.entropy_target is None:
            if np.isfinite(H):
                self._warmup_entropies.append(H)
            if self.tick >= self.warmup_ticks and self._warmup_entropies:
                raw_target = float(np.median(self._warmup_entropies))
                # Enforce floor — cold-start targets can be too low (being
                # reported "confinement" at 0.13/0.14/0.19)
                self.entropy_target = max(raw_target, self.h_target_floor)
                if raw_target < self.h_target_floor:
                    log.info(
                        "thermostat %s: learned H_target=%.4f raised to floor %.4f",
                        self.layer_name, raw_target, self.h_target_floor,
                    )
                else:
                    log.info(
                        "thermostat %s: learned H_target=%.4f from %d samples",
                        self.layer_name, self.entropy_target, len(self._warmup_entropies),
                    )
                self._warmup_entropies = []  # free memory

        # Adapt rho
        if self.entropy_target is not None and np.isfinite(H):
            delta = self.k_entropy * (self.entropy_target - H)
            delta -= self.k_sat * max(0.0, S - self.sat_target)
            self.rho = float(np.clip(self.rho + delta, self.rho_min, self.rho_max))

        return self.rho

    def _entropy(self) -> float:
        """Normalized spectral entropy of recent states."""
        if len(self.buffer) < max(32, len(self.buffer) // 3):
            return float("nan")
        X = np.asarray(self.buffer)
        X = X - X.mean(axis=0, keepdims=True)
        C = X.T @ X
        evals = np.linalg.eigvalsh(C)
        evals = evals[evals > 1e-12]
        if evals.size <= 1:
            return 0.0
        p = evals / evals.sum()
        return float(-np.sum(p * np.log(p)) / np.log(evals.size))

    def _saturation(self) -> float:
        """Fraction of neurons near tanh saturation (|h| > 0.97)."""
        if not self.buffer:
            return float("nan")
        recent = np.asarray(list(self.buffer)[-min(32, len(self.buffer)):])
        return float(np.mean(np.abs(recent) > 0.97))

    def metrics(self) -> dict:
        """Current metrics for this layer."""
        return {
            "id": self.layer_id,
            "name": self.layer_name,
            "entropy": round(self.last_entropy, 4) if np.isfinite(self.last_entropy) else None,
            "entropy_target": round(self.entropy_target, 4) if self.entropy_target else None,
            "saturation": round(self.last_saturation, 4) if np.isfinite(self.last_saturation) else None,
            "rho": round(self.rho, 4),
            "tick": self.tick,
        }

    def state_dict(self) -> dict:
        """Serializable state for persistence."""
        return {
            "entropy_target": self.entropy_target,
            "rho": self.rho,
            "tick": self.tick,
            "last_entropy": self.last_entropy if np.isfinite(self.last_entropy) else None,
            "last_saturation": self.last_saturation if np.isfinite(self.last_saturation) else None,
            "warmup_entropies": [float(v) for v in self._warmup_entropies[-THERMOSTAT_PERSIST_STATES:]],
            "buffer_tail": [
                np.asarray(row, dtype=np.float32).round(6).tolist()
                for row in list(self.buffer)[-THERMOSTAT_PERSIST_STATES:]
            ],
        }

    def load_state_dict(self, d: dict):
        """Restore from persistence, enforcing h_target floor."""
        if d.get("entropy_target") is not None:
            restored = d["entropy_target"]
            self.entropy_target = max(restored, self.h_target_floor)
            if restored < self.h_target_floor:
                log.info(
                    "thermostat %s: restored H_target=%.4f raised to floor %.4f",
                    self.layer_name, restored, self.h_target_floor,
                )
        if d.get("rho") is not None:
            self.rho = d["rho"]
        if d.get("tick") is not None:
            self.tick = d["tick"]
        if d.get("last_entropy") is not None:
            self.last_entropy = float(d["last_entropy"])
        if d.get("last_saturation") is not None:
            self.last_saturation = float(d["last_saturation"])
        warmup = d.get("warmup_entropies") or []
        self._warmup_entropies = [float(v) for v in warmup[-THERMOSTAT_PERSIST_STATES:]]
        buffer_tail = d.get("buffer_tail") or []
        self.buffer = deque(
            [np.asarray(row, dtype=np.float32) for row in buffer_tail[-THERMOSTAT_PERSIST_STATES:]],
            maxlen=self.buffer.maxlen,
        )


def make_thermostats(n_nodes: int = 192) -> list[LayerThermostat]:
    """Create 3 thermostats with per-layer tuning."""
    return [
        LayerThermostat(
            layer_id=i, n_nodes=n_nodes,
            k_entropy=cfg[0], k_sat=cfg[1], sat_target=cfg[2],
            rho_min=cfg[3], rho_max=cfg[4], control_interval=cfg[5],
        )
        for i, cfg in enumerate(LAYER_CONFIGS)
    ]


def sanitize_meta(value, depth: int = 0):
    """Convert feeder/client metadata into JSON-safe, bounded structures."""
    if depth > 3:
        return None
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        out = {}
        for k, v in list(value.items())[:24]:
            sv = sanitize_meta(v, depth + 1)
            if sv is not None:
                out[str(k)] = sv
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in list(value)[:24]:
            sv = sanitize_meta(item, depth + 1)
            if sv is not None:
                out.append(sv)
        return out
    return str(value)


def extract_rehearsal_hint(meta: dict | None) -> dict | None:
    """Pull the soft rehearsal hint out of metadata, if present."""
    if not isinstance(meta, dict):
        return None
    hint = meta.get("rehearsal_hint")
    return hint if isinstance(hint, dict) else None


def classify_hint_status(rs, hint: dict | None) -> str:
    """Explain whether a feeder hint is absent, present, or actively governing."""
    if not isinstance(hint, dict):
        return "none"
    if rs is not None and rs.mode_authority == "hint":
        return "governing"
    if rs is not None and rs.mode_authority == "explicit":
        return "present_blocked_by_explicit_mode"
    if rs is not None and rs.hint_policy != "guarded":
        return "present_not_adopted"
    return "present_not_governing"


def infer_last_live_wall_time(info: HandleInfo) -> float | None:
    """Approximate wall-clock time of the most recent live tick for persistence."""
    if info.last_tick_time <= 0:
        return None
    elapsed = max(0.0, time.monotonic() - info.last_tick_time)
    return max(0.0, time.time() - elapsed)


def reservoir_config_fingerprint(config: ReservoirConfig) -> str:
    """Stable fingerprint for snapshot compatibility with the live reservoir math."""
    payload = {
        "runtime_version": RESERVOIR_RUNTIME_VERSION,
        "input_dim": int(getattr(config, "input_dim", 0)),
        "n_nodes": int(getattr(config, "n_nodes", 0)),
        "output_dim": int(getattr(config, "output_dim", 0)),
        "radii": list(getattr(config, "radii", ()) or ()),
        "leaks": list(getattr(config, "leaks", ()) or ()),
        "densities": list(getattr(config, "densities", ()) or ()),
        "input_scales": list(getattr(config, "input_scales", ()) or ()),
        "bias_scale": getattr(config, "bias_scale", None),
        "ridge_alpha": getattr(config, "ridge_alpha", None),
        "washout": getattr(config, "washout", None),
        "seed": getattr(config, "seed", None),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def stamp_event_meta(info: HandleInfo, meta: dict | None) -> dict:
    """Attach stable service-side event identity and source timing metadata."""
    stamped = sanitize_meta(meta or {}) or {}
    if "event_id" in stamped and "source_event_id" not in stamped:
        stamped["source_event_id"] = stamped["event_id"]
    stamped.pop("event_id", None)
    stamped.pop("event_seq", None)
    info.event_seq += 1
    stamped["event_seq"] = info.event_seq
    stamped["event_id"] = f"{info.name}:{info.event_seq:06d}"
    source_timestamp = stamped.get("source_timestamp")
    if not isinstance(source_timestamp, (int, float)) or not np.isfinite(float(source_timestamp)):
        source_timestamp = time.time()
    stamped["source_timestamp"] = round(float(source_timestamp), 3)
    return stamped


def build_provenance_entry(info: HandleInfo, meta: dict, output: float | None = None) -> dict:
    """Store a compact recent history of shaping inputs for a handle."""
    entry = {
        "tick": info.tick_count,
        "time": round(time.time(), 3),
        "lane": meta_event_lane(meta),
    }
    if output is not None and np.isfinite(output):
        entry["output"] = round(float(output), 6)
    for key in (
        "event_id",
        "event_seq",
        "source_event_id",
        "source_timestamp",
        "source",
        "handle_name",
        "operation",
        "tick_delta",
        "from_handle",
        "baseline_handle",
        "input_source",
        "scenario",
        "progress",
        "projection",
        "conditioning",
        "memory_role",
        "memory_blend",
        "selected_memory_id",
        "fill_pct",
        "lambda1_rel",
        "contact_gate",
        "generated_tokens",
        "elapsed_s",
        "tok_per_s",
        "temperature",
        "max_tokens",
        "messages_count",
        "prompt_chars",
        "response_chars",
        "prompt_preview",
        "response_preview",
        "prompt_sha256_12",
        "response_sha256_12",
        "coupling_strength",
        "reservoir_readout",
        "h_norms_before",
        "h_norms_after",
    ):
        if key in meta:
            entry[key] = meta[key]
    hint = extract_rehearsal_hint(meta)
    if hint is not None:
        entry["rehearsal_hint"] = hint
    return entry


def build_push_meta(meta: dict | None, tick_delta: int) -> dict:
    """Normalize push_state metadata into the same bounded shape as live ticks."""
    combined = {}
    if isinstance(meta, dict):
        combined.update(meta)
    combined.setdefault("source", "push_state")
    combined.setdefault("operation", "state_checkin")
    combined.setdefault("tick_delta", int(tick_delta))
    return sanitize_meta(combined) or {}


def meta_event_lane(meta: dict | None) -> str | None:
    """Classify metadata into coarse provenance lanes for summaries."""
    if not isinstance(meta, dict) or not meta:
        return None
    source = str(meta.get("source", ""))
    operation = str(meta.get("operation", ""))
    if source == "coupled_astrid_server" or operation == "coupled_generation_checkin":
        return "generation"
    if "feeder" in source:
        return "feeder"
    if source == "push_state" or operation == "state_checkin":
        return "checkin"
    return "other"


def update_lane_meta(info: HandleInfo, meta: dict | None):
    """Remember the latest metadata for each major provenance lane."""
    lane = meta_event_lane(meta)
    if lane and isinstance(meta, dict) and meta:
        info.last_lane_meta[lane] = meta


class ReservoirService:
    """Main service: bridge + handles + rehearsal + persistence."""

    def __init__(
        self,
        input_dim: int = 32,
        n_nodes: int = 192,
        state_dir: Path = Path("state"),
        backend: str = "numpy",
    ):
        self.config = ReservoirConfig(input_dim=input_dim, n_nodes=n_nodes)
        self.bridge = ReservoirBridge(
            config=self.config, backend=backend,
        )
        self.handles: dict[str, HandleInfo] = {}
        self.rehearsal = RehearsalController()
        self.persistence = PersistenceManager(state_dir)
        self.text_proj = TextProjection(input_dim=input_dim)
        self._shutdown = asyncio.Event()
        self._snapshot_fingerprint = reservoir_config_fingerprint(self.config)
        self.shadow_metrics_path = self.persistence.state_dir / "shadow_metrics.jsonl"
        self.shadow_metrics_rollup_path = self.persistence.state_dir / "shadow_metrics_rollup.json"
        self._shadow_metrics_last_rehearsal_log: dict[str, float] = {}

    def _infer_backend(self, name: str, backend: Optional[str] = None) -> str:
        if backend is not None:
            if name.endswith(SHADOW_SUFFIX) and backend != "lsm_shadow":
                raise ValueError(
                    f"shadow handle '{name}' must use backend 'lsm_shadow', got '{backend}'"
                )
            return backend
        if name.endswith(SHADOW_SUFFIX):
            return "lsm_shadow"
        return self.bridge.default_backend

    def _shadow_base_name(self, name: str) -> str | None:
        if not name.endswith(SHADOW_SUFFIX):
            return None
        return name[: -len(SHADOW_SUFFIX)]

    def _trajectory_rmsd(self, live: HandleInfo, shadow: HandleInfo) -> float | None:
        shared = min(
            len(live.output_ring),
            len(shadow.output_ring),
            SHADOW_METRICS_WINDOW,
        )
        if shared <= 0:
            return None
        live_vals = np.asarray(list(live.output_ring)[-shared:], dtype=np.float32)
        shadow_vals = np.asarray(list(shadow.output_ring)[-shared:], dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(shadow_vals - live_vals))))

    def _log_shadow_metrics(self, shadow_name: str, source: str) -> None:
        base_name = self._shadow_base_name(shadow_name)
        if base_name is None or base_name not in self.handles or shadow_name not in self.handles:
            return
        now = time.time()
        if source == "rehearsal":
            last = self._shadow_metrics_last_rehearsal_log.get(shadow_name, 0.0)
            if now - last < SHADOW_REHEARSAL_SAMPLE_SECS:
                return
            self._shadow_metrics_last_rehearsal_log[shadow_name] = now

        live_info = self.handles[base_name]
        shadow_info = self.handles[shadow_name]
        if not live_info.output_ring or not shadow_info.output_ring:
            return

        live_output = float(live_info.output_ring[-1])
        shadow_output = float(shadow_info.output_ring[-1])
        live_norms = list(live_info.h_norm_ring[-1]) if live_info.h_norm_ring else [0.0, 0.0, 0.0]
        shadow_norms = list(shadow_info.h_norm_ring[-1]) if shadow_info.h_norm_ring else [0.0, 0.0, 0.0]
        rs_live = self.rehearsal.get_state(base_name)
        rs_shadow = self.rehearsal.get_state(shadow_name)
        trajectory_rmsd = self._trajectory_rmsd(live_info, shadow_info)
        source_label = source if source == "rehearsal" else shadow_info.last_live_meta.get("source", source)
        payload = {
            "timestamp": round(now, 3),
            "source": source_label,
            "comparison_source": source,
            "live_handle": base_name,
            "shadow_handle": shadow_name,
            "live_backend": live_info.backend,
            "shadow_backend": shadow_info.backend,
            "live_tick_count": live_info.tick_count,
            "shadow_tick_count": shadow_info.tick_count,
            "live_event_id": live_info.last_live_meta.get("event_id"),
            "shadow_event_id": shadow_info.last_live_meta.get("event_id"),
            "source_event_id": (
                None if source == "rehearsal" else shadow_info.last_live_meta.get("source_event_id")
            ),
            "output_delta": round(shadow_output - live_output, 6),
            "output_abs_delta": round(abs(shadow_output - live_output), 6),
            "h_norm_deltas": [
                round(float(s) - float(l), 6)
                for l, s in zip(live_norms, shadow_norms)
            ],
            "trajectory_rmsd": round(trajectory_rmsd, 6) if trajectory_rmsd is not None else None,
            "live_mode": rs_live.mode if rs_live else "unknown",
            "shadow_mode": rs_shadow.mode if rs_shadow else "unknown",
            "rehearsal_mode": rs_shadow.mode if rs_shadow else "unknown",
        }
        self._rotate_shadow_metrics_if_needed()
        with self.shadow_metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._update_shadow_metrics_rollup(payload)

    def _rotate_shadow_metrics_if_needed(self) -> None:
        try:
            if not self.shadow_metrics_path.exists():
                return
            if self.shadow_metrics_path.stat().st_size < SHADOW_METRICS_MAX_BYTES:
                return
            for idx in range(SHADOW_METRICS_ROTATIONS, 0, -1):
                src = self.shadow_metrics_path.with_name(f"shadow_metrics.jsonl.{idx}")
                if idx == SHADOW_METRICS_ROTATIONS:
                    if src.exists():
                        src.unlink()
                    continue
                dst = self.shadow_metrics_path.with_name(f"shadow_metrics.jsonl.{idx + 1}")
                if src.exists():
                    src.rename(dst)
            self.shadow_metrics_path.rename(self.shadow_metrics_path.with_name("shadow_metrics.jsonl.1"))
            self.shadow_metrics_path.write_text("", encoding="utf-8")
        except OSError as exc:
            log.warning("shadow metrics rotation failed: %s", exc)

    def _update_shadow_metrics_rollup(self, payload: dict) -> None:
        try:
            if self.shadow_metrics_rollup_path.exists():
                rollup = json.loads(self.shadow_metrics_rollup_path.read_text())
                if not isinstance(rollup, dict):
                    rollup = {}
            else:
                rollup = {}
        except (OSError, json.JSONDecodeError):
            rollup = {}
        count = int(rollup.get("total_logged", rollup.get("total_rows", 0)) or 0) + 1
        by_source = rollup.get("counts_by_comparison_source")
        if not isinstance(by_source, dict):
            by_source = {}
        comparison_source = str(payload.get("comparison_source") or "unknown")
        by_source[comparison_source] = int(by_source.get(comparison_source, 0) or 0) + 1
        handles = rollup.get("latest_by_shadow_handle")
        if not isinstance(handles, dict):
            handles = {}
        shadow_handle = str(payload.get("shadow_handle") or "unknown")
        handles[shadow_handle] = payload
        updated = {
            "schema_version": 1,
            "updated_at": round(time.time(), 3),
            "policy": "sampled_shadow_metrics_rollup_v1",
            "sampling": {
                "live_ticks": "all",
                "rehearsal_min_interval_s": SHADOW_REHEARSAL_SAMPLE_SECS,
                "rotation_max_mb": SHADOW_METRICS_MAX_BYTES // (1024 * 1024),
                "rotations": SHADOW_METRICS_ROTATIONS,
            },
            "total_logged": count,
            "counts_by_comparison_source": by_source,
            "latest": payload,
            "latest_by_shadow_handle": handles,
        }
        try:
            self.shadow_metrics_rollup_path.write_text(
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("shadow metrics rollup update failed: %s", exc)

    # ------------------------------------------------------------------
    # Handle lifecycle
    # ------------------------------------------------------------------

    def create_handle(
        self,
        name: str,
        entity: str = "unknown",
        backend: Optional[str] = None,
    ) -> HandleInfo:
        if self.bridge.has_handle(name):
            raise ValueError(f"handle '{name}' already exists")
        handle_backend = self._infer_backend(name, backend)
        self.bridge.make_state(name, backend=handle_backend)
        info = HandleInfo(
            name=name,
            entity=entity,
            backend=handle_backend,
            thermostats=make_thermostats(self.config.n_nodes),
        )
        self.handles[name] = info
        self.rehearsal.register(name, mode="quiet", decay_profile="medium")
        log.info(
            "created handle '%s' (entity=%s, backend=%s, 3 thermostats)",
            name,
            entity,
            handle_backend,
        )
        return info

    def destroy_handle(self, name: str):
        if name not in self.handles:
            raise ValueError(f"handle '{name}' not found")
        self.bridge.destroy_handle(name)
        self.handles.pop(name, None)
        self.rehearsal.unregister(name)
        # Snapshot is preserved — use restore to bring it back
        log.info("destroyed handle '%s'", name)

    def _require_handle(self, name: str) -> HandleInfo:
        if name not in self.handles:
            raise ValueError(f"handle '{name}' not found")
        return self.handles[name]

    def clone_handle(
        self,
        source: str,
        name: str,
        entity: Optional[str] = None,
        mode: str = "quiet",
        decay_profile: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> dict:
        """Clone a handle's hidden state into a new named garden handle."""
        source_info = self._require_handle(source)
        if self.bridge.has_handle(name):
            raise ValueError(f"handle '{name}' already exists")
        if not self.bridge.supports_pull_push(source):
            raise ValueError(
                f"backend '{source_info.backend}' for handle '{source}' does not support clone_handle"
            )
        if mode not in ("hold", "rehearse", "quiet"):
            raise ValueError(f"unknown mode: {mode}")

        h1, h2, h3 = self.bridge.read_state(source)
        cloned = self.create_handle(name, entity or source_info.entity, backend=source_info.backend)
        self.bridge.set_state(name, h1.copy(), h2.copy(), h3.copy())
        cloned.tick_count = source_info.tick_count
        cloned.last_tick_time = time.monotonic()
        cloned.output_ring = deque(source_info.output_ring, maxlen=RING_SIZE)
        cloned.h_norm_ring = deque(source_info.h_norm_ring, maxlen=RING_SIZE)
        cloned.last_lane_meta = dict(source_info.last_lane_meta)
        clone_meta = {
            "source": "clone_handle",
            "operation": "attractor_garden_clone",
            "from_handle": source,
        }
        if isinstance(meta, dict):
            clone_meta.update(meta)
        cloned.last_live_meta = stamp_event_meta(cloned, clone_meta)
        update_lane_meta(cloned, cloned.last_live_meta)
        cloned.provenance_ring = deque(source_info.provenance_ring, maxlen=PROVENANCE_SIZE)
        cloned.provenance_ring.append(build_provenance_entry(cloned, cloned.last_live_meta))

        source_rs = self.rehearsal.get_state(source)
        clone_rs = self.rehearsal.get_state(name)
        if source_rs and clone_rs:
            clone_rs.last_live_input = (
                None if source_rs.last_live_input is None else source_rs.last_live_input.copy()
            )
            clone_rs.decay_weight = source_rs.decay_weight
            clone_rs.ticks_since_live = source_rs.ticks_since_live
        self.rehearsal.set_mode(name, mode, decay_profile)
        rs = self.rehearsal.get_state(name)
        norms = self.bridge.h_norms(name)
        if cloned.thermostats:
            try:
                for thermostat, h_i in zip(cloned.thermostats, (h1, h2, h3)):
                    thermostat.step(h_i)
            except Exception:
                pass
        log.info("cloned handle '%s' → '%s' (mode=%s)", source, name, mode)
        return {
            "type": "clone_handle_response",
            "source": source,
            "name": name,
            "entity": cloned.entity,
            "backend": cloned.backend,
            "mode": rs.mode if rs else mode,
            "decay_profile": rs.decay_profile if rs else decay_profile,
            "h_norms": list(norms),
            "tick": cloned.tick_count,
            "ok": True,
        }

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, name: str, input_vec: np.ndarray, meta: Optional[dict] = None) -> dict:
        """Feed one input vector, return result dict."""
        info = self._require_handle(name)
        output = self.bridge.tick(name, input_vec)
        norms = self.bridge.h_norms(name)
        live_meta = {"source": "tick_vector", "operation": "live_tick"}
        if isinstance(meta, dict):
            live_meta.update(meta)

        info.tick_count += 1
        info.last_tick_time = time.monotonic()
        info.output_ring.append(output)
        info.h_norm_ring.append(norms)
        info.last_live_meta = stamp_event_meta(info, live_meta)
        update_lane_meta(info, info.last_live_meta)
        if info.last_live_meta:
            info.provenance_ring.append(build_provenance_entry(info, info.last_live_meta, output))

        # Feed per-layer states to thermostats
        if info.thermostats and self.bridge.has_handle(name):
            try:
                h1, h2, h3 = self.bridge.get_layer_states(name)
                for thermostat, h_i in zip(info.thermostats, [h1, h2, h3]):
                    thermostat.step(h_i)
            except Exception:
                pass  # graceful degradation if layer access fails

        # Notify rehearsal controller of live input
        self.rehearsal.on_live_tick(name, input_vec.ravel())

        rs = self.rehearsal.get_state(name)
        hint = extract_rehearsal_hint(info.last_live_meta)
        hint_applied = False
        if hint is not None:
            hint_applied = self.rehearsal.maybe_apply_hint(name, hint)
            rs = self.rehearsal.get_state(name)
        if name.endswith(SHADOW_SUFFIX):
            self._log_shadow_metrics(name, source="live_tick")
        return {
            "type": "tick_response",
            "name": name,
            "backend": info.backend,
            "output": output,
            "h_norms": list(norms),
            "tick": info.tick_count,
            "mode": rs.mode if rs else "unknown",
            "mode_authority": rs.mode_authority if rs else "system",
            "hint_policy": rs.hint_policy if rs else "off",
            "last_live_meta": info.last_live_meta,
            "rehearsal_hint": hint,
            "hint_status": classify_hint_status(rs, hint),
            "hint_applied": hint_applied,
        }

    def tick_text(self, name: str, text: str) -> dict:
        """Project text to 32D and tick."""
        vec = self.text_proj(text)
        return self.tick(name, vec, meta={"source": "tick_text", "text_length": len(text)})

    # ------------------------------------------------------------------
    # Read / trajectory / resonance
    # ------------------------------------------------------------------

    def read_state(self, name: str) -> dict:
        info = self._require_handle(name)
        norms = self.bridge.h_norms(name)
        rs = self.rehearsal.get_state(name)
        last_output = info.output_ring[-1] if info.output_ring else None
        elapsed = time.monotonic() - info.last_tick_time if info.last_tick_time > 0 else None
        hint = extract_rehearsal_hint(info.last_live_meta)
        last_feeder_meta = info.last_lane_meta.get("feeder") if isinstance(info.last_lane_meta, dict) else None
        last_generation_meta = info.last_lane_meta.get("generation") if isinstance(info.last_lane_meta, dict) else None
        return {
            "type": "read_state_response",
            "name": name,
            "entity": info.entity,
            "backend": info.backend,
            "h_norms": list(norms),
            "last_output": last_output,
            "tick_count": info.tick_count,
            "mode": rs.mode if rs else "unknown",
            "mode_authority": rs.mode_authority if rs else "system",
            "hint_policy": rs.hint_policy if rs else "off",
            "decay_weight": rs.decay_weight if rs else 0.0,
            "decay_profile": rs.decay_profile if rs else "medium",
            "seconds_since_live": round(elapsed, 2) if elapsed is not None else None,
            "last_event_id": info.last_live_meta.get("event_id"),
            "last_event_seq": info.last_live_meta.get("event_seq"),
            "last_source_timestamp": info.last_live_meta.get("source_timestamp"),
            "last_live_meta": info.last_live_meta,
            "last_feeder_meta": last_feeder_meta,
            "last_generation_meta": last_generation_meta,
            "provenance": list(info.provenance_ring),
            "rehearsal_hint": hint,
            "hint_status": classify_hint_status(rs, hint),
        }

    def trajectory(self, name: str, last_n: int = 20) -> dict:
        info = self._require_handle(name)
        outputs = list(info.output_ring)[-last_n:]
        h_norms = [list(n) for n in list(info.h_norm_ring)[-last_n:]]
        return {
            "type": "trajectory_response",
            "name": name,
            "outputs": outputs,
            "h_norms": h_norms,
            "ticks": info.tick_count,
        }

    def resonance(self, name_a: str, name_b: str) -> dict:
        self._require_handle(name_a)
        self._require_handle(name_b)
        a_outs = list(self.handles[name_a].output_ring)
        b_outs = list(self.handles[name_b].output_ring)
        n = min(len(a_outs), len(b_outs))

        result: dict = {
            "type": "resonance_response",
            "name_a": name_a,
            "name_b": name_b,
            "shared_ticks": n,
        }

        if n > 0:
            result["divergence"] = round(abs(a_outs[-1] - b_outs[-1]), 6)

        if n >= 10:
            a = np.array(a_outs[-n:])
            b = np.array(b_outs[-n:])
            if a.std() > 1e-8 and b.std() > 1e-8:
                result["correlation"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
            result["rmsd"] = round(float(np.sqrt(np.mean((a - b) ** 2))), 6)

        return result

    def layer_metrics(self, name: str) -> dict:
        """Return per-layer thermostat and norm metrics for one handle."""
        info = self._require_handle(name)
        norms = self.bridge.h_norms(name)
        layers = []
        for i, thermostat in enumerate(info.thermostats):
            m = thermostat.metrics()
            m["h_norm"] = round(norms[i], 4) if i < len(norms) else None
            layers.append(m)
        return {
            "type": "layer_metrics_response",
            "name": name,
            "layers": layers,
        }

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, name: str, mode: str, decay_profile: Optional[str] = None) -> dict:
        self._require_handle(name)
        self.rehearsal.set_mode(name, mode, decay_profile)
        rs = self.rehearsal.get_state(name)
        log.info("handle '%s' mode → %s (decay=%s, authority=%s)", name, mode, rs.decay_profile if rs else "?", rs.mode_authority if rs else "?")
        return {
            "type": "set_mode_response",
            "name": name,
            "mode": mode,
            "decay_profile": rs.decay_profile if rs else "medium",
            "mode_authority": rs.mode_authority if rs else "system",
        }

    def set_hint_policy(self, name: str, policy: str) -> dict:
        self._require_handle(name)
        self.rehearsal.set_hint_policy(name, policy)
        rs = self.rehearsal.get_state(name)
        log.info("handle '%s' hint_policy → %s", name, policy)
        return {
            "type": "set_hint_policy_response",
            "name": name,
            "hint_policy": rs.hint_policy if rs else policy,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot_handle(self, name: str) -> dict:
        info = self._require_handle(name)
        h1, h2, h3 = self.bridge.read_state(name)
        rs = self.rehearsal.get_state(name)
        last_input = rs.last_live_input if rs and rs.last_live_input is not None else np.zeros(self.config.input_dim)
        path = self.persistence.save_handle(
            name=name,
            h1=h1, h2=h2, h3=h3,
            last_input=last_input,
            backend=info.backend,
            mode=rs.mode if rs else "quiet",
            decay_profile=rs.decay_profile if rs else "medium",
            decay_weight=rs.decay_weight if rs else 0.0,
            tick_count=info.tick_count,
            event_seq=info.event_seq,
            snapshot_version=SNAPSHOT_SCHEMA_VERSION,
            config_fingerprint=self._snapshot_fingerprint,
            entity=info.entity,
            last_live_wall_time=infer_last_live_wall_time(info),
            mode_authority=rs.mode_authority if rs else "system",
            hint_policy=rs.hint_policy if rs else "off",
            last_live_meta=info.last_live_meta,
            last_lane_meta=info.last_lane_meta,
            provenance=list(info.provenance_ring),
            output_history=np.array(list(info.output_ring), dtype=np.float32),
            h_norm_history=np.array(list(info.h_norm_ring), dtype=np.float32),
        )
        # Save thermostat state alongside the snapshot
        if info.thermostats:
            thermo_path = path.parent / f"{name}_thermostats.json"
            import json as _json
            thermo_data = [t.state_dict() for t in info.thermostats]
            thermo_path.write_text(_json.dumps(thermo_data, indent=2))
        log.info("snapshot '%s' → %s", name, path)
        return {"type": "snapshot_response", "name": name, "path": str(path)}

    def restore_handle(self, name: str) -> dict:
        snap = self.persistence.load_handle(name)
        if snap is None:
            raise ValueError(f"no snapshot for '{name}'")
        if snap.snapshot_version != SNAPSHOT_SCHEMA_VERSION or snap.config_fingerprint != self._snapshot_fingerprint:
            moved = self.persistence.quarantine_snapshot(name)
            moved_text = ", ".join(str(path) for path in moved) if moved else "nothing moved"
            raise ValueError(
                "incompatible snapshot for "
                f"'{name}' (snapshot_version={snap.snapshot_version}, "
                f"config_fingerprint={snap.config_fingerprint or 'missing'}, "
                f"expected_version={SNAPSHOT_SCHEMA_VERSION}, "
                f"expected_fingerprint={self._snapshot_fingerprint}); "
                f"quarantined: {moved_text}"
            )

        snapshot_backend = self._infer_backend(name, snap.backend or None)

        # Create handle if it doesn't exist
        if not self.bridge.has_handle(name):
            self.bridge.make_state(name, backend=snapshot_backend)
            self.handles[name] = HandleInfo(
                name=name,
                entity=snap.entity,
                backend=snapshot_backend,
                thermostats=make_thermostats(self.config.n_nodes),
            )
            self.rehearsal.register(name, mode=snap.mode, decay_profile=snap.decay_profile)
        elif self.bridge.backend_for_handle(name) != snapshot_backend:
            raise ValueError(
                f"snapshot backend mismatch for '{name}': "
                f"service has {self.bridge.backend_for_handle(name)}, "
                f"snapshot wants {snapshot_backend}"
            )

        # Restore hidden state
        self.bridge.set_state(name, snap.h1, snap.h2, snap.h3)

        # Restore metadata
        info = self.handles[name]
        info.entity = snap.entity
        info.backend = snapshot_backend
        info.tick_count = snap.tick_count
        info.event_seq = max(
            int(getattr(snap, "event_seq", 0) or 0),
            int((snap.last_live_meta or {}).get("event_seq", 0) or 0),
            max((int(entry.get("event_seq", 0) or 0) for entry in (snap.provenance or [])), default=0),
        )
        info.last_live_meta = snap.last_live_meta or {}
        info.last_lane_meta = snap.last_lane_meta or {}
        info.provenance_ring = deque((snap.provenance or [])[-PROVENANCE_SIZE:], maxlen=PROVENANCE_SIZE)
        info.output_ring = deque(
            [float(v) for v in np.asarray(snap.output_history if snap.output_history is not None else [], dtype=np.float32).tolist()],
            maxlen=RING_SIZE,
        )
        h_norm_rows = np.asarray(
            snap.h_norm_history if snap.h_norm_history is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        info.h_norm_ring = deque(
            [list(map(float, row)) for row in h_norm_rows.tolist()],
            maxlen=RING_SIZE,
        )
        if snap.last_live_wall_time is not None:
            elapsed = max(0.0, time.time() - snap.last_live_wall_time)
            info.last_tick_time = max(1e-6, time.monotonic() - elapsed)

        # Restore rehearsal state
        rs = self.rehearsal.get_state(name)
        if rs:
            rs.mode = snap.mode
            rs.decay_profile = snap.decay_profile
            rs.decay_weight = snap.decay_weight
            rs.last_live_input = snap.last_input
            rs.mode_authority = snap.mode_authority
            rs.hint_policy = snap.hint_policy
            if snap.last_live_wall_time is not None:
                rs.last_live_time = max(0.0, time.monotonic() - max(0.0, time.time() - snap.last_live_wall_time))

        # Restore thermostat state if available
        if not info.thermostats:
            info.thermostats = make_thermostats(self.config.n_nodes)
        thermo_path = self.persistence.state_dir / f"{name}_thermostats.json"
        if thermo_path.exists():
            try:
                import json as _json
                thermo_data = _json.loads(thermo_path.read_text())
                for thermostat, d in zip(info.thermostats, thermo_data):
                    thermostat.load_state_dict(d)
                log.info("  thermostats restored: %s",
                    ", ".join(f"{t.layer_name}:rho={t.rho:.3f}" for t in info.thermostats))
            except Exception as e:
                log.warning("thermostat restore failed for '%s': %s", name, e)

        log.info(
            "restored '%s' (entity=%s, backend=%s, ticks=%d, mode=%s)",
            name,
            snap.entity,
            snapshot_backend,
            snap.tick_count,
            snap.mode,
        )
        return {
            "type": "restore_response",
            "name": name,
            "entity": snap.entity,
            "backend": snapshot_backend,
            "tick_count": snap.tick_count,
        }

    def snapshot_all(self):
        """Snapshot every active handle."""
        for name in list(self.handles):
            try:
                self.snapshot_handle(name)
            except Exception as e:
                log.error("snapshot_all failed for '%s': %s", name, e)

    def restore_all(self):
        """Restore all handles from saved snapshots."""
        names = self.persistence.list_snapshots()
        for name in names:
            try:
                self.restore_handle(name)
            except Exception as e:
                log.error("restore_all failed for '%s': %s", name, e)
        if names:
            log.info("restored %d handles from disk: %s", len(names), ", ".join(names))

    # ------------------------------------------------------------------
    # Full state transfer (checkout / checkin)
    # ------------------------------------------------------------------

    def pull_handle_state(self, name: str) -> dict:
        """Return the full h1/h2/h3 hidden state as base64-encoded numpy arrays.

        This is the 'checkout' operation — the caller gets the complete
        576-dimensional state to evolve locally (e.g., during coupled generation).
        """
        import base64
        info = self._require_handle(name)
        if not self.bridge.supports_pull_push(name):
            raise ValueError(
                f"backend '{info.backend}' for handle '{name}' does not support "
                "pull_state in phase 1; LSM shadow handles only expose public trace state"
            )
        h1, h2, h3 = self.bridge.read_state(name)
        rs = self.rehearsal.get_state(name)
        return {
            "type": "pull_state_response",
            "name": name,
            "backend": info.backend,
            "h1": base64.b64encode(h1.astype(np.float32).tobytes()).decode(),
            "h2": base64.b64encode(h2.astype(np.float32).tobytes()).decode(),
            "h3": base64.b64encode(h3.astype(np.float32).tobytes()).decode(),
            "n_nodes": self.config.n_nodes,
            "tick_count": info.tick_count,
            "mode": rs.mode if rs else "unknown",
        }

    def push_handle_state(
        self,
        name: str,
        h1_b64: str,
        h2_b64: str,
        h3_b64: str,
        tick_delta: int = 0,
        last_input: Optional[list] = None,
        meta: Optional[dict] = None,
    ) -> dict:
        """Overwrite a handle's hidden state from base64-encoded numpy arrays.

        This is the 'checkin' operation — the caller returns the evolved state
        after coupled generation. The handle now carries the full dynamical
        imprint of what was said.

        tick_delta: number of ticks the caller ran (e.g. tokens generated).
        last_input: representative input vector for rehearsal afterimage.
        """
        import base64
        info = self._require_handle(name)
        if not self.bridge.supports_pull_push(name):
            raise ValueError(
                f"backend '{info.backend}' for handle '{name}' does not support "
                "push_state in phase 1; LSM shadow handles only expose public trace state"
            )
        n = self.config.n_nodes
        h1 = np.frombuffer(base64.b64decode(h1_b64), dtype=np.float32).reshape(1, n)
        h2 = np.frombuffer(base64.b64decode(h2_b64), dtype=np.float32).reshape(1, n)
        h3 = np.frombuffer(base64.b64decode(h3_b64), dtype=np.float32).reshape(1, n)
        self.bridge.set_state(name, h1, h2, h3)

        # Credit the generation ticks to the handle
        info.tick_count += tick_delta
        info.last_tick_time = time.monotonic()

        # Update rehearsal — use the caller's last input as the afterimage
        # so rehearsal replays something from the generation, not stale feeder data
        rs = self.rehearsal.get_state(name)
        if rs:
            rs.last_live_time = time.monotonic()
            rs.ticks_since_live = 0
            rs.decay_weight = 1.0
            if rs.mode == "quiet":
                rs.mode = "rehearse"
            if last_input is not None:
                rs.last_live_input = np.array(last_input, dtype=np.float32).ravel()

        norms = self.bridge.h_norms(name)
        info.h_norm_ring.append(norms)
        info.last_live_meta = stamp_event_meta(info, build_push_meta(meta, tick_delta))
        update_lane_meta(info, info.last_live_meta)
        if info.last_live_meta:
            info.provenance_ring.append(build_provenance_entry(info, info.last_live_meta))

        # Feed current layers into thermostats so post-checkin metrics reflect
        # the new state even though we do not synthesize a scalar trajectory point.
        if info.thermostats:
            try:
                for thermostat, h_i in zip(info.thermostats, (h1, h2, h3)):
                    thermostat.step(h_i)
            except Exception:
                pass

        hint = extract_rehearsal_hint(info.last_live_meta)
        hint_applied = False
        if hint is not None:
            hint_applied = self.rehearsal.maybe_apply_hint(name, hint)
            rs = self.rehearsal.get_state(name)

        log.info(
            "push_state '%s': source=%s, h_norms=[%.3f, %.3f, %.3f], +%d ticks",
            name,
            info.last_live_meta.get("source", "push_state"),
            *norms,
            tick_delta,
        )
        return {
            "type": "push_state_response",
            "name": name,
            "backend": info.backend,
            "ok": True,
            "h_norms": list(norms),
            "tick": info.tick_count,
            "mode": rs.mode if rs else "unknown",
            "mode_authority": rs.mode_authority if rs else "system",
            "hint_policy": rs.hint_policy if rs else "off",
            "last_live_meta": info.last_live_meta,
            "rehearsal_hint": hint,
            "hint_status": classify_hint_status(rs, hint),
            "hint_applied": hint_applied,
        }

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_handles(self) -> dict:
        now = time.monotonic()
        handles = []
        for name, info in self.handles.items():
            rs = self.rehearsal.get_state(name)
            elapsed = now - info.last_tick_time if info.last_tick_time > 0 else None
            last_feeder_meta = info.last_lane_meta.get("feeder") if isinstance(info.last_lane_meta, dict) else None
            last_generation_meta = info.last_lane_meta.get("generation") if isinstance(info.last_lane_meta, dict) else None
            memory_meta = last_feeder_meta or info.last_live_meta
            handles.append({
                "name": name,
                "entity": info.entity,
                "backend": info.backend,
                "mode": rs.mode if rs else "unknown",
                "mode_authority": rs.mode_authority if rs else "system",
                "hint_policy": rs.hint_policy if rs else "off",
                "decay_weight": round(rs.decay_weight, 4) if rs else 0.0,
                "tick_count": info.tick_count,
                "last_tick_ago": round(elapsed, 1) if elapsed is not None else None,
                "last_source": info.last_live_meta.get("source"),
                "last_event_id": info.last_live_meta.get("event_id"),
                "last_event_seq": info.last_live_meta.get("event_seq"),
                "last_source_timestamp": info.last_live_meta.get("source_timestamp"),
                "last_feeder_source": last_feeder_meta.get("source") if isinstance(last_feeder_meta, dict) else None,
                "last_generation_source": last_generation_meta.get("source") if isinstance(last_generation_meta, dict) else None,
                "memory_role": memory_meta.get("memory_role") if isinstance(memory_meta, dict) else None,
                "rehearsal_hint": extract_rehearsal_hint(info.last_live_meta),
                "hint_status": classify_hint_status(rs, extract_rehearsal_hint(info.last_live_meta)),
                "last_live_meta": info.last_live_meta,
                "last_feeder_meta": last_feeder_meta,
                "last_generation_meta": last_generation_meta,
                "recent_sources": [entry.get("source") for entry in list(info.provenance_ring)[-3:] if entry.get("source")],
                "recent_event_ids": [entry.get("event_id") for entry in list(info.provenance_ring)[-3:] if entry.get("event_id")],
                "provenance": list(info.provenance_ring)[-3:],
            })
        return {"type": "list_handles_response", "handles": handles}

    # ------------------------------------------------------------------
    # Protocol dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, msg: dict) -> dict:
        """Route a JSON message to the appropriate handler."""
        msg_type = msg.get("type", "")
        try:
            if msg_type == "create_handle":
                info = self.create_handle(
                    msg["name"],
                    msg.get("entity", "unknown"),
                    backend=msg.get("backend"),
                )
                return {
                    "type": "create_handle_response",
                    "name": msg["name"],
                    "backend": info.backend,
                    "ok": True,
                }

            elif msg_type == "clone_handle":
                return self.clone_handle(
                    msg["source"],
                    msg["name"],
                    entity=msg.get("entity"),
                    mode=msg.get("mode", "quiet"),
                    decay_profile=msg.get("decay_profile"),
                    meta=msg.get("meta"),
                )

            elif msg_type == "tick":
                vec = np.array(msg["input"], dtype=np.float32)
                return self.tick(msg["name"], vec, meta=msg.get("meta"))

            elif msg_type == "tick_text":
                return self.tick_text(msg["name"], msg["text"])

            elif msg_type == "read_state":
                return self.read_state(msg["name"])

            elif msg_type == "set_mode":
                return self.set_mode(msg["name"], msg["mode"], msg.get("decay_profile"))

            elif msg_type == "set_hint_policy":
                return self.set_hint_policy(msg["name"], msg["policy"])

            elif msg_type == "trajectory":
                return self.trajectory(msg["name"], msg.get("last_n", 20))

            elif msg_type == "resonance":
                return self.resonance(msg["name_a"], msg["name_b"])

            elif msg_type == "snapshot":
                return self.snapshot_handle(msg["name"])

            elif msg_type == "restore":
                return self.restore_handle(msg["name"])

            elif msg_type == "list_handles":
                return self.list_handles()

            elif msg_type == "pull_state":
                return self.pull_handle_state(msg["name"])

            elif msg_type == "push_state":
                return self.push_handle_state(
                    msg["name"], msg["h1"], msg["h2"], msg["h3"],
                    tick_delta=msg.get("tick_delta", 0),
                    last_input=msg.get("last_input"),
                    meta=msg.get("meta"),
                )

            elif msg_type == "destroy_handle":
                self.destroy_handle(msg["name"])
                return {"type": "destroy_handle_response", "name": msg["name"], "ok": True}

            elif msg_type == "layer_metrics":
                return self.layer_metrics(msg["name"])

            else:
                return {"type": "error", "message": f"unknown message type: {msg_type}"}

        except (KeyError, ValueError, TypeError) as e:
            return {"type": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def ws_handler(self, websocket):
        peer = websocket.remote_address
        log.info("client connected: %s", peer)
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "invalid JSON"}))
                    continue
                result = await self.dispatch(msg)
                await websocket.send(json.dumps(result))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            log.info("client disconnected: %s", peer)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def rehearsal_loop(self):
        """Background task: tick all non-quiet handles at regular intervals."""
        while not self._shutdown.is_set():
            try:
                inputs = self.rehearsal.get_rehearsal_inputs()
                for name, vec in sorted(
                    inputs.items(),
                    key=lambda item: item[0].endswith(SHADOW_SUFFIX),
                ):
                    if name in self.handles:
                        output = self.bridge.tick(name, vec.reshape(1, -1))
                        norms = self.bridge.h_norms(name)
                        info = self.handles[name]
                        info.tick_count += 1
                        info.output_ring.append(output)
                        info.h_norm_ring.append(norms)

                        # Per-layer thermostatic decay: each layer adapts
                        # independently based on its own entropy + saturation.
                        if info.thermostats:
                            try:
                                h1, h2, h3 = self.bridge.get_layer_states(name)
                                layers = [h1, h2, h3]
                                for i, (thermostat, h_i) in enumerate(zip(info.thermostats, layers)):
                                    rho_i = thermostat.step(h_i)
                                    # Scale this layer's state by its adaptive rho
                                    # (replaces the uniform decay from rehearsal)
                                    layers[i] = h_i * rho_i
                                self.bridge.set_state(name, *[l.reshape(1, -1) for l in layers])
                            except Exception:
                                pass
                        if name.endswith(SHADOW_SUFFIX):
                            self._log_shadow_metrics(name, source="rehearsal")
            except Exception as e:
                log.error("rehearsal tick error: %s", e)

            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=REHEARSAL_INTERVAL)
                break  # shutdown signaled
            except asyncio.TimeoutError:
                pass  # normal: interval elapsed, loop again

    async def auto_snapshot_loop(self):
        """Background task: snapshot all handles periodically."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=AUTO_SNAPSHOT_INTERVAL)
                break
            except asyncio.TimeoutError:
                pass
            if self.handles:
                self.snapshot_all()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    async def run(self, host: str = "127.0.0.1", port: int = 7881):
        """Start the service."""
        # Restore saved handles
        self.restore_all()

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: self._shutdown.set())

        # Start background tasks
        rehearsal_task = asyncio.create_task(self.rehearsal_loop())
        snapshot_task = asyncio.create_task(self.auto_snapshot_loop())

        backend_name = self.bridge.backend_label()
        log.info("listening on %s:%d (input_dim=%d, n_nodes=%d, backend=%s)",
                 host, port, self.config.input_dim, self.config.n_nodes, backend_name)

        async with serve(self.ws_handler, host, port):
            await self._shutdown.wait()

        log.info("shutting down...")
        rehearsal_task.cancel()
        snapshot_task.cancel()

        # Final snapshot
        if self.handles:
            self.snapshot_all()
            log.info("final snapshot complete")

        log.info("goodbye")


def main():
    ap = argparse.ArgumentParser(description="ANE triple reservoir service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7881)
    ap.add_argument("--state-dir", type=Path, default=Path("state"))
    ap.add_argument("--input-dim", type=int, default=32)
    ap.add_argument("--n-nodes", type=int, default=192)
    ap.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "mlx", "coreml", "lsm_shadow"],
        help="Reservoir backend (default: numpy)",
    )
    args = ap.parse_args()

    service = ReservoirService(
        input_dim=args.input_dim,
        n_nodes=args.n_nodes,
        state_dir=args.state_dir,
        backend=args.backend,
    )
    asyncio.run(service.run(host=args.host, port=args.port))


if __name__ == "__main__":
    main()
