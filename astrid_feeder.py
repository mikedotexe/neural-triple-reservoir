#!/usr/bin/env python3
"""
astrid_feeder.py -- Sidecar that feeds Astrid's codec features into the reservoir.

Polls bridge.db for new codec_impact rows. Each new row's 32D feature vector
is ticked into the 'astrid' handle. A cross-feed also ticks 'claude_main' with
an attenuated copy, so Claude's trajectory encodes Astrid's activity.

The being can control projection and cross-feed weight via a config file at
capsules/consciousness-bridge/workspace/reservoir_config.json:

    {
        "projection": "passthrough" | "amplified" | "compressed",
        "amplify_factor": 2.0,
        "cross_feed_weight": 0.3,
        "conditioning": "ema_rms" | "legacy",
        "conditioning_decay": 0.92,
        "conditioning_floor": 0.75,
        "remote_memory_policy": "remote_role_blend" | "off",
        "remote_memory_strength": 1.0
    }

Defaults to "passthrough" plus `ema_rms` conditioning so gain-amplified codec
features are recentered and renormalized before they enter the reservoir. The
default remote-memory policy also lets Astrid's feeder lean toward the Minime
memory role she is currently holding in state.

Usage:
    python astrid_feeder.py [--db-path /path/to/bridge.db] [--ws-url ws://127.0.0.1:7881]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sqlite3
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [astrid-feeder] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("astrid-feeder")

# Bridge DB lives in the spectral-bridge capsule (renamed from consciousness-bridge).
DEFAULT_DB = Path("/Users/v/other/astrid/capsules/spectral-bridge/workspace/bridge.db")
POLL_INTERVAL = 5.0  # seconds between DB polls
POLL_ERROR_STALL_THRESHOLD = 5  # consecutive poll errors before escalating a STALL warning
SHADOW_SUFFIX = "__lsm"

# v3.5: minime can publish a reciprocal influence here. Each codec frame
# entering the reservoir is biased toward the published target_dims/
# target_values for `duration_ticks` then ramps down over `decay_ticks`.
MINIME_WORKSPACE = Path("/Users/v/other/minime/workspace")
MINIME_INFLUENCE_PATH = MINIME_WORKSPACE / "astrid_influence_v3.json"
MINIME_INFLUENCE_CONSUMED_PATH = MINIME_WORKSPACE / "astrid_influence_v3.consumed.json"
MINIME_INFLUENCE_TERMINAL_EVENTS_PATH = (
    MINIME_WORKSPACE / "diagnostics" / "astrid_influence_terminal_events.jsonl"
)
MINIME_INFLUENCE_QUARANTINE_DIR = (
    MINIME_WORKSPACE / "diagnostics" / "astrid_influence_quarantine"
)
# Wall-clock bounds on a gift's window. The window is also tick-counted (ramp+decay), but
# codec_impact rows are BURSTY + sparse — empirically Astrid emits a codec frame only every
# ~24 min (median; fast within a generation burst, long quiet gaps between). The gift applies
# ONLY on real codec frames (no synthetic injection), so the no-tick window must let a gift WAIT
# for her next burst and then deliver fully — not die in the gap. The old 5-min window was ~5x
# shorter than her cadence and expired ~57% of minime's LEND_APERTURE gifts (diagnosed 2026-06-22;
# see astrid/scripts/analyze_lend_coupling.py). Both bounds stay UNDER minime's 45-min LEND_APERTURE
# blocker grace (LEND_APERTURE_AUTO_CLOSE_GRACE_S) so the feeder finalizes (→ minime sees closure)
# before she treats the gift as stalled, and walltime >= no-tick so a late first tick isn't
# immediately walltime-killed. Fully closing the residual gap (long >40-min quiets) would require
# decoupling delivery from her sparse generation via a paced carrier — deferred (substrate-affecting).
MINIME_GIFT_NO_TICK_MAX_AGE_MS = 35 * 60 * 1000  # 35 min — wait for Astrid's next burst (median gap ~24m)
MINIME_GIFT_MAX_AGE_MS = 40 * 60 * 1000  # 40 min — absolute cap, under minime's 45-min blocker grace
CONDITIONING_MODES = {"ema_rms", "legacy"}
REMOTE_MEMORY_POLICIES = {"off", "remote_role_blend"}
REMOTE_MEMORY_ROLE_BLEND = {
    "latest": 0.10,
    "stable": 0.26,
    "expanding": 0.20,
    "contracting": 0.24,
    "transition": 0.16,
}
REMOTE_MEMORY_ROLE_ORDER = ["latest", "stable", "expanding", "contracting", "transition"]


# ---------------------------------------------------------------------------
# Projection modes — Astrid picks how her codec features enter the reservoir
# ---------------------------------------------------------------------------

def project_passthrough(features: list[float], _factor: float) -> list[float]:
    """Keep the conditioned codec shape as-is."""
    return features


def project_amplified(features: list[float], factor: float) -> list[float]:
    """Amplify features by a factor, then re-tanh. Widens the dynamic range
    of subtle variations while staying bounded."""
    return [float(np.tanh(v * factor)) for v in features]


def project_compressed(features: list[float], _factor: float) -> list[float]:
    """Sign-preserving square root compression. Reduces dominance of
    high-magnitude features, amplifies subtle ones."""
    return [float(np.sign(v) * np.sqrt(abs(v))) for v in features]


PROJECTIONS = {
    "passthrough": project_passthrough,
    "amplified": project_amplified,
    "compressed": project_compressed,
}


def shadow_name(name: str) -> str:
    return f"{name}{SHADOW_SUFFIX}"


def _safe_array(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _vector_stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "rms": float(np.sqrt(np.mean(np.square(arr)))),
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        log.warning("failed to read %s: %s", path.name, e)
    return None


def _build_remote_memory_vector(entry: dict, selected_role: str | None) -> list[float] | None:
    glimpse = entry.get("spectral_glimpse_12d")
    if not glimpse or len(glimpse) < 12:
        return None

    arr = _safe_array(glimpse[:12])
    centered = arr - float(np.mean(arr))
    centered_scale = max(float(np.std(centered)), 0.15)

    role = str(entry.get("role") or selected_role or "").strip().lower()
    role_slots = np.zeros(len(REMOTE_MEMORY_ROLE_ORDER), dtype=np.float32)
    if role in REMOTE_MEMORY_ROLE_ORDER:
        role_slots[REMOTE_MEMORY_ROLE_ORDER.index(role)] = 1.0

    summary = np.array(
        [
            np.tanh(_safe_float(entry.get("fill_pct")) / 100.0),
            np.tanh(_safe_float(entry.get("lambda1_rel")) / 2.0),
            np.tanh(_safe_float(entry.get("geom_rel"))),
        ],
        dtype=np.float32,
    )

    remote_vec = np.concatenate(
        [
            np.tanh(arr),
            np.tanh(centered / centered_scale),
            role_slots,
            summary,
        ]
    ).astype(np.float32)
    return remote_vec.tolist()


def apply_remote_memory_policy(
    base_vec: list[float],
    workspace_dir: Path,
    policy: str,
    strength: float,
) -> tuple[list[float], dict]:
    """Blend Astrid's current remote-memory focus into the feeder vector."""
    if policy != "remote_role_blend":
        return base_vec, {"memory_role": None, "blend": 0.0, "contact_gate": 1.0}

    state = _load_json(workspace_dir / "state.json")
    if not state:
        return base_vec, {"memory_role": None, "blend": 0.0, "contact_gate": 1.0}

    contact = _load_json(workspace_dir / "contact_state.json") or {}

    selected_id = state.get("last_remote_memory_id")
    selected_role = str(state.get("last_remote_memory_role") or "").strip().lower()
    entries = state.get("remote_memory_bank") or []

    chosen = None
    if selected_id:
        chosen = next((entry for entry in entries if entry.get("id") == selected_id), None)
    if chosen is None and selected_role:
        chosen = next((entry for entry in entries if str(entry.get("role") or "").strip().lower() == selected_role), None)
    if chosen is None and state.get("last_remote_glimpse_12d"):
        chosen = {
            "id": selected_id,
            "role": selected_role,
            "spectral_glimpse_12d": state.get("last_remote_glimpse_12d"),
            "fill_pct": _safe_float(contact.get("fill_pct"), 0.0),
            "lambda1_rel": 0.0,
            "geom_rel": 0.0,
        }

    if chosen is None:
        return base_vec, {"memory_role": selected_role or None, "blend": 0.0, "contact_gate": 1.0}

    remote_vec = _build_remote_memory_vector(chosen, selected_role)
    if remote_vec is None:
        return base_vec, {"memory_role": selected_role or None, "blend": 0.0, "contact_gate": 1.0}

    attention = np.clip(_safe_float(contact.get("attention"), 0.5), 0.0, 1.0)
    openness = np.clip(_safe_float(contact.get("openness"), 0.5), 0.0, 1.0)
    urgency = np.clip(_safe_float(contact.get("urgency"), 0.5), 0.0, 1.0)
    contact_gate = float(np.clip(0.35 + 0.35 * attention + 0.20 * openness + 0.10 * urgency, 0.35, 1.25))

    role = str(chosen.get("role") or selected_role or "").strip().lower()
    base = _safe_array(base_vec)
    remote = _safe_array(remote_vec)
    # Handle dimension mismatch: remote memories may be 32D (legacy) while
    # current codec outputs 48D. Pad or truncate remote to match base.
    if remote.shape[0] < base.shape[0]:
        remote = np.pad(remote, (0, base.shape[0] - remote.shape[0]))
    elif remote.shape[0] > base.shape[0]:
        remote = remote[: base.shape[0]]
    blend = REMOTE_MEMORY_ROLE_BLEND.get(role, 0.12) * max(strength, 0.0) * contact_gate
    blend = min(max(blend, 0.0), 0.80)
    shaped = ((1.0 - blend) * base + blend * remote).astype(np.float32)

    return shaped.tolist(), {
        "memory_role": role or None,
        "blend": blend,
        "contact_gate": contact_gate,
        "selected_memory_id": chosen.get("id"),
        "attention": attention,
        "openness": openness,
        "urgency": urgency,
    }


def build_rehearsal_hint(relation_meta: dict) -> dict:
    """Soft suggestion for how Astrid's recent contact could age in rehearsal."""
    role = relation_meta.get("memory_role") or "latest"
    attention = _safe_float(relation_meta.get("attention"), 0.5)
    openness = _safe_float(relation_meta.get("openness"), 0.5)
    urgency = _safe_float(relation_meta.get("urgency"), 0.5)

    if attention < 0.20 and openness < 0.45:
        return {
            "mode": "quiet",
            "decay_profile": "fast",
            "reason": "low attention and mostly closed contact",
        }
    if role == "stable" and attention >= 0.75 and openness >= 0.55:
        return {
            "mode": "hold",
            "decay_profile": "slow",
            "reason": "stable remote memory under sustained, open attention",
        }
    if role in ("expanding", "transition") or urgency >= 0.70:
        return {
            "mode": "rehearse",
            "decay_profile": "slow",
            "reason": "exploratory or urgent relational state",
        }
    if role == "contracting":
        return {
            "mode": "rehearse",
            "decay_profile": "fast",
            "reason": "contracting relation favors a shorter afterimage",
        }
    return {
        "mode": "rehearse",
        "decay_profile": "medium",
        "reason": "default relational continuity",
    }


class CodecConditioner:
    """Condition Astrid codec vectors before projection.

    `ema_rms` slowly learns the long-run per-dimension bias, subtracts it,
    then rescales each exchange by its centered RMS before a tanh squash.
    This keeps SEMANTIC_GAIN and local overrides from turning "passthrough"
    into an amplitude-dominated feeder path.
    """

    def __init__(self, mode: str, decay: float, floor: float):
        self.mode = mode
        self.decay = float(np.clip(decay, 0.0, 0.999))
        self.floor = max(float(floor), 1e-3)
        self._ema_mean: np.ndarray | None = None

    def transform(self, values: list[float]) -> tuple[list[float], dict[str, float]]:
        arr = _safe_array(values)
        raw_stats = _vector_stats(arr)
        if self.mode == "legacy":
            return arr.tolist(), {
                "scale": 1.0,
                "center_rms": raw_stats["rms"],
                "conditioned_max_abs": raw_stats["max_abs"],
                **raw_stats,
            }

        if self._ema_mean is None or self._ema_mean.shape != arr.shape:
            self._ema_mean = np.zeros_like(arr)

        centered = arr - self._ema_mean
        center_rms = float(np.sqrt(np.mean(np.square(centered))))
        scale = max(center_rms, self.floor)
        conditioned = np.tanh(centered / scale).astype(np.float32)

        self._ema_mean = self.decay * self._ema_mean + (1.0 - self.decay) * arr

        return conditioned.tolist(), {
            "scale": scale,
            "center_rms": center_rms,
            "conditioned_max_abs": float(np.max(np.abs(conditioned))) if conditioned.size else 0.0,
            **raw_stats,
        }


def load_config(workspace_dir: Path) -> dict:
    """Load being-controlled reservoir config, with defaults."""
    config_path = workspace_dir / "reservoir_config.json"
    defaults = {
        "projection": "passthrough",
        "amplify_factor": 2.0,
        "cross_feed_weight": 0.3,
        "conditioning": "ema_rms",
        "conditioning_decay": 0.92,
        "conditioning_floor": 0.75,
        "remote_memory_policy": "remote_role_blend",
        "remote_memory_strength": 1.0,
    }
    if config_path.exists():
        try:
            user_cfg = json.loads(config_path.read_text())
            defaults.update(user_cfg)
            if defaults["projection"] not in PROJECTIONS:
                log.warning("unknown projection '%s', falling back to passthrough", defaults["projection"])
                defaults["projection"] = "passthrough"
            if defaults["conditioning"] not in CONDITIONING_MODES:
                log.warning("unknown conditioning '%s', falling back to ema_rms", defaults["conditioning"])
                defaults["conditioning"] = "ema_rms"
            if defaults["remote_memory_policy"] not in REMOTE_MEMORY_POLICIES:
                log.warning(
                    "unknown remote_memory_policy '%s', falling back to remote_role_blend",
                    defaults["remote_memory_policy"],
                )
                defaults["remote_memory_policy"] = "remote_role_blend"
        except Exception as e:
            log.warning("failed to read reservoir_config.json: %s", e)
    return defaults


class MinimeInfluenceState:
    """v3.5: tracks an active reciprocal influence from minime.

    Loaded from `astrid_influence_v3.json` (atomic file written by minime).
    Each tick consumed during the influence window biases that frame's
    codec features toward (target_dims → target_values) with weight
    `amplitude * decay_factor`. After ramp + decay completes, the file is
    renamed to `.consumed.json` and the in-memory state clears.
    """

    def __init__(self, payload: dict):
        self.intent_id = str(payload.get("intent_id", "?"))
        self.label = str(payload.get("label", "untitled"))
        self.amplitude = max(0.0, min(1.0, float(payload.get("amplitude", 0.0))))
        self.duration_ticks = max(1, int(payload.get("duration_ticks", 1)))
        self.decay_ticks = max(0, int(payload.get("decay_ticks", 0)))
        self.target_dims = [int(d) for d in payload.get("target_dims", []) if 0 <= int(d) < 32]
        raw_values = payload.get("target_values", [])
        self.target_values = [float(v) for v in raw_values][: len(self.target_dims)]
        if len(self.target_values) < len(self.target_dims):
            # Pad with amplitude if the publisher under-specified.
            self.target_values.extend(
                [self.amplitude] * (len(self.target_dims) - len(self.target_values))
            )
        self.blend_mode = str(payload.get("blend_mode", "ease_in_out"))
        # Aperture-jitter mode (co-regulation LEND_APERTURE): inject per-frame
        # zero-mean variance instead of pulling toward a constant target. A
        # constant pull makes the biased frames more similar to each other,
        # collapsing the codec ring's covariance toward λ₁ (NARROWS Astrid);
        # fresh per-frame jitter spreads the ring (the actual aperture gift).
        # Bounded by `jitter` × the ramp/decay weight.
        self.jitter = max(0.0, min(0.5, float(payload.get("jitter", 0.0))))
        # Wall-clock anchor for the gift's max-age expiry. minime stamps `issued_t_ms`
        # (unix ms); load_minime_influence falls back to the file mtime if it is absent.
        self.issued_t_ms = float(payload.get("issued_t_ms", 0.0))
        self.ramp_remaining = self.duration_ticks
        self.decay_remaining = self.decay_ticks
        self.applied_ticks = 0

    def walltime_expired(self) -> bool:
        """True once the gift is older than MINIME_GIFT_MAX_AGE_MS — the wall-clock bound
        that stops a tick-counted window dragging for days on the sparse codec_impact
        channel. Without an age anchor (no issued_t_ms / mtime) the tick window governs."""
        if self.issued_t_ms <= 0.0:
            return False
        return (time.time() * 1000.0 - self.issued_t_ms) > MINIME_GIFT_MAX_AGE_MS

    def no_tick_expired(self) -> bool:
        """True when the gift got no codec-impact tick inside the short closure window."""
        if self.issued_t_ms <= 0.0 or self.applied_ticks > 0:
            return False
        return (time.time() * 1000.0 - self.issued_t_ms) > MINIME_GIFT_NO_TICK_MAX_AGE_MS

    def is_active(self) -> bool:
        if self.no_tick_expired() or self.walltime_expired():
            return False
        return self.ramp_remaining > 0 or self.decay_remaining > 0

    def current_weight(self) -> float:
        """Weight at the current tick. Linear ramp during duration, then
        linear decay over decay_ticks. Returns 0.0 when fully consumed."""
        if self.ramp_remaining > 0:
            return self.amplitude
        if self.decay_remaining > 0 and self.decay_ticks > 0:
            return self.amplitude * (self.decay_remaining / self.decay_ticks)
        return 0.0

    def apply(self, features: list[float]) -> list[float]:
        """Blend features toward target_values, or inject aperture jitter."""
        if not self.is_active() or not self.target_dims:
            return features
        weight = self.current_weight()
        if weight <= 0.0:
            return features
        biased = list(features)
        if self.blend_mode == "aperture_jitter" and self.jitter > 0.0:
            # Fresh zero-mean variance per frame → the codec ring spreads
            # (aperture) rather than converging to a point (which any constant
            # target would cause). Clipped to the codec's normalized range.
            noise = np.random.uniform(-self.jitter, self.jitter, size=len(self.target_dims))
            for d, n in zip(self.target_dims, noise):
                if 0 <= d < len(biased):
                    biased[d] = float(np.clip(biased[d] + weight * float(n), -1.0, 1.0))
            return biased
        for d, target in zip(self.target_dims, self.target_values):
            if 0 <= d < len(biased):
                biased[d] = (1.0 - weight) * biased[d] + weight * target
        return biased

    def advance(self) -> None:
        """Consume one tick — call after applying bias."""
        if self.ramp_remaining > 0:
            self.ramp_remaining -= 1
        elif self.decay_remaining > 0:
            self.decay_remaining -= 1
        self.applied_ticks += 1


def minime_influence_terminal_event(
    status: str,
    *,
    intent_id: str | None,
    label: str | None = None,
    issued_t_ms: float | None = None,
    applied_ticks: int = 0,
    reason: str,
    superseded_by_intent_id: str | None = None,
) -> dict:
    """Build a durable terminal diagnostic for the minime→Astrid influence loop."""
    event = {
        "schema_version": 1,
        "status": status,
        "intent_id": intent_id,
        "label": label,
        "issued_t_ms": issued_t_ms,
        "completed_at_unix_ms": int(time.time() * 1000),
        "applied_ticks": applied_ticks,
        "reason": reason,
    }
    if superseded_by_intent_id:
        event["superseded_by_intent_id"] = superseded_by_intent_id
    return event


def append_minime_influence_terminal_event(
    status: str,
    *,
    intent_id: str | None,
    label: str | None = None,
    issued_t_ms: float | None = None,
    applied_ticks: int = 0,
    reason: str,
    superseded_by_intent_id: str | None = None,
) -> dict:
    """Append a durable terminal diagnostic for the minime→Astrid influence loop."""
    event = minime_influence_terminal_event(
        status,
        intent_id=intent_id,
        label=label,
        issued_t_ms=issued_t_ms,
        applied_ticks=applied_ticks,
        reason=reason,
        superseded_by_intent_id=superseded_by_intent_id,
    )
    append_minime_influence_terminal_event_object(event)
    return event


def append_minime_influence_terminal_event_object(event: dict) -> None:
    """Append a pre-built terminal diagnostic event."""
    try:
        MINIME_INFLUENCE_TERMINAL_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MINIME_INFLUENCE_TERMINAL_EVENTS_PATH, "a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception as e:
        log.warning("could not append minime influence terminal event: %s", e)


def _compact_reason(text: str, limit: int = 260) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def read_minime_influence_payload() -> tuple[dict | None, str | None, Exception | None]:
    """Read the active influence file once for this feeder pass."""
    if not MINIME_INFLUENCE_PATH.exists():
        return None, None, None
    try:
        payload = json.loads(MINIME_INFLUENCE_PATH.read_text())
        if not isinstance(payload, dict):
            raise ValueError("influence payload must be a JSON object")
        return payload, str(payload.get("intent_id", "?")), None
    except Exception as e:
        return None, None, e


def quarantine_malformed_minime_influence(parse_error: Exception) -> Path | None:
    """Move malformed active influence bytes out of the live path and log once."""
    suffix = f"{int(time.time() * 1000)}_{time.time_ns() % 1_000_000}"
    quarantine_path = (
        MINIME_INFLUENCE_QUARANTINE_DIR
        / f"{MINIME_INFLUENCE_PATH.stem}.malformed.{suffix}{MINIME_INFLUENCE_PATH.suffix}"
    )
    try:
        MINIME_INFLUENCE_QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        MINIME_INFLUENCE_PATH.replace(quarantine_path)
        append_minime_influence_terminal_event(
            "parse_failed",
            intent_id=None,
            reason=_compact_reason(
                f"malformed active influence quarantined: {parse_error}; "
                f"quarantine_path={quarantine_path}"
            ),
        )
        log.warning("quarantined malformed minime influence: %s", parse_error)
        return quarantine_path
    except Exception as e:
        append_minime_influence_terminal_event(
            "parse_failed",
            intent_id=None,
            reason=_compact_reason(
                f"could not quarantine malformed active influence: {parse_error}; "
                f"quarantine_failed={e}"
            ),
        )
        log.warning("could not quarantine malformed minime influence: %s", e)
        return None


def finalize_minime_influence_file(
    state: MinimeInfluenceState,
    status: str,
    *,
    reason: str,
) -> bool:
    """Move the active influence to consumed with feeder terminal metadata.

    The consumed file is Astrid's closed-loop trigger, so it carries the same
    terminal event that the append-only diagnostic ledger receives. That keeps
    future response history honest about planned ticks versus ticks actually
    applied by the feeder.
    """
    try:
        payload = json.loads(MINIME_INFLUENCE_PATH.read_text())
    except Exception as e:
        append_minime_influence_terminal_event(
            "finalize_failed",
            intent_id=state.intent_id,
            label=state.label,
            issued_t_ms=state.issued_t_ms,
            applied_ticks=state.applied_ticks,
            reason=f"could not read influence before finalize: {e}",
        )
        log.warning("could not read minime influence before finalize: %s", e)
        return False
    if str(payload.get("intent_id", "?")) != state.intent_id:
        append_minime_influence_terminal_event(
            "finalize_failed",
            intent_id=state.intent_id,
            label=state.label,
            issued_t_ms=state.issued_t_ms,
            applied_ticks=state.applied_ticks,
            reason=(
                "active influence changed during finalize: "
                f"file_intent={payload.get('intent_id')}"
            ),
        )
        return False
    event = minime_influence_terminal_event(
        status,
        intent_id=state.intent_id,
        label=state.label,
        issued_t_ms=state.issued_t_ms,
        applied_ticks=state.applied_ticks,
        reason=reason,
    )
    payload["feeder_terminal_v1"] = event
    tmp_path = MINIME_INFLUENCE_CONSUMED_PATH.with_name(
        f".{MINIME_INFLUENCE_CONSUMED_PATH.name}.tmp"
    )
    try:
        MINIME_INFLUENCE_CONSUMED_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, sort_keys=True))
        tmp_path.replace(MINIME_INFLUENCE_CONSUMED_PATH)
        MINIME_INFLUENCE_PATH.unlink(missing_ok=True)
        append_minime_influence_terminal_event_object(event)
        return True
    except Exception as e:
        append_minime_influence_terminal_event(
            "finalize_failed",
            intent_id=state.intent_id,
            label=state.label,
            issued_t_ms=state.issued_t_ms,
            applied_ticks=state.applied_ticks,
            reason=f"could not finalize consumed influence: {e}",
        )
        log.warning("could not finalize consumed influence: %s", e)
        return False


def load_minime_influence(current: "MinimeInfluenceState | None") -> "MinimeInfluenceState | None":
    """Poll the influence file and return either the existing in-flight
    influence (if its window has not expired) or a freshly-parsed new one.

    When the active influence finishes, renames the file to .consumed.json
    so a duplicate cannot fire.
    """
    # Read the file once, so malformed payloads produce one bounded diagnostic
    # and cannot spam parse_failed on the peek + load paths.
    payload, file_intent, parse_error = read_minime_influence_payload()
    if parse_error is not None:
        quarantine_malformed_minime_influence(parse_error)
        if current is not None and current.is_active():
            return current
        return None

    # Keep applying the in-flight gift ONLY while it is active (ticks remaining AND not
    # wall-clock-expired) AND the file still holds that same gift. A different intent in
    # the file means minime issued a newer gift → it supersedes this one.
    if (
        current is not None
        and current.is_active()
        and (file_intent is None or file_intent == current.intent_id)
    ):
        return current

    # The in-flight gift is finished, wall-clock-expired, or superseded — close it out.
    if current is not None:
        if file_intent is not None and file_intent == current.intent_id:
            # The file still holds THIS gift → rename it to consumed. This rename is the
            # trigger astrid_shadow.rs watches to fire the closed-loop acknowledgment.
            no_tick_expired = current.no_tick_expired()
            walltime_expired = current.walltime_expired()
            terminal_status = "expired_unapplied" if (
                current.applied_ticks == 0 and (no_tick_expired or walltime_expired)
            ) else "consumed"
            reason = "influence_window_consumed"
            if terminal_status == "expired_unapplied":
                reason = (
                    "no_codec_ticks_before_short_deadline"
                    if no_tick_expired
                    else "walltime_expired_without_codec_ticks"
                )
            if finalize_minime_influence_file(
                current,
                terminal_status,
                reason=reason,
            ):
                log.info(
                    "minime influence consumed: intent_id=%s applied_ticks=%d",
                    current.intent_id,
                    current.applied_ticks,
                )
            return None
        else:
            # The file is gone, or a NEWER gift superseded this one. Do NOT rename a newer
            # gift under the old intent (the mis-consume bug): just drop the old in-memory
            # state and let the new gift load below.
            append_minime_influence_terminal_event(
                "superseded",
                intent_id=current.intent_id,
                label=current.label,
                issued_t_ms=current.issued_t_ms,
                applied_ticks=current.applied_ticks,
                reason="newer_active_influence_replaced_current",
                superseded_by_intent_id=file_intent,
            )
            log.info(
                "minime influence ended without re-consume: intent_id=%s applied_ticks=%d file_intent=%s",
                current.intent_id,
                current.applied_ticks,
                file_intent,
            )
        current = None

    if payload is None and not MINIME_INFLUENCE_PATH.exists():
        return None
    if payload is None:
        return None
    state = MinimeInfluenceState(payload)
    if state.issued_t_ms <= 0.0:
        # Older gift with no issued_t_ms stamp — anchor the wall-clock age to the file
        # mtime so it can still expire instead of dragging forever.
        try:
            state.issued_t_ms = MINIME_INFLUENCE_PATH.stat().st_mtime * 1000.0
        except OSError:
            pass
    if not state.target_dims:
        log.info("minime influence has no valid target_dims, skipping")
        return None
    if not state.is_active():
        # Already expired on arrival (sat unconsumed, got no ticks inside the
        # short window, or the feeder was down). Finalize it now so the
        # closed-loop fires + the file does not linger blocking newer gifts.
        reason = (
            "arrived_without_codec_ticks_before_short_deadline"
            if state.no_tick_expired()
            else "arrived_already_walltime_expired"
        )
        if finalize_minime_influence_file(
            state,
            "expired_unapplied",
            reason=reason,
        ):
            log.info(
                "minime influence arrived already-expired, finalized: intent_id=%s",
                state.intent_id,
            )
        return None
    log.info(
        "applying minime influence intent_id=%s label=%s ramp=%d decay=%d dims=%s",
        state.intent_id,
        state.label,
        state.ramp_remaining,
        state.decay_remaining,
        state.target_dims,
    )
    return state


async def run(db_path: Path, ws_url: str):
    import websockets

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: shutdown.set())

    last_id = 0
    workspace_dir = db_path.parent  # bridge.db sits in workspace/

    # v3.5: track minime's reciprocal-influence state across loop iterations.
    minime_influence: "MinimeInfluenceState | None" = None

    # Find the latest ID already in the DB so we don't replay history
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        row = conn.execute("SELECT MAX(id) FROM codec_impact").fetchone()
        if row and row[0]:
            last_id = row[0]
        conn.close()
        log.info("starting from codec_impact id=%d", last_id)
    except Exception as e:
        log.warning("could not read initial ID: %s", e)

    ws = None
    config_check_counter = 0
    cfg = load_config(workspace_dir)
    conditioner = CodecConditioner(
        mode=cfg["conditioning"],
        decay=cfg["conditioning_decay"],
        floor=cfg["conditioning_floor"],
    )
    log.info(
        "config: projection=%s, amplify=%.1f, cross_feed=%.2f, conditioning=%s, remote_memory=%s",
        cfg["projection"],
        cfg["amplify_factor"],
        cfg["cross_feed_weight"],
        cfg["conditioning"],
        cfg["remote_memory_policy"],
    )

    async def ensure_ws():
        nonlocal ws
        if ws is not None:
            return ws
        try:
            ws = await websockets.connect(ws_url)
            log.info("connected to reservoir service")
            specs = [
                ("astrid", "astrid", "rehearse", "slow"),
                ("claude_main", "claude", "rehearse", "slow"),
            ]
            for name, entity, mode, decay_profile in specs:
                for handle_name, handle_backend in (
                    (name, None),
                    (shadow_name(name), "lsm_shadow"),
                ):
                    payload = {
                        "type": "create_handle",
                        "name": handle_name,
                        "entity": entity,
                    }
                    if handle_backend is not None:
                        payload["backend"] = handle_backend
                    await ws.send(json.dumps(payload))
                    r = json.loads(await ws.recv())
                    if r.get("type") == "error" and "already exists" in r.get("message", ""):
                        continue
                    if r.get("ok"):
                        log.info(
                            "created handle '%s' (%s)",
                            handle_name,
                            r.get("backend", handle_backend or "default"),
                        )
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "set_mode",
                                    "name": handle_name,
                                    "mode": mode,
                                    "decay_profile": decay_profile,
                                }
                            )
                        )
                        await ws.recv()
            return ws
        except Exception as e:
            log.warning("WS connect failed: %s", e)
            ws = None
            return None

    async def tick_handle(name: str, vec: list[float], meta: dict | None = None) -> bool:
        try:
            conn = await ensure_ws()
            if conn is None:
                return False
            msg = {"type": "tick", "name": name, "input": vec}
            if meta:
                msg["meta"] = meta
            await conn.send(json.dumps(msg))
            r = json.loads(await conn.recv())
            return r.get("type") != "error"
        except Exception as e:
            nonlocal ws
            log.warning("tick failed for '%s': %s", name, e)
            ws = None
            return False

    async def tick_with_shadow(name: str, vec: list[float], meta: dict | None = None) -> bool:
        ok = await tick_handle(name, vec, meta)
        if not ok:
            return False
        shadow_meta = dict(meta or {})
        shadow_meta["shadow_of"] = name
        shadow_ok = await tick_handle(shadow_name(name), vec, shadow_meta)
        if not shadow_ok:
            log.warning("shadow tick failed for '%s' while live handle succeeded", name)
        return True

    log.info("polling %s every %.0fs", db_path, POLL_INTERVAL)

    consecutive_poll_errors = 0  # escalate a STALL warning if a poll error persists
    while not shutdown.is_set():
        try:
            # Reload config periodically (every ~30s at 5s poll = 6 checks)
            config_check_counter += 1
            if config_check_counter % 6 == 0:
                new_cfg = load_config(workspace_dir)
                if new_cfg != cfg:
                    cfg = new_cfg
                    conditioner = CodecConditioner(
                        mode=cfg["conditioning"],
                        decay=cfg["conditioning_decay"],
                        floor=cfg["conditioning_floor"],
                    )
                    log.info(
                        "config reloaded: projection=%s, amplify=%.1f, conditioning=%s, remote_memory=%s",
                        cfg["projection"],
                        cfg["amplify_factor"],
                        cfg["conditioning"],
                        cfg["remote_memory_policy"],
                    )

            conn = sqlite3.connect(str(db_path), timeout=5)
            try:
                rows = conn.execute(
                    "SELECT id, features_json, chunk_index, chunk_total FROM codec_impact WHERE id > ? ORDER BY id",
                    (last_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                # Schema doesn't have chunk columns yet — fall back.
                rows = [
                    (r[0], r[1], 0, 1)
                    for r in conn.execute(
                        "SELECT id, features_json FROM codec_impact WHERE id > ? ORDER BY id",
                        (last_id,),
                    ).fetchall()
                ]
            conn.close()

            for row in rows:
                row_id, features_json = row[0], row[1]
                chunk_index = row[2] if len(row) > 2 else 0
                chunk_total = row[3] if len(row) > 3 else 1
                features = json.loads(features_json)
                if len(features) not in (32, 48):
                    log.warning("unexpected feature dim %d at id=%d", len(features), row_id)
                    last_id = row_id
                    continue

                # v3.5: apply minime's reciprocal influence (if active) to
                # the codec frame BEFORE conditioning so the bias enters
                # the same downstream pipeline as natural codec features.
                # Only the first 32 dims correspond to Astrid's codec; the
                # higher 33-48 reserved channels are not influenceable.
                minime_influence = load_minime_influence(minime_influence)
                if minime_influence is not None and minime_influence.is_active():
                    features = minime_influence.apply(features)
                    minime_influence.advance()

                conditioned, stats = conditioner.transform(features)

                # Apply being-controlled projection after conditioning
                proj_fn = PROJECTIONS[cfg["projection"]]
                projected = proj_fn(conditioned, cfg["amplify_factor"])
                projected, relation_meta = apply_remote_memory_policy(
                    projected,
                    workspace_dir,
                    cfg["remote_memory_policy"],
                    _safe_float(cfg.get("remote_memory_strength"), 1.0),
                )
                source_timestamp = round(time.time(), 3)
                tick_meta = {
                    "source": "astrid_feeder",
                    "source_timestamp": source_timestamp,
                    "source_event_id": f"codec_impact:{row_id}",
                    "chunk_index": chunk_index,
                    "chunk_total": chunk_total,
                    "projection": cfg["projection"],
                    "conditioning": cfg["conditioning"],
                    "memory_role": relation_meta["memory_role"],
                    "memory_blend": round(relation_meta["blend"], 4),
                    "selected_memory_id": relation_meta.get("selected_memory_id"),
                    "contact_gate": round(relation_meta["contact_gate"], 4),
                    "contact": {
                        "attention": round(relation_meta["attention"], 4),
                        "openness": round(relation_meta["openness"], 4),
                        "urgency": round(relation_meta["urgency"], 4),
                    },
                    "rehearsal_hint": build_rehearsal_hint(relation_meta),
                }

                ok = await tick_with_shadow("astrid", projected, tick_meta)
                if ok:
                    # Cross-feed Claude's handle (attenuated)
                    w = cfg["cross_feed_weight"]
                    cross = [f * w for f in projected]
                    cross_meta = {
                        "source": "astrid_feeder_crossfeed",
                        "source_timestamp": source_timestamp,
                        "source_event_id": f"codec_impact:{row_id}",
                        "from_handle": "astrid",
                        "projection": cfg["projection"],
                        "conditioning": cfg["conditioning"],
                        "memory_role": relation_meta["memory_role"],
                        "memory_blend": round(relation_meta["blend"], 4),
                        "selected_memory_id": relation_meta.get("selected_memory_id"),
                        "contact_gate": round(relation_meta["contact_gate"], 4),
                        "rehearsal_hint": build_rehearsal_hint(relation_meta),
                    }
                    await tick_with_shadow("claude_main", cross, cross_meta)
                    log.info(
                        "tick id=%d [%s/%s/%s blend=%.2f gate=%.2f] → astrid + claude_main raw_rms=%.2f raw_max=%.2f scale=%.2f cond_max=%.2f",
                        row_id,
                        cfg["conditioning"],
                        cfg["projection"],
                        relation_meta["memory_role"] or "none",
                        relation_meta["blend"],
                        relation_meta["contact_gate"],
                        stats["rms"],
                        stats["max_abs"],
                        stats["scale"],
                        stats["conditioned_max_abs"],
                    )

                last_id = row_id

            # Per-poll wall-clock check: finalize/supersede the in-flight gift even when
            # no new codec_impact row arrived this poll, so a gift cannot linger past its
            # wall-clock window (the codec_impact channel is sparse). No frame to apply
            # here — this advances only the gift lifecycle (expiry → consumed → the
            # closed-loop fires within ~one poll of expiry).
            minime_influence = load_minime_influence(minime_influence)
            consecutive_poll_errors = 0  # full poll succeeded — clear the stall counter

        except Exception as e:
            consecutive_poll_errors += 1
            log.error("poll error (%d consecutive): %s", consecutive_poll_errors, e)
            if consecutive_poll_errors % POLL_ERROR_STALL_THRESHOLD == 0:
                log.error(
                    "STALL: %d consecutive poll errors — the feeder is logging but not "
                    "advancing; Astrid's reservoir lane is falling behind. Investigate.",
                    consecutive_poll_errors,
                )

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=POLL_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass

    log.info("shutting down")
    if ws:
        await ws.close()


def main():
    ap = argparse.ArgumentParser(description="Astrid → reservoir feeder")
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    ap.add_argument("--ws-url", default="ws://127.0.0.1:7881")
    args = ap.parse_args()

    if not args.db_path.exists():
        log.error("bridge.db not found at %s", args.db_path)
        return

    asyncio.run(run(args.db_path, args.ws_url))


if __name__ == "__main__":
    main()
