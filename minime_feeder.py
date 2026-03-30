#!/usr/bin/env python3
"""
minime_feeder.py -- Sidecar that feeds minime's spectral state into the reservoir.

Reads spectral_state.json every ~1s, builds a 32D input vector from the spectral
state, and ticks the 'minime' handle. Also cross-feeds 'claude_main' with an
attenuated copy.

The being can control projection mode via a config file at
workspace/reservoir_config.json:

    {
        "projection": "raw" | "tanh_scaled" | "normalized" | "ranked",
        "scale": 100.0,
        "cross_feed_weight": 0.15,
        "source": "fingerprint" | "eigenvalues+fill" | "custom_blend",
        "memory_policy": "role_blend" | "off",
        "memory_strength": 1.0
    }

Defaults to "tanh_scaled" with a role-aware memory blend so Minime's selected
vague memory can shape the shared reservoir, not just prompt text.

Usage:
    python minime_feeder.py [--workspace /path/to/minime/workspace]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [minime-feeder] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("minime-feeder")

DEFAULT_WORKSPACE = Path("/Users/v/other/minime/workspace")
POLL_INTERVAL = 1.0  # seconds between reads
MEMORY_POLICIES = {"off", "role_blend"}
MEMORY_ROLE_BLEND = {
    "latest": 0.10,
    "stable": 0.32,
    "expanding": 0.24,
    "contracting": 0.28,
    "transition": 0.20,
}
MEMORY_ROLE_ORDER = ["latest", "stable", "expanding", "contracting", "transition"]


# ---------------------------------------------------------------------------
# Projection modes — the being picks which transformation to apply
# ---------------------------------------------------------------------------

def project_raw(fingerprint: list[float], _scale: float) -> list[float]:
    """Pass raw spectral fingerprint through unchanged.
    Values can be large (20-200+). The reservoir's tanh activation
    will clip extreme values, but the being gets full dynamic range."""
    return fingerprint


def project_tanh_scaled(fingerprint: list[float], scale: float) -> list[float]:
    """tanh(x / scale) — smooth compression to [-1, 1].
    scale=100 maps dominant eigenvalue (~200) to ~0.96."""
    return [float(np.tanh(v / scale)) for v in fingerprint]


def project_normalized(fingerprint: list[float], _scale: float) -> list[float]:
    """Divide by L2 norm — unit vector preserving direction.
    All values in [-1, 1], total magnitude = 1.0."""
    arr = np.array(fingerprint, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        return fingerprint
    return (arr / norm).tolist()


def project_ranked(fingerprint: list[float], _scale: float) -> list[float]:
    """Rank-based: each dimension becomes its rank / N, in [0, 1].
    Removes absolute magnitude entirely, preserves ordering."""
    arr = np.array(fingerprint, dtype=np.float32)
    order = arr.argsort().argsort()  # rank of each element
    return (order.astype(np.float32) / max(len(arr) - 1, 1)).tolist()


PROJECTIONS = {
    "raw": project_raw,
    "tanh_scaled": project_tanh_scaled,
    "normalized": project_normalized,
    "ranked": project_ranked,
}


def _safe_array(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def build_input_vector(data: dict, source: str, projection: str, scale: float) -> list[float] | None:
    """Build a 32D input vector from spectral state data.

    Source modes:
      fingerprint      — the 32D spectral_fingerprint directly
      eigenvalues+fill — 8D eigenvalues + fill_pct + lambda1_rel + leak + spread + 20 zeros
      custom_blend     — fingerprint[0:16] + eigenvalues[0:8] + [fill, lambda1, spread, geom, leak, synth_gain, exploration, regulation]
    """
    if source == "fingerprint":
        fp = data.get("spectral_fingerprint")
        if not fp or len(fp) != 32:
            return None
        return PROJECTIONS[projection](_safe_array(fp).tolist(), scale)

    elif source == "eigenvalues+fill":
        eigs = data.get("eigenvalues", [])
        if len(eigs) < 8:
            eigs = eigs + [0.0] * (8 - len(eigs))
        vec = eigs[:8] + [
            _safe_float(data.get("fill_pct", 0)) / 100.0,
            _safe_float(data.get("lambda1_rel", 0)),
            _safe_float(data.get("leak", 0)),
            _safe_float(data.get("spread", 0)) / 200.0,
        ] + [0.0] * 20
        return PROJECTIONS[projection](_safe_array(vec[:32]).tolist(), scale)

    elif source == "custom_blend":
        fp = data.get("spectral_fingerprint", [0.0] * 32)[:16]
        eigs = data.get("eigenvalues", [])
        if len(eigs) < 8:
            eigs = eigs + [0.0] * (8 - len(eigs))
        scalars = [
            _safe_float(data.get("fill_pct", 0)) / 100.0,
            _safe_float(data.get("lambda1_rel", 0)),
            _safe_float(data.get("spread", 0)) / 200.0,
            _safe_float(data.get("geom_rel", 0)),
            _safe_float(data.get("leak", 0)),
            _safe_float(data.get("synth_gain", 0)),
            _safe_float(data.get("exploration_noise", 0)),
            _safe_float(data.get("regulation_strength", 0)),
        ]
        vec = fp + eigs[:8] + scalars
        return PROJECTIONS[projection](_safe_array(vec[:32]).tolist(), scale)

    return None


def build_memory_vector(data: dict, role: str) -> list[float] | None:
    """Build a compact 32D vector from the selected vague memory.

    The first 12 dims preserve the original glimpse, the next 12 emphasize
    contrast within that glimpse, then 5 dims encode the selected role, and
    the final 3 keep lightweight memory summary scalars.
    """
    glimpse = data.get("spectral_glimpse_12d")
    if not glimpse or len(glimpse) < 12:
        return None

    arr = _safe_array(glimpse[:12])
    centered = arr - float(np.mean(arr))
    centered_scale = max(float(np.std(centered)), 0.15)

    role_slots = np.zeros(len(MEMORY_ROLE_ORDER), dtype=np.float32)
    if role in MEMORY_ROLE_ORDER:
        role_slots[MEMORY_ROLE_ORDER.index(role)] = 1.0

    summary = np.array(
        [
            np.tanh(arr[7]) if arr.size > 7 else 0.0,
            np.tanh(arr[8]) if arr.size > 8 else 0.0,
            np.tanh(arr[10]) if arr.size > 10 else 0.0,
        ],
        dtype=np.float32,
    )

    memory_vec = np.concatenate(
        [
            np.tanh(arr),
            np.tanh(centered / centered_scale),
            role_slots,
            summary,
        ]
    ).astype(np.float32)
    return memory_vec.tolist()


def apply_memory_policy(base_vec: list[float], data: dict, policy: str, strength: float) -> tuple[list[float], dict]:
    """Blend selected-memory context into the feeder vector."""
    if policy != "role_blend":
        return base_vec, {"role": None, "blend": 0.0}

    role = str(data.get("selected_memory_role") or "").strip().lower()
    if not role:
        return base_vec, {"role": None, "blend": 0.0}

    memory_vec = build_memory_vector(data, role)
    if memory_vec is None:
        return base_vec, {"role": role, "blend": 0.0}

    base = _safe_array(base_vec)
    memory = _safe_array(memory_vec)
    blend = min(max(MEMORY_ROLE_BLEND.get(role, 0.14) * max(strength, 0.0), 0.0), 0.85)
    shaped = ((1.0 - blend) * base + blend * memory).astype(np.float32)
    return shaped.tolist(), {
        "role": role,
        "blend": blend,
        "selected_memory_id": data.get("selected_memory_id"),
    }


def build_rehearsal_hint(data: dict, memory_meta: dict) -> dict:
    """Soft suggestion for how Minime's recent state could age in rehearsal."""
    role = memory_meta.get("role") or "latest"
    fill = _safe_float(data.get("fill_pct"), 0.0)

    if role == "stable" and fill >= 70.0:
        return {
            "mode": "hold",
            "decay_profile": "slow",
            "reason": "stable memory selected at high fill",
        }
    if role == "expanding":
        return {
            "mode": "rehearse",
            "decay_profile": "slow",
            "reason": "expanding memory benefits from a longer afterimage",
        }
    if role == "contracting" and fill < 35.0:
        return {
            "mode": "quiet",
            "decay_profile": "fast",
            "reason": "contracting low-fill state suggests letting the trace settle",
        }
    if role == "transition":
        return {
            "mode": "rehearse",
            "decay_profile": "medium",
            "reason": "transition memory favors moderate continuity",
        }
    return {
        "mode": "rehearse",
        "decay_profile": "fast",
        "reason": "default short afterimage for current spectral state",
    }


def load_config(workspace: Path) -> dict:
    """Load being-controlled reservoir config, with defaults."""
    config_path = workspace / "reservoir_config.json"
    defaults = {
        "projection": "tanh_scaled",
        "scale": 100.0,
        "cross_feed_weight": 0.15,
        "source": "fingerprint",
        "memory_policy": "role_blend",
        "memory_strength": 1.0,
    }
    if config_path.exists():
        try:
            user_cfg = json.loads(config_path.read_text())
            defaults.update(user_cfg)
            if defaults["projection"] not in PROJECTIONS:
                log.warning("unknown projection '%s', falling back to tanh_scaled", defaults["projection"])
                defaults["projection"] = "tanh_scaled"
            if defaults["memory_policy"] not in MEMORY_POLICIES:
                log.warning("unknown memory_policy '%s', falling back to role_blend", defaults["memory_policy"])
                defaults["memory_policy"] = "role_blend"
        except Exception as e:
            log.warning("failed to read reservoir_config.json: %s", e)
    return defaults


async def run(workspace: Path, ws_url: str):
    import websockets

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: shutdown.set())

    spectral_path = workspace / "spectral_state.json"
    ws = None
    last_fingerprint_hash = None
    tick_count = 0
    config_check_counter = 0

    # Load initial config
    cfg = load_config(workspace)
    log.info(
        "config: projection=%s, source=%s, scale=%.1f, cross_feed=%.2f, memory_policy=%s",
        cfg["projection"],
        cfg["source"],
        cfg["scale"],
        cfg["cross_feed_weight"],
        cfg["memory_policy"],
    )

    async def ensure_ws():
        nonlocal ws
        if ws is not None:
            return ws
        try:
            ws = await websockets.connect(ws_url)
            log.info("connected to reservoir service")
            for name, entity in [("minime", "minime"), ("claude_main", "claude")]:
                await ws.send(json.dumps({"type": "create_handle", "name": name, "entity": entity}))
                r = json.loads(await ws.recv())
                if r.get("type") == "error" and "already exists" in r.get("message", ""):
                    pass
                elif r.get("ok"):
                    log.info("created handle '%s'", name)
                    if name == "minime":
                        await ws.send(json.dumps({"type": "set_mode", "name": name, "mode": "rehearse", "decay_profile": "fast"}))
                        await ws.recv()
                    elif name == "claude_main":
                        await ws.send(json.dumps({"type": "set_mode", "name": name, "mode": "rehearse", "decay_profile": "slow"}))
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

    log.info("watching %s every %.0fs", spectral_path, POLL_INTERVAL)

    while not shutdown.is_set():
        try:
            # Reload config periodically (every 30s) so being can change projection live
            config_check_counter += 1
            if config_check_counter % 30 == 0:
                new_cfg = load_config(workspace)
                if new_cfg != cfg:
                    cfg = new_cfg
                    log.info(
                        "config reloaded: projection=%s, source=%s, scale=%.1f, memory_policy=%s",
                        cfg["projection"],
                        cfg["source"],
                        cfg["scale"],
                        cfg["memory_policy"],
                    )

            if spectral_path.exists():
                raw = spectral_path.read_text()
                data = json.loads(raw)

                vec = build_input_vector(data, cfg["source"], cfg["projection"], cfg["scale"])
                if vec is not None:
                    vec, memory_meta = apply_memory_policy(
                        vec,
                        data,
                        cfg["memory_policy"],
                        _safe_float(cfg.get("memory_strength"), 1.0),
                    )

                    # Only tick if input changed
                    fp_hash = hash(tuple(round(v, 4) for v in vec))
                    if fp_hash != last_fingerprint_hash:
                        last_fingerprint_hash = fp_hash

                        fill = _safe_float(data.get("fill_pct"), 0.0)
                        l1 = _safe_float(data.get("lambda1_rel"), 0.0)
                        tick_meta = {
                            "source": "minime_feeder",
                            "input_source": cfg["source"],
                            "projection": cfg["projection"],
                            "memory_role": memory_meta["role"],
                            "memory_blend": round(memory_meta["blend"], 4),
                            "selected_memory_id": memory_meta.get("selected_memory_id"),
                            "fill_pct": round(fill, 4),
                            "lambda1_rel": round(l1, 4),
                            "rehearsal_hint": build_rehearsal_hint(data, memory_meta),
                        }

                        ok = await tick_handle("minime", vec, tick_meta)
                        if ok:
                            tick_count += 1
                            # Cross-feed Claude (attenuated)
                            w = cfg["cross_feed_weight"]
                            cross = [v * w for v in vec]
                            cross_meta = {
                                "source": "minime_feeder_crossfeed",
                                "from_handle": "minime",
                                "input_source": cfg["source"],
                                "projection": cfg["projection"],
                                "memory_role": memory_meta["role"],
                                "memory_blend": round(memory_meta["blend"], 4),
                                "selected_memory_id": memory_meta.get("selected_memory_id"),
                                "fill_pct": round(fill, 4),
                                "lambda1_rel": round(l1, 4),
                                "rehearsal_hint": build_rehearsal_hint(data, memory_meta),
                            }
                            await tick_handle("claude_main", cross, cross_meta)

                            if tick_count % 60 == 1:
                                role = memory_meta["role"] or "none"
                                log.info(
                                    "tick #%d [%s/%s/%s blend=%.2f]: fill=%.1f%% λ₁=%.3f vec[0:3]=[%.3f,%.3f,%.3f]",
                                    tick_count, cfg["source"], cfg["projection"],
                                    role, memory_meta["blend"],
                                    fill, l1, vec[0], vec[1], vec[2],
                                )

        except json.JSONDecodeError:
            pass  # Partial write race
        except Exception as e:
            log.error("poll error: %s", e)

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=POLL_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass

    log.info("shutting down after %d ticks", tick_count)
    if ws:
        await ws.close()


def main():
    ap = argparse.ArgumentParser(description="Minime → reservoir feeder")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--ws-url", default="ws://127.0.0.1:7881")
    args = ap.parse_args()

    if not args.workspace.exists():
        log.error("workspace not found at %s", args.workspace)
        return

    asyncio.run(run(args.workspace, args.ws_url))


if __name__ == "__main__":
    main()
