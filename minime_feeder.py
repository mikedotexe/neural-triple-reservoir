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
        "source": "fingerprint" | "eigenvalues+fill" | "custom_blend"
    }

Defaults to "tanh_scaled" with scale=100 if no config file exists.

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
        return PROJECTIONS[projection](fp, scale)

    elif source == "eigenvalues+fill":
        eigs = data.get("eigenvalues", [])
        if len(eigs) < 8:
            eigs = eigs + [0.0] * (8 - len(eigs))
        vec = eigs[:8] + [
            data.get("fill_pct", 0) / 100.0,
            data.get("lambda1_rel", 0),
            data.get("leak", 0),
            data.get("spread", 0) / 200.0,
        ] + [0.0] * 20
        return PROJECTIONS[projection](vec[:32], scale)

    elif source == "custom_blend":
        fp = data.get("spectral_fingerprint", [0.0] * 32)[:16]
        eigs = data.get("eigenvalues", [])
        if len(eigs) < 8:
            eigs = eigs + [0.0] * (8 - len(eigs))
        scalars = [
            data.get("fill_pct", 0) / 100.0,
            data.get("lambda1_rel", 0),
            data.get("spread", 0) / 200.0,
            data.get("geom_rel", 0),
            data.get("leak", 0),
            data.get("synth_gain", 0),
            data.get("exploration_noise", 0),
            data.get("regulation_strength", 0),
        ]
        vec = fp + eigs[:8] + scalars
        return PROJECTIONS[projection](vec[:32], scale)

    return None


def load_config(workspace: Path) -> dict:
    """Load being-controlled reservoir config, with defaults."""
    config_path = workspace / "reservoir_config.json"
    defaults = {
        "projection": "tanh_scaled",
        "scale": 100.0,
        "cross_feed_weight": 0.15,
        "source": "fingerprint",
    }
    if config_path.exists():
        try:
            user_cfg = json.loads(config_path.read_text())
            defaults.update(user_cfg)
            if defaults["projection"] not in PROJECTIONS:
                log.warning("unknown projection '%s', falling back to tanh_scaled", defaults["projection"])
                defaults["projection"] = "tanh_scaled"
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
    log.info("config: projection=%s, source=%s, scale=%.1f, cross_feed=%.2f",
             cfg["projection"], cfg["source"], cfg["scale"], cfg["cross_feed_weight"])

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

    async def tick_handle(name: str, vec: list[float]) -> bool:
        try:
            conn = await ensure_ws()
            if conn is None:
                return False
            await conn.send(json.dumps({"type": "tick", "name": name, "input": vec}))
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
                    log.info("config reloaded: projection=%s, source=%s, scale=%.1f",
                             cfg["projection"], cfg["source"], cfg["scale"])

            if spectral_path.exists():
                raw = spectral_path.read_text()
                data = json.loads(raw)

                vec = build_input_vector(data, cfg["source"], cfg["projection"], cfg["scale"])
                if vec is not None:
                    # Only tick if input changed
                    fp_hash = hash(tuple(round(v, 4) for v in vec))
                    if fp_hash != last_fingerprint_hash:
                        last_fingerprint_hash = fp_hash

                        ok = await tick_handle("minime", vec)
                        if ok:
                            tick_count += 1
                            # Cross-feed Claude (attenuated)
                            w = cfg["cross_feed_weight"]
                            cross = [v * w for v in vec]
                            await tick_handle("claude_main", cross)

                            if tick_count % 60 == 1:
                                fill = data.get("fill_pct", 0)
                                l1 = data.get("lambda1_rel", 0)
                                log.info(
                                    "tick #%d [%s/%s]: fill=%.1f%% λ₁=%.3f vec[0:3]=[%.3f,%.3f,%.3f]",
                                    tick_count, cfg["source"], cfg["projection"],
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
