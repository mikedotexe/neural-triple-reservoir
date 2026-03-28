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
        "cross_feed_weight": 0.3
    }

Defaults to "passthrough" (codec features are already tanh-bounded).

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

DEFAULT_DB = Path("/Users/v/other/astrid/capsules/consciousness-bridge/workspace/bridge.db")
POLL_INTERVAL = 5.0  # seconds between DB polls


# ---------------------------------------------------------------------------
# Projection modes — Astrid picks how her codec features enter the reservoir
# ---------------------------------------------------------------------------

def project_passthrough(features: list[float], _factor: float) -> list[float]:
    """Codec features as-is. Already tanh-bounded from the bridge codec."""
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


def load_config(workspace_dir: Path) -> dict:
    """Load being-controlled reservoir config, with defaults."""
    config_path = workspace_dir / "reservoir_config.json"
    defaults = {
        "projection": "passthrough",
        "amplify_factor": 2.0,
        "cross_feed_weight": 0.3,
    }
    if config_path.exists():
        try:
            user_cfg = json.loads(config_path.read_text())
            defaults.update(user_cfg)
            if defaults["projection"] not in PROJECTIONS:
                log.warning("unknown projection '%s', falling back to passthrough", defaults["projection"])
                defaults["projection"] = "passthrough"
        except Exception as e:
            log.warning("failed to read reservoir_config.json: %s", e)
    return defaults


async def run(db_path: Path, ws_url: str):
    import websockets

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: shutdown.set())

    last_id = 0
    workspace_dir = db_path.parent  # bridge.db sits in workspace/

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
    log.info("config: projection=%s, amplify=%.1f, cross_feed=%.2f",
             cfg["projection"], cfg["amplify_factor"], cfg["cross_feed_weight"])

    async def ensure_ws():
        nonlocal ws
        if ws is not None:
            return ws
        try:
            ws = await websockets.connect(ws_url)
            log.info("connected to reservoir service")
            for name, entity in [("astrid", "astrid"), ("claude_main", "claude")]:
                await ws.send(json.dumps({"type": "create_handle", "name": name, "entity": entity}))
                r = json.loads(await ws.recv())
                if r.get("type") == "error" and "already exists" in r.get("message", ""):
                    pass
                elif r.get("ok"):
                    log.info("created handle '%s'", name)
                    if name == "astrid":
                        await ws.send(json.dumps({"type": "set_mode", "name": name, "mode": "rehearse", "decay_profile": "slow"}))
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

    log.info("polling %s every %.0fs", db_path, POLL_INTERVAL)

    while not shutdown.is_set():
        try:
            # Reload config periodically (every ~30s at 5s poll = 6 checks)
            config_check_counter += 1
            if config_check_counter % 6 == 0:
                new_cfg = load_config(workspace_dir)
                if new_cfg != cfg:
                    cfg = new_cfg
                    log.info("config reloaded: projection=%s, amplify=%.1f",
                             cfg["projection"], cfg["amplify_factor"])

            conn = sqlite3.connect(str(db_path), timeout=5)
            rows = conn.execute(
                "SELECT id, features_json FROM codec_impact WHERE id > ? ORDER BY id",
                (last_id,),
            ).fetchall()
            conn.close()

            for row_id, features_json in rows:
                features = json.loads(features_json)
                if len(features) != 32:
                    log.warning("unexpected feature dim %d at id=%d", len(features), row_id)
                    last_id = row_id
                    continue

                # Apply being-controlled projection
                proj_fn = PROJECTIONS[cfg["projection"]]
                projected = proj_fn(features, cfg["amplify_factor"])

                ok = await tick_handle("astrid", projected)
                if ok:
                    # Cross-feed Claude's handle (attenuated)
                    w = cfg["cross_feed_weight"]
                    cross = [f * w for f in projected]
                    await tick_handle("claude_main", cross)
                    log.info("tick id=%d [%s] → astrid + claude_main", row_id, cfg["projection"])

                last_id = row_id

        except Exception as e:
            log.error("poll error: %s", e)

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
