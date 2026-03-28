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
REHEARSAL_INTERVAL = 0.5  # seconds between rehearsal ticks
AUTO_SNAPSHOT_INTERVAL = 300  # seconds between auto-snapshots


@dataclass
class HandleInfo:
    """Metadata for one named handle."""
    name: str
    entity: str
    tick_count: int = 0
    last_tick_time: float = 0.0
    output_ring: deque = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    h_norm_ring: deque = field(default_factory=lambda: deque(maxlen=RING_SIZE))


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
            config=self.config, use_mlx=(backend == "mlx"),
        )
        self.handles: dict[str, HandleInfo] = {}
        self.rehearsal = RehearsalController()
        self.persistence = PersistenceManager(state_dir)
        self.text_proj = TextProjection(input_dim=input_dim)
        self._shutdown = asyncio.Event()

    # ------------------------------------------------------------------
    # Handle lifecycle
    # ------------------------------------------------------------------

    def create_handle(self, name: str, entity: str = "unknown") -> HandleInfo:
        if self.bridge.has_handle(name):
            raise ValueError(f"handle '{name}' already exists")
        self.bridge.make_state(name)
        info = HandleInfo(name=name, entity=entity)
        self.handles[name] = info
        self.rehearsal.register(name, mode="quiet", decay_profile="medium")
        log.info("created handle '%s' (entity=%s)", name, entity)
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

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, name: str, input_vec: np.ndarray) -> dict:
        """Feed one input vector, return result dict."""
        info = self._require_handle(name)
        output = self.bridge.tick(name, input_vec)
        norms = self.bridge.h_norms(name)

        info.tick_count += 1
        info.last_tick_time = time.monotonic()
        info.output_ring.append(output)
        info.h_norm_ring.append(norms)

        # Notify rehearsal controller of live input
        self.rehearsal.on_live_tick(name, input_vec.ravel())

        rs = self.rehearsal.get_state(name)
        return {
            "type": "tick_response",
            "name": name,
            "output": output,
            "h_norms": list(norms),
            "tick": info.tick_count,
            "mode": rs.mode if rs else "unknown",
        }

    def tick_text(self, name: str, text: str) -> dict:
        """Project text to 32D and tick."""
        vec = self.text_proj(text)
        return self.tick(name, vec)

    # ------------------------------------------------------------------
    # Read / trajectory / resonance
    # ------------------------------------------------------------------

    def read_state(self, name: str) -> dict:
        info = self._require_handle(name)
        norms = self.bridge.h_norms(name)
        rs = self.rehearsal.get_state(name)
        last_output = info.output_ring[-1] if info.output_ring else None
        elapsed = time.monotonic() - info.last_tick_time if info.last_tick_time > 0 else None
        return {
            "type": "read_state_response",
            "name": name,
            "entity": info.entity,
            "h_norms": list(norms),
            "last_output": last_output,
            "tick_count": info.tick_count,
            "mode": rs.mode if rs else "unknown",
            "decay_weight": rs.decay_weight if rs else 0.0,
            "decay_profile": rs.decay_profile if rs else "medium",
            "seconds_since_live": round(elapsed, 2) if elapsed is not None else None,
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

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, name: str, mode: str, decay_profile: Optional[str] = None) -> dict:
        self._require_handle(name)
        self.rehearsal.set_mode(name, mode, decay_profile)
        rs = self.rehearsal.get_state(name)
        log.info("handle '%s' mode → %s (decay=%s)", name, mode, rs.decay_profile if rs else "?")
        return {
            "type": "set_mode_response",
            "name": name,
            "mode": mode,
            "decay_profile": rs.decay_profile if rs else "medium",
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
            mode=rs.mode if rs else "quiet",
            decay_profile=rs.decay_profile if rs else "medium",
            decay_weight=rs.decay_weight if rs else 0.0,
            tick_count=info.tick_count,
            entity=info.entity,
        )
        log.info("snapshot '%s' → %s", name, path)
        return {"type": "snapshot_response", "name": name, "path": str(path)}

    def restore_handle(self, name: str) -> dict:
        snap = self.persistence.load_handle(name)
        if snap is None:
            raise ValueError(f"no snapshot for '{name}'")

        # Create handle if it doesn't exist
        if not self.bridge.has_handle(name):
            self.bridge.make_state(name)
            self.handles[name] = HandleInfo(name=name, entity=snap.entity)
            self.rehearsal.register(name, mode=snap.mode, decay_profile=snap.decay_profile)

        # Restore hidden state
        self.bridge.set_state(name, snap.h1, snap.h2, snap.h3)

        # Restore metadata
        info = self.handles[name]
        info.entity = snap.entity
        info.tick_count = snap.tick_count

        # Restore rehearsal state
        rs = self.rehearsal.get_state(name)
        if rs:
            rs.mode = snap.mode
            rs.decay_profile = snap.decay_profile
            rs.decay_weight = snap.decay_weight
            rs.last_live_input = snap.last_input

        log.info("restored '%s' (entity=%s, ticks=%d, mode=%s)", name, snap.entity, snap.tick_count, snap.mode)
        return {"type": "restore_response", "name": name, "entity": snap.entity, "tick_count": snap.tick_count}

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
        h1, h2, h3 = self.bridge.read_state(name)
        rs = self.rehearsal.get_state(name)
        return {
            "type": "pull_state_response",
            "name": name,
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
        n = self.config.n_nodes
        h1 = np.frombuffer(base64.b64decode(h1_b64), dtype=np.float32).reshape(1, n)
        h2 = np.frombuffer(base64.b64decode(h2_b64), dtype=np.float32).reshape(1, n)
        h3 = np.frombuffer(base64.b64decode(h3_b64), dtype=np.float32).reshape(1, n)
        self.bridge.set_state(name, h1, h2, h3)

        # Credit the generation ticks to the handle
        info.tick_count += tick_delta

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
        log.info("push_state '%s': h_norms=[%.3f, %.3f, %.3f], +%d ticks", name, *norms, tick_delta)
        return {"type": "push_state_response", "name": name, "ok": True, "h_norms": list(norms)}

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_handles(self) -> dict:
        now = time.monotonic()
        handles = []
        for name, info in self.handles.items():
            rs = self.rehearsal.get_state(name)
            elapsed = now - info.last_tick_time if info.last_tick_time > 0 else None
            handles.append({
                "name": name,
                "entity": info.entity,
                "mode": rs.mode if rs else "unknown",
                "decay_weight": round(rs.decay_weight, 4) if rs else 0.0,
                "tick_count": info.tick_count,
                "last_tick_ago": round(elapsed, 1) if elapsed is not None else None,
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
                self.create_handle(msg["name"], msg.get("entity", "unknown"))
                return {"type": "create_handle_response", "name": msg["name"], "ok": True}

            elif msg_type == "tick":
                vec = np.array(msg["input"], dtype=np.float32)
                return self.tick(msg["name"], vec)

            elif msg_type == "tick_text":
                return self.tick_text(msg["name"], msg["text"])

            elif msg_type == "read_state":
                return self.read_state(msg["name"])

            elif msg_type == "set_mode":
                return self.set_mode(msg["name"], msg["mode"], msg.get("decay_profile"))

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
                )

            elif msg_type == "destroy_handle":
                self.destroy_handle(msg["name"])
                return {"type": "destroy_handle_response", "name": msg["name"], "ok": True}

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
                for name, vec in inputs.items():
                    if name in self.handles:
                        output = self.bridge.tick(name, vec.reshape(1, -1))
                        norms = self.bridge.h_norms(name)
                        info = self.handles[name]
                        info.tick_count += 1
                        info.output_ring.append(output)
                        info.h_norm_ring.append(norms)
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

        backend_name = "MLX (Metal)" if self.bridge.use_mlx else (
            "Core ML (ANE)" if self.bridge.use_coreml else "NumPy"
        )
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
        choices=["numpy", "mlx", "coreml"],
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
