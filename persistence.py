"""
persistence.py -- State save/load for named reservoir handles via numpy .npz files.

Each handle snapshot stores h1, h2, h3 hidden state arrays plus metadata
(mode, decay_weight, tick_count, entity, timestamp). Snapshots survive
service restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HandleSnapshot:
    """Everything needed to restore a handle."""
    h1: np.ndarray
    h2: np.ndarray
    h3: np.ndarray
    last_input: np.ndarray
    mode: str
    decay_profile: str
    decay_weight: float
    tick_count: int
    event_seq: int
    snapshot_version: int
    config_fingerprint: str | None
    entity: str
    timestamp: float
    last_live_wall_time: Optional[float] = None
    mode_authority: str = "system"
    hint_policy: str = "off"
    last_live_meta: dict | None = None
    last_lane_meta: dict | None = None
    provenance: list | None = None
    output_history: np.ndarray | None = None
    h_norm_history: np.ndarray | None = None


class PersistenceManager:
    """Save and load handle state as .npz files."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str) -> Path:
        safe_name = name.replace("/", "_").replace("..", "_")
        return self.state_dir / f"{safe_name}.npz"

    def save_handle(
        self,
        name: str,
        h1: np.ndarray,
        h2: np.ndarray,
        h3: np.ndarray,
        last_input: np.ndarray,
        mode: str,
        decay_profile: str,
        decay_weight: float,
        tick_count: int,
        event_seq: int,
        snapshot_version: int,
        config_fingerprint: str | None,
        entity: str,
        last_live_wall_time: Optional[float] = None,
        mode_authority: str = "system",
        hint_policy: str = "off",
        last_live_meta: Optional[dict] = None,
        last_lane_meta: Optional[dict] = None,
        provenance: Optional[list] = None,
        output_history: Optional[np.ndarray] = None,
        h_norm_history: Optional[np.ndarray] = None,
    ) -> Path:
        """Persist one handle to disk. Returns the file path."""
        path = self._path_for(name)
        np.savez(
            path,
            h1=h1,
            h2=h2,
            h3=h3,
            last_input=last_input,
            mode=np.array(mode),
            decay_profile=np.array(decay_profile),
            decay_weight=np.array(decay_weight, dtype=np.float64),
            tick_count=np.array(tick_count, dtype=np.int64),
            event_seq=np.array(event_seq, dtype=np.int64),
            snapshot_version=np.array(snapshot_version, dtype=np.int64),
            config_fingerprint=np.array(config_fingerprint or ""),
            entity=np.array(entity),
            timestamp=np.array(time.time(), dtype=np.float64),
            last_live_wall_time=np.array(
                -1.0 if last_live_wall_time is None else float(last_live_wall_time),
                dtype=np.float64,
            ),
            mode_authority=np.array(mode_authority),
            hint_policy=np.array(hint_policy),
            last_live_meta_json=np.array(json.dumps(last_live_meta or {})),
            last_lane_meta_json=np.array(json.dumps(last_lane_meta or {})),
            provenance_json=np.array(json.dumps(provenance or [])),
            output_history=np.asarray(output_history if output_history is not None else [], dtype=np.float32),
            h_norm_history=np.asarray(h_norm_history if h_norm_history is not None else [], dtype=np.float32),
        )
        return path

    def load_handle(self, name: str) -> Optional[HandleSnapshot]:
        """Load one handle from disk. Returns None if not found."""
        path = self._path_for(name)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            last_live_wall_time = (
                float(data["last_live_wall_time"])
                if "last_live_wall_time" in data and float(data["last_live_wall_time"]) >= 0.0
                else None
            )
            return HandleSnapshot(
                h1=data["h1"],
                h2=data["h2"],
                h3=data["h3"],
                last_input=data["last_input"],
                mode=str(data["mode"]),
                decay_profile=str(data["decay_profile"]),
                decay_weight=float(data["decay_weight"]),
                tick_count=int(data["tick_count"]),
                event_seq=int(data["event_seq"]) if "event_seq" in data else 0,
                snapshot_version=int(data["snapshot_version"]) if "snapshot_version" in data else 1,
                config_fingerprint=(
                    str(data["config_fingerprint"])
                    if "config_fingerprint" in data and str(data["config_fingerprint"])
                    else None
                ),
                entity=str(data["entity"]),
                timestamp=float(data["timestamp"]),
                last_live_wall_time=last_live_wall_time,
                mode_authority=str(data["mode_authority"]) if "mode_authority" in data else "system",
                hint_policy=str(data["hint_policy"]) if "hint_policy" in data else "off",
                last_live_meta=json.loads(str(data["last_live_meta_json"])) if "last_live_meta_json" in data else {},
                last_lane_meta=json.loads(str(data["last_lane_meta_json"])) if "last_lane_meta_json" in data else {},
                provenance=json.loads(str(data["provenance_json"])) if "provenance_json" in data else [],
                output_history=data["output_history"] if "output_history" in data else np.array([], dtype=np.float32),
                h_norm_history=data["h_norm_history"] if "h_norm_history" in data else np.empty((0, 3), dtype=np.float32),
            )
        except Exception as e:
            print(f"[persistence] failed to load {name}: {e}")
            return None

    def list_snapshots(self) -> list[str]:
        """Return names of all saved handles."""
        names = []
        for p in self.state_dir.glob("*.npz"):
            names.append(p.stem)
        return sorted(names)

    def delete_snapshot(self, name: str) -> bool:
        """Remove a snapshot file. Returns True if it existed."""
        path = self._path_for(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def quarantine_snapshot(self, name: str) -> list[Path]:
        """Move an incompatible snapshot aside so it is not retried on startup."""
        safe_name = name.replace("/", "_").replace("..", "_")
        invalid_dir = self.state_dir / "invalid"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        moved: list[Path] = []
        for src in (
            self._path_for(name),
            self.state_dir / f"{safe_name}_thermostats.json",
        ):
            if not src.exists():
                continue
            target = invalid_dir / f"{src.name}.{stamp}.invalid"
            src.rename(target)
            moved.append(target)
        return moved
