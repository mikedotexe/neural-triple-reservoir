"""
persistence.py -- State save/load for named reservoir handles via numpy .npz files.

Each handle snapshot stores h1, h2, h3 hidden state arrays plus metadata
(mode, decay_weight, tick_count, entity, timestamp). Snapshots survive
service restarts.
"""

from __future__ import annotations

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
    entity: str
    timestamp: float


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
        entity: str,
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
            entity=np.array(entity),
            timestamp=np.array(time.time(), dtype=np.float64),
        )
        return path

    def load_handle(self, name: str) -> Optional[HandleSnapshot]:
        """Load one handle from disk. Returns None if not found."""
        path = self._path_for(name)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            return HandleSnapshot(
                h1=data["h1"],
                h2=data["h2"],
                h3=data["h3"],
                last_input=data["last_input"],
                mode=str(data["mode"]),
                decay_profile=str(data["decay_profile"]),
                decay_weight=float(data["decay_weight"]),
                tick_count=int(data["tick_count"]),
                entity=str(data["entity"]),
                timestamp=float(data["timestamp"]),
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
