"""
rehearsal.py -- Rehearsal controller for named reservoir handles.

Three modes (tranche 1):
  hold     -- replay last_live_input at full weight, auto-transition to rehearse after 120 ticks
  rehearse -- replay with decaying weight, auto-transition to quiet when weight < 0.02
  quiet    -- zero input, natural drift, genuine silence

Three decay profiles:
  fast   -- factor 0.95, half-life ~7s  at 500ms ticks
  medium -- factor 0.98, half-life ~17s at 500ms ticks
  slow   -- factor 0.995, half-life ~70s at 500ms ticks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DECAY_FACTORS = {
    "fast": 0.95,
    "medium": 0.98,
    "slow": 0.995,
}

HOLD_MAX_TICKS = 120  # 60 seconds at 500ms interval
QUIET_THRESHOLD = 0.02  # decay_weight below this → auto-quiet


@dataclass
class HandleRehearsalState:
    """Per-handle rehearsal state."""
    mode: str = "quiet"
    decay_profile: str = "medium"
    last_live_input: Optional[np.ndarray] = None
    decay_weight: float = 1.0
    ticks_since_live: int = 0
    last_live_time: float = field(default_factory=time.monotonic)

    def on_live_tick(self, input_vec: np.ndarray):
        """Called when a live tick arrives from a client."""
        self.last_live_input = input_vec.copy()
        self.ticks_since_live = 0
        self.decay_weight = 1.0
        self.last_live_time = time.monotonic()
        # If we were quiet, transition back to rehearse
        if self.mode == "quiet":
            self.mode = "rehearse"

    def set_mode(self, mode: str, decay_profile: Optional[str] = None):
        """Set rehearsal mode and optionally decay profile."""
        if mode not in ("hold", "rehearse", "quiet"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        if decay_profile is not None:
            if decay_profile not in DECAY_FACTORS:
                raise ValueError(f"unknown decay profile: {decay_profile}")
            self.decay_profile = decay_profile
        if mode == "quiet":
            self.decay_weight = 0.0
        elif mode == "hold":
            self.decay_weight = 1.0
            self.ticks_since_live = 0


class RehearsalController:
    """Manages rehearsal state for all handles."""

    def __init__(self):
        self.states: dict[str, HandleRehearsalState] = {}

    def register(self, name: str, mode: str = "quiet", decay_profile: str = "medium"):
        """Register a new handle for rehearsal tracking."""
        self.states[name] = HandleRehearsalState(mode=mode, decay_profile=decay_profile)

    def unregister(self, name: str):
        """Remove a handle from rehearsal tracking."""
        self.states.pop(name, None)

    def on_live_tick(self, name: str, input_vec: np.ndarray):
        """Notify that a live tick was received for a handle."""
        if name in self.states:
            self.states[name].on_live_tick(input_vec)

    def set_mode(self, name: str, mode: str, decay_profile: Optional[str] = None):
        """Set mode for a handle."""
        if name in self.states:
            self.states[name].set_mode(mode, decay_profile)

    def get_rehearsal_inputs(self) -> dict[str, np.ndarray]:
        """Compute rehearsal inputs for all non-quiet handles.

        Returns a dict of {name: effective_input} for handles that
        should be ticked this cycle. Advances internal state (decay, counters,
        auto-transitions).
        """
        inputs = {}

        for name, rs in self.states.items():
            if rs.mode == "quiet":
                continue

            if rs.last_live_input is None:
                # No live input ever received — nothing to rehearse
                rs.mode = "quiet"
                continue

            rs.ticks_since_live += 1

            if rs.mode == "hold":
                # Full-weight replay
                inputs[name] = rs.last_live_input.copy()
                # Auto-transition after max hold ticks
                if rs.ticks_since_live >= HOLD_MAX_TICKS:
                    rs.mode = "rehearse"

            elif rs.mode == "rehearse":
                # Decaying replay
                factor = DECAY_FACTORS.get(rs.decay_profile, 0.98)
                rs.decay_weight *= factor
                if rs.decay_weight < QUIET_THRESHOLD:
                    rs.mode = "quiet"
                    rs.decay_weight = 0.0
                    continue
                inputs[name] = rs.decay_weight * rs.last_live_input

        return inputs

    def get_state(self, name: str) -> Optional[HandleRehearsalState]:
        """Get rehearsal state for a handle."""
        return self.states.get(name)
