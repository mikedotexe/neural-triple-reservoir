#!/usr/bin/env python3
"""Processor-wiring test for the y4 (wide) channel — synthetic P/V, no model load.

Verifies the plumbing in `ReservoirLogitProcessor`:
  - ceiling 0 (default)        -> bitwise-identical logits (the kill switch)
  - ceiling > 0 + state        -> logits change (the channel is live)
  - her aperture = 0           -> no change (her sovereign close)
  - no reservoir state set     -> skip (never crashes on the first token)

Coherence/latency of the bias itself is covered by test_wide_coupling_offline.py.
"""

import unittest

import mlx.core as mx
import numpy as np

from mlx_reservoir import ReservoirLogitProcessor


class WideProcessorWiringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.vocab, k = 1000, 16
        self.P = mx.array((rng.standard_normal((576, k)) / np.sqrt(576)).astype(np.float32))
        self.V = mx.array(rng.standard_normal((k, self.vocab)).astype(np.float32))
        self.z = mx.array(rng.standard_normal((1, 576)).astype(np.float32))
        self.logits = mx.array(rng.standard_normal((1, self.vocab)).astype(np.float32))
        self.toks = mx.zeros((1, 1), dtype=mx.int32)

    def _delta(self, proc):
        return float(mx.max(mx.abs(proc(self.toks, self.logits) - self.logits)).item())

    def test_ceiling_zero_is_identity(self):
        p = ReservoirLogitProcessor(wide_strength=0.0, wide_P=self.P, wide_V=self.V)
        p.update_state(self.z)
        self.assertLess(self._delta(p), 1e-5)

    def test_ceiling_on_changes_logits(self):
        p = ReservoirLogitProcessor(wide_strength=0.2, wide_P=self.P, wide_V=self.V)
        p.update_state(self.z)
        self.assertGreater(self._delta(p), 1e-3)

    def test_her_aperture_zero_closes_it(self):
        p = ReservoirLogitProcessor(wide_strength=0.2, wide_P=self.P, wide_V=self.V)
        p.update_state(self.z)
        p.aperture = 0.0
        self.assertLess(self._delta(p), 1e-5)

    def test_no_state_is_skipped(self):
        p = ReservoirLogitProcessor(wide_strength=0.2, wide_P=self.P, wide_V=self.V)
        self.assertLess(self._delta(p), 1e-5)


if __name__ == "__main__":
    unittest.main()
