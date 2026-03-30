#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from astrid_feeder import CodecConditioner, apply_remote_memory_policy
from minime_feeder import apply_memory_policy, build_memory_vector


class AstridConditioningTests(unittest.TestCase):
    def test_ema_rms_conditioning_bounds_gain_amplified_features(self):
        conditioner = CodecConditioner(mode="ema_rms", decay=0.92, floor=0.75)
        raw = [4.5, -3.8, 2.2, -1.6] * 8

        conditioned, stats = conditioner.transform(raw)

        self.assertEqual(len(conditioned), 32)
        self.assertTrue(all(abs(v) <= 1.0 for v in conditioned))
        self.assertGreater(stats["max_abs"], 1.0)
        self.assertLessEqual(stats["conditioned_max_abs"], 1.0)
        self.assertGreater(stats["scale"], 0.0)

    def test_legacy_conditioning_preserves_raw_magnitude(self):
        conditioner = CodecConditioner(mode="legacy", decay=0.92, floor=0.75)
        raw = [2.0, -1.5] * 16

        conditioned, stats = conditioner.transform(raw)

        self.assertEqual(conditioned, raw)
        self.assertAlmostEqual(stats["scale"], 1.0)

    def test_remote_memory_policy_blends_selected_remote_memory(self):
        import json
        import tempfile
        from pathlib import Path

        base = [0.2] * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "state.json").write_text(json.dumps({
                "last_remote_memory_id": "memory_stable_1",
                "last_remote_memory_role": "stable",
                "remote_memory_bank": [
                    {
                        "id": "memory_stable_1",
                        "role": "stable",
                        "spectral_glimpse_12d": np.linspace(0.1, 1.4, 12, dtype=np.float32).tolist(),
                        "fill_pct": 72.0,
                        "lambda1_rel": 0.8,
                        "geom_rel": 1.4,
                    }
                ],
            }))
            (workspace / "contact_state.json").write_text(json.dumps({
                "attention": 0.9,
                "openness": 0.7,
                "urgency": 0.6,
            }))

            shaped, meta = apply_remote_memory_policy(base, workspace, "remote_role_blend", 1.0)

        self.assertEqual(len(shaped), 32)
        self.assertEqual(meta["memory_role"], "stable")
        self.assertGreater(meta["blend"], 0.0)
        self.assertGreater(meta["contact_gate"], 0.5)
        self.assertGreater(float(np.mean(np.abs(np.asarray(shaped) - np.asarray(base)))), 0.0)

    def test_remote_memory_policy_off_is_contract_stable(self):
        import tempfile
        from pathlib import Path

        base = [0.2] * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            shaped, meta = apply_remote_memory_policy(base, workspace, "off", 1.0)

        self.assertEqual(shaped, base)
        self.assertIn("memory_role", meta)
        self.assertIsNone(meta["memory_role"])
        self.assertEqual(meta["blend"], 0.0)

    def test_remote_memory_policy_without_state_keeps_selected_role_shape(self):
        import json
        import tempfile
        from pathlib import Path

        base = [0.2] * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "state.json").write_text(json.dumps({
                "last_remote_memory_role": "expanding",
                "remote_memory_bank": [],
            }))
            shaped, meta = apply_remote_memory_policy(base, workspace, "remote_role_blend", 1.0)

        self.assertEqual(shaped, base)
        self.assertEqual(meta["memory_role"], "expanding")
        self.assertEqual(meta["blend"], 0.0)


class MinimeMemoryPolicyTests(unittest.TestCase):
    def _spectral_state(self, role: str) -> dict:
        glimpse = np.linspace(0.05, 1.2, 12, dtype=np.float32).tolist()
        return {
            "selected_memory_role": role,
            "selected_memory_id": f"memory_{role}_123",
            "spectral_glimpse_12d": glimpse,
        }

    def test_build_memory_vector_is_32d(self):
        vec = build_memory_vector(self._spectral_state("stable"), "stable")
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 32)

    def test_role_blend_shapes_vector_more_for_stable_than_latest(self):
        base = [0.4] * 32

        stable_vec, stable_meta = apply_memory_policy(base, self._spectral_state("stable"), "role_blend", 1.0)
        latest_vec, latest_meta = apply_memory_policy(base, self._spectral_state("latest"), "role_blend", 1.0)

        stable_delta = float(np.mean(np.abs(np.asarray(stable_vec) - np.asarray(base))))
        latest_delta = float(np.mean(np.abs(np.asarray(latest_vec) - np.asarray(base))))

        self.assertGreater(stable_meta["blend"], latest_meta["blend"])
        self.assertGreater(stable_delta, latest_delta)

    def test_memory_policy_off_leaves_vector_unchanged(self):
        base = [0.25] * 32
        out, meta = apply_memory_policy(base, self._spectral_state("transition"), "off", 1.0)

        self.assertEqual(out, base)
        self.assertEqual(meta["blend"], 0.0)


if __name__ == "__main__":
    unittest.main()
