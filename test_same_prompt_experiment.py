#!/usr/bin/env python3
from __future__ import annotations

import unittest

import experiment_same_prompt_different_state as exp1


class SamePromptExperimentTests(unittest.TestCase):
    def test_each_scenario_builds_bounded_vectors(self):
        for spec in exp1.SCENARIOS:
            vectors = exp1.build_vectors(spec, steps=9, input_dim=32)
            self.assertEqual(len(vectors), 9)
            self.assertEqual(vectors[0].shape, (32,))
            for vec in vectors:
                self.assertLessEqual(float(vec.max()), 1.0)
                self.assertGreaterEqual(float(vec.min()), -1.0)

    def test_handle_name_is_stable_and_safe(self):
        name = exp1.build_handle_name("exp 1", "stable", "20260330_120000")
        self.assertEqual(name, "exp_1_stable_20260330_120000")

    def test_render_markdown_includes_pairwise_and_output_sections(self):
        report = {
            "captured_at": "2026-03-30 12:20:00",
            "baseline_handle": "astrid",
            "prompt": "Describe continuity in one sentence.",
            "results": [{
                "scenario": "stable",
                "description": "Anchored regime",
                "handle": "exp1_stable_20260330_120000",
                "output_text": "Continuity feels held together by a stable thread.",
                "trajectory": {"trend": "rising", "latest": 0.3, "delta": 0.1, "spread": 0.2},
                "generation_meta": {
                    "generated_tokens": 12,
                    "coupling_strength": 0.1,
                    "event_id": "astrid:000012",
                    "source_timestamp": 1774940400.0,
                    "response_sha256_12": "abc123def456",
                    "response_preview": "Continuity feels held together by a stable thread.",
                    "reservoir_readout": {"y1_final": 0.1, "y2_final": 0.2, "y3_final": 0.3},
                },
                "feeder_meta": {},
                "layer_metrics": [{
                    "name": "h1_fast",
                    "h_norm": 1.2,
                    "entropy": 0.21,
                    "entropy_target": 0.2,
                    "saturation": 0.01,
                    "rho": 0.95,
                }],
            }],
            "pairwise": [],
        }
        markdown = exp1.render_markdown(report)
        self.assertIn("# Same Prompt, Different State", markdown)
        self.assertIn("## Stable", markdown)
        self.assertIn("Continuity feels held together by a stable thread.", markdown)
        self.assertIn("| Scenario | Handle | Tokens | Coupling | y1 | y2 | y3 | Trend | Event | Response Hash |", markdown)
        self.assertIn("astrid:000012 @", markdown)


if __name__ == "__main__":
    unittest.main()
