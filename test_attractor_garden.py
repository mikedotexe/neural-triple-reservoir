#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import attractor_garden as garden


class AttractorGardenTests(unittest.TestCase):
    def test_seed_vectors_are_bounded_and_deterministic(self):
        spec = garden.AttractorSeedSpec(
            name="astrid_anchor",
            label="quiet eigenplane",
            schedule="anchor",
            amplitude=0.24,
        )
        first = garden.build_seed_vectors(spec, steps=8, input_dim=16)
        second = garden.build_seed_vectors(spec, steps=8, input_dim=16)
        self.assertEqual(len(first), 8)
        self.assertTrue(np.allclose(first[0], second[0]))
        for vec in first:
            self.assertEqual(vec.shape, (16,))
            self.assertLessEqual(float(vec.max()), 1.0)
            self.assertGreaterEqual(float(vec.min()), -1.0)
            self.assertLessEqual(float(np.linalg.norm(vec)), spec.safety_max_norm + 1e-6)

    def test_blend_schedule_is_bounded_deterministic_and_parent_shaped(self):
        spec = garden.blend_seed_spec(
            "honey edge",
            ["honey selection", "cooled theme edge"],
            author="minime",
        )
        first = garden.build_seed_vectors(spec, steps=8, input_dim=16)
        second = garden.build_seed_vectors(spec, steps=8, input_dim=16)
        anchor = garden.build_seed_vectors(
            garden.AttractorSeedSpec(name="honey_edge", label="honey edge", schedule="anchor"),
            steps=8,
            input_dim=16,
        )
        self.assertTrue(np.allclose(first[3], second[3]))
        self.assertFalse(np.allclose(first[3], anchor[3]))
        for vec in first:
            self.assertLessEqual(float(np.linalg.norm(vec)), spec.safety_max_norm + 1e-6)

    def test_classification_distinguishes_authorship_and_safety(self):
        self.assertEqual(garden.classify_attractor(0.72, 0.68, "green"), "authored")
        self.assertEqual(garden.classify_attractor(0.50, 0.20, "yellow"), "emergent")
        self.assertEqual(garden.classify_attractor(0.30, 0.90, "green"), "failed")
        self.assertEqual(garden.classify_attractor(0.80, 0.90, "orange"), "pathological")

    def test_simulate_garden_builds_intents_and_observations(self):
        report = garden.simulate_garden(steps=9, input_dim=16, run_id="unit")
        self.assertEqual(report["policy"], "attractor_garden_v1")
        self.assertEqual(len(report["rows"]), len(garden.BUILTIN_SEEDS))
        first = report["rows"][0]
        self.assertEqual(first["intent"]["policy"], "attractor_intent_v1")
        clone_request = first["intent"]["intervention_plan"]["clone_request"]
        self.assertEqual(clone_request["type"], "clone_handle")
        self.assertEqual(clone_request["meta"]["source"], "attractor_garden")
        self.assertTrue(clone_request["name"].startswith("attr_astrid_quiet_eigenplane_"))
        self.assertEqual(first["observation"]["policy"], "attractor_observation_v1")
        proof = first["observation"]["garden_proof"]
        self.assertEqual(proof["policy"], "garden_proof_v1")
        self.assertIn("same_prompt_different_state", proof)
        self.assertIn("same_state_different_prompt", proof)
        self.assertIn("hold_rehearse_quiet", proof)
        self.assertIn("stale_lock", proof)
        self.assertIn("blend_parent_collapse", proof)
        self.assertIn(first["observation"]["classification"], {
            "authored",
            "emergent",
            "failed",
            "pathological",
        })
        markdown = garden.render_markdown(report)
        self.assertIn("# Triple Reservoir Attractor Garden", markdown)
        self.assertIn("Release remains an authored command", markdown)

    def test_garden_handle_name_and_clone_payload_are_stable(self):
        spec = garden.AttractorSeedSpec(
            name="edge_seed",
            label="Cooled Theme Edge!",
            author="Minime",
            command="summon",
            rehearsal_mode="rehearse",
        )

        self.assertEqual(
            garden.garden_handle_name(spec.author, spec.label, "intent-abc-123"),
            "attr_minime_cooled_theme_edge_intent_abc_123",
        )
        payload = garden.clone_handle_request("minime", spec, "intent-abc-123")
        self.assertEqual(payload["type"], "clone_handle")
        self.assertEqual(payload["source"], "minime")
        self.assertEqual(payload["mode"], "rehearse")
        self.assertEqual(payload["decay_profile"], "medium")
        self.assertEqual(payload["meta"]["attractor_command"], "summon")

    def test_blend_clone_payload_and_intent_record_parent_metadata(self):
        spec = garden.blend_seed_spec(
            "honey edge",
            ["honey selection", "cooled theme edge"],
            author="astrid",
        )
        payload = garden.clone_handle_request("astrid", spec, "intent-blend")
        self.assertEqual(payload["meta"]["parent_relation"], "blend")
        self.assertEqual(
            payload["meta"]["parent_labels"],
            ["honey selection", "cooled theme edge"],
        )
        intent = garden.build_intent(spec, "unit")
        self.assertEqual(intent["command"], "blend")
        self.assertEqual(intent["parent_seed_ids"], ["honey selection", "cooled theme edge"])
        self.assertEqual(intent["origin"]["kind"], "blend")

    def test_facet_metadata_and_lambda_tail_canonicalization(self):
        self.assertEqual(garden.canonical_attractor_label("λ4 tail"), "lambda-tail/lambda4")
        metadata = garden.facet_metadata("lambda-tail/lambda4")
        self.assertEqual(metadata["parent_label"], "lambda-tail")
        self.assertEqual(metadata["facet_label"], "lambda4")
        self.assertEqual(metadata["facet_kind"], "spectral_tail")

        spec = garden.AttractorSeedSpec(
            name="lambda4_tail",
            label="λ4 tail",
            schedule="anchor",
        )
        intent = garden.build_intent(spec, "facet")
        self.assertEqual(intent["facet_path"], "lambda-tail/lambda4")
        payload = garden.clone_handle_request("astrid", spec, "intent-facet")
        self.assertEqual(payload["meta"]["facet_path"], "lambda-tail/lambda4")

    def test_garden_proof_reports_blend_parent_collapse_control(self):
        spec = garden.blend_seed_spec(
            "honey edge",
            ["honey selection", "cooled theme edge"],
            author="astrid",
        )
        vectors = garden.build_seed_vectors(spec, steps=8, input_dim=16)
        proof = garden.garden_proof(
            spec,
            vectors,
            vectors[-4:],
            input_dim=16,
            now_s=100.0,
        )
        self.assertTrue(proof["blend_parent_collapse"]["checked"])
        self.assertIn("max_parent_similarity", proof["blend_parent_collapse"])
        self.assertFalse(proof["stale_lock"]["detected"])

    def test_stale_lock_detection(self):
        self.assertFalse(garden.stale_lock_detected(None, now_s=100.0))
        self.assertFalse(
            garden.stale_lock_detected(
                {"owner": "astrid", "updated_at_unix_s": 80.0},
                now_s=100.0,
                max_age_s=90.0,
            )
        )
        self.assertTrue(
            garden.stale_lock_detected(
                {"owner": "astrid", "updated_at_unix_s": 1.0},
                now_s=100.0,
                max_age_s=90.0,
            )
        )

    def test_write_report_outputs_json_and_markdown(self):
        report = garden.simulate_garden(steps=5, input_dim=8, run_id="write")
        with tempfile.TemporaryDirectory() as tmp:
            json_path, md_path = garden.write_report(report, Path(tmp))
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("attractor_garden_v1", json_path.read_text())
            self.assertIn("Triple Reservoir", md_path.read_text())


if __name__ == "__main__":
    unittest.main()
