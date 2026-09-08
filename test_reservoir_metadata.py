#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def b64_state(layer: np.ndarray) -> str:
    return base64.b64encode(np.asarray(layer, dtype=np.float32).tobytes()).decode()


def load_reservoir_service_with_stubs():
    fake_dual = types.ModuleType("dual_ai_bridge")
    fake_coreml = types.ModuleType("triple_reservoir_coreml")

    class FakeReservoirConfig:
        def __init__(self, input_dim: int = 32, n_nodes: int = 8):
            self.input_dim = input_dim
            self.n_nodes = n_nodes

    class FakeTextProjection:
        def __init__(self, input_dim: int = 32):
            self.input_dim = input_dim

        def __call__(self, text: str):
            return np.zeros((1, self.input_dim), dtype=np.float32)

    class FakeBridge:
        def __init__(
            self,
            config,
            mlpackage=None,
            compute_units="cpu_and_ne",
            use_mlx=False,
            backend=None,
        ):
            self.config = config
            self.states = {}
            self.outputs = {}
            self.handle_backends = {}
            self.default_backend = backend or ("mlx" if use_mlx else "numpy")

        def has_handle(self, name):
            return name in self.states

        def make_state(self, name, backend=None):
            zeros = np.zeros((1, self.config.n_nodes), dtype=np.float32)
            self.states[name] = (zeros.copy(), zeros.copy(), zeros.copy())
            self.outputs[name] = 0.0
            self.handle_backends[name] = backend or self.default_backend

        def destroy_handle(self, name):
            self.states.pop(name, None)
            self.outputs.pop(name, None)
            self.handle_backends.pop(name, None)

        def tick(self, name, input_vec):
            value = float(np.asarray(input_vec, dtype=np.float32).sum())
            if self.handle_backends.get(name) == "lsm_shadow":
                value *= 0.8
            h1 = np.full((1, self.config.n_nodes), value, dtype=np.float32)
            h2 = np.full((1, self.config.n_nodes), value * 0.5, dtype=np.float32)
            h3 = np.full((1, self.config.n_nodes), value * 0.25, dtype=np.float32)
            self.states[name] = (h1, h2, h3)
            self.outputs[name] = value
            return value

        def h_norms(self, name):
            return [float(np.linalg.norm(layer)) for layer in self.states[name]]

        def get_layer_states(self, name):
            return self.states[name]

        def set_state(self, name, h1, h2, h3):
            self.states[name] = (
                np.asarray(h1, dtype=np.float32),
                np.asarray(h2, dtype=np.float32),
                np.asarray(h3, dtype=np.float32),
            )

        def read_state(self, name):
            return self.states[name]

        def backend_for_handle(self, name):
            return self.handle_backends[name]

        def supports_pull_push(self, name):
            return self.handle_backends.get(name) != "lsm_shadow"

        def backend_label(self, name=None):
            backend_name = self.default_backend if name is None else self.handle_backends[name]
            labels = {
                "numpy": "NumPy",
                "mlx": "MLX (Metal)",
                "coreml": "Core ML (ANE)",
                "lsm_shadow": "LSM Shadow",
            }
            return labels.get(backend_name, backend_name)

        def available_backends(self):
            return ["lsm_shadow", "numpy"]

    fake_dual.ReservoirBridge = FakeBridge
    fake_dual.TextProjection = FakeTextProjection
    fake_coreml.ReservoirConfig = FakeReservoirConfig

    sys.modules["dual_ai_bridge"] = fake_dual
    sys.modules["triple_reservoir_coreml"] = fake_coreml
    sys.modules.pop("reservoir_service", None)
    return importlib.import_module("reservoir_service")


class ReservoirMetadataTests(unittest.TestCase):
    _STUBBED_MODULES = (
        "dual_ai_bridge",
        "triple_reservoir_coreml",
        "reservoir_service",
    )

    def setUp(self):
        self._module_snapshot = {
            name: sys.modules.get(name) for name in self._STUBBED_MODULES
        }

    def tearDown(self):
        for name, module in self._module_snapshot.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_tick_metadata_is_visible_in_read_and_list(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")

            meta = {
                "source": "astrid_feeder",
                "memory_role": "stable",
                "rehearsal_hint": {
                    "mode": "hold",
                    "decay_profile": "slow",
                    "reason": "stable remote memory under sustained attention",
                },
            }
            service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta=meta)

            read = service.read_state("astrid")
            self.assertEqual(read["last_live_meta"]["source"], "astrid_feeder")
            self.assertEqual(read["last_feeder_meta"]["source"], "astrid_feeder")
            self.assertEqual(read["last_live_meta"]["memory_role"], "stable")
            self.assertEqual(read["last_live_meta"]["event_id"], "astrid:000001")
            self.assertEqual(read["last_event_id"], "astrid:000001")
            self.assertEqual(read["last_live_meta"]["event_seq"], 1)
            self.assertIsInstance(read["last_live_meta"]["source_timestamp"], float)
            self.assertEqual(read["rehearsal_hint"]["mode"], "hold")
            self.assertEqual(read["hint_status"], "present_not_adopted")

            listed = service.list_handles()["handles"][0]
            self.assertEqual(listed["last_source"], "astrid_feeder")
            self.assertEqual(listed["last_event_id"], "astrid:000001")
            self.assertEqual(listed["recent_event_ids"], ["astrid:000001"])
            self.assertEqual(listed["last_feeder_source"], "astrid_feeder")
            self.assertEqual(listed["memory_role"], "stable")
            self.assertEqual(listed["rehearsal_hint"]["decay_profile"], "slow")
            self.assertEqual(listed["hint_status"], "present_not_adopted")
            self.assertEqual(read["provenance"][-1]["source"], "astrid_feeder")
            self.assertEqual(read["provenance"][-1]["event_id"], "astrid:000001")
            self.assertEqual(listed["recent_sources"], ["astrid_feeder"])

    def test_backend_is_visible_in_read_and_list_for_shadow_handles(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")
            service.create_handle("astrid__lsm", "astrid")

            live = service.read_state("astrid")
            shadow = service.read_state("astrid__lsm")
            listed = {entry["name"]: entry for entry in service.list_handles()["handles"]}

            self.assertEqual(live["backend"], "numpy")
            self.assertEqual(shadow["backend"], "lsm_shadow")
            self.assertEqual(listed["astrid"]["backend"], "numpy")
            self.assertEqual(listed["astrid__lsm"]["backend"], "lsm_shadow")

    def test_dispatch_create_handle_accepts_explicit_backend(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))

            result = asyncio.run(
                service.dispatch(
                    {
                        "type": "create_handle",
                        "name": "minime__lsm",
                        "entity": "minime",
                        "backend": "lsm_shadow",
                    }
                )
            )

            self.assertEqual(result["type"], "create_handle_response")
            self.assertEqual(result["backend"], "lsm_shadow")
            self.assertEqual(service.read_state("minime__lsm")["backend"], "lsm_shadow")

    def test_pull_state_rejects_lsm_shadow_handles(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid__lsm", "astrid")

            result = asyncio.run(service.dispatch({"type": "pull_state", "name": "astrid__lsm"}))

            self.assertEqual(result["type"], "error")
            self.assertIn("phase 1", result["message"])
            self.assertIn("lsm_shadow", result["message"])

    def test_shadow_metrics_log_is_written_for_mirrored_ticks(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            service.create_handle("astrid__lsm", "astrid")

            service.tick(
                "astrid",
                np.ones((1, 32), dtype=np.float32),
                meta={"source": "astrid_feeder", "source_event_id": "codec_impact:42"},
            )
            service.tick(
                "astrid__lsm",
                np.ones((1, 32), dtype=np.float32),
                meta={
                    "source": "astrid_feeder",
                    "source_event_id": "codec_impact:42",
                    "shadow_of": "astrid",
                },
            )

            metrics_path = state_dir / "shadow_metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            lines = metrics_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["live_handle"], "astrid")
            self.assertEqual(payload["shadow_handle"], "astrid__lsm")
            self.assertEqual(payload["live_backend"], "numpy")
            self.assertEqual(payload["shadow_backend"], "lsm_shadow")
            self.assertEqual(payload["source_event_id"], "codec_impact:42")
            self.assertIn("trajectory_rmsd", payload)
            rollup = json.loads((state_dir / "shadow_metrics_rollup.json").read_text())
            self.assertEqual(rollup["latest"]["shadow_handle"], "astrid__lsm")
            self.assertEqual(rollup["counts_by_comparison_source"]["live_tick"], 1)

    def test_shadow_metrics_rehearsal_is_sampled_and_rotated(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            service.create_handle("astrid__lsm", "astrid")

            service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta={"source": "seed"})
            service.tick(
                "astrid__lsm",
                np.ones((1, 32), dtype=np.float32),
                meta={"source": "seed", "shadow_of": "astrid"},
            )
            service._log_shadow_metrics("astrid__lsm", source="rehearsal")
            service._log_shadow_metrics("astrid__lsm", source="rehearsal")
            lines = (state_dir / "shadow_metrics.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

            old_max = reservoir_service.SHADOW_METRICS_MAX_BYTES
            try:
                reservoir_service.SHADOW_METRICS_MAX_BYTES = 8
                service._rotate_shadow_metrics_if_needed()
            finally:
                reservoir_service.SHADOW_METRICS_MAX_BYTES = old_max
            self.assertTrue((state_dir / "shadow_metrics.jsonl.1").exists())
            self.assertEqual((state_dir / "shadow_metrics.jsonl").read_text(), "")

    def test_snapshot_restore_preserves_shadow_backend(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid__lsm", "astrid")
            service.tick(
                "astrid__lsm",
                np.ones((1, 32), dtype=np.float32),
                meta={"source": "astrid_feeder", "source_event_id": "codec_impact:99"},
            )
            service.snapshot_handle("astrid__lsm")

            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            restored.restore_handle("astrid__lsm")

            self.assertEqual(restored.read_state("astrid__lsm")["backend"], "lsm_shadow")

    def test_guarded_hint_policy_applies_only_when_enabled(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")

            meta = {
                "source": "astrid_feeder",
                "rehearsal_hint": {
                    "mode": "hold",
                    "decay_profile": "slow",
                    "reason": "stable remote memory under sustained attention",
                },
            }
            first = service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta=meta)
            self.assertFalse(first["hint_applied"])
            self.assertEqual(first["mode"], "rehearse")

            service.set_hint_policy("astrid", "guarded")
            second = service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta=meta)
            self.assertTrue(second["hint_applied"])
            self.assertEqual(second["mode"], "hold")
            self.assertEqual(second["mode_authority"], "hint")
            self.assertEqual(second["hint_status"], "governing")

    def test_explicit_mode_blocks_guarded_hint_adoption(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")
            service.set_hint_policy("astrid", "guarded")
            service.set_mode("astrid", "rehearse", "fast")

            meta = {
                "source": "astrid_feeder",
                "rehearsal_hint": {
                    "mode": "hold",
                    "decay_profile": "slow",
                    "reason": "stable remote memory under sustained attention",
                },
            }
            result = service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta=meta)

            self.assertFalse(result["hint_applied"])
            self.assertEqual(result["mode"], "rehearse")
            self.assertEqual(result["mode_authority"], "explicit")
            self.assertEqual(result["hint_status"], "present_blocked_by_explicit_mode")

    def test_layer_metrics_are_available_for_each_handle(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")
            service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta={"source": "astrid_feeder"})

            result = asyncio.run(service.dispatch({"type": "layer_metrics", "name": "astrid"}))

            self.assertEqual(result["type"], "layer_metrics_response")
            self.assertEqual(result["name"], "astrid")
            self.assertEqual(len(result["layers"]), 3)
            self.assertEqual(result["layers"][0]["name"], "h1_fast")
            self.assertIn("rho", result["layers"][0])
            self.assertIn("h_norm", result["layers"][0])

    def test_clone_handle_copies_state_and_diverges_without_mutating_source(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("minime", "minime")
            service.tick("minime", np.ones((1, 32), dtype=np.float32), meta={"source": "unit_seed"})

            result = asyncio.run(
                service.dispatch(
                    {
                        "type": "clone_handle",
                        "source": "minime",
                        "name": "attr_minime_lambda_edge_seed",
                        "entity": "minime",
                        "mode": "hold",
                        "decay_profile": "slow",
                        "meta": {
                            "intent_id": "intent-1",
                            "attractor_label": "lambda edge",
                        },
                    }
                )
            )

            self.assertEqual(result["type"], "clone_handle_response")
            self.assertEqual(result["source"], "minime")
            self.assertEqual(result["name"], "attr_minime_lambda_edge_seed")
            self.assertEqual(result["mode"], "hold")
            self.assertEqual(result["decay_profile"], "slow")
            base = service.read_state("minime")
            clone = service.read_state("attr_minime_lambda_edge_seed")
            self.assertEqual(base["backend"], clone["backend"])
            self.assertTrue(np.allclose(base["h_norms"], clone["h_norms"]))
            self.assertEqual(clone["last_live_meta"]["source"], "clone_handle")
            self.assertEqual(clone["last_live_meta"]["from_handle"], "minime")
            self.assertEqual(clone["last_live_meta"]["intent_id"], "intent-1")

            service.tick(
                "attr_minime_lambda_edge_seed",
                np.full((1, 32), 0.5, dtype=np.float32),
                meta={"source": "garden_shape"},
            )
            diverged = service.resonance("minime", "attr_minime_lambda_edge_seed")
            self.assertIn("divergence", diverged)
            self.assertEqual(service.read_state("minime")["tick_count"], 1)

            service.destroy_handle("attr_minime_lambda_edge_seed")
            self.assertEqual(service.read_state("minime")["name"], "minime")

    def test_clone_handle_rejects_shadow_backend(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("minime__lsm", "minime", backend="lsm_shadow")

            result = asyncio.run(
                service.dispatch(
                    {
                        "type": "clone_handle",
                        "source": "minime__lsm",
                        "name": "attr_shadow_blocked",
                    }
                )
            )

            self.assertEqual(result["type"], "error")
            self.assertIn("does not support clone_handle", result["message"])

    def test_push_state_meta_is_visible_without_synthetic_output(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")

            n = service.config.n_nodes
            h1 = np.full((1, n), 0.5, dtype=np.float32)
            h2 = np.full((1, n), 0.25, dtype=np.float32)
            h3 = np.full((1, n), 0.125, dtype=np.float32)
            meta = {
                "source": "coupled_astrid_server",
                "operation": "coupled_generation_checkin",
                "generated_tokens": 17,
                "elapsed_s": 1.8,
                "tok_per_s": 9.4,
                "coupling_strength": 0.15,
                "response_preview": "Astrid left a compact generation preview.",
                "reservoir_readout": {
                    "y1_final": 0.11,
                    "y2_final": -0.04,
                    "y3_final": 0.22,
                },
                "rehearsal_hint": {
                    "mode": "rehearse",
                    "decay_profile": "medium",
                    "reason": "generation afterimage should linger softly",
                },
            }

            result = service.push_handle_state(
                "astrid",
                b64_state(h1),
                b64_state(h2),
                b64_state(h3),
                tick_delta=17,
                last_input=[0.1] * 32,
                meta=meta,
            )
            read = service.read_state("astrid")
            listed = service.list_handles()["handles"][0]
            trajectory = service.trajectory("astrid", last_n=5)

            self.assertEqual(result["tick"], 17)
            self.assertEqual(result["last_live_meta"]["source"], "coupled_astrid_server")
            self.assertEqual(result["last_live_meta"]["event_id"], "astrid:000001")
            self.assertEqual(result["hint_status"], "present_not_adopted")
            self.assertEqual(read["last_live_meta"]["generated_tokens"], 17)
            self.assertEqual(read["last_generation_meta"]["source"], "coupled_astrid_server")
            self.assertEqual(read["last_live_meta"]["reservoir_readout"]["y3_final"], 0.22)
            self.assertEqual(read["provenance"][-1]["source"], "coupled_astrid_server")
            self.assertEqual(read["provenance"][-1]["event_id"], "astrid:000001")
            self.assertEqual(read["provenance"][-1]["response_preview"], "Astrid left a compact generation preview.")
            self.assertIsNotNone(read["seconds_since_live"])
            self.assertEqual(listed["last_source"], "coupled_astrid_server")
            self.assertEqual(listed["last_event_id"], "astrid:000001")
            self.assertEqual(listed["last_generation_source"], "coupled_astrid_server")
            self.assertEqual(trajectory["outputs"], [])
            self.assertEqual(len(trajectory["h_norms"]), 1)

    def test_push_state_without_meta_records_default_provenance(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")

            zeros = np.zeros((1, service.config.n_nodes), dtype=np.float32)
            service.push_handle_state(
                "astrid",
                b64_state(zeros),
                b64_state(zeros),
                b64_state(zeros),
                tick_delta=3,
                last_input=[0.0] * 32,
            )

            read = service.read_state("astrid")
            listed = service.list_handles()["handles"][0]

            self.assertEqual(read["last_live_meta"]["source"], "push_state")
            self.assertEqual(read["last_live_meta"]["operation"], "state_checkin")
            self.assertEqual(read["last_live_meta"]["tick_delta"], 3)
            self.assertEqual(read["last_live_meta"]["event_id"], "astrid:000001")
            self.assertIsInstance(read["last_live_meta"]["source_timestamp"], float)
            self.assertEqual(read["provenance"][-1]["source"], "push_state")
            self.assertEqual(read["provenance"][-1]["event_id"], "astrid:000001")
            self.assertEqual(listed["recent_sources"], ["push_state"])
            self.assertEqual(listed["recent_event_ids"], ["astrid:000001"])
            self.assertEqual(read["hint_status"], "none")
            self.assertIsNotNone(read["seconds_since_live"])

    def test_split_provenance_lanes_keep_feeder_and_generation_separately(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            service = reservoir_service.ReservoirService(state_dir=Path(tmp))
            service.create_handle("astrid", "astrid")
            service.tick(
                "astrid",
                np.ones((1, 32), dtype=np.float32),
                meta={
                    "source": "astrid_feeder",
                    "projection": "passthrough",
                    "memory_role": "stable",
                },
            )
            h = np.full((1, service.config.n_nodes), 0.5, dtype=np.float32)
            service.push_handle_state(
                "astrid",
                b64_state(h),
                b64_state(h * 0.5),
                b64_state(h * 0.25),
                tick_delta=11,
                last_input=[0.1] * 32,
                meta={
                    "source": "coupled_astrid_server",
                    "operation": "coupled_generation_checkin",
                    "generated_tokens": 11,
                    "response_preview": "Short coupled completion.",
                },
            )

            read = service.read_state("astrid")
            listed = service.list_handles()["handles"][0]

            self.assertEqual(read["last_live_meta"]["source"], "coupled_astrid_server")
            self.assertEqual(read["last_live_meta"]["event_id"], "astrid:000002")
            self.assertEqual(read["last_feeder_meta"]["source"], "astrid_feeder")
            self.assertEqual(read["last_feeder_meta"]["event_id"], "astrid:000001")
            self.assertEqual(read["last_generation_meta"]["source"], "coupled_astrid_server")
            self.assertEqual(read["last_generation_meta"]["event_id"], "astrid:000002")
            self.assertEqual(listed["last_source"], "coupled_astrid_server")
            self.assertEqual(listed["last_event_id"], "astrid:000002")
            self.assertEqual(listed["last_feeder_source"], "astrid_feeder")
            self.assertEqual(listed["last_generation_source"], "coupled_astrid_server")
            self.assertEqual(listed["memory_role"], "stable")

    def test_snapshot_restore_preserves_metrics_context(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            service.tick(
                "astrid",
                np.ones((1, 32), dtype=np.float32),
                meta={
                    "source": "astrid_feeder",
                    "memory_role": "stable",
                    "rehearsal_hint": {
                        "mode": "rehearse",
                        "decay_profile": "slow",
                        "reason": "snapshot restore should keep context",
                    },
                },
            )
            service.snapshot_handle("astrid")

            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            restored.restore_handle("astrid")

            read = restored.read_state("astrid")
            listed = restored.list_handles()["handles"][0]
            trajectory = restored.trajectory("astrid", last_n=5)

            self.assertEqual(read["last_live_meta"]["source"], "astrid_feeder")
            self.assertEqual(read["last_feeder_meta"]["source"], "astrid_feeder")
            self.assertEqual(read["last_live_meta"]["event_id"], "astrid:000001")
            self.assertEqual(read["last_event_id"], "astrid:000001")
            self.assertEqual(read["rehearsal_hint"]["mode"], "rehearse")
            self.assertIsNotNone(read["seconds_since_live"])
            self.assertEqual(listed["last_source"], "astrid_feeder")
            self.assertEqual(listed["last_feeder_source"], "astrid_feeder")
            self.assertGreaterEqual(len(read["provenance"]), 1)
            self.assertGreaterEqual(len(trajectory["outputs"]), 1)

    def test_snapshot_restore_preserves_push_state_generation_digest(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            h = np.full((1, service.config.n_nodes), 0.75, dtype=np.float32)
            service.push_handle_state(
                "astrid",
                b64_state(h),
                b64_state(h * 0.5),
                b64_state(h * 0.25),
                tick_delta=21,
                last_input=[0.2] * 32,
                meta={
                    "source": "coupled_astrid_server",
                    "operation": "coupled_generation_checkin",
                    "generated_tokens": 21,
                    "response_preview": "A compact completion survives restart.",
                    "response_sha256_12": "deadbeefcafe",
                    "coupling_strength": 0.18,
                },
            )
            service.snapshot_handle("astrid")

            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            restored.restore_handle("astrid")

            read = restored.read_state("astrid")
            listed = restored.list_handles()["handles"][0]

            self.assertEqual(read["last_live_meta"]["source"], "coupled_astrid_server")
            self.assertEqual(read["last_generation_meta"]["source"], "coupled_astrid_server")
            self.assertEqual(read["last_live_meta"]["generated_tokens"], 21)
            self.assertEqual(read["last_live_meta"]["response_sha256_12"], "deadbeefcafe")
            self.assertEqual(read["last_live_meta"]["event_id"], "astrid:000001")
            self.assertIsNotNone(read["seconds_since_live"])
            self.assertEqual(listed["last_source"], "coupled_astrid_server")
            self.assertEqual(listed["last_generation_source"], "coupled_astrid_server")
            self.assertEqual(read["provenance"][-1]["response_preview"], "A compact completion survives restart.")

    def test_restore_rejects_incompatible_snapshot_and_quarantines_it(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta={"source": "astrid_feeder"})
            service.snapshot_handle("astrid")

            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            restored._snapshot_fingerprint = "different-fingerprint"

            with self.assertRaisesRegex(ValueError, "incompatible snapshot"):
                restored.restore_handle("astrid")

            invalid_dir = state_dir / "invalid"
            self.assertTrue(invalid_dir.exists())
            quarantined = sorted(path.name for path in invalid_dir.iterdir())
            self.assertTrue(any(name.startswith("astrid.npz.") and name.endswith(".invalid") for name in quarantined))
            self.assertTrue(any(name.startswith("astrid_thermostats.json.") and name.endswith(".invalid") for name in quarantined))
            self.assertFalse((state_dir / "astrid.npz").exists())

    def test_restore_rejects_legacy_snapshot_without_fingerprint(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta={"source": "astrid_feeder"})
            snapshot = Path(service.snapshot_handle("astrid")["path"])

            with np.load(snapshot, allow_pickle=False) as data:
                rewritten = {key: data[key] for key in data.files if key not in {"snapshot_version", "config_fingerprint"}}
            np.savez(snapshot, **rewritten)

            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            with self.assertRaisesRegex(ValueError, "incompatible snapshot"):
                restored.restore_handle("astrid")

            self.assertTrue((state_dir / "invalid").exists())

    def test_event_sequence_continues_after_restore(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            service.tick("astrid", np.ones((1, 32), dtype=np.float32), meta={"source": "astrid_feeder"})
            service.snapshot_handle("astrid")

            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            restored.restore_handle("astrid")
            restored.tick("astrid", np.ones((1, 32), dtype=np.float32), meta={"source": "astrid_feeder"})

            read = restored.read_state("astrid")
            self.assertEqual(read["last_live_meta"]["event_id"], "astrid:000002")
            self.assertEqual(read["last_live_meta"]["event_seq"], 2)

    def test_thermostat_metrics_survive_restart_without_warming_up_again(self):
        reservoir_service = load_reservoir_service_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            service = reservoir_service.ReservoirService(state_dir=state_dir)
            service.create_handle("astrid", "astrid")
            info = service.handles["astrid"]
            for thermostat in info.thermostats:
                thermostat.warmup_ticks = 10
                thermostat.control_interval = 5

            for i in range(40):
                vec = np.full((1, 32), 0.2 + 0.01 * i, dtype=np.float32)
                service.tick("astrid", vec, meta={"source": "astrid_feeder"})

            before = service.layer_metrics("astrid")["layers"]
            self.assertTrue(all(layer["entropy"] is not None for layer in before))
            self.assertTrue(all(layer["entropy_target"] is not None for layer in before))

            service.snapshot_handle("astrid")
            restored = reservoir_service.ReservoirService(state_dir=state_dir)
            restored.restore_handle("astrid")
            restored_info = restored.handles["astrid"]
            for thermostat in restored_info.thermostats:
                thermostat.control_interval = 5

            for _ in range(5):
                restored.tick("astrid", np.full((1, 32), 0.7, dtype=np.float32), meta={"source": "astrid_feeder"})

            after = restored.layer_metrics("astrid")["layers"]
            self.assertTrue(all(layer["entropy"] is not None for layer in after))
            self.assertTrue(all(layer["entropy_target"] is not None for layer in after))

    def test_metrics_snapshot_renders_coupled_generation_summary(self):
        import metrics_snapshot

        snapshot = {
            "captured_at": "2026-03-30 22:45:00",
            "handles": [{
                "name": "astrid",
                "entity": "astrid",
                "mode": "rehearse",
                "last_tick_ago": 2.4,
                "tick_count": 144,
                "latest_output": 0.12,
                "trend": "rising",
                "trend_delta": 0.08,
                "spread": 0.15,
                "last_source": "coupled_astrid_server",
                "last_event_id": "astrid:000042",
                "recent_sources": ["coupled_astrid_server"],
                "last_feeder_meta": {
                    "source": "astrid_feeder",
                    "event_id": "astrid:000041",
                    "source_timestamp": 1774939500.0,
                    "memory_role": "stable",
                    "projection": "passthrough",
                },
                "last_live_meta": {
                    "source": "coupled_astrid_server",
                    "event_id": "astrid:000042",
                    "source_timestamp": 1774939502.0,
                    "generated_tokens": 42,
                    "elapsed_s": 2.3,
                    "tok_per_s": 18.2,
                    "coupling_strength": 0.17,
                    "response_preview": "A compact summary of the last coupled completion.",
                    "reservoir_readout": {
                        "y1_final": 0.12,
                        "y2_final": -0.03,
                        "y3_final": 0.41,
                    },
                },
                "rehearsal_hint": None,
                "hint_status": "none",
                "layers": [],
                "layer_states": [],
            }],
            "resonances": [],
        }

        markdown = metrics_snapshot.render_markdown(snapshot)
        text = metrics_snapshot.render_text(snapshot)

        self.assertIn("Last generation: 42 tokens in 2.3s (18.2 tok/s), coupling=0.170", markdown)
        self.assertIn("Feeder lane: astrid_feeder, memory=stable, projection=passthrough", markdown)
        self.assertIn("evt=astrid:000041 @ 2026-03-30 23:45:00", markdown)
        self.assertIn("evt=astrid:000042 @ 2026-03-30 23:45:02", markdown)
        self.assertIn("preview='A compact summary of the last coupled completion.'", markdown)
        self.assertIn("last event: evt=astrid:000042 @ 2026-03-30 23:45:02", text)
        self.assertIn("feeder lane: astrid_feeder, memory=stable, projection=passthrough", text)
        self.assertIn("last generation: 42 tokens in 2.3s (18.2 tok/s), coupling=0.170", text)


if __name__ == "__main__":
    unittest.main()
