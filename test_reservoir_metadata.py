#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


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
        def __init__(self, config, use_mlx=False):
            self.config = config
            self.states = {}
            self.outputs = {}

        def has_handle(self, name):
            return name in self.states

        def make_state(self, name):
            zeros = np.zeros((1, self.config.n_nodes), dtype=np.float32)
            self.states[name] = (zeros.copy(), zeros.copy(), zeros.copy())
            self.outputs[name] = 0.0

        def destroy_handle(self, name):
            self.states.pop(name, None)
            self.outputs.pop(name, None)

        def tick(self, name, input_vec):
            value = float(np.asarray(input_vec, dtype=np.float32).sum())
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

    fake_dual.ReservoirBridge = FakeBridge
    fake_dual.TextProjection = FakeTextProjection
    fake_coreml.ReservoirConfig = FakeReservoirConfig

    sys.modules["dual_ai_bridge"] = fake_dual
    sys.modules["triple_reservoir_coreml"] = fake_coreml
    sys.modules.pop("reservoir_service", None)
    return importlib.import_module("reservoir_service")


class ReservoirMetadataTests(unittest.TestCase):
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
            self.assertEqual(read["last_live_meta"]["memory_role"], "stable")
            self.assertEqual(read["rehearsal_hint"]["mode"], "hold")

            listed = service.list_handles()["handles"][0]
            self.assertEqual(listed["last_source"], "astrid_feeder")
            self.assertEqual(listed["memory_role"], "stable")
            self.assertEqual(listed["rehearsal_hint"]["decay_profile"], "slow")
            self.assertEqual(read["provenance"][-1]["source"], "astrid_feeder")
            self.assertEqual(listed["recent_sources"], ["astrid_feeder"])

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


if __name__ == "__main__":
    unittest.main()
