#!/usr/bin/env python3
from __future__ import annotations

import sys
import types
import unittest

import numpy as np

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Module:
        pass

    class _Linear:
        def __init__(self, *args, **kwargs):
            pass

    def _no_grad():
        def decorator(fn):
            return fn
        return decorator

    torch_stub.nn = types.SimpleNamespace(Module=_Module, Linear=_Linear)
    torch_stub.no_grad = _no_grad
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub

IMPORT_ERROR = None
try:
    from dual_ai_bridge import ReservoirBridge
    from triple_reservoir_coreml import ReservoirConfig
except ModuleNotFoundError as exc:
    ReservoirBridge = None
    ReservoirConfig = None
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"optional dependency unavailable: {IMPORT_ERROR}")
class LSMShadowTests(unittest.TestCase):
    def make_config(self) -> ReservoirConfig:
        return ReservoirConfig(input_dim=8, n_nodes=16, washout=8)

    def test_bridge_supports_live_and_shadow_handles(self):
        bridge = ReservoirBridge(config=self.make_config(), backend="numpy")
        bridge.make_state("live")
        bridge.make_state("live__lsm", backend="lsm_shadow")

        x = np.full((1, 8), 0.25, dtype=np.float32)
        live_output = bridge.tick("live", x)
        shadow_output = bridge.tick("live__lsm", x)
        shadow_state = bridge.read_state("live__lsm")

        self.assertEqual(bridge.backend_for_handle("live"), "numpy")
        self.assertEqual(bridge.backend_for_handle("live__lsm"), "lsm_shadow")
        self.assertTrue(np.isfinite(live_output))
        self.assertTrue(np.isfinite(shadow_output))
        self.assertTrue(all(layer.shape == (1, 16) for layer in shadow_state))
        self.assertTrue(bridge.supports_pull_push("live"))
        self.assertFalse(bridge.supports_pull_push("live__lsm"))

    def test_lsm_set_state_round_trips_public_trace_state(self):
        bridge = ReservoirBridge(config=self.make_config(), backend="numpy")
        bridge.make_state("shadow", backend="lsm_shadow")

        h1 = np.full((1, 16), 0.3, dtype=np.float32)
        h2 = np.full((1, 16), 0.2, dtype=np.float32)
        h3 = np.full((1, 16), 0.1, dtype=np.float32)
        bridge.set_state("shadow", h1, h2, h3)
        read_h1, read_h2, read_h3 = bridge.read_state("shadow")

        np.testing.assert_allclose(read_h1, h1)
        np.testing.assert_allclose(read_h2, h2)
        np.testing.assert_allclose(read_h3, h3)


if __name__ == "__main__":
    unittest.main()
