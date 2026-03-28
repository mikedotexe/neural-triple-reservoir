#!/usr/bin/env python3
"""
test_multi_headed.py -- Smoke tests for multi-headed reservoir and state sync.

Exercises:
  1. fit_multi_readout produces per-layer weights
  2. MLX init_multi_readout copies them correctly
  3. step_multi returns three distinct per-layer scalars
  4. State round-trip: numpy ↔ mlx preserves exact values
  5. Base64 round-trip: encode/decode preserves bitwise identity
  6. ReservoirLogitProcessor three-head modulation works

Run:
    python test_multi_headed.py
"""

from __future__ import annotations

import base64
import sys

import mlx.core as mx
import numpy as np

from mlx_reservoir import EmbeddingProjection, MLXTripleReservoir, ReservoirLogitProcessor
from triple_reservoir_coreml import ReservoirConfig, TripleReservoir

SEED = 99
N_NODES = 64
INPUT_DIM = 32
STEPS = 100

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def build_reservoir():
    """Build and train a reservoir with both single and multi readouts."""
    cfg = ReservoirConfig(input_dim=INPUT_DIM, n_nodes=N_NODES)
    model = TripleReservoir(cfg)
    rng = np.random.default_rng(SEED)
    x = np.tanh(rng.standard_normal((3000, INPUT_DIM))).astype(np.float32)
    y = np.zeros((3000, cfg.output_dim), dtype=np.float32)
    latent = 0.0
    for t in range(3000):
        latent = 0.9 * latent + 0.3 * float(x[t, :min(4, INPUT_DIM)].sum())
        y[t, 0] = latent
    model.fit_readout(x, y)
    model.fit_multi_readout(x, y)
    return model, cfg


# ===========================================================================
print("=" * 60)
print("1. fit_multi_readout — per-layer weights exist")
print("=" * 60)

np_model, cfg = build_reservoir()

check("readout_w1 exists", hasattr(np_model, "readout_w1"))
check("readout_w2 exists", hasattr(np_model, "readout_w2"))
check("readout_w3 exists", hasattr(np_model, "readout_w3"))
check("readout_b1 exists", hasattr(np_model, "readout_b1"))
check("readout_b2 exists", hasattr(np_model, "readout_b2"))
check("readout_b3 exists", hasattr(np_model, "readout_b3"))

check(
    "w1 shape matches h1",
    np_model.readout_w1.shape == (1, N_NODES),
    f"got {np_model.readout_w1.shape}",
)
check(
    "w2 shape matches h2",
    np_model.readout_w2.shape == (1, N_NODES),
    f"got {np_model.readout_w2.shape}",
)
check(
    "w3 shape matches h3",
    np_model.readout_w3.shape == (1, N_NODES),
    f"got {np_model.readout_w3.shape}",
)

# Weights should differ (trained on different temporal targets)
check(
    "w1 != w2",
    not np.allclose(np_model.readout_w1, np_model.readout_w2),
)
check(
    "w2 != w3",
    not np.allclose(np_model.readout_w2, np_model.readout_w3),
)

# ===========================================================================
print()
print("=" * 60)
print("2. MLX init_multi_readout — weights transfer correctly")
print("=" * 60)

mlx_model = MLXTripleReservoir(np_model)
mlx_model.init_multi_readout(np_model)

for name in ["readout_w1", "readout_b1", "readout_w2", "readout_b2", "readout_w3", "readout_b3"]:
    np_val = getattr(np_model, name)
    mx_val = np.array(getattr(mlx_model, name))
    diff = float(np.max(np.abs(np_val.flatten() - mx_val.flatten())))
    check(f"{name} transfer", diff < 1e-6, f"max_diff={diff:.2e}")

# ===========================================================================
print()
print("=" * 60)
print("3. step_multi — three distinct per-layer scalars")
print("=" * 60)

rng = np.random.default_rng(42)
state = mlx_model.zero_state()
y1_vals, y2_vals, y3_vals = [], [], []

for t in range(STEPS):
    inp = mx.array(np.tanh(rng.standard_normal((1, INPUT_DIM))).astype(np.float32))
    (y1, y2, y3), state = mlx_model.step_multi(inp, state)
    if t % 64 == 0:
        mx.eval(*state)
    y1_vals.append(float(y1.item()))
    y2_vals.append(float(y2.item()))
    y3_vals.append(float(y3.item()))

# After 100 steps, all three should be non-zero
check("y1 non-zero", abs(y1_vals[-1]) > 1e-8, f"y1={y1_vals[-1]:.6f}")
check("y2 non-zero", abs(y2_vals[-1]) > 1e-8, f"y2={y2_vals[-1]:.6f}")
check("y3 non-zero", abs(y3_vals[-1]) > 1e-8, f"y3={y3_vals[-1]:.6f}")

# They should differ from each other (different temporal dynamics)
check(
    "y1 != y2 trajectory",
    not np.allclose(y1_vals, y2_vals, atol=1e-4),
    f"final: y1={y1_vals[-1]:.4f} y2={y2_vals[-1]:.4f}",
)
check(
    "y2 != y3 trajectory",
    not np.allclose(y2_vals, y3_vals, atol=1e-4),
    f"final: y2={y2_vals[-1]:.4f} y3={y3_vals[-1]:.4f}",
)

# Variance check — each layer should show non-trivial variation
for name, vals in [("y1", y1_vals), ("y2", y2_vals), ("y3", y3_vals)]:
    var = np.var(vals)
    check(f"{name} has variance", var > 1e-8, f"var={var:.2e}")

print(f"  (final values: y1={y1_vals[-1]:+.4f}  y2={y2_vals[-1]:+.4f}  y3={y3_vals[-1]:+.4f})")

# ===========================================================================
print()
print("=" * 60)
print("4. State round-trip: mlx → numpy → mlx")
print("=" * 60)

# Save state as numpy
h1_np, h2_np, h3_np = mlx_model.state_to_numpy(state)
check("h1 shape", h1_np.shape == (1, N_NODES), f"got {h1_np.shape}")
check("h2 shape", h2_np.shape == (1, N_NODES), f"got {h2_np.shape}")
check("h3 shape", h3_np.shape == (1, N_NODES), f"got {h3_np.shape}")

# Restore and tick
restored_state = mlx_model.state_from_numpy((h1_np, h2_np, h3_np))
next_inp = mx.array(np.tanh(rng.standard_normal((1, INPUT_DIM))).astype(np.float32))

(y1_orig, y2_orig, y3_orig), _ = mlx_model.step_multi(next_inp, state)
(y1_rest, y2_rest, y3_rest), _ = mlx_model.step_multi(next_inp, restored_state)

mx.eval(y1_orig, y2_orig, y3_orig, y1_rest, y2_rest, y3_rest)

for name, orig, rest in [("y1", y1_orig, y1_rest), ("y2", y2_orig, y2_rest), ("y3", y3_orig, y3_rest)]:
    diff = abs(float(orig.item()) - float(rest.item()))
    check(f"{name} round-trip exact", diff < 1e-6, f"diff={diff:.2e}")

# ===========================================================================
print()
print("=" * 60)
print("5. Base64 round-trip (wire format for pull/push)")
print("=" * 60)

for name, h_np in [("h1", h1_np), ("h2", h2_np), ("h3", h3_np)]:
    encoded = base64.b64encode(h_np.astype(np.float32).tobytes()).decode()
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32).reshape(h_np.shape)
    check(f"{name} base64 bitwise", np.array_equal(h_np, decoded))

# Verify the full cycle: mlx → numpy → base64 → numpy → mlx → tick
h1_b64 = base64.b64encode(h1_np.astype(np.float32).tobytes()).decode()
h2_b64 = base64.b64encode(h2_np.astype(np.float32).tobytes()).decode()
h3_b64 = base64.b64encode(h3_np.astype(np.float32).tobytes()).decode()

h1_back = np.frombuffer(base64.b64decode(h1_b64), dtype=np.float32).reshape(1, N_NODES)
h2_back = np.frombuffer(base64.b64decode(h2_b64), dtype=np.float32).reshape(1, N_NODES)
h3_back = np.frombuffer(base64.b64decode(h3_b64), dtype=np.float32).reshape(1, N_NODES)

wire_state = mlx_model.state_from_numpy((h1_back, h2_back, h3_back))
(y1_wire, y2_wire, y3_wire), _ = mlx_model.step_multi(next_inp, wire_state)
mx.eval(y1_wire, y2_wire, y3_wire)

for name, orig, wire in [("y1", y1_orig, y1_wire), ("y2", y2_orig, y2_wire), ("y3", y3_orig, y3_wire)]:
    diff = abs(float(orig.item()) - float(wire.item()))
    check(f"{name} full wire round-trip", diff < 1e-6, f"diff={diff:.2e}")

# ===========================================================================
print()
print("=" * 60)
print("6. ReservoirLogitProcessor — three-head modulation")
print("=" * 60)

proc = ReservoirLogitProcessor(coupling_strength=0.2)
fake_tokens = mx.array([1, 2, 3])
fake_logits = mx.ones((1, 100)) * 5.0  # uniform logits

# Identity check: zero energy should barely change logits
proc.update(0.0, 0.0, 0.0)
out_zero = proc(fake_tokens, fake_logits)
mx.eval(out_zero)
max_diff_zero = float(mx.max(mx.abs(out_zero - fake_logits)).item())
check("zero energy ~ identity", max_diff_zero < 0.1, f"max_diff={max_diff_zero:.4f}")

# Positive energy: should change logits
proc.update(3.0, 3.0, 3.0)
out_pos = proc(fake_tokens, fake_logits)
mx.eval(out_pos)
max_diff_pos = float(mx.max(mx.abs(out_pos - fake_logits)).item())
check("positive energy modifies logits", max_diff_pos > 0.01, f"max_diff={max_diff_pos:.4f}")

# Negative energy: should change logits in opposite direction
proc.update(-3.0, -3.0, -3.0)
out_neg = proc(fake_tokens, fake_logits)
mx.eval(out_neg)
max_diff_neg = float(mx.max(mx.abs(out_neg - fake_logits)).item())
check("negative energy modifies logits", max_diff_neg > 0.01, f"max_diff={max_diff_neg:.4f}")

# Positive and negative should differ from each other
diff_pos_neg = float(mx.max(mx.abs(out_pos - out_neg)).item())
check("pos != neg modulation", diff_pos_neg > 0.01, f"diff={diff_pos_neg:.4f}")

# Layer isolation: changing only y1 vs only y3 should produce different patterns
proc.update(3.0, 0.0, 0.0)
out_y1_only = proc(fake_tokens, fake_logits)
proc.update(0.0, 0.0, 3.0)
out_y3_only = proc(fake_tokens, fake_logits)
mx.eval(out_y1_only, out_y3_only)
diff_layers = float(mx.max(mx.abs(out_y1_only - out_y3_only)).item())
check("y1-only != y3-only", diff_layers > 0.001, f"diff={diff_layers:.4f}")

# Backward compat: update_single
proc.update_single(2.0)
check("update_single sets all three", proc._y1 == 2.0 and proc._y2 == 2.0 and proc._y3 == 2.0)

# ===========================================================================
print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

sys.exit(1 if failed > 0 else 0)
