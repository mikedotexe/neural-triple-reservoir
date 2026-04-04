#!/usr/bin/env python3
"""
lsm_reservoir.py -- Simple liquid-state shadow reservoir for phase-1 comparison.

The public state is three trace layers `(h1, h2, h3)` that behave like the
existing reservoir's readable hidden state. Private membrane and spike tensors
stay internal to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from triple_reservoir_coreml import CANONICAL_SEED, ReservoirConfig


@dataclass
class LSMState:
    mem1: np.ndarray
    mem2: np.ndarray
    mem3: np.ndarray
    spk1: np.ndarray
    spk2: np.ndarray
    spk3: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    h3: np.ndarray


def _make_sparse(
    rows: int,
    cols: int,
    density: float,
    target_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mat = rng.standard_normal((rows, cols)).astype(np.float32)
    if density < 0.999:
        mask = rng.random((rows, cols)) < density
        mat *= mask.astype(np.float32)
    row_sum = np.max(np.sum(np.abs(mat), axis=1))
    if not np.isfinite(row_sum) or row_sum < 1e-6:
        row_sum = 1.0
    return (mat * (target_scale / row_sum)).astype(np.float32)


class LSMReservoir:
    """Three-layer LIF-style liquid state reservoir with trace readout."""

    def __init__(self, config: ReservoirConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        n = config.n_nodes
        d = config.input_dim

        self.x_mean = np.zeros((d,), dtype=np.float32)
        self.x_scale = np.ones((d,), dtype=np.float32)

        radii = tuple(float(r) * 0.34 for r in config.radii)
        self.w_in1 = _make_sparse(n, d, 1.0, config.input_scales[0], rng)
        self.w1 = _make_sparse(n, n, config.densities[0], radii[0], rng)
        self.b1 = (config.bias_scale * rng.standard_normal((n,))).astype(np.float32)

        self.w_in2 = _make_sparse(n, n, 1.0, config.input_scales[1], rng)
        self.w2 = _make_sparse(n, n, config.densities[1], radii[1], rng)
        self.b2 = (config.bias_scale * rng.standard_normal((n,))).astype(np.float32)

        self.w_in3 = _make_sparse(n, n, 1.0, config.input_scales[2], rng)
        self.w3 = _make_sparse(n, n, config.densities[2], radii[2], rng)
        self.b3 = (config.bias_scale * rng.standard_normal((n,))).astype(np.float32)

        self.mem_decay = tuple(
            float(np.clip(1.0 - leak * 1.6, 0.18, 0.88)) for leak in config.leaks
        )
        self.trace_decay = tuple(
            float(np.clip(1.0 - leak * 0.45, 0.68, 0.97)) for leak in config.leaks
        )
        self.thresholds = (0.82, 0.94, 1.05)
        self.readout_w = np.zeros((config.output_dim, 3 * n), dtype=np.float32)
        self.readout_b = np.zeros((config.output_dim,), dtype=np.float32)

    def zero_state(self, batch: int = 1) -> LSMState:
        n = self.config.n_nodes
        zeros = np.zeros((batch, n), dtype=np.float32)
        return LSMState(
            mem1=zeros.copy(),
            mem2=zeros.copy(),
            mem3=zeros.copy(),
            spk1=zeros.copy(),
            spk2=zeros.copy(),
            spk3=zeros.copy(),
            h1=zeros.copy(),
            h2=zeros.copy(),
            h3=zeros.copy(),
        )

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_scale).astype(np.float32)

    def _lif_step(
        self,
        drive: np.ndarray,
        recurrent_source: np.ndarray,
        mem: np.ndarray,
        trace: np.ndarray,
        w_rec: np.ndarray,
        bias: np.ndarray,
        mem_decay: float,
        trace_decay: float,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current = drive + recurrent_source @ w_rec.T + bias.reshape(1, -1)
        mem_next = np.clip(mem * mem_decay + current, -2.5, 2.5)
        spk_next = (mem_next >= threshold).astype(np.float32)
        mem_next = np.where(spk_next > 0.0, 0.0, mem_next)
        trace_next = trace * trace_decay + spk_next
        return mem_next.astype(np.float32), spk_next.astype(np.float32), trace_next.astype(
            np.float32
        )

    def step(self, x: np.ndarray, state: LSMState) -> tuple[np.ndarray, np.ndarray, LSMState]:
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.config.input_dim)
        x = self._normalize(x)

        mem1, spk1, h1 = self._lif_step(
            x @ self.w_in1.T,
            state.h1,
            state.mem1,
            state.h1,
            self.w1,
            self.b1,
            self.mem_decay[0],
            self.trace_decay[0],
            self.thresholds[0],
        )
        mem2, spk2, h2 = self._lif_step(
            h1 @ self.w_in2.T,
            state.h2,
            state.mem2,
            state.h2,
            self.w2,
            self.b2,
            self.mem_decay[1],
            self.trace_decay[1],
            self.thresholds[1],
        )
        mem3, spk3, h3 = self._lif_step(
            h2 @ self.w_in3.T,
            state.h3,
            state.mem3,
            state.h3,
            self.w3,
            self.b3,
            self.mem_decay[2],
            self.trace_decay[2],
            self.thresholds[2],
        )

        z = np.concatenate([h1, h2, h3], axis=1).astype(np.float32)
        y = z @ self.readout_w.T + self.readout_b.reshape(1, -1)
        next_state = LSMState(
            mem1=mem1,
            mem2=mem2,
            mem3=mem3,
            spk1=spk1,
            spk2=spk2,
            spk3=spk3,
            h1=h1,
            h2=h2,
            h3=h3,
        )
        return y.astype(np.float32), z, next_state

    def public_state(self, state: LSMState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return state.h1.copy(), state.h2.copy(), state.h3.copy()

    def state_from_public(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        h3: np.ndarray,
    ) -> LSMState:
        h1 = np.asarray(h1, dtype=np.float32).reshape(1, self.config.n_nodes)
        h2 = np.asarray(h2, dtype=np.float32).reshape(1, self.config.n_nodes)
        h3 = np.asarray(h3, dtype=np.float32).reshape(1, self.config.n_nodes)
        return LSMState(
            mem1=np.clip(h1 * 0.35, -1.0, 1.0),
            mem2=np.clip(h2 * 0.35, -1.0, 1.0),
            mem3=np.clip(h3 * 0.35, -1.0, 1.0),
            spk1=np.zeros_like(h1),
            spk2=np.zeros_like(h2),
            spk3=np.zeros_like(h3),
            h1=h1.copy(),
            h2=h2.copy(),
            h3=h3.copy(),
        )

    def collect_states(self, x_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_seq = np.asarray(x_seq, dtype=np.float32).reshape(-1, self.config.input_dim)
        state = self.zero_state()
        zs = []
        ys = []
        for row in x_seq:
            y, z, state = self.step(row[None, :], state)
            zs.append(z[0])
            ys.append(y[0])
        return np.stack(zs), np.stack(ys)

    def fit_readout(
        self,
        x_seq: np.ndarray,
        y_seq: np.ndarray,
        washout: Optional[int] = None,
    ) -> float:
        x_seq = np.asarray(x_seq, dtype=np.float32).reshape(-1, self.config.input_dim)
        y_seq = np.asarray(y_seq, dtype=np.float32)
        if y_seq.ndim == 1:
            y_seq = y_seq[:, None]
        if y_seq.shape[0] != x_seq.shape[0]:
            raise ValueError(f"x/y length mismatch: {x_seq.shape[0]} vs {y_seq.shape[0]}")

        self.x_mean = x_seq.mean(axis=0).astype(np.float32)
        self.x_scale = x_seq.std(axis=0).astype(np.float32)
        self.x_scale[self.x_scale < 1e-6] = 1.0

        washout = self.config.washout if washout is None else int(washout)
        washout = max(0, min(washout, x_seq.shape[0] - 1))

        z_seq, _ = self.collect_states(x_seq)
        x_fit = z_seq[washout:]
        y_fit = y_seq[washout:]

        ridge = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        ridge.fit(x_fit.astype(np.float64), y_fit.astype(np.float64))
        coef = ridge.coef_
        if coef.ndim == 1:
            coef = coef[None, :]
        self.readout_w = coef.astype(np.float32)
        self.readout_b = np.atleast_1d(ridge.intercept_).astype(np.float32)

        pred = x_fit @ self.readout_w.T + self.readout_b
        return float(np.mean((pred - y_fit) ** 2))


def build_trained_lsm(config: ReservoirConfig) -> LSMReservoir:
    """Build a deterministic shadow model with a valid scalar readout."""
    model = LSMReservoir(config)
    rng = np.random.default_rng(CANONICAL_SEED)
    d = config.input_dim
    x = np.tanh(rng.standard_normal((3000, d))).astype(np.float32)
    y = np.zeros((3000, config.output_dim), dtype=np.float32)
    latent = 0.0
    for t in range(3000):
        latent = 0.9 * latent + 0.3 * float(x[t, : min(4, d)].sum())
        y[t, 0] = latent
    model.fit_readout(x, y)
    return model
