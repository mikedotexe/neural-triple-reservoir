from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge


def massage_to_directional_vector(
    price_dev: float,
    gas_dev: float,
    liquidity_dev: float,
    oracle_price: float = 80_000.0,
    comfort_gas: float = 0.5,
    liquidity_scale: float = 80_000.0,
) -> np.ndarray:
    p = np.tanh(price_dev / max(oracle_price, 1e-6))
    g = np.tanh(-gas_dev / max(comfort_gas, 1e-6))
    l = np.tanh(liquidity_dev / max(liquidity_scale, 1e-6))
    return np.array([p, g, l], dtype=np.float32)


def massage_batch(
    rows: np.ndarray,
    oracle_price: float = 80_000.0,
    comfort_gas: float = 0.5,
    liquidity_scale: float = 80_000.0,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"expected shape (n, 3), got {rows.shape}")
    out = np.empty_like(rows, dtype=np.float32)
    out[:, 0] = np.tanh(rows[:, 0] / max(oracle_price, 1e-6))
    out[:, 1] = np.tanh(-rows[:, 1] / max(comfort_gas, 1e-6))
    out[:, 2] = np.tanh(rows[:, 2] / max(liquidity_scale, 1e-6))
    return out


@dataclass(frozen=True)
class ReservoirConfig:
    input_dim: int = 3
    n_nodes: int = 192
    output_dim: int = 1
    radii: Tuple[float, float, float] = (0.98, 0.92, 0.85)
    leaks: Tuple[float, float, float] = (0.25, 0.18, 0.12)
    densities: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    input_scales: Tuple[float, float, float] = (0.8, 0.7, 0.6)
    bias_scale: float = 0.05
    ridge_alpha: float = 1e-2
    washout: int = 128
    seed: int = 7


def _orthogonal_matrix(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    q = q * np.sign(np.diag(r))
    return (radius * q).astype(np.float32)


def _sparse_matrix(n: int, density: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    for _ in range(16):
        w = rng.standard_normal((n, n)).astype(np.float64)
        w *= (rng.random((n, n)) < density)
        rho = float(np.max(np.abs(np.linalg.eigvals(w))))
        if rho > 1e-8:
            w *= radius / rho
            return w.astype(np.float32)
    raise RuntimeError("failed to initialize a non-degenerate sparse reservoir")


def _make_recurrent(n: int, density: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    if density >= 0.999:
        return _orthogonal_matrix(n, radius, rng)
    return _sparse_matrix(n, density, radius, rng)


class TripleReservoir:
    def __init__(self, config: ReservoirConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        n = config.n_nodes
        d = config.input_dim

        self.x_mean = np.zeros((d,), dtype=np.float32)
        self.x_scale = np.ones((d,), dtype=np.float32)

        self.w_in1 = (config.input_scales[0] * rng.standard_normal((n, d))).astype(np.float32)
        self.w1 = _make_recurrent(n, config.densities[0], config.radii[0], rng)
        self.b1 = (config.bias_scale * rng.standard_normal((n,))).astype(np.float32)

        self.w_in2 = (config.input_scales[1] * rng.standard_normal((n, n))).astype(np.float32)
        self.w2 = _make_recurrent(n, config.densities[1], config.radii[1], rng)
        self.b2 = (config.bias_scale * rng.standard_normal((n,))).astype(np.float32)

        self.w_in3 = (config.input_scales[2] * rng.standard_normal((n, n))).astype(np.float32)
        self.w3 = _make_recurrent(n, config.densities[2], config.radii[2], rng)
        self.b3 = (config.bias_scale * rng.standard_normal((n,))).astype(np.float32)

        self.readout_w = np.zeros((config.output_dim, 3 * n), dtype=np.float32)
        self.readout_b = np.zeros((config.output_dim,), dtype=np.float32)

    def zero_state(self, batch: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.n_nodes
        z = np.zeros((batch, n), dtype=np.float32)
        return z.copy(), z.copy(), z.copy()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_scale

    def step_numpy(
        self,
        x: np.ndarray,
        state: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.config.input_dim)
        x = self._normalize(x)
        h1, h2, h3 = state
        l1, l2, l3 = self.config.leaks

        h1 = (1.0 - l1) * h1 + l1 * np.tanh(x @ self.w_in1.T + h1 @ self.w1.T + self.b1)
        h2 = (1.0 - l2) * h2 + l2 * np.tanh(h1 @ self.w_in2.T + h2 @ self.w2.T + self.b2)
        h3 = (1.0 - l3) * h3 + l3 * np.tanh(h2 @ self.w_in3.T + h3 @ self.w3.T + self.b3)

        z = np.concatenate([h1, h2, h3], axis=1)
        y = z @ self.readout_w.T + self.readout_b
        next_state = (
            h1.astype(np.float32),
            h2.astype(np.float32),
            h3.astype(np.float32),
        )
        return y.astype(np.float32), z.astype(np.float32), next_state

    def collect_states(self, x_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_seq = np.asarray(x_seq, dtype=np.float32).reshape(-1, self.config.input_dim)
        state = self.zero_state()
        zs = []
        ys = []
        for row in x_seq:
            y, z, state = self.step_numpy(row[None, :], state)
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

    def fit_multi_readout(
        self,
        x_seq: np.ndarray,
        y_seq: np.ndarray,
        washout: Optional[int] = None,
    ):
        """Train three per-layer readouts with different temporal targets.

        y1 (h1, fast):   short-horizon latent (original target)
        y2 (h2, medium): medium-horizon (5-step moving average)
        y3 (h3, slow):   slow trend (20-step exponential moving average)

        Call after fit_readout() — uses the same normalization.
        """
        x_seq = np.asarray(x_seq, dtype=np.float32).reshape(-1, self.config.input_dim)
        y_seq = np.asarray(y_seq, dtype=np.float32)
        if y_seq.ndim == 1:
            y_seq = y_seq[:, None]

        washout = self.config.washout if washout is None else int(washout)
        washout = max(0, min(washout, x_seq.shape[0] - 1))

        # Collect per-layer hidden states
        n = self.config.n_nodes
        state = self.zero_state()
        h1_seq, h2_seq, h3_seq = [], [], []
        for row in x_seq:
            _, _, state = self.step_numpy(row[None, :], state)
            h1_seq.append(state[0][0])  # (n_nodes,)
            h2_seq.append(state[1][0])
            h3_seq.append(state[2][0])
        h1_arr = np.stack(h1_seq)[washout:]
        h2_arr = np.stack(h2_seq)[washout:]
        h3_arr = np.stack(h3_seq)[washout:]

        # Build temporal targets
        y_raw = y_seq[washout:].astype(np.float64)
        T = len(y_raw)

        # y1 target: original (short-horizon)
        y1_target = y_raw

        # y2 target: 5-step moving average
        y2_target = np.copy(y_raw)
        for t in range(T):
            start = max(0, t - 4)
            y2_target[t] = y_raw[start:t + 1].mean(axis=0)

        # y3 target: exponential moving average (alpha=0.05, ~20-step window)
        y3_target = np.copy(y_raw)
        ema = y_raw[0].copy()
        for t in range(T):
            ema = 0.95 * ema + 0.05 * y_raw[t]
            y3_target[t] = ema

        # Fit three Ridge regressors
        alpha = self.config.ridge_alpha
        for name, h_arr, y_tgt, attr_w, attr_b in [
            ("h1", h1_arr, y1_target, "readout_w1", "readout_b1"),
            ("h2", h2_arr, y2_target, "readout_w2", "readout_b2"),
            ("h3", h3_arr, y3_target, "readout_w3", "readout_b3"),
        ]:
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(h_arr.astype(np.float64), y_tgt)
            coef = ridge.coef_
            if coef.ndim == 1:
                coef = coef[None, :]
            setattr(self, attr_w, coef.astype(np.float32))
            setattr(self, attr_b, np.atleast_1d(ridge.intercept_).astype(np.float32))

    def predict_sequence(self, x_seq: np.ndarray) -> np.ndarray:
        _, y = self.collect_states(x_seq)
        return y

    def build_torch_model(self) -> "StatefulTripleReservoir":
        return StatefulTripleReservoir(self)


CANONICAL_SEED = 99


def build_canonical_reservoir(
    input_dim: int = 32, n_nodes: int = 192,
) -> tuple["TripleReservoir", "ReservoirConfig"]:
    """Build and train a reservoir with canonical seed and synthetic data.

    All backends must use this to ensure identical readout weights.
    Returns (model, config).
    """
    cfg = ReservoirConfig(input_dim=input_dim, n_nodes=n_nodes)
    model = TripleReservoir(cfg)
    rng = np.random.default_rng(CANONICAL_SEED)
    d = cfg.input_dim
    x = np.tanh(rng.standard_normal((3000, d))).astype(np.float32)
    y = np.zeros((3000, cfg.output_dim), dtype=np.float32)
    latent = 0.0
    for t in range(3000):
        latent = 0.9 * latent + 0.3 * float(x[t, : min(4, d)].sum())
        y[t, 0] = latent
    model.fit_readout(x, y)
    model.fit_multi_readout(x, y)
    return model, cfg

    def export_coreml(self, package_path: str | Path, compute_units: str = "cpu_and_ne"):
        try:
            import coremltools as ct
        except ImportError as e:
            raise RuntimeError("coremltools is required for export: pip install coremltools") from e

        package_path = Path(package_path)
        model = self.build_torch_model().eval()
        model.reset_state()
        scripted = torch.jit.script(model)

        compute_unit_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        }
        if compute_units not in compute_unit_map:
            raise ValueError(
                f"compute_units must be one of {sorted(compute_unit_map)}, got {compute_units}"
            )

        n = self.config.n_nodes
        mlmodel = ct.convert(
            scripted,
            inputs=[ct.TensorType(name="x", shape=(1, self.config.input_dim), dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, n), dtype=np.float16),
                    name="h1",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, n), dtype=np.float16),
                    name="h2",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, n), dtype=np.float16),
                    name="h3",
                ),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_units=compute_unit_map[compute_units],
        )
        mlmodel.save(str(package_path))
        return mlmodel


class StatefulTripleReservoir(torch.nn.Module):
    def __init__(self, bundle: TripleReservoir):
        super().__init__()
        n = bundle.config.n_nodes
        to_t = lambda a: torch.from_numpy(np.asarray(a)).to(torch.float16)

        self.register_buffer("x_mean", to_t(bundle.x_mean).reshape(1, -1))
        self.register_buffer("x_scale", to_t(bundle.x_scale).reshape(1, -1))

        self.register_buffer("w_in1", to_t(bundle.w_in1))
        self.register_buffer("w1", to_t(bundle.w1))
        self.register_buffer("b1", to_t(bundle.b1).reshape(1, -1))

        self.register_buffer("w_in2", to_t(bundle.w_in2))
        self.register_buffer("w2", to_t(bundle.w2))
        self.register_buffer("b2", to_t(bundle.b2).reshape(1, -1))

        self.register_buffer("w_in3", to_t(bundle.w_in3))
        self.register_buffer("w3", to_t(bundle.w3))
        self.register_buffer("b3", to_t(bundle.b3).reshape(1, -1))

        self.register_buffer("leak1", torch.tensor(bundle.config.leaks[0], dtype=torch.float16))
        self.register_buffer("leak2", torch.tensor(bundle.config.leaks[1], dtype=torch.float16))
        self.register_buffer("leak3", torch.tensor(bundle.config.leaks[2], dtype=torch.float16))

        self.register_buffer("h1", torch.zeros((1, n), dtype=torch.float16))
        self.register_buffer("h2", torch.zeros((1, n), dtype=torch.float16))
        self.register_buffer("h3", torch.zeros((1, n), dtype=torch.float16))

        self.readout = torch.nn.Linear(3 * n, bundle.config.output_dim, bias=True, dtype=torch.float16)
        with torch.no_grad():
            self.readout.weight.copy_(to_t(bundle.readout_w))
            self.readout.bias.copy_(to_t(bundle.readout_b).reshape(-1))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.h1.zero_()
        self.h2.zero_()
        self.h3.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        x = (x - self.x_mean) / self.x_scale

        h1_drive = x @ self.w_in1.T + self.h1 @ self.w1.T + self.b1
        self.h1.mul_(1.0 - self.leak1).add_(self.leak1 * torch.tanh(h1_drive))

        h2_drive = self.h1 @ self.w_in2.T + self.h2 @ self.w2.T + self.b2
        self.h2.mul_(1.0 - self.leak2).add_(self.leak2 * torch.tanh(h2_drive))

        h3_drive = self.h2 @ self.w_in3.T + self.h3 @ self.w3.T + self.b3
        self.h3.mul_(1.0 - self.leak3).add_(self.leak3 * torch.tanh(h3_drive))

        z = torch.cat([self.h1, self.h2, self.h3], dim=-1)
        return self.readout(z)


def build_stub_dataset(steps: int = 5000, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = np.column_stack(
        [
            rng.normal(0.0, 900.0, size=steps),
            rng.normal(0.0, 0.08, size=steps),
            rng.normal(0.0, 1200.0, size=steps),
        ]
    ).astype(np.float32)
    x = massage_batch(raw)
    y = np.zeros((steps, 1), dtype=np.float32)
    latent = 0.0
    for t in range(steps):
        latent = 0.92 * latent + 0.85 * x[t, 0] - 0.45 * x[t, 1] + 0.30 * x[t, 2]
        if t >= 3:
            latent += 0.25 * x[t - 3, 0] - 0.10 * x[t - 1, 2]
        y[t, 0] = latent + 0.02 * rng.normal()
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--n-nodes", type=int, default=192)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="cpu_and_ne",
    )
    args = parser.parse_args()

    cfg = ReservoirConfig(n_nodes=args.n_nodes)
    model = TripleReservoir(cfg)

    x_seq, y_seq = build_stub_dataset(steps=args.steps)
    train_mse = model.fit_readout(x_seq, y_seq)
    print(f"train_mse={train_mse:.8f}")

    torch_model = model.build_torch_model().eval()
    torch_model.reset_state()
    for i in range(3):
        tick = torch.from_numpy(x_seq[i : i + 1]).to(torch.float16)
        pred = torch_model(tick).detach().cpu().numpy()
        print(f"tick={i} pred={pred.ravel()[0]:.6f}")

    if args.export is not None:
        mlmodel = model.export_coreml(args.export, compute_units=args.compute_units)
        loaded = type(mlmodel).__name__
        print(f"saved={args.export} loader={loaded}")
        try:
            import coremltools as ct

            runtime_model = ct.models.MLModel(
                str(args.export),
                compute_units={
                    "all": ct.ComputeUnit.ALL,
                    "cpu_only": ct.ComputeUnit.CPU_ONLY,
                    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
                    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
                }[args.compute_units],
            )
            state = runtime_model.make_state()
            sample = x_seq[0:1].astype(np.float16)
            out = runtime_model.predict({"x": sample}, state=state)["y"]
            print(f"coreml_pred={np.asarray(out).reshape(-1)[0]:.6f}")
        except Exception as e:
            print(f"post_export_runtime_check_skipped={e}")


if __name__ == "__main__":
    main()
