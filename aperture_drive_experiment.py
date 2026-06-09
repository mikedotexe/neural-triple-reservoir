#!/usr/bin/env python3
"""Offline reservoir drive experiment (fully offline, zero live impact).

Question: is the triple reservoir's measured ~2% utilization (~4 effective modes
of 192) a DYNAMICS CEILING (the recurrent dynamics collapse any input to ~4 modes)
or an IDLE/INPUT artifact (effective-D rises when the drive is richer/more varied)?

Method: instantiate a FRESH TripleReservoir at the production config (input_dim=32,
n_nodes=192, radii 0.98/0.92/0.85, leaks 0.35/0.22/0.14), drive it under input
regimes of increasing effective rank + temporal variability, and measure each
layer's participation ratio PR=(Σλ)²/Σλ² over the driven state window. Also
confirms the thermostat's rho (applied live as a uniform per-layer state scale) is
participation-ratio-neutral.

Run:  /Users/v/other/neural-triple-reservoir/.venv/bin/python aperture_drive_experiment.py
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/v/other/neural-triple-reservoir")
from triple_reservoir_coreml import ReservoirConfig, TripleReservoir  # noqa: E402

EPS = 1e-12


def participation_ratio(eigs: np.ndarray) -> float:
    e = np.asarray([x for x in eigs if x > EPS], dtype=float)
    if e.size == 0:
        return 0.0
    ssq = float((e * e).sum())
    return float(e.sum() ** 2 / ssq) if ssq > 0 else 0.0


def norm_entropy(eigs: np.ndarray) -> float:
    e = np.asarray([x for x in eigs if x > EPS], dtype=float)
    if e.size <= 1:
        return 0.0
    p = e / e.sum()
    return float(-np.sum(p * np.log(p)) / np.log(e.size))


def cov_eigs(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    Ac = A - A.mean(axis=0, keepdims=True)
    ev = np.linalg.eigvalsh(Ac.T @ Ac)
    return ev[ev > EPS]


def make_input(rng, T: int, dim: int, rank: int, variability: str) -> np.ndarray:
    """T×dim input with effective rank `rank` and given temporal variability.
    Built from a rank-dim latent projected up, then tanh (like the real codec)."""
    P = rng.standard_normal((rank, dim))
    if variability == "constant":
        latent = np.repeat(rng.standard_normal((1, rank)), T, axis=0)
    elif variability == "slow":  # smooth random walk
        latent = np.cumsum(rng.standard_normal((T, rank)) * 0.06, axis=0)
    else:  # "iid" — fast, fully fresh each tick
        latent = rng.standard_normal((T, rank))
    X = np.tanh((latent @ P) / np.sqrt(rank))
    return X.astype(np.float32)


def drive_and_measure(config, X, washout=150):
    res = TripleReservoir(config)
    state = res.zero_state()
    hs = [[], [], []]
    for row in X:
        _, _, state = res.step_numpy(row[None, :], state)
        for i in range(3):
            hs[i].append(state[i][0].copy())
    out = {}
    for i, name in enumerate(("h1", "h2", "h3")):
        W = np.stack(hs[i])[washout:]
        ev = cov_eigs(W)
        out[name] = (participation_ratio(ev), norm_entropy(ev))
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    config = ReservoirConfig(input_dim=32, n_nodes=192)  # production dims
    T = 1400
    print("Offline TripleReservoir drive experiment (fresh instance, zero live impact)")
    print(f"  config: input_dim={config.input_dim}, n_nodes={config.n_nodes}, "
          f"radii={config.radii}, leaks={config.leaks}, input_scales={config.input_scales}")
    print(f"  driving T={T} ticks per regime; PR = effective modes / 192\n")
    print(f"  {'input regime':<30}{'h1_fast':<14}{'h2_med':<14}{'h3_slow':<14}")
    print(f"  {'':<30}{'PR / Hnorm':<14}{'PR / Hnorm':<14}{'PR / Hnorm':<14}")
    regimes = [
        ("constant (idle/frozen)", 1, "constant"),
        ("rank1 slow-varying", 1, "slow"),
        ("rank4 slow (eig+fill-ish)", 4, "slow"),
        ("rank8 slow", 8, "slow"),
        ("rank16 slow", 16, "slow"),
        ("rank32 slow (fingerprint)", 32, "slow"),
        ("rank32 iid (rich/active)", 32, "iid"),
    ]
    for label, rank, var in regimes:
        X = make_input(rng, T, 32, rank, var)
        r = drive_and_measure(config, X)
        cells = "".join(f"{r[l][0]:.1f} / {r[l][1]:.2f}".ljust(14) for l in ("h1", "h2", "h3"))
        print(f"  {label:<30}{cells}")

    # PR-neutrality of the thermostat's rho (live applied as layers[i] = h_i * rho_i)
    X = make_input(rng, T, 32, 16, "slow")
    res = TripleReservoir(config)
    state = res.zero_state()
    H = []
    for row in X:
        _, _, state = res.step_numpy(row[None, :], state)
        H.append(state[0][0].copy())
    W = np.stack(H)[150:]
    pr_base = participation_ratio(cov_eigs(W))
    pr_scaled = participation_ratio(cov_eigs(W * 0.5))
    print(f"\n  thermostat-rho PR-neutrality check (uniform scale ×0.5): "
          f"base={pr_base:.2f} vs scaled={pr_scaled:.2f} → "
          f"{'PR-NEUTRAL (confirmed)' if abs(pr_base - pr_scaled) < 0.1 else 'DIFFERS'}")
    print("\n  Read: if PR rises strongly with input rank/variability → the live ~4 is an "
          "idle/input artifact (enrichment helps). If it stays ~4 across regimes → a dynamics ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
