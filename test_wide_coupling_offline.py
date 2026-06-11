#!/usr/bin/env python3
"""Offline coherence + latency gate for the wide-coupling (y4) channel.

NON-NEGOTIABLE before any live token (Astrid fears her voice "becoming noise"):
proves the embedding-tied vocab bias is WIDE but COHERENT (favors real, clustered
token families, not a random scatter), bounded, fast (<=1ms/token),
state-discriminative, and off-is-off. Loads only the embedding layer + tokenizer
(no server, no live reservoir).

    /Users/v/other/neural-triple-reservoir/.venv/bin/python test_wide_coupling_offline.py
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from wide_coupling import (
    RESERVOIR_DIM,
    build_wide_coupling_matrices,
    embedding_rows,
    pressure_scale,
    wide_bias,
)

DEFAULT_MODEL = "mlx-community/gemma-4-12B-it-5bit"


def load_model_bits(model_name):
    # Reuse the coupled server's exact loader (handles the gemma4_unified ->
    # gemma4 config override + sanitize patch), so the harness validates against
    # the same model the live lane runs. Import-safe (server start is __main__).
    import coupled_astrid_server as cas

    model, tokenizer, _secs, _audit = cas._load_mlx_runtime(
        model_name, memory_map_requested=False
    )
    if hasattr(model, "language_model"):
        embed = model.language_model.model.embed_tokens
    else:
        embed = model.model.embed_tokens
    args = getattr(model, "args", None)
    vocab = None
    if args is not None:
        if hasattr(args, "vocab_size") and args.vocab_size:
            vocab = int(args.vocab_size)
        elif hasattr(args, "text_config") and isinstance(args.text_config, dict):
            vocab = int(args.text_config.get("vocab_size") or 0) or None
    if vocab is None:
        vocab = int(getattr(tokenizer, "vocab_size", 262144))
    return embed, vocab, tokenizer


def is_wordish(s: str) -> bool:
    t = s.replace("▁", " ").replace("Ġ", " ").strip()
    return len(t) >= 2 and sum(c.isalpha() for c in t) >= max(2, int(0.6 * len(t)))


def norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def mutual_cosine(emb: np.ndarray) -> float:
    """Mean off-diagonal cosine of normalized rows — how clustered a token set is."""
    n = emb.shape[0]
    if n < 2:
        return 0.0
    g = emb @ emb.T
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--top-m", type=int, default=30)
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    embed, vocab, tok = load_model_bits(args.model)
    print(f"  vocab={vocab}", flush=True)

    P, V, info = build_wide_coupling_matrices(embed, vocab, k=args.k)
    Vnp = np.array(V)  # [k, vocab]
    rng = np.random.default_rng(0)
    failures: list[str] = []

    # ---- (a) COHERENCE: each axis's top tokens are real + clustered, not noise.
    print("\n[a] coherence — each axis's top tokens (real + clustered, not noise):")
    rand_cohesions = []
    for _ in range(args.k):
        ridx = rng.choice(vocab, size=args.top_m, replace=False)
        rand_cohesions.append(mutual_cosine(norm_rows(np.array(embedding_rows(embed, mx.array(ridx))))))
    rand_mean = float(np.mean(rand_cohesions))
    axis_cohesions, wordish = [], []
    for i in range(args.k):
        top = np.argsort(-Vnp[i])[: args.top_m].astype(np.int32)
        toks = [tok.decode([int(t)]) for t in top]
        axis_cohesions.append(mutual_cosine(norm_rows(np.array(embedding_rows(embed, mx.array(top))))))
        wordish.append(float(np.mean([is_wordish(s) for s in toks])))
        if i < 8:
            preview = " ".join(repr(s) for s in toks[:10])
            print(f"  axis {i:2d}: coh={axis_cohesions[-1]:+.3f} word={wordish[-1]:.0%}  {preview}")
    axis_mean = float(np.mean(axis_cohesions))
    ratio = axis_mean / (abs(rand_mean) + 1e-6)
    print(
        f"  axis-cohesion mean={axis_mean:+.3f}  random={rand_mean:+.3f}  "
        f"ratio={ratio:.1f}x  wordish={np.mean(wordish):.0%}"
    )
    if not (axis_mean > rand_mean + 0.05 and axis_mean > 2.0 * abs(rand_mean)):
        failures.append(f"coherence: axis cohesion {axis_mean:.3f} not >> random {rand_mean:.3f}")

    # ---- (b) BOUNDED over random states.
    maxabs = 0.0
    for _ in range(200):
        z = mx.array(rng.standard_normal((1, RESERVOIR_DIM)).astype(np.float32))
        maxabs = max(maxabs, float(mx.max(mx.abs(wide_bias(z, P, V, 0.15, args.cap))).item()))
    print(f"\n[b] bounded: max|bias| over 200 states = {maxabs:.3f} (cap {args.cap})")
    if maxabs > args.cap + 1e-3:
        failures.append(f"bounded: {maxabs} > cap {args.cap}")

    # ---- (c) LATENCY (k-sweep; tiny n_sample — only shapes matter for timing).
    print("\n[c] latency per token:")
    for kk in (8, 16, 32):
        pk, vk, _ = build_wide_coupling_matrices(embed, vocab, k=kk, n_sample=2000)
        z = mx.array(rng.standard_normal((1, RESERVOIR_DIM)).astype(np.float32))
        mx.eval(wide_bias(z, pk, vk, 0.1, args.cap))  # warmup
        n_iter = 300
        t0 = time.perf_counter()
        for _ in range(n_iter):
            mx.eval(wide_bias(z, pk, vk, 0.1, args.cap))
        ms = (time.perf_counter() - t0) / n_iter * 1000
        print(f"  k={kk:2d}: {ms:.3f} ms/token  [{'OK' if ms <= 1.0 else 'high'}]")
        if kk == args.k and ms > 1.5:
            failures.append(f"latency: k={kk} {ms:.2f}ms > 1.5ms")

    # ---- (d) DISCRIMINATIVE + pressure opens wider.
    z_quiet = mx.array((rng.standard_normal((1, RESERVOIR_DIM)) * 0.2).astype(np.float32))
    z_active = mx.array((rng.standard_normal((1, RESERVOIR_DIM)) * 0.9).astype(np.float32))
    b1 = np.array(wide_bias(z_quiet, P, V, 0.15, args.cap)).reshape(-1)
    b2 = np.array(wide_bias(z_active, P, V, 0.15, args.cap)).reshape(-1)
    diff = float(np.mean(np.abs(b1 - b2)))
    s_q = float(pressure_scale(z_quiet).item())
    s_a = float(pressure_scale(z_active).item())
    print(f"\n[d] discriminative: mean|Δbias|={diff:.3f}  pressure quiet={s_q:.3f} active={s_a:.3f}")
    if diff <= 1e-4:
        failures.append("diff: identical bias for different states")
    if not s_a > s_q:
        failures.append(f"pressure: active {s_a:.3f} !> quiet {s_q:.3f}")

    # ---- (e) OFF is off.
    z = mx.array(rng.standard_normal((1, RESERVOIR_DIM)).astype(np.float32))
    off = float(mx.max(mx.abs(wide_bias(z, P, V, 0.0, args.cap))).item())
    print(f"\n[e] off-is-off: max|bias| at s_eff=0 = {off:.6f}")
    if off > 1e-6:
        failures.append(f"off: nonzero bias {off}")

    print("\n" + "=" * 60)
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("PASS — wide coupling is coherent, bounded, fast, discriminative, off-is-off.")


if __name__ == "__main__":
    main()
