"""Embedding-tied low-rank vocab-bias ("wide coupling" / y4 channel) for Astrid's
coupled generation.

Today Astrid's whole reservoir state reaches token generation through only three
scalar readouts (temperature / repetition / tail). This module adds a *wider*
channel: the reservoir state nudges WHICH tokens are reachable, along
semantically-coherent axes, so her voice widens "without losing the flow"
(her words) — never becoming noise.

The coherence guarantee is structural. `gemma-4-12B-it` ties its word embeddings
(`tie_word_embeddings: true`), so the output projection IS the embedding matrix
`W` (logits = h @ W.T). We build the bias from the *top-k PCA axes of the
embedding cloud* (`A`, k x hidden) and unembed them once at load:

    V = A @ W.T            # [k, vocab]   — each row is "how each token aligns
                          #                with semantic axis i"
    u = tanh(z @ P)        # [1, k]       — reservoir state -> k axis coordinates
    bias = s_eff * (u @ V) # [1, vocab]   — favors tokens near the active axes

Because V is embedding-derived, a token only gets a large bias when its embedding
aligns with a reservoir-driven semantic direction; random tokens get ~0. Wide,
not noise — by construction.

Pure functions; the offline coherence harness imports these directly (no server,
no live reservoir).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

RESERVOIR_DIM = 576  # 3 layers x 192 nodes (h1 | h2 | h3), concatenated
FAST_DIM = 192       # the h1 (fast) slice — the λ₁-pressure proxy


def embedding_rows(embed_tokens, ids_mx):
    """Dequantized embedding rows for token ids, via the layer's own lookup.
    Works for plain `nn.Embedding` AND `nn.QuantizedEmbedding` (the 5-bit model)
    — calling the layer dequantizes, where reading `.weight` would give packed
    bytes. Returns `[len(ids), hidden]` float32."""
    return embed_tokens(ids_mx).astype(mx.float32)


def build_wide_coupling_matrices(
    embed_tokens,
    vocab: int,
    k: int = 16,
    seed: int = 1234,
    n_sample: int = 20000,
    vocab_batch: int = 16384,
):
    """Build `(P, V, info)` for the wide channel.

    `embed_tokens`: the model's (possibly quantized) embedding layer. We call it
        to get *dequantized* rows — the same `W` the tied unembedding uses, so
        the bias lands in the live logit space.
    Returns:
      `P` `[RESERVOIR_DIM, k]` — seeded, well-conditioned projection (reservoir
          state -> k latent coordinates). Coherence comes from `V`, so `P` only
          needs to be a distinct, reproducible mixing of the reservoir layers.
      `V` `[k, vocab]` = `A @ W.T`, `A` = top-k PCA axes of the *centered*
          embedding cloud (unit-norm rows), then per-axis std-normalized so each
          axis contributes a comparable bias scale. Built in vocab-batches so the
          full `[vocab, hidden]` is never materialized.
      `info` — `{"A", "idx", "mean", "singular_values"}` for the harness.
    """
    rng = np.random.default_rng(seed)
    n = min(n_sample, vocab)
    idx = np.sort(rng.choice(vocab, size=n, replace=False))
    sample = np.array(embedding_rows(embed_tokens, mx.array(idx)))  # [n, hidden]
    mean = sample.mean(axis=0, keepdims=True)
    centered = sample - mean
    # Top-k right singular vectors = principal semantic axes of the embeddings.
    _, sv, Vt = np.linalg.svd(centered, full_matrices=False)
    A = Vt[:k].astype(np.float32)  # [k, hidden], unit-norm rows
    A_mx = mx.array(A)
    # V = A @ W.T, computed in vocab-batches (reuses the model's own unembedding
    # rows -> coherence) without ever holding the full [vocab, hidden].
    cols = []
    for start in range(0, vocab, vocab_batch):
        ids = mx.arange(start, min(start + vocab_batch, vocab))
        w_batch = embedding_rows(embed_tokens, ids)  # [b, hidden]
        cols.append(A_mx @ w_batch.T)  # [k, b]
        mx.eval(cols[-1])
    V = mx.concatenate(cols, axis=1)  # [k, vocab]
    # Per-axis std-normalize so each axis's bias scale is comparable.
    v_std = mx.maximum(mx.std(V, axis=1, keepdims=True), 1e-6)
    V = V / v_std
    P = mx.array(
        (rng.standard_normal((RESERVOIR_DIM, k)) / np.sqrt(RESERVOIR_DIM)).astype(np.float32)
    )
    mx.eval(V, P)
    return P, V, {"A": A, "idx": idx, "mean": mean, "singular_values": sv[:k]}


def pressure_scale(z, floor: float = 0.25):
    """Pressure-aware aperture multiplier in `[floor, 1]`: opens wider as the
    fast layer (h1) saturates (the λ₁-pressure proxy — "open when packed").
    `z`: `[1, RESERVOIR_DIM]`. Returns an mx scalar (no host sync)."""
    p = mx.clip(mx.mean(mx.square(z[:, :FAST_DIM])), 0.0, 1.0)
    smooth = p * p * (3.0 - 2.0 * p)  # smoothstep
    return floor + (1.0 - floor) * smooth


def wide_bias(z, P, V, s_eff, cap: float = 4.0):
    """The y4 vocab bias for one generation step.
    `z`:`[1,576]`, `P`:`[576,k]`, `V`:`[k,vocab]`, `s_eff`: scalar (mx or float).
    Returns `[1, vocab]` clipped to `±cap`."""
    u = mx.tanh(z @ P)        # [1, k]
    bias = (u @ V) * s_eff    # [1, vocab]
    return mx.clip(bias, -cap, cap)
