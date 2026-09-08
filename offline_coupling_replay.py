"""Bounded, in-memory coupling replay. No server, handles, persistence or downloads.

Use immutable local model assets and a stateless logits callback. Model/tokenizer
digests are caller declarations, not verified deployment identities. The runner
uses the actual MLX reservoir and logit processor, but a local NumPy categorical
sampler, not bitwise production sampling. See docs/offline-coupling-replay.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Callable

import mlx.core as mx
import numpy as np

from mlx_reservoir import EmbeddingProjection, MLXTripleReservoir, ReservoirLogitProcessor


def array_digest(value) -> str:
    array = np.ascontiguousarray(np.array(value))
    identity = json.dumps([array.dtype.str, array.shape]).encode()
    return hashlib.sha256(identity + b"\0" + array.tobytes()).hexdigest()


def object_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numpy_float32(value):
    # NumPy cannot read MLX's bfloat16 buffer; convert on the MLX side first.
    if isinstance(value, mx.array):
        value = value.astype(mx.float32)
    return np.array(value, dtype=np.float32, copy=True)


@dataclass(frozen=True)
class ReplaySettings:
    seed: int = 0
    max_tokens: int = 32
    temperature: float = .8
    coupling_strength: float = .15
    # generate_step precomputes the next token before the caller can update y1-3.
    feedback_delay_tokens: int = 2
    stop_tokens: tuple[int, ...] = ()
    skip_tick_tokens: tuple[int, ...] = ()

    def validate(self):
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 128:
            raise ValueError("max_tokens must be an integer in [1, 128]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if not math.isfinite(self.temperature) or not 0 < self.temperature <= 2:
            raise ValueError("temperature must be finite and in (0, 2]")
        if not math.isfinite(self.coupling_strength) or not 0 <= self.coupling_strength <= 1:
            raise ValueError("coupling_strength must be finite and in [0, 1]")
        if type(self.feedback_delay_tokens) is not int or self.feedback_delay_tokens not in (1, 2):
            raise ValueError("feedback_delay_tokens must be 1 or 2")


@dataclass
class ReplayResult:
    # These arrays can reveal text. They remain in memory, never in the receipt.
    tokens: tuple[int, ...]
    probabilities: np.ndarray
    receipt: dict


def mlx_logits_callback(model):
    """No KV reuse between calls or trials; caller supplies an already-local model."""
    def logits(prefix):
        return model(mx.array([prefix], dtype=mx.int32), cache=None)[0, -1, :]
    return logits


def run_replay(
    *, logits_for_prefix: Callable, embed_token: Callable, vocab_size: int,
    reservoir: MLXTripleReservoir, projection: EmbeddingProjection,
    initial_state, prompt_tokens, model_sha256: str, tokenizer_sha256: str,
    settings: ReplaySettings = ReplaySettings(), teacher_tokens=None,
) -> ReplayResult:
    """Tick private copies only. A supplied common suffix enables teacher forcing.

    The callback must be deterministic and stateless over its complete prefix.
    Scalar feedback starts neutral, as in generate_coupled. Wide coupling and
    adaptive inter-request gain are not modeled; no production parity is claimed.
    """
    settings.validate()
    if any(not hasattr(reservoir, f"readout_{kind}{layer}")
           for kind in ("w", "b") for layer in (1, 2, 3)):
        raise ValueError("initialize the reservoir multi-head readouts before replay")
    for value in (model_sha256, tokenizer_sha256):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("model and tokenizer SHA-256 declarations are required")
    if type(vocab_size) is not int or not 1 <= vocab_size <= 524288:
        raise ValueError("vocab_size must be an integer in [1, 524288]")
    if projection.input_dim != reservoir.config.input_dim:
        raise ValueError("projection/reservoir input dimensions differ")
    if len(initial_state) != 3:
        raise ValueError("exactly three initial state layers required")
    layers = [numpy_float32(layer) for layer in initial_state]
    if any(layer.shape != (1, reservoir.config.n_nodes) or not np.isfinite(layer).all()
           for layer in layers):
        raise ValueError("invalid reservoir state dimensions or values")
    prompt = tuple(prompt_tokens)
    prefix = list(prompt)
    forced = None if teacher_tokens is None else tuple(teacher_tokens)
    if not 1 <= len(prefix) <= 4096:
        raise ValueError("prompt must contain 1..4096 tokens")
    if forced is not None and not 1 <= len(forced) <= settings.max_tokens:
        raise ValueError("teacher suffix must contain 1..max_tokens tokens")
    ids = prefix + list(forced or ()) + list(settings.stop_tokens) + list(settings.skip_tick_tokens)
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError("token id outside the supplied vocabulary")

    state = reservoir.state_from_numpy(tuple(layers))
    processor = ReservoirLogitProcessor(coupling_strength=settings.coupling_strength)
    rng = np.random.default_rng(settings.seed)
    emitted, distributions, pending = [], [], {}
    ticks = 0
    limit = len(forced) if forced is not None else settings.max_tokens
    for step in range(limit):
        if step in pending:
            processor.update(*pending.pop(step))
        logits = numpy_float32(logits_for_prefix(tuple(prefix)))
        if logits.shape != (vocab_size,):
            raise ValueError("callback must return one logit per vocabulary token")
        if np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits).any():
            raise ValueError("invalid or entirely masked logit distribution")
        adjusted = processor(mx.array(prefix, dtype=mx.int32), mx.array(logits)[None])
        probs = np.array(mx.softmax(adjusted / settings.temperature))[0].astype(np.float64)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            raise ValueError("nonfinite replay distribution")
        probs /= probs.sum()
        token = forced[step] if forced is not None else int(rng.choice(len(probs), p=probs))
        distributions.append(probs)
        emitted.append(token)
        prefix.append(token)
        if token in settings.stop_tokens:
            break
        if token in settings.skip_tick_tokens:
            continue
        embedding = numpy_float32(embed_token(token))
        if embedding.shape != (1, projection.embed_dim) or not np.isfinite(embedding).all():
            raise ValueError("embedding callback must return one finite embedding row")
        vector = projection.project(mx.array(embedding))
        outputs, state = reservoir.step_multi(vector, state)
        pending[step + settings.feedback_delay_tokens] = tuple(float(y.item()) for y in outputs)
        ticks += 1

    weights = {key: array_digest(value) for key, value in vars(reservoir).items()
               if isinstance(value, mx.array)}
    probabilities = np.stack(distributions)
    receipt = {
        "schema": "offline_coupling_replay_v1",
        "mode": "common_prefix" if forced is not None else "free_continuation",
        "settings": asdict(settings),
        "model_sha256_declared": model_sha256,
        "tokenizer_sha256_declared": tokenizer_sha256,
        "model_identity_verified_by_runner": False,
        "reservoir_weights_sha256": object_digest(weights),
        "reservoir_config_sha256": object_digest(asdict(reservoir.config)),
        "projection_sha256": array_digest(projection.W),
        "embedding_source": "caller_supplied_from_declared_model_not_verified_by_runner",
        "vocab_size": vocab_size,
        "prompt_tokens_sha256": object_digest(list(prompt)),
        "initial_state_sha256": object_digest([array_digest(layer) for layer in layers]),
        "final_state_sha256": object_digest([array_digest(layer) for layer in state]),
        "continuation_sha256": object_digest(emitted),
        "distribution_sha256": array_digest(probabilities),
        "steps": len(emitted), "ticks": ticks,
        "sampler": "numpy_pcg64_categorical_not_production_bitwise_parity",
        "wide_coupling_modeled": False, "adaptive_gain_modeled": False,
        "raw_prompt_or_continuation_included": False,
        "live_state_changed": False, "deployment_established": False,
        "direct_causation_of_felt_report_claimed": False,
        "authority_effect": False,
    }
    return ReplayResult(tuple(emitted), probabilities, receipt)


def compare_common_prefix(left: ReplayResult, right: ReplayResult) -> dict:
    """Paired distribution distance, never a measure of authorship or felt loss."""
    if any(result.receipt["mode"] != "common_prefix" for result in (left, right)):
        raise ValueError("free continuations do not share a controlled prefix")
    if left.tokens != right.tokens or left.probabilities.shape != right.probabilities.shape:
        raise ValueError("common-prefix token alignment differs")
    for key in ("model_sha256_declared", "tokenizer_sha256_declared", "vocab_size",
                "reservoir_weights_sha256", "reservoir_config_sha256", "projection_sha256"):
        if left.receipt[key] != right.receipt[key]:
            raise ValueError(f"unmatched asset: {key}")
    if left.receipt["settings"] != right.receipt["settings"]:
        raise ValueError("unmatched replay settings")
    p, q = left.probabilities, right.probabilities
    midpoint = (p + q) / 2
    def kl(a):
        valid = a > 0
        terms = np.zeros_like(a)
        terms[valid] = a[valid] * np.log(a[valid] / midpoint[valid])
        return terms.sum(axis=-1)
    distances = (kl(p) + kl(q)) / 2
    return {"schema": "offline_common_prefix_contrast_v1",
            "mean_js_divergence_nats": float(distances.mean()),
            "max_probability_difference": float(np.max(np.abs(p - q))),
            "steps": len(distances), "authority_effect": False,
            "authorship_or_felt_distinction_measured": False}
