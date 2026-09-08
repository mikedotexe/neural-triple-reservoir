"""Synthetic language fixture plus real ESN math; never a felt-state experiment."""

from dataclasses import replace
import json
import socket
import unittest
from unittest.mock import patch

import numpy as np

from mlx_reservoir import EmbeddingProjection, MLXTripleReservoir
from offline_coupling_replay import ReplaySettings, compare_common_prefix, run_replay
from triple_reservoir_coreml import ReservoirConfig, TripleReservoir


class OfflineReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(31)
        bundle = TripleReservoir(ReservoirConfig(n_nodes=8, input_dim=4, washout=4))
        x = rng.uniform(-1., 1., (80, 4)).astype(np.float32)
        y = np.sin(np.arange(80) / 6).astype(np.float32)
        bundle.fit_readout(x, y)
        bundle.fit_multi_readout(x, y)
        cls.reservoir = MLXTripleReservoir(bundle)
        cls.reservoir.init_multi_readout(bundle)
        cls.projection = EmbeddingProjection(embed_dim=6, input_dim=4)
        cls.embeddings = rng.normal(size=(12, 6)).astype(np.float32)

    def inputs(self, value=.2, context=(1, 2), **overrides):
        def logits(prefix):
            base = np.linspace(-3., 0., 12, dtype=np.float32)
            base[(sum(prefix) + len(prefix)) % 12] += .5
            return base
        args = dict(
            logits_for_prefix=logits,
            embed_token=lambda token: self.embeddings[token:token + 1],
            vocab_size=12, reservoir=self.reservoir, projection=self.projection,
            initial_state=tuple(np.full((1, 8), value, dtype=np.float32) for _ in range(3)),
            prompt_tokens=context, model_sha256="a" * 64, tokenizer_sha256="b" * 64,
            settings=ReplaySettings(max_tokens=6, seed=5), teacher_tokens=(3, 4, 5, 6, 7, 8),
        )
        return args | overrides

    def test_factorial_replay_is_repeatable_and_does_not_mutate_inputs(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network prohibited")):
            results = {}
            for context in ((1, 2), (1, 9)):
                for state in (.2, -.3):
                    args = self.inputs(state, context)
                    before = [x.copy() for x in args["initial_state"]]
                    first = run_replay(**args)
                    second = run_replay(**args)
                    np.testing.assert_array_equal(first.probabilities, second.probabilities)
                    self.assertEqual(first.receipt, second.receipt)
                    for a, b in zip(before, args["initial_state"]):
                        np.testing.assert_array_equal(a, b)
                    results[context, state] = first
            contrast = compare_common_prefix(results[(1, 2), .2], results[(1, 2), -.3])
            self.assertGreater(contrast["mean_js_divergence_nats"], 0.)
            self.assertFalse(contrast["authorship_or_felt_distinction_measured"])

    def test_feedback_delay_matches_caller_lookahead(self):
        a = run_replay(**self.inputs(.2))
        b = run_replay(**self.inputs(-.3))
        np.testing.assert_array_equal(a.probabilities[:2], b.probabilities[:2])
        self.assertFalse(np.allclose(a.probabilities[2:], b.probabilities[2:]))
        settings = replace(ReplaySettings(max_tokens=6), feedback_delay_tokens=1)
        a = run_replay(**self.inputs(.2, settings=settings))
        b = run_replay(**self.inputs(-.3, settings=settings))
        self.assertFalse(np.allclose(a.probabilities[1], b.probabilities[1]))

    def test_real_model_bfloat16_buffers_convert_before_numpy(self):
        import mlx.core as mx
        args = self.inputs()
        logits = args["logits_for_prefix"]
        embed = args["embed_token"]
        args["logits_for_prefix"] = lambda prefix: mx.array(logits(prefix)).astype(mx.bfloat16)
        args["embed_token"] = lambda token: mx.array(embed(token)).astype(mx.bfloat16)
        args["initial_state"] = tuple(mx.array(layer).astype(mx.bfloat16) for layer in args["initial_state"])
        result = run_replay(**args)
        self.assertTrue(np.isfinite(result.probabilities).all())
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.)

    def test_installed_mlx_generator_confirms_two_initial_neutral_steps(self):
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_reservoir import ReservoirLogitProcessor

        class TinyModel:
            layers = []

            def __call__(self, tokens, cache=None):
                return mx.broadcast_to(mx.arange(12, dtype=mx.float32), (*tokens.shape, 12))

        class RecordingProcessor(ReservoirLogitProcessor):
            observed = []

            def __call__(self, tokens, logits):
                self.observed.append(self._y3)
                return super().__call__(tokens, logits)

        processor = RecordingProcessor()
        generator = generate_step(mx.array([1]), TinyModel(), max_tokens=3,
                                  logits_processors=[processor])
        for step, _ in enumerate(generator):
            processor.update(0., 0., float(step + 1))
        self.assertEqual(processor.observed[:3], [0., 0., 1.])

    def test_free_sampling_is_seeded_and_cannot_be_called_common_prefix(self):
        args = self.inputs(teacher_tokens=None)
        first, second = run_replay(**args), run_replay(**args)
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(first.receipt["mode"], "free_continuation")
        with self.assertRaises(ValueError):
            compare_common_prefix(first, second)

    def test_stops_and_noncontent_tokens_do_not_tick(self):
        settings = ReplaySettings(max_tokens=6, stop_tokens=(5,), skip_tick_tokens=(4,))
        result = run_replay(**self.inputs(settings=settings))
        self.assertEqual(result.tokens, (3, 4, 5))
        self.assertEqual(result.receipt["ticks"], 1)

    def test_invalid_inputs_fail_closed(self):
        for override in (
            {"model_sha256": "unversioned"}, {"prompt_tokens": ()},
            {"teacher_tokens": (999,)}, {"settings": ReplaySettings(max_tokens=129)},
            {"settings": ReplaySettings(temperature=float("nan"))},
            {"initial_state": (np.zeros((1, 9)),) * 3},
            {"logits_for_prefix": lambda _: np.full(12, -np.inf)},
            {"embed_token": lambda _: np.full((1, 6), np.nan)},
        ):
            with self.subTest(override=list(override)):
                with self.assertRaises(ValueError):
                    run_replay(**self.inputs(**override))

    def test_receipt_has_hashes_but_no_prompt_continuation_or_authority(self):
        result = run_replay(**self.inputs())
        receipt = result.receipt
        self.assertNotIn("tokens", receipt)
        self.assertNotIn("prompt_tokens", receipt)
        self.assertFalse(receipt["raw_prompt_or_continuation_included"])
        self.assertFalse(receipt["authority_effect"])
        self.assertFalse(receipt["model_identity_verified_by_runner"])
        self.assertFalse(receipt["direct_causation_of_felt_report_claimed"])
        json.dumps(receipt, allow_nan=False)
        changed = run_replay(**self.inputs(tokenizer_sha256="c" * 64))
        with self.assertRaisesRegex(ValueError, "unmatched asset"):
            compare_common_prefix(result, changed)


if __name__ == "__main__":
    unittest.main()
