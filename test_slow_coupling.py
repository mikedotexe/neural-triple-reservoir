"""Slow-channel probability invariants; no model load or service connection."""

import unittest

import mlx.core as mx
import numpy as np

from mlx_reservoir import ReservoirLogitProcessor


class SlowCouplingTests(unittest.TestCase):
    def apply(self, values, y3, strength=.15, y1=0., y2=0., tokens=()):
        p = ReservoirLogitProcessor(coupling_strength=strength)
        p.update(y1, y2, y3)
        return p(mx.array(tokens, dtype=mx.int32), mx.array(values, dtype=mx.float32))

    def test_direction_and_offset_invariance(self):
        for values in ([-4., -2., -1., 0.], [1., 2., 4., 9.],
                       [-12., -8., -2., -1.], [-3., -3., 0., 7.]):
            x = mx.array([values])
            mask = x < mx.median(x)
            baseline = float(mx.sum(mx.softmax(x) * mask).item())
            for strength in (0., .1, .15, .5, 1.):
                for y3 in (-8., -2., 0., 2., 8.):
                    with self.subTest(values=values, strength=strength, y3=y3):
                        out = self.apply(x, y3, strength)
                        probs = mx.softmax(out)
                        mass = float(mx.sum(probs * mask).item())
                        if y3 > 0:
                            self.assertLessEqual(mass, baseline + 1e-6)
                        elif y3 < 0:
                            self.assertGreaterEqual(mass, baseline - 1e-6)
                        if y3 == 0 or strength == 0:
                            np.testing.assert_array_equal(np.array(out), np.array(x))
                        for offset in (-100., -10., 10., 100.):
                            shifted = mx.softmax(self.apply(x + offset, y3, strength))
                            np.testing.assert_allclose(np.array(shifted), np.array(probs), atol=3e-6)

    def test_top_half_ties_and_shape_preserved(self):
        for shape in ((4,), (1, 4)):
            x = mx.array([-4., -2., -1., 0.]).reshape(shape)
            out = self.apply(x, 2.)
            self.assertEqual(out.shape, x.shape)
            np.testing.assert_array_equal(np.array(out)[..., -2:], np.array(x)[..., -2:])
        x = mx.ones((1, 4)) * 3.
        np.testing.assert_array_equal(np.array(self.apply(x, 2.)), np.array(x))

    def test_masked_tokens_remain_impossible(self):
        for values in ([-float("inf"), -2., -1., 0.],
                       [-float("inf"), -float("inf"), -float("inf"), 0.]):
            out = self.apply([values], 2.)
            probs = np.array(mx.softmax(out))[0]
            self.assertTrue(np.isfinite(probs).all())
            self.assertEqual(probs[0], 0.)

    def test_fast_and_medium_interactions_keep_offset_invariance(self):
        x = mx.array([[-4., -2., -1., 0.]])
        for y1, y2 in ((2., 0.), (0., 2.), (-2., -2.), (2., 2.)):
            a = self.apply(x, 2., y1=y1, y2=y2, tokens=(0, 0, 2))
            b = self.apply(x + 10., 2., y1=y1, y2=y2, tokens=(0, 0, 2))
            np.testing.assert_allclose(np.array(mx.softmax(a)), np.array(mx.softmax(b)), atol=1e-6)

    def test_wide_channel_is_additive_after_slow_and_sync_is_retained(self):
        from wide_coupling import pressure_scale, wide_bias
        p = mx.ones((576, 2)) * .01
        v = mx.array([[1., -1., 2., 0.], [0., 1., -1., 2.]])
        z = mx.ones((1, 576)) * .1
        syncs = []
        processor = ReservoirLogitProcessor(
            coupling_strength=.15, wide_strength=.2, wide_P=p, wide_V=v,
            sync_observer=lambda kind, elapsed: syncs.append((kind, elapsed)),
        )
        processor.update(0., 0., 2.)
        processor.update_state(z)
        x = mx.array([[-4., -2., -1., 0.]])
        actual = processor(mx.array([], dtype=mx.int32), x)
        expected = self.apply(x, 2.) + wide_bias(z, p, v, .2 * pressure_scale(z), 4.)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)
        self.assertEqual([kind for kind, _ in syncs], ["median_item"])


if __name__ == "__main__":
    unittest.main()
