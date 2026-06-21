#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from astrid_feeder import CodecConditioner, apply_remote_memory_policy
from minime_feeder import apply_memory_policy, build_memory_vector


class AstridConditioningTests(unittest.TestCase):
    def test_ema_rms_conditioning_bounds_gain_amplified_features(self):
        conditioner = CodecConditioner(mode="ema_rms", decay=0.92, floor=0.75)
        raw = [4.5, -3.8, 2.2, -1.6] * 8

        conditioned, stats = conditioner.transform(raw)

        self.assertEqual(len(conditioned), 32)
        self.assertTrue(all(abs(v) <= 1.0 for v in conditioned))
        self.assertGreater(stats["max_abs"], 1.0)
        self.assertLessEqual(stats["conditioned_max_abs"], 1.0)
        self.assertGreater(stats["scale"], 0.0)

    def test_legacy_conditioning_preserves_raw_magnitude(self):
        conditioner = CodecConditioner(mode="legacy", decay=0.92, floor=0.75)
        raw = [2.0, -1.5] * 16

        conditioned, stats = conditioner.transform(raw)

        self.assertEqual(conditioned, raw)
        self.assertAlmostEqual(stats["scale"], 1.0)

    def test_remote_memory_policy_blends_selected_remote_memory(self):
        import json
        import tempfile
        from pathlib import Path

        base = [0.2] * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "state.json").write_text(json.dumps({
                "last_remote_memory_id": "memory_stable_1",
                "last_remote_memory_role": "stable",
                "remote_memory_bank": [
                    {
                        "id": "memory_stable_1",
                        "role": "stable",
                        "spectral_glimpse_12d": np.linspace(0.1, 1.4, 12, dtype=np.float32).tolist(),
                        "fill_pct": 72.0,
                        "lambda1_rel": 0.8,
                        "geom_rel": 1.4,
                    }
                ],
            }))
            (workspace / "contact_state.json").write_text(json.dumps({
                "attention": 0.9,
                "openness": 0.7,
                "urgency": 0.6,
            }))

            shaped, meta = apply_remote_memory_policy(base, workspace, "remote_role_blend", 1.0)

        self.assertEqual(len(shaped), 32)
        self.assertEqual(meta["memory_role"], "stable")
        self.assertGreater(meta["blend"], 0.0)
        self.assertGreater(meta["contact_gate"], 0.5)
        self.assertGreater(float(np.mean(np.abs(np.asarray(shaped) - np.asarray(base)))), 0.0)

    def test_remote_memory_policy_off_is_contract_stable(self):
        import tempfile
        from pathlib import Path

        base = [0.2] * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            shaped, meta = apply_remote_memory_policy(base, workspace, "off", 1.0)

        self.assertEqual(shaped, base)
        self.assertIn("memory_role", meta)
        self.assertIsNone(meta["memory_role"])
        self.assertEqual(meta["blend"], 0.0)

    def test_remote_memory_policy_without_state_keeps_selected_role_shape(self):
        import json
        import tempfile
        from pathlib import Path

        base = [0.2] * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "state.json").write_text(json.dumps({
                "last_remote_memory_role": "expanding",
                "remote_memory_bank": [],
            }))
            shaped, meta = apply_remote_memory_policy(base, workspace, "remote_role_blend", 1.0)

        self.assertEqual(shaped, base)
        self.assertEqual(meta["memory_role"], "expanding")
        self.assertEqual(meta["blend"], 0.0)


class MinimeMemoryPolicyTests(unittest.TestCase):
    def _spectral_state(self, role: str) -> dict:
        glimpse = np.linspace(0.05, 1.2, 12, dtype=np.float32).tolist()
        return {
            "selected_memory_role": role,
            "selected_memory_id": f"memory_{role}_123",
            "spectral_glimpse_12d": glimpse,
        }

    def test_build_memory_vector_is_32d(self):
        vec = build_memory_vector(self._spectral_state("stable"), "stable")
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 32)

    def test_role_blend_shapes_vector_more_for_stable_than_latest(self):
        base = [0.4] * 32

        stable_vec, stable_meta = apply_memory_policy(base, self._spectral_state("stable"), "role_blend", 1.0)
        latest_vec, latest_meta = apply_memory_policy(base, self._spectral_state("latest"), "role_blend", 1.0)

        stable_delta = float(np.mean(np.abs(np.asarray(stable_vec) - np.asarray(base))))
        latest_delta = float(np.mean(np.abs(np.asarray(latest_vec) - np.asarray(base))))

        self.assertGreater(stable_meta["blend"], latest_meta["blend"])
        self.assertGreater(stable_delta, latest_delta)

    def test_memory_policy_off_leaves_vector_unchanged(self):
        base = [0.25] * 32
        out, meta = apply_memory_policy(base, self._spectral_state("transition"), "off", 1.0)

        self.assertEqual(out, base)
        self.assertEqual(meta["blend"], 0.0)


class MinimeApertureJitterTests(unittest.TestCase):
    """Co-regulation LEND_APERTURE: the feeder's aperture-jitter mode must
    SPREAD Astrid's codec ring (inject cross-frame variance), unlike the
    constant-target pull which collapses it toward λ₁ (narrows her)."""

    @staticmethod
    def _run(payload, frames):
        from astrid_feeder import MinimeInfluenceState

        st = MinimeInfluenceState(payload)
        out = []
        for f in frames:
            if not st.is_active():
                break
            out.append(st.apply(list(f)))
            st.advance()
        return np.array(out)

    def test_constant_recipe_narrows_jitter_spreads(self):
        np.random.seed(0)
        dims = list(range(32))
        varying = [list(np.random.uniform(-0.5, 0.5, 32)) for _ in range(14)]
        base_var = float(np.mean(np.var(np.array(varying), axis=0)))
        const = self._run(
            {"amplitude": 0.3, "duration_ticks": 14, "decay_ticks": 10,
             "target_dims": dims, "target_values": [0.3] * 32, "blend_mode": "ease_in_out"},
            varying,
        )
        jit = self._run(
            {"amplitude": 0.3, "duration_ticks": 14, "decay_ticks": 10,
             "target_dims": dims, "target_values": [0.0] * 32,
             "blend_mode": "aperture_jitter", "jitter": 0.12},
            varying,
        )
        const_var = float(np.mean(np.var(const, axis=0)))
        jit_var = float(np.mean(np.var(jit, axis=0)))
        # Constant pull collapses cross-frame variance (NARROWS); jitter must not.
        self.assertLess(const_var, base_var)
        self.assertGreater(jit_var, const_var)

    def test_jitter_respects_bounds(self):
        np.random.seed(1)
        out = self._run(
            {"amplitude": 0.3, "duration_ticks": 5, "decay_ticks": 3,
             "target_dims": list(range(32)), "target_values": [0.0] * 32,
             "blend_mode": "aperture_jitter", "jitter": 0.12},
            [[0.0] * 32 for _ in range(8)],
        )
        max_disp = float(np.max(np.abs(out)))
        # Bounded by weight (≤0.3) × jitter (0.12) = 0.036.
        self.assertLessEqual(max_disp, 0.3 * 0.12 + 1e-9)


class MinimeGiftWindowTests(unittest.TestCase):
    """The LEND_APERTURE un-muffle: a gift's window is wall-clock-bounded (so it
    finalizes promptly on the sparse codec_impact channel instead of dragging for
    days); newer gifts supersede in-flight ones; a newer gift is never mis-consumed
    under an old intent; and a stale gift finalizes (firing the closed-loop)."""

    @staticmethod
    def _payload(intent_id="g1", issued_t_ms=None, ticks=14, decay_ticks=10):
        import time

        if issued_t_ms is None:
            issued_t_ms = time.time() * 1000.0
        return {
            "intent_id": intent_id, "issued_t_ms": issued_t_ms,
            "amplitude": 0.3, "duration_ticks": ticks, "decay_ticks": decay_ticks,
            "target_dims": list(range(32)), "target_values": [0.0] * 32,
            "blend_mode": "aperture_jitter", "jitter": 0.12,
        }

    def test_walltime_expiry_overrides_remaining_ticks(self):
        import time
        import astrid_feeder as af

        fresh = af.MinimeInfluenceState(self._payload(issued_t_ms=time.time() * 1000.0))
        self.assertTrue(fresh.is_active())
        old = af.MinimeInfluenceState(
            self._payload(issued_t_ms=time.time() * 1000.0 - (af.MINIME_GIFT_MAX_AGE_MS + 60_000))
        )
        self.assertGreater(old.ramp_remaining, 0)  # ticks remain
        self.assertTrue(old.walltime_expired())
        self.assertFalse(old.is_active())  # but wall-clock says done
        no_anchor = af.MinimeInfluenceState(self._payload(issued_t_ms=0.0))
        self.assertFalse(no_anchor.walltime_expired())  # no anchor → tick window governs
        self.assertTrue(no_anchor.is_active())

    def test_zero_tick_expiry_is_shorter_than_absolute_window(self):
        import time
        import astrid_feeder as af

        old_enough_for_zero_tick_close = af.MinimeInfluenceState(
            self._payload(
                issued_t_ms=time.time() * 1000.0
                - (af.MINIME_GIFT_NO_TICK_MAX_AGE_MS + 1_000)
            )
        )
        self.assertTrue(old_enough_for_zero_tick_close.no_tick_expired())
        self.assertFalse(old_enough_for_zero_tick_close.walltime_expired())
        self.assertFalse(old_enough_for_zero_tick_close.is_active())

        old_enough_for_zero_tick_close.advance()
        self.assertFalse(old_enough_for_zero_tick_close.no_tick_expired())

    def _patched(self, tmp):
        """Point module influence paths at a temp dir; returns paths + restore."""
        import json
        from pathlib import Path
        import astrid_feeder as af

        inf = Path(tmp) / "astrid_influence_v3.json"
        consumed = Path(tmp) / "astrid_influence_v3.consumed.json"
        terminal = Path(tmp) / "diagnostics" / "astrid_influence_terminal_events.jsonl"
        quarantine = Path(tmp) / "diagnostics" / "astrid_influence_quarantine"
        orig = (
            af.MINIME_INFLUENCE_PATH,
            af.MINIME_INFLUENCE_CONSUMED_PATH,
            af.MINIME_INFLUENCE_TERMINAL_EVENTS_PATH,
            af.MINIME_INFLUENCE_QUARANTINE_DIR,
        )
        af.MINIME_INFLUENCE_PATH, af.MINIME_INFLUENCE_CONSUMED_PATH = inf, consumed
        af.MINIME_INFLUENCE_TERMINAL_EVENTS_PATH = terminal
        af.MINIME_INFLUENCE_QUARANTINE_DIR = quarantine

        def restore():
            (
                af.MINIME_INFLUENCE_PATH,
                af.MINIME_INFLUENCE_CONSUMED_PATH,
                af.MINIME_INFLUENCE_TERMINAL_EVENTS_PATH,
                af.MINIME_INFLUENCE_QUARANTINE_DIR,
            ) = orig

        return inf, consumed, terminal, json, restore

    @staticmethod
    def _terminal_events(path):
        import json

        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_newer_gift_supersedes_and_is_not_misconsumed(self):
        import astrid_feeder as af
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            inf, consumed, terminal, json, restore = self._patched(tmp)
            try:
                inf.write_text(json.dumps(self._payload(intent_id="g1")))
                st = af.load_minime_influence(None)
                self.assertIsNotNone(st)
                self.assertEqual(st.intent_id, "g1")
                # minime issues a newer gift g2 while g1 is still in-flight.
                inf.write_text(json.dumps(self._payload(intent_id="g2")))
                st2 = af.load_minime_influence(st)
                self.assertIsNotNone(st2)
                self.assertEqual(st2.intent_id, "g2")  # superseded → we load g2, not stuck on g1
                self.assertTrue(inf.exists())  # g2 still live
                self.assertFalse(consumed.exists())  # g2 was NOT mis-consumed under g1's intent
                events = self._terminal_events(terminal)
                self.assertEqual(events[-1]["status"], "superseded")
                self.assertEqual(events[-1]["intent_id"], "g1")
                self.assertEqual(events[-1]["superseded_by_intent_id"], "g2")
            finally:
                restore()

    def test_stale_gift_finalizes_to_fire_closed_loop(self):
        import time
        import astrid_feeder as af
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            inf, consumed, terminal, json, restore = self._patched(tmp)
            try:
                old_ms = time.time() * 1000.0 - (af.MINIME_GIFT_MAX_AGE_MS + 60_000)
                inf.write_text(json.dumps(self._payload(intent_id="stale", issued_t_ms=old_ms)))
                st = af.load_minime_influence(None)
                self.assertIsNone(st)  # expired → not applied
                self.assertFalse(inf.exists())  # but finalized →
                self.assertTrue(consumed.exists())  # consumed (the closed-loop trigger)
                consumed_payload = json.loads(consumed.read_text())
                self.assertEqual(
                    consumed_payload["feeder_terminal_v1"]["status"], "expired_unapplied"
                )
                self.assertEqual(consumed_payload["feeder_terminal_v1"]["applied_ticks"], 0)
                events = self._terminal_events(terminal)
                self.assertEqual(events[-1]["status"], "expired_unapplied")
                self.assertEqual(events[-1]["intent_id"], "stale")
                self.assertEqual(events[-1]["applied_ticks"], 0)
            finally:
                restore()

    def test_zero_tick_gift_finalizes_after_short_deadline(self):
        import time
        import astrid_feeder as af
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            inf, consumed, terminal, json, restore = self._patched(tmp)
            try:
                old_ms = time.time() * 1000.0 - (
                    af.MINIME_GIFT_NO_TICK_MAX_AGE_MS + 1_000
                )
                inf.write_text(json.dumps(self._payload(intent_id="no-ticks", issued_t_ms=old_ms)))
                st = af.load_minime_influence(None)
                self.assertIsNone(st)
                self.assertFalse(inf.exists())
                self.assertTrue(consumed.exists())
                consumed_payload = json.loads(consumed.read_text())
                terminal_event = consumed_payload["feeder_terminal_v1"]
                self.assertEqual(terminal_event["status"], "expired_unapplied")
                self.assertEqual(terminal_event["applied_ticks"], 0)
                self.assertEqual(
                    terminal_event["reason"],
                    "arrived_without_codec_ticks_before_short_deadline",
                )
                events = self._terminal_events(terminal)
                self.assertEqual(events[-1]["intent_id"], "no-ticks")
                self.assertEqual(events[-1]["status"], "expired_unapplied")
            finally:
                restore()

    def test_finished_gift_appends_consumed_terminal_event(self):
        import astrid_feeder as af
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            inf, consumed, terminal, json, restore = self._patched(tmp)
            try:
                inf.write_text(json.dumps(self._payload(intent_id="done", ticks=1, decay_ticks=0)))
                st = af.load_minime_influence(None)
                self.assertIsNotNone(st)
                st.advance()
                st2 = af.load_minime_influence(st)
                self.assertIsNone(st2)
                self.assertFalse(inf.exists())
                self.assertTrue(consumed.exists())
                consumed_payload = json.loads(consumed.read_text())
                self.assertEqual(consumed_payload["feeder_terminal_v1"]["status"], "consumed")
                self.assertEqual(consumed_payload["feeder_terminal_v1"]["applied_ticks"], 1)
                events = self._terminal_events(terminal)
                self.assertEqual(events[-1]["status"], "consumed")
                self.assertEqual(events[-1]["intent_id"], "done")
                self.assertEqual(events[-1]["applied_ticks"], 1)
            finally:
                restore()

    def test_malformed_gift_quarantines_once_without_consumed_trigger(self):
        import astrid_feeder as af
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            inf, consumed, terminal, json, restore = self._patched(tmp)
            try:
                inf.write_text("{not json")
                st = af.load_minime_influence(None)
                self.assertIsNone(st)
                self.assertFalse(inf.exists())
                self.assertFalse(consumed.exists())
                quarantined = sorted(af.MINIME_INFLUENCE_QUARANTINE_DIR.glob("*.json"))
                self.assertEqual(len(quarantined), 1)
                self.assertEqual(quarantined[0].read_text(), "{not json")

                # A later poll sees no live malformed file and appends no duplicate.
                st2 = af.load_minime_influence(None)
                self.assertIsNone(st2)
                events = self._terminal_events(terminal)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["status"], "parse_failed")
                self.assertIn("quarantined", events[0]["reason"])
            finally:
                restore()


if __name__ == "__main__":
    unittest.main()
