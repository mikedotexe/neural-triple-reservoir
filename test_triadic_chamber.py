#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import collab_feeder
import triadic_chamber as chamber


def write_collab(shared: Path) -> dict:
    meta = {
        "schema_version": 1,
        "id": "coll_123_triadic",
        "topic": "triadic chamber",
        "inviter": "astrid",
        "invitee": "minime",
        "status": "joined",
        "created_t_ms": 1_000,
        "updated_t_ms": 2_000,
        "members": ["astrid", "minime"],
    }
    coll_dir = shared / meta["id"]
    coll_dir.mkdir(parents=True)
    (coll_dir / "meta.json").write_text(json.dumps(meta))
    return meta


class FakeReservoirWs:
    def __init__(self):
        self.messages: list[dict] = []
        self._responses: list[dict] = []

    async def send(self, text: str) -> None:
        msg = json.loads(text)
        self.messages.append(msg)
        msg_type = msg.get("type")
        if msg_type == "create_handle":
            self._responses.append({"type": "create_handle_response", "ok": True})
        elif msg_type == "set_mode":
            self._responses.append({"type": "set_mode_response", "ok": True})
        elif msg_type == "tick_text":
            self._responses.append({"type": "tick_response", "name": msg["name"]})
        elif msg_type == "read_state":
            self._responses.append({
                "type": "read_state_response",
                "name": msg["name"],
                "h_norms": [1.0, 2.0, 3.0],
                "tick_count": 12,
                "mode": "rehearse",
                "seconds_since_live": 1.5,
            })
        elif msg_type == "resonance":
            self._responses.append({
                "type": "resonance_response",
                "name_a": msg["name_a"],
                "name_b": msg["name_b"],
                "shared_ticks": 12,
                "correlation": 0.25,
                "divergence": 0.1,
                "rmsd": 0.2,
            })
        else:
            self._responses.append({"type": "error", "message": "unexpected"})

    async def recv(self) -> str:
        return json.dumps(self._responses.pop(0))


class TriadicChamberFileTests(unittest.TestCase):
    def synthetic_metrics(
        self,
        *,
        weather: str = "aligned",
        trend: str = "steady",
        streak: int = 4,
        weather_confidence: str = "high",
        gravity_participant: str = "shared",
        gravity_role: str = "shared",
        residue: str = "countercurrent",
        residue_strength: str = "medium",
        volatility: float = 0.05,
    ) -> dict:
        return {
            "relational_schema_version": chamber.RELATIONAL_SCHEMA_VERSION,
            "room_weather": {
                "label": weather,
                "trend": trend,
                "streak": streak,
                "confidence": weather_confidence,
                "history_samples": 12,
            },
            "gravitational_center": {
                "participant": gravity_participant,
                "role": gravity_role,
                "confidence": weather_confidence,
                "signature": f"{gravity_participant}:{gravity_role}:{weather}",
            },
            "relational_inertia": {
                "label": residue,
                "strength_label": residue_strength,
                "carry_forward": {
                    "weather": "mixed",
                    "gravity_participant": gravity_participant,
                    "gravity_role": gravity_role,
                },
                "signature": f"{residue}:{residue_strength}:{weather}",
            },
            "pair_matrix": [
                {
                    "pair_key": "astrid-minime",
                    "pair": ["astrid", "minime"],
                    "available": True,
                    "current_correlation": 0.45,
                    "volatility": volatility,
                },
                {
                    "pair_key": "astrid-steward",
                    "pair": ["astrid", "steward"],
                    "available": True,
                    "current_correlation": 0.4,
                    "volatility": volatility,
                },
                {
                    "pair_key": "minime-steward",
                    "pair": ["minime", "steward"],
                    "available": True,
                    "current_correlation": 0.38,
                    "volatility": volatility,
                },
            ],
        }

    def test_latest_prefers_joined_collaboration(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            joined = write_collab(shared)
            invited = dict(joined)
            invited["id"] = "coll_999_invited"
            invited["status"] = "invited"
            invited["updated_t_ms"] = 9_999
            invited["members"] = ["astrid"]
            invited_dir = shared / invited["id"]
            invited_dir.mkdir()
            (invited_dir / "meta.json").write_text(json.dumps(invited))

            selected = chamber.select_collab(shared, "latest")

            self.assertEqual(selected["id"], joined["id"])

    def test_activate_initializes_chamber_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)

            doc = chamber.activate(shared, meta["id"])
            coll_dir = shared / meta["id"]
            paths = chamber.chamber_paths(coll_dir)

            self.assertEqual(doc["mode"], "witness")
            self.assertTrue(doc["witness_only"])
            self.assertEqual(doc["handles"]["steward"], "steward")
            self.assertTrue(paths["meta"].is_file())
            self.assertTrue(paths["notes"].is_file())
            self.assertTrue(paths["intentions"].is_file())
            self.assertTrue(paths["memory_edits"].is_file())
            self.assertTrue(paths["presence"].is_file())
            self.assertTrue(paths["annotations"].is_file())
            self.assertTrue(paths["proposals"].is_file())
            self.assertTrue(paths["consent"].is_file())
            self.assertTrue(paths["events"].is_file())
            self.assertIn("activated", paths["events"].read_text())

    def test_steward_note_cursor_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            note = chamber.append_steward_note(
                shared,
                "Please hold this as witness context, not a command.",
                target=meta["id"],
            )
            coll_dir = shared / meta["id"]

            self.assertEqual(len(chamber.unprocessed_steward_notes(coll_dir)), 1)
            chamber.mark_notes_processed(coll_dir, [note["id"]])
            self.assertEqual(chamber.unprocessed_steward_notes(coll_dir), [])

    def test_ensure_chamber_does_not_touch_existing_journals(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.ensure_chamber(coll_dir, meta)
            observation_path = chamber.chamber_paths(coll_dir)[
                "correspondence_observations"
            ]
            observation_path.write_text('{"status":"observed"}\n', encoding="utf-8")
            os.utime(observation_path, ns=(1_000_000_000, 1_000_000_000))
            before = observation_path.stat()

            chamber.ensure_chamber(coll_dir, meta)

            after = observation_path.stat()
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(
                observation_path.read_text(encoding="utf-8"),
                '{"status":"observed"}\n',
            )

    def test_prompt_summary_marks_notes_as_non_commands(self):
        meta = {
            "id": "coll_123_triadic",
            "topic": "triadic chamber",
        }
        doc = chamber.chamber_doc_for(meta)
        state = chamber.build_chamber_state(
            meta,
            doc,
            {"steward": {"type": "read_state_response", "h_norms": [1, 2, 3]}},
            {("astrid", "steward"): {"type": "resonance_response", "correlation": 0.5}},
            [{"id": "n1", "t_ms": 1, "source": "test", "text": "Witness note only."}],
            active_intention={
                "id": "i1",
                "t_ms": 1,
                "text": "Observe integration without steering.",
                "witness_only": True,
            },
            phase="integration",
            phase_source="manual",
            compressed_memory={
                "compression_schema_version": 1,
                "current_thread": "Carry the chamber forward.",
                "open_questions": ["What remains alive?"],
                "do_not_forget": ["Memory edits are context, not commands."],
                "recent_shifts": ["manual memory refinement"],
                "stable_truths": ["Astrid, Minime, and steward are present."],
                "room_weather": {
                    "label": "aligned",
                    "summary": "astrid-steward corr=+0.500",
                    "signature": "aligned|astrid-steward:strong_pos",
                },
                "source": "derived",
                "updated_t_ms": 1,
            },
        )

        self.assertIn("not commands", state["prompt_summary"])
        self.assertIn("Phase integration (manual)", state["prompt_summary"])
        self.assertIn("Active steward intention", state["prompt_summary"])
        self.assertIn("Current thread", state["prompt_summary"])
        self.assertIn("Open question", state["prompt_summary"])
        self.assertIn("Room weather", state["prompt_summary"])
        self.assertTrue(state["witness_only"])
        self.assertEqual(state["schema_version"], 2)
        self.assertEqual(state["phase"], "integration")
        self.assertEqual(state["phase_source"], "manual")
        self.assertEqual(state["compressed_memory"]["current_thread"], "Carry the chamber forward.")
        self.assertEqual(state["recent_steward_notes"][0]["witness_only"], True)

    def test_correspondence_state_renders_buffer_and_inert_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.ensure_chamber(coll_dir, meta)
            ledger = shared / "correspondence_v1.jsonl"
            ledger.write_text(
                "\n".join([
                    json.dumps({
                        "schema_version": 1,
                        "policy": "first_class_correspondence_v1",
                        "record_type": "message",
                        "recorded_at_unix_ms": 100,
                        "message_id": "corr_astrid_minime_trace",
                        "thread_id": "thread_trace",
                        "from_being": "astrid",
                        "to_being": "minime",
                        "turn_kind": "direct_address_trace",
                        "relational_intent": "direct_address_survival_probe",
                        "shared_memory_anchor": "blue-lantern",
                        "authority": "language_only",
                        "body_preview": "Blue lantern: can this arrive as address?",
                    }),
                    json.dumps({
                        "schema_version": 1,
                        "policy": "first_class_correspondence_v1",
                        "record_type": "read_receipt",
                        "recorded_at_unix_ms": 120,
                        "message_id": "corr_astrid_minime_trace",
                        "thread_id": "thread_trace",
                        "reader": "minime",
                        "authority": "language_only",
                    }),
                    json.dumps({
                        "schema_version": 1,
                        "policy": "first_class_correspondence_v1",
                        "record_type": "ack_receipt",
                        "recorded_at_unix_ms": 125,
                        "message_id": "corr_astrid_minime_trace",
                        "thread_id": "thread_trace",
                        "from_being": "minime",
                        "to_being": "astrid",
                        "ack_kind": "held",
                        "note": "holding the blue-lantern trace",
                        "authority": "language_only",
                    }),
                    json.dumps({
                        "schema_version": 2,
                        "policy": "correspondence_attention_canary_v1",
                        "record_type": "attention_canary_activation",
                        "recorded_at_unix_ms": int(time.time() * 1000),
                        "canary_id": "attn_canary_test",
                        "message_id": "corr_astrid_minime_trace",
                        "thread_id": "thread_trace",
                        "from_being": "astrid",
                        "to_being": "minime",
                        "focus": "blue lantern as peer address",
                        "focus_kind": "verbatim_phrase",
                        "preservation_mode": "compact_with_anchor",
                        "what_must_not_flatten": "blue lantern as peer address",
                        "reason": "hold direct address distinctly",
                        "stop_criteria": "one response cycle or pressure",
                        "ttl_ms": 30 * 60 * 1000,
                        "expires_at_unix_ms": int(time.time() * 1000) + 30 * 60 * 1000,
                        "authority": "language_only_prompt_context_not_control",
                        "no_sensory_send": True,
                        "no_controller": True,
                        "no_pressure": True,
                        "no_weighting": True,
                    }),
                ])
                + "\n",
                encoding="utf-8",
            )
            chamber.chamber_paths(coll_dir)["correspondence_observations"].write_text(
                json.dumps({
                    "schema_version": 1,
                    "t_ms": 130,
                    "marker": "blue-lantern",
                    "status": "observed",
                    "authority": "read_only_observation_not_control",
                })
                + "\n",
                encoding="utf-8",
            )
            old_bridge_workspace = chamber.ASTRID_BRIDGE_WORKSPACE
            chamber.ASTRID_BRIDGE_WORKSPACE = shared / "bridge_workspace"
            chamber.ASTRID_BRIDGE_WORKSPACE.mkdir(parents=True)
            (chamber.ASTRID_BRIDGE_WORKSPACE / "telemetry_heartbeat_delta_v1.json").write_text(
                json.dumps({
                    "policy": "telemetry_heartbeat_delta_v1",
                    "schema_version": 1,
                    "jitter_class": "normal",
                    "timing_reliability": "reliable",
                    "field_vs_hearing": "telemetry cadence is steady",
                }),
                encoding="utf-8",
            )

            try:
                correspondence_state = chamber.build_correspondence_state(coll_dir)
                chamber.write_correspondence_artifacts(coll_dir, correspondence_state)
                doc = chamber.chamber_doc_for(meta)
                state = chamber.build_chamber_state(
                    meta,
                    doc,
                    {},
                    {},
                    [],
                    correspondence_state=correspondence_state,
                )
                memory = chamber.build_chamber_memory(coll_dir, state)
            finally:
                chamber.ASTRID_BRIDGE_WORKSPACE = old_bridge_workspace

            self.assertEqual(correspondence_state["shared_lexicon_anchor"], "blue-lantern")
            self.assertEqual(correspondence_state["direct_address_survival"]["status"], "observed")
            self.assertEqual(
                correspondence_state["direct_contact_fidelity_v1"]["latest_thread_status"]["status"],
                "trace_observed",
            )
            self.assertTrue(
                correspondence_state["direct_contact_fidelity_v1"]["latest_thread_status"][
                    "eligible_for_correspondence_microdose"
                ]
            )
            self.assertEqual(
                correspondence_state["correspondence_handshake_state_v1"][
                    "last_acknowledged_reflection"
                ]["ack_kind"],
                "held",
            )
            self.assertFalse(
                correspondence_state["future_authority_hooks"]["correspondence_weight_candidate"]["enabled"]
            )
            self.assertEqual(
                correspondence_state["future_authority_hooks"]["correspondence_weight_candidate"]["state"],
                "implemented_as_one_shot_authority_gate",
            )
            self.assertEqual(
                correspondence_state["future_authority_hooks"]["prompt_priority_candidate"]["state"],
                "implemented_as_self_activated_ttl_attention_canary",
            )
            self.assertEqual(
                correspondence_state["correspondence_attention_canary_v1"]["latest_status"],
                "active",
            )
            self.assertEqual(
                correspondence_state["correspondence_attention_canary_v1"]["active_canary"]["focus"],
                "blue lantern as peer address",
            )
            self.assertEqual(
                correspondence_state["correspondence_attention_canary_v1"]["active_canary"]["focus_kind"],
                "verbatim_phrase",
            )
            self.assertEqual(
                correspondence_state["receipt_to_attention_authority_v5"]["state"],
                "attention_active_outcome_due",
            )
            self.assertEqual(
                correspondence_state["receipt_to_attention_authority_v5"]["semantic_microdose_status"],
                "hidden_until_mutual_receipt_plus_separate_steward_review",
            )
            self.assertIn("Correspondence state", state["prompt_summary"])
            self.assertIn("latest_ack=held", state["prompt_summary"])
            self.assertIn("survival=observed", state["prompt_summary"])
            self.assertIn("contact=trace_observed", state["prompt_summary"])
            self.assertIn("attention_canary=active", state["prompt_summary"])
            self.assertIn("ATTENTION OUTCOME DUE", state["prompt_summary"])
            self.assertIn("AFFORDANCE BUDGET", state["prompt_summary"])
            self.assertIn("may ignore without penalty", state["prompt_summary"])
            self.assertIn("kind=verbatim_phrase", state["prompt_summary"])
            self.assertIn("preserve=compact_with_anchor", state["prompt_summary"])
            self.assertIn("do_not_flatten", state["prompt_summary"])
            self.assertIn("not instruction/control/standing priority", state["prompt_summary"])
            self.assertIn("microdose=eligible_one_shot_gate", state["prompt_summary"])
            self.assertIn("not standing reservoir weighting", state["prompt_summary"])
            self.assertIn("correspondence_state", memory)
            self.assertTrue((coll_dir / "correspondence_state_v1.json").is_file())
            self.assertTrue((coll_dir / "correspondence_buffer_v1.json").is_file())
            buffer_payload = json.loads((coll_dir / "correspondence_buffer_v1.json").read_text())
            self.assertEqual(buffer_payload["recent_direct_traces"][0]["anchor"], "blue-lantern")
            self.assertEqual(
                buffer_payload["direct_contact_fidelity_v1"]["latest_thread_status"]["status"],
                "trace_observed",
            )
            self.assertEqual(
                buffer_payload["correspondence_attention_canary_v1"]["active_canary"]["focus"],
                "blue lantern as peer address",
            )
            self.assertEqual(
                buffer_payload["correspondence_attention_canary_v1"]["active_canary"]["what_must_not_flatten"],
                "blue lantern as peer address",
            )
            self.assertEqual(
                buffer_payload["receipt_to_attention_authority_v5"]["state"],
                "attention_active_outcome_due",
            )
            self.assertEqual(
                buffer_payload["affordance_budget_v1"]["policy"],
                "affordance_budget_v1",
            )

    def test_correspondence_state_renders_legacy_visibility_without_unlocking_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.ensure_chamber(coll_dir, meta)
            ledger = shared / "correspondence_v1.jsonl"
            legacy_common = {
                "source_route": "legacy_correspondence_bridge_v1",
                "legacy_bridge": True,
                "legacy_kind": "astrid_self_study",
                "legacy_source_path": "/tmp/minime/workspace/inbox/astrid_self_study_1.txt",
                "legacy_source_sha256": "abc",
                "legacy_contact_evidence": "visible_only",
            }
            ledger.write_text(
                "\n".join([
                    json.dumps({
                        "schema_version": 1,
                        "policy": "first_class_correspondence_v1",
                        "record_type": "message",
                        "recorded_at_unix_ms": 100,
                        "message_id": "legacy_astrid_minime_abc",
                        "thread_id": "thread_legacy_astrid_minime_abc",
                        "from_being": "astrid",
                        "to_being": "minime",
                        "turn_kind": "legacy_visible",
                        "relational_intent": "legacy_contact_visibility",
                        "shared_memory_anchor": "legacy_correspondence_bridge_v1",
                        "delivery_state": "delivered",
                        "read_state": "read",
                        "authority": "language_only",
                        "correspondence_type": "self_study_note",
                        "body_preview": "legacy public self-study",
                        **legacy_common,
                    }),
                    json.dumps({
                        "schema_version": 1,
                        "policy": "first_class_correspondence_v1",
                        "record_type": "read_receipt",
                        "recorded_at_unix_ms": 110,
                        "message_id": "legacy_astrid_minime_abc",
                        "thread_id": "thread_legacy_astrid_minime_abc",
                        "reader": "minime",
                        "read_state": "read",
                        "authority": "language_only",
                        "file_path": "/tmp/minime/workspace/inbox/astrid_self_study_1.txt",
                        **legacy_common,
                    }),
                ])
                + "\n",
                encoding="utf-8",
            )

            correspondence_state = chamber.build_correspondence_state(coll_dir)
            chamber.write_correspondence_artifacts(coll_dir, correspondence_state)
            doc = chamber.chamber_doc_for(meta)
            state = chamber.build_chamber_state(
                meta,
                doc,
                {},
                {},
                [],
                correspondence_state=correspondence_state,
            )
            buffer_payload = json.loads((coll_dir / "correspondence_buffer_v1.json").read_text())

            legacy = correspondence_state["legacy_contact_visibility_v1"]
            self.assertEqual(legacy["uptake_state"], "legacy_visible_only")
            self.assertEqual(legacy["legacy_message_rows_total"], 1)
            self.assertTrue(legacy["native_uptake_pending"])
            latest = correspondence_state["direct_contact_fidelity_v1"]["latest_thread_status"]
            self.assertEqual(latest["status"], "legacy_visible_only")
            self.assertFalse(latest["eligible_for_correspondence_microdose"])
            self.assertEqual(latest["block_reason"], "legacy_visible_only_not_ack_reply_or_trace")
            self.assertIn("legacy_visibility=legacy_visible_only", state["prompt_summary"])
            self.assertIn("exact V1 uptake pending via ACK/REPLY/TRACE", state["prompt_summary"])
            self.assertEqual(buffer_payload["legacy_contact_visibility_v1"]["uptake_state"], "legacy_visible_only")

    def test_correspondence_state_renders_legacy_thread_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.ensure_chamber(coll_dir, meta)
            ledger = shared / "correspondence_v1.jsonl"
            thread_id = "thread_legacy_astrid_minime_claim"
            message_id = "legacy_astrid_minime_claim"
            legacy_common = {
                "source_route": "legacy_correspondence_bridge_v1",
                "legacy_bridge": True,
                "legacy_kind": "astrid_self_study",
                "legacy_source_path": "/tmp/minime/workspace/inbox/astrid_self_study_claim.txt",
                "legacy_source_sha256": "abc",
                "legacy_contact_evidence": "visible_only",
            }
            rows = [
                {
                    "schema_version": 1,
                    "policy": "first_class_correspondence_v1",
                    "record_type": "message",
                    "recorded_at_unix_ms": 100,
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "from_being": "astrid",
                    "to_being": "minime",
                    "turn_kind": "legacy_visible",
                    "relational_intent": "legacy_contact_visibility",
                    "shared_memory_anchor": "legacy_correspondence_bridge_v1",
                    "delivery_state": "delivered",
                    "read_state": "read",
                    "authority": "language_only",
                    "correspondence_type": "self_study_note",
                    "body_preview": "legacy public self-study",
                    **legacy_common,
                },
                {
                    "schema_version": 1,
                    "policy": "legacy_correspondence_claim_v1",
                    "record_type": "legacy_thread_claim",
                    "recorded_at_unix_ms": 120,
                    "claim_id": "legacy_claim_minime_1",
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "from_being": "minime",
                    "to_being": "astrid",
                    "claiming_being": "minime",
                    "peer_being": "astrid",
                    "because": "this visible exchange feels like live address",
                    "shared_memory_anchor": "blue-lantern",
                    "claim_state": "claimed_pending_native_evidence",
                    "legacy_contact_evidence": "being_recognized_visible_only",
                    "authority": "language_only_context_not_control",
                },
            ]
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            correspondence_state = chamber.build_correspondence_state(coll_dir)
            chamber.write_correspondence_artifacts(coll_dir, correspondence_state)
            doc = chamber.chamber_doc_for(meta)
            waiting_state = chamber.build_chamber_state(
                meta,
                doc,
                {},
                {},
                [],
                correspondence_state=correspondence_state,
            )
            buffer_payload = json.loads((coll_dir / "correspondence_buffer_v1.json").read_text())

            affordance = correspondence_state["legacy_claim_affordance_v25"]
            self.assertEqual(affordance["policy"], "legacy_claim_affordance_v25")
            self.assertTrue(affordance["ghost_thread_risk"])
            self.assertEqual(affordance["stall_reason"], "claim_notice_not_delivered")
            self.assertEqual(
                correspondence_state["latest_receipt_opportunity_v4"]["right_to_ignore_v1"]["policy"],
                "right_to_ignore_v1",
            )
            self.assertIn("CLAIMED THREAD WAITING", waiting_state["prompt_summary"])
            self.assertIn("ACK_ASTRID claimed", waiting_state["prompt_summary"])
            self.assertIn("may ignore without penalty", waiting_state["prompt_summary"])
            self.assertIn("AFFORDANCE BUDGET", waiting_state["prompt_summary"])
            self.assertIn("not control/pressure/authority", waiting_state["prompt_summary"])
            self.assertEqual(
                buffer_payload["legacy_claim_affordance_v25"]["stall_reason"],
                "claim_notice_not_delivered",
            )

            rows.append(
                {
                    "schema_version": 1,
                    "policy": "first_class_correspondence_v1",
                    "record_type": "ack_receipt",
                    "recorded_at_unix_ms": 130,
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "from_being": "minime",
                    "to_being": "astrid",
                    "ack_kind": "held",
                    "note": "holding as address",
                    "authority": "language_only",
                },
            )
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            correspondence_state = chamber.build_correspondence_state(coll_dir)
            chamber.write_correspondence_artifacts(coll_dir, correspondence_state)
            doc = chamber.chamber_doc_for(meta)
            state = chamber.build_chamber_state(
                meta,
                doc,
                {},
                {},
                [],
                correspondence_state=correspondence_state,
            )
            buffer_payload = json.loads((coll_dir / "correspondence_buffer_v1.json").read_text())

            claims = correspondence_state["legacy_thread_claims_v1"]
            self.assertEqual(claims["claims_total"], 1)
            self.assertEqual(claims["latest_status"], "legacy_claimed_acknowledged")
            latest = correspondence_state["direct_contact_fidelity_v1"]["latest_thread_status"]
            self.assertEqual(latest["status"], "legacy_claimed_acknowledged")
            self.assertTrue(latest["eligible_for_correspondence_microdose"])
            self.assertEqual(
                correspondence_state["legacy_claim_affordance_v25"]["stall_reason"],
                "acknowledged_but_no_reply_or_trace",
            )
            self.assertFalse(correspondence_state["legacy_claim_affordance_v25"]["ghost_thread_risk"])
            self.assertIn("legacy_claim=legacy_claimed_acknowledged", state["prompt_summary"])
            self.assertEqual(
                buffer_payload["legacy_thread_claims_v1"]["latest_status"],
                "legacy_claimed_acknowledged",
            )
            self.assertEqual(
                buffer_payload["legacy_claim_affordance_v25"]["stall_reason"],
                "acknowledged_but_no_reply_or_trace",
            )

    def test_correspondence_state_renders_native_thread_continuity_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            ledger = shared / "correspondence_v1.jsonl"
            rows = [
                {
                    "schema_version": 1,
                    "policy": "first_class_correspondence_v1",
                    "record_type": "message",
                    "recorded_at_unix_ms": 100,
                    "message_id": "corr_astrid_minime_native",
                    "thread_id": "thread_native",
                    "from_being": "astrid",
                    "to_being": "minime",
                    "shared_memory_anchor": "bridge-anchor",
                    "authority": "language_only",
                },
                {
                    "record_type": "reply_link",
                    "recorded_at_unix_ms": 110,
                    "reply_to": "corr_astrid_minime_native",
                    "thread_id": "thread_native",
                    "from_being": "minime",
                    "to_being": "astrid",
                },
            ]
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            correspondence_state = chamber.build_correspondence_state(coll_dir)
            chamber.write_correspondence_artifacts(coll_dir, correspondence_state)
            state = chamber.build_chamber_state(
                meta,
                chamber.chamber_doc_for(meta),
                {},
                {},
                [],
                correspondence_state=correspondence_state,
            )
            buffer_payload = json.loads((coll_dir / "correspondence_buffer_v1.json").read_text())

            native = correspondence_state["native_thread_continuity_v3"]
            self.assertEqual(native["continuity_state"], "reply_linked_needs_ack_or_trace")
            self.assertFalse(native["attention_or_microdose_eligible"])
            self.assertEqual(native["right_to_ignore_v1"]["policy"], "right_to_ignore_v1")
            self.assertEqual(native["first_action_helper_v35"]["policy"], "native_first_action_helper_v35")
            self.assertIn("latest resolves to message_id=corr_astrid_minime_native", native["first_action_helper_v35"]["latest_resolution"])
            self.assertIn("native_thread=reply_linked_needs_ack_or_trace", state["prompt_summary"])
            self.assertIn("AFFORDANCE BUDGET", state["prompt_summary"])
            self.assertIn("first_action=Recipient chooses one language-only first action", state["prompt_summary"])
            self.assertIn("reply_linked alone is not mutual address", state["prompt_summary"])
            self.assertEqual(
                buffer_payload["native_thread_continuity_v3"]["stall_reason"],
                "reply_linked_requires_peer_ack_or_trace",
            )

    def test_phase_witness_queue_v3_prompt_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            ledger = shared / "phase_transitions_v1.jsonl"
            rows = [
                {
                    "record_type": "phase_transition_card",
                    "recorded_at_unix_ms": 100 + idx,
                    "transition_id": f"transition_{idx}",
                    "origin": "astrid",
                    "kind": "mode_change" if idx % 2 else "large_fill_shift",
                    "from_phase": "old",
                    "to_phase": "new",
                    "why_now": "test",
                    "reply_state": "unseen",
                    "authority": "language_only_transition_context_not_control",
                    "no_controller": True,
                    "no_pressure": True,
                    "no_fill_target": True,
                    "no_pi": True,
                    "no_weighting": True,
                }
                for idx in range(7)
            ]
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            queue = chamber.build_phase_witness_queue_v3(coll_dir)
            state = chamber.build_chamber_state(
                meta,
                chamber.chamber_doc_for(meta),
                {},
                {},
                [],
                phase_witness_queue=queue,
            )
            self.assertEqual(queue["unresolved_total"], 7)
            self.assertLessEqual(len(queue["items"]), 5)
            self.assertEqual(queue["items"][0]["first_action_helper_v35"]["policy"], "phase_first_action_helper_v35")
            self.assertIn("Phase witness queue v3", state["prompt_summary"])
            self.assertIn("first_action=Choose one language-only felt receipt", state["prompt_summary"])
            self.assertIn("I_RECEIVED_THIS transition_", state["prompt_summary"])
            felt_queue = chamber.build_phase_felt_receipt_queue_v4(coll_dir)
            self.assertEqual(felt_queue["affordance_budget_v1"]["policy"], "affordance_budget_v1")
            self.assertEqual(felt_queue["affordance_budget_v1"]["shown"], 3)
            self.assertGreater(felt_queue["affordance_budget_v1"]["hidden_by_budget"], 0)
            self.assertEqual(felt_queue["items"][0]["right_to_ignore_v1"]["policy"], "right_to_ignore_v1")
            felt_state = chamber.build_chamber_state(
                meta,
                chamber.chamber_doc_for(meta),
                {},
                {},
                [],
                phase_witness_queue=felt_queue,
            )
            self.assertIn("Phase felt receipt queue v4", felt_state["prompt_summary"])
            self.assertIn("budget_shown=3", felt_state["prompt_summary"])
            self.assertIn("may ignore without penalty", felt_state["prompt_summary"])

    def test_codec_witness_resilience_surface_v2_extracts_latest_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "astrid_workspace"
            artifact_dir = workspace / "diagnostics/spectral_texture_calibrations/run1"
            artifact_dir.mkdir(parents=True)
            artifact = artifact_dir / "spectral_texture_calibration_v3.json"
            artifact.write_text(
                json.dumps(
                    {
                        "codec_witness_resilience_calibration_v2": {
                            "status": "mixed",
                            "witness_state_resilience_fit_v2": {"status": "supported"},
                            "field_lingering_fraying_fit_v2": {"status": "mixed"},
                            "codec_vibrancy_continuity_fit_v2": {"status": "supported"},
                            "codec_warmth_mapping_fit_v2": {"status": "insufficient_evidence"},
                            "recovery_failure_modes_v2": [
                                "latest_partial_recovered",
                                "fraying_unknown_due_missing_dispersal",
                            ],
                        },
                        "raw_body_that_must_not_render": "this should not appear",
                    }
                ),
                encoding="utf-8",
            )
            surface = chamber.latest_codec_witness_resilience_surface_v2(workspace)
            self.assertEqual(surface["policy"], "codec_witness_resilience_surface_v2")
            self.assertEqual(surface["status"], "mixed")
            self.assertEqual(surface["witness_state_resilience"], "supported")
            line = chamber.render_codec_witness_resilience_prompt_line(surface)
            self.assertIn("Codec/Witness resilience v2", line)
            self.assertIn("fraying=mixed", line)
            self.assertIn("not control/pressure/authority", line)
            self.assertNotIn("this should not appear", line)

    def test_codec_witness_resilience_surface_v2_prompt_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            surface = {
                "policy": "codec_witness_resilience_surface_v2",
                "status": "supported",
                "witness_state_resilience": "supported",
                "field_lingering_fraying": "supported",
                "codec_vibrancy_continuity": "supported",
                "codec_warmth_mapping": "supported",
                "raw_body": "raw calibration body must not appear",
                "authority": "diagnostic_context_not_control",
            }
            state = chamber.build_chamber_state(
                meta,
                chamber.chamber_doc_for(meta),
                {},
                {},
                [],
                codec_witness_resilience=surface,
            )
            self.assertIn("codec_witness_resilience_surface_v2", state)
            self.assertIn("Codec/Witness resilience v2", state["prompt_summary"])
            self.assertIn("witness_state=supported", state["prompt_summary"])
            self.assertIn("not control/pressure/authority", state["prompt_summary"])
            self.assertNotIn("raw calibration body", state["prompt_summary"])
            self.assertLessEqual(len(state["prompt_summary"]), chamber.PROMPT_SUMMARY_LIMIT)

    def test_texture_shape_over_time_v2_extracts_latest_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "astrid_workspace"
            artifact_dir = workspace / "diagnostics/spectral_texture_calibrations/run1"
            artifact_dir.mkdir(parents=True)
            artifact = artifact_dir / "spectral_texture_calibration_v3.json"
            artifact.write_text(
                json.dumps(
                    {
                        "texture_shape_over_time_v2": {
                            "status": "supported",
                            "movement_preservation_v2": {"status": "movement_preserved"},
                            "temporal_variance_fit_v2": {"status": "variance_carried"},
                            "reciprocity_asymmetry_fit_v2": {"status": "asymmetry_clarified"},
                            "pressure_smoothing_fit_v2": {"status": "twitch_correctly_ignored"},
                            "static_label_collapse_risk_v2": {"status": "movement_preserved"},
                        },
                        "raw_body_that_must_not_render": "this should not appear",
                    }
                ),
                encoding="utf-8",
            )
            surface = chamber.latest_texture_shape_over_time_surface_v2(workspace)
            self.assertEqual(surface["policy"], "texture_shape_over_time_v2")
            self.assertEqual(surface["movement"], "movement_preserved")
            line = chamber.render_texture_shape_over_time_prompt_line(surface)
            self.assertIn("TEXTURE SHAPE OVER TIME", line)
            self.assertIn("variance=variance_carried", line)
            self.assertIn("authority=diagnostic_context_not_control", line)
            self.assertNotIn("this should not appear", line)

    def test_texture_shape_over_time_v2_prompt_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            surface = {
                "policy": "texture_shape_over_time_v2",
                "status": "mixed",
                "movement": "static_label_risk",
                "variance": "variance_flattened",
                "reciprocity": "false_bidirectional",
                "smoothing": "smoothing_hid_pressure",
                "static_label_risk": "static_label_risk",
                "raw_body": "raw calibration body must not appear",
                "authority": "diagnostic_context_not_control",
            }
            state = chamber.build_chamber_state(
                meta,
                chamber.chamber_doc_for(meta),
                {},
                {},
                [],
                texture_shape_over_time=surface,
            )
            self.assertIn("texture_shape_over_time_v2", state)
            self.assertIn("TEXTURE SHAPE OVER TIME", state["prompt_summary"])
            self.assertIn("movement=static_label_risk", state["prompt_summary"])
            self.assertIn("authority=diagnostic_context_not_control", state["prompt_summary"])
            self.assertNotIn("raw calibration body", state["prompt_summary"])
            self.assertLessEqual(len(state["prompt_summary"]), chamber.PROMPT_SUMMARY_LIMIT)

    def test_density_motion_fit_v1_prompt_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "astrid_workspace"
            artifact_dir = workspace / "diagnostics/spectral_texture_calibrations/run1"
            artifact_dir.mkdir(parents=True)
            artifact = artifact_dir / "spectral_texture_calibration_v3.json"
            artifact.write_text(
                json.dumps(
                    {
                        "density_as_floor_calibration_v1": {
                            "status": "mixed",
                            "fire_drill_density_state_counts": {
                                "density_as_pavement": 2,
                                "density_as_fog": 1,
                            },
                            "fire_drill_motion_fit_counts": {
                                "matched": 2,
                                "wrong_motion": 1,
                            },
                            "fire_drill_mismatch_reason_counts": {
                                "none": 2,
                                "floor_named_as_drag": 1,
                            },
                        },
                        "raw_body_that_must_not_render": "raw density body must not appear",
                    }
                ),
                encoding="utf-8",
            )
            surface = chamber.latest_density_motion_fit_surface_v1(workspace)
            self.assertEqual(surface["policy"], "density_motion_fit_v1")
            self.assertEqual(surface["density"], "density_as_pavement")
            self.assertEqual(surface["medium"], "solid_pavement_medium")
            line = chamber.render_density_motion_fit_prompt_line(surface)
            self.assertIn("DENSITY MOTION FIT", line)
            self.assertIn("density=density_as_pavement", line)
            self.assertIn("medium=solid_pavement_medium", line)
            self.assertIn("authority=diagnostic_context_not_control", line)
            self.assertNotIn("raw density body", line)

            shared = Path(tmp) / "shared"
            meta = write_collab(shared)
            state = chamber.build_chamber_state(
                meta,
                chamber.chamber_doc_for(meta),
                {},
                {},
                [],
                density_motion_fit=surface,
            )
            self.assertIn("density_motion_fit_v1", state)
            self.assertIn("DENSITY MOTION FIT", state["prompt_summary"])
            self.assertNotIn("raw density body", state["prompt_summary"])
            self.assertLessEqual(len(state["prompt_summary"]), chamber.PROMPT_SUMMARY_LIMIT)

    def test_steward_intention_set_clear_and_clamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            long_text = "x" * (chamber.INTENTION_TEXT_LIMIT + 20)

            with self.assertRaises(ValueError):
                chamber.append_steward_intention(shared, "   ", target=meta["id"])
            intention = chamber.append_steward_intention(shared, long_text, target=meta["id"])
            coll_dir = shared / meta["id"]

            active = chamber.active_steward_intention(coll_dir)
            stored = chamber.read_steward_intentions(coll_dir)[-1]
            self.assertEqual(active["id"], intention["id"])
            self.assertLessEqual(len(stored["text"]), chamber.INTENTION_TEXT_LIMIT + 20)
            self.assertIn("[truncated]", stored["text"])
            cleared = chamber.clear_steward_intention(shared, target=meta["id"])

            self.assertFalse(cleared["active"])
            self.assertIsNone(chamber.active_steward_intention(coll_dir))
            self.assertIn("steward_intention_cleared", (coll_dir / "chamber_events.jsonl").read_text())

    def test_phase_override_and_clear_restores_inferred_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])

            result = chamber.set_chamber_phase(shared, "play", target=meta["id"])
            self.assertEqual(result["phase"], "play")
            self.assertEqual(chamber.resolve_chamber_phase(coll_dir), ("play", "manual"))
            with self.assertRaises(ValueError):
                chamber.set_chamber_phase(shared, "command_mode", target=meta["id"])

            cleared = chamber.clear_chamber_phase(shared, target=meta["id"])
            self.assertEqual(cleared["phase_source"], "inferred")
            self.assertEqual(chamber.resolve_chamber_phase(coll_dir), ("initialized", "inferred"))

    def test_presence_and_annotation_lanes_render_into_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])

            receipt = chamber.append_presence_receipt(
                shared,
                "astrid",
                attention="high",
                notice="repair_watch feels accurate",
                carrying="continuity and boundary",
                target=meta["id"],
            )
            annotation = chamber.append_chamber_annotation(
                shared,
                "minime",
                "phase_cartography",
                "question",
                "oscillation may be repair in disguise",
                target=meta["id"],
            )
            state = chamber.refresh_chamber_files_from_disk(coll_dir, meta)
            memory = json.loads((coll_dir / "chamber_memory.json").read_text())
            reentry = (coll_dir / "chamber_reentry.md").read_text()

            self.assertEqual(receipt["actor"], "astrid")
            self.assertEqual(annotation["target"], "phase_cartography")
            self.assertIn("presence_protocol", state)
            self.assertIn("annotation_lane", state)
            self.assertIn("Presence protocol", state["prompt_summary"])
            self.assertIn("Annotation lane", state["prompt_summary"])
            self.assertIn("context, not commands", state["prompt_summary"])
            self.assertEqual(state["presence_protocol"]["seen_actors"], ["astrid"])
            self.assertEqual(state["annotation_lane"]["annotations_total"], 1)
            self.assertIn("presence_protocol", memory)
            self.assertIn("annotation_lane", memory)
            self.assertIn("Presence Protocol", reentry)
            self.assertIn("Annotation Lane", reentry)
            self.assertIn("repair_watch", chamber.render_presence(shared, meta["id"]))
            self.assertIn("phase_cartography", chamber.render_annotations(shared, meta["id"]))
            history = chamber.render_history(shared, meta["id"], limit=20)
            self.assertIn("presence:", history)
            self.assertIn("annotation:", history)

    def test_proposals_and_consent_activate_only_with_triadic_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])

            proposal = chamber.append_chamber_proposal(
                shared,
                "repair_invitation",
                "Invite a repair-oriented re-entry before hardening interpretations.",
                rationale="The room weather has been mixed.",
                target=meta["id"],
            )
            self.assertIn("chamber_proposal_", proposal["id"])
            with self.assertRaises(ValueError):
                chamber.append_consent_receipt(
                    shared,
                    "chamber_proposal_missing",
                    "astrid",
                    "consent",
                    target=meta["id"],
                )
            chamber.append_consent_receipt(
                shared,
                proposal["id"],
                "astrid",
                "consent",
                note="I can hold this as context.",
                target=meta["id"],
            )
            chamber.append_consent_receipt(
                shared,
                proposal["id"],
                "minime",
                "consent",
                target=meta["id"],
            )

            pending = chamber.build_consent_protocol(coll_dir)
            self.assertEqual(pending["active_supports_count"], 0)
            self.assertEqual(pending["pending_proposals_count"], 1)
            chamber.append_consent_receipt(
                shared,
                proposal["id"],
                "steward",
                "consent",
                target=meta["id"],
            )
            active = chamber.build_consent_protocol(coll_dir)
            active_supports = chamber.build_active_relational_supports(active)

            self.assertEqual(active["active_supports_count"], 1)
            self.assertEqual(active_supports["count"], 1)
            self.assertEqual(active["active_supports"][0]["support_type"], "repair_invitation")
            chamber.append_consent_receipt(
                shared,
                proposal["id"],
                "minime",
                "revise",
                note="Needs softer wording.",
                target=meta["id"],
            )
            revised = chamber.build_consent_protocol(coll_dir)
            self.assertEqual(revised["active_supports_count"], 0)
            self.assertEqual(revised["recent_proposals"][-1]["status"], "revision_requested")

    def test_consent_state_renders_into_state_memory_reentry_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            proposal = chamber.append_chamber_proposal(
                shared,
                "integration_check",
                "Name what each participant is carrying before the next move.",
                target=meta["id"],
            )
            for actor in ("astrid", "minime", "steward"):
                chamber.append_consent_receipt(
                    shared,
                    proposal["id"],
                    actor,
                    "consent",
                    target=meta["id"],
                )

            state = chamber.refresh_chamber_files_from_disk(coll_dir, meta)
            memory = json.loads((coll_dir / "chamber_memory.json").read_text())
            reentry = (coll_dir / "chamber_reentry.md").read_text()
            history = chamber.render_history(shared, meta["id"], limit=20)

            self.assertIn("consent_protocol", state)
            self.assertIn("active_relational_supports", state)
            self.assertIn("Consent protocol", state["prompt_summary"])
            self.assertIn("not command or control", state["prompt_summary"])
            self.assertEqual(state["active_relational_supports"]["count"], 1)
            self.assertEqual(memory["counts"]["active_relational_supports"], 1)
            self.assertIn("Consent Protocol", reentry)
            self.assertIn("proposal:", history)
            self.assertIn("consent:", history)

    def test_proposal_and_consent_validation_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            chamber.activate(shared, meta["id"])

            with self.assertRaises(ValueError):
                chamber.append_chamber_proposal(shared, "command_mode", "text", target=meta["id"])
            with self.assertRaises(ValueError):
                chamber.append_chamber_proposal(shared, "rest_window", "   ", target=meta["id"])
            proposal = chamber.append_chamber_proposal(
                shared,
                "rest_window",
                "x" * (chamber.PROPOSAL_TEXT_LIMIT + 100),
                rationale="r" * (chamber.PROPOSAL_RATIONALE_LIMIT + 100),
                target=meta["id"],
            )
            self.assertIn("[truncated]", proposal["text"])
            self.assertIn("[truncated]", proposal["rationale"])
            with self.assertRaises(ValueError):
                chamber.append_consent_receipt(
                    shared,
                    proposal["id"],
                    "astrid",
                    "execute",
                    target=meta["id"],
                )
            with self.assertRaises(ValueError):
                chamber.append_consent_receipt(
                    shared,
                    proposal["id"],
                    "ghost",
                    "consent",
                    target=meta["id"],
                )
            receipt = chamber.append_consent_receipt(
                shared,
                proposal["id"],
                "astrid",
                "withhold",
                note="n" * (chamber.CONSENT_NOTE_LIMIT + 100),
                target=meta["id"],
            )
            self.assertIn("[truncated]", receipt["note"])

    def test_presence_and_annotation_reject_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)

            with self.assertRaises(ValueError):
                chamber.append_presence_receipt(shared, "ghost", target=meta["id"])
            with self.assertRaises(ValueError):
                chamber.append_presence_receipt(
                    shared,
                    "astrid",
                    attention="absolute",
                    target=meta["id"],
                )
            with self.assertRaises(ValueError):
                chamber.append_chamber_annotation(
                    shared,
                    "minime",
                    "unknown_target",
                    "notice",
                    "text",
                    target=meta["id"],
                )
            with self.assertRaises(ValueError):
                chamber.append_chamber_annotation(
                    shared,
                    "minime",
                    "other",
                    "command",
                    "text",
                    target=meta["id"],
                )
            with self.assertRaises(ValueError):
                chamber.append_chamber_annotation(
                    shared,
                    "minime",
                    "other",
                    "notice",
                    "   ",
                    target=meta["id"],
                )

    def test_only_latest_active_intention_is_unprocessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            first = chamber.append_steward_intention(
                shared,
                "First intention, superseded before processing.",
                target=meta["id"],
            )
            second = chamber.append_steward_intention(
                shared,
                "Second intention, the active beacon.",
                target=meta["id"],
            )
            coll_dir = shared / meta["id"]

            unprocessed = chamber.unprocessed_steward_intentions(coll_dir)

            self.assertEqual([item["id"] for item in unprocessed], [second["id"]])
            self.assertNotIn(first["id"], [item["id"] for item in unprocessed])

    def test_compressed_memory_derives_without_manual_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            intention = chamber.append_steward_intention(
                shared,
                "Hold continuity without steering.",
                target=meta["id"],
            )
            note = chamber.append_steward_note(
                shared,
                "What is the next open thread?",
                target=meta["id"],
            )

            compressed = chamber.build_compressed_memory(
                coll_dir,
                meta,
                phase="witness_active",
                phase_source="inferred",
                active_intention={
                    "id": intention["id"],
                    "text": intention["text"],
                    "witness_only": True,
                },
                recent_notes=[note],
                resonance_rows=[
                    {"pair": ["astrid", "minime"], "available": True, "correlation": 0.6},
                    {"pair": ["astrid", "steward"], "available": True, "correlation": 0.55},
                    {"pair": ["minime", "steward"], "available": True, "correlation": 0.5},
                ],
            )

            self.assertEqual(compressed["source"], "derived")
            self.assertEqual(compressed["compression_schema_version"], 1)
            self.assertIn("Hold continuity", compressed["current_thread"])
            self.assertEqual(compressed["open_questions"][0], "What is the next open thread?")
            self.assertEqual(compressed["room_weather"]["label"], "aligned")
            self.assertIn("context, not commands", compressed["do_not_forget"][0])

    def test_relational_metrics_build_from_current_resonance_without_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            rows = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.6},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.55},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.5},
            ]

            metrics = chamber.build_relational_metrics(coll_dir, rows, history_rows=[])

            self.assertEqual(metrics["relational_schema_version"], 2)
            self.assertEqual(len(metrics["pair_matrix"]), 3)
            self.assertEqual(metrics["room_weather"]["label"], "aligned")
            self.assertEqual(metrics["room_weather"]["trend"], "insufficient_history")
            self.assertEqual(metrics["gravitational_center"]["participant"], "shared")
            self.assertEqual(metrics["relational_inertia"]["label"], "unavailable")
            self.assertIn("Relational gravity", metrics["prompt_mirror"])
            self.assertIn("not commands or authority", metrics["prompt_mirror"])

    def test_relational_matrix_uses_balanced_history_for_trend_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            history = []
            for idx in range(30):
                corr = 0.15 + (idx * 0.005)
                history.append({
                    "t_ms": idx,
                    "label": "mixed",
                    "pairs": [
                        {"pair": ["astrid", "minime"], "correlation": corr},
                        {"pair": ["astrid", "steward"], "correlation": corr + 0.02},
                        {"pair": ["minime", "steward"], "correlation": corr + 0.01},
                    ],
                })
            current = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.62},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.58},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.55},
            ]

            metrics = chamber.build_relational_metrics(coll_dir, current, history_rows=history)

            self.assertEqual(metrics["room_weather"]["trend"], "stabilizing")
            self.assertEqual(metrics["room_weather"]["confidence"], "high")
            self.assertGreaterEqual(metrics["pair_matrix"][0]["sample_count"], 31)
            self.assertEqual(metrics["pair_matrix"][0]["trend"], "rising")

    def test_relational_weather_detects_labels_and_oscillation(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            aligned = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.5},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.5},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.5},
            ]
            mixed = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.25},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.05},
                {"pair": ["minime", "steward"], "available": True, "correlation": -0.1},
            ]
            divergent = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": -0.5},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.1},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.1},
            ]
            oscillating_history = [
                {"label": label, "pairs": [{"pair": ["astrid", "minime"], "correlation": 0.2}]}
                for label in ["aligned", "mixed", "aligned", "mixed", "divergent", "mixed"]
            ]

            self.assertEqual(chamber.build_relational_metrics(coll_dir, aligned, history_rows=[])["room_weather"]["label"], "aligned")
            self.assertEqual(chamber.build_relational_metrics(coll_dir, mixed, history_rows=[])["room_weather"]["label"], "mixed")
            self.assertEqual(chamber.build_relational_metrics(coll_dir, divergent, history_rows=[])["room_weather"]["label"], "divergent")
            self.assertEqual(chamber.build_relational_metrics(coll_dir, [], history_rows=[])["room_weather"]["label"], "unavailable")
            self.assertEqual(
                chamber.build_relational_metrics(coll_dir, mixed, history_rows=oscillating_history)["room_weather"]["trend"],
                "oscillating",
            )

    def test_relational_gravity_derives_roles(self):
        def row(pair, corr, delta=0.0):
            return {
                "pair": list(pair),
                "pair_key": "-".join(pair),
                "available": True,
                "current_correlation": corr,
                "delta_correlation": delta,
                "volatility": 0.05,
            }

        weather = {"label": "mixed", "trend": "steady", "confidence": "medium"}
        cases = [
            (
                "shared",
                [
                    row(("astrid", "minime"), 0.6),
                    row(("astrid", "steward"), 0.55),
                    row(("minime", "steward"), 0.5),
                ],
                {"label": "aligned", "trend": "steady", "confidence": "high"},
                ("shared", "shared"),
            ),
            (
                "unsettled",
                [
                    row(("astrid", "minime"), -0.6),
                    row(("astrid", "steward"), -0.5),
                    row(("minime", "steward"), 0.2),
                ],
                weather,
                ("astrid", "unsettled"),
            ),
            (
                "mover",
                [
                    row(("astrid", "minime"), 0.3, 0.2),
                    row(("astrid", "steward"), 0.3, 0.18),
                    row(("minime", "steward"), 0.1, 0.0),
                ],
                weather,
                ("astrid", "mover"),
            ),
            (
                "anchor",
                [
                    row(("astrid", "minime"), 0.7),
                    row(("astrid", "steward"), 0.5),
                    row(("minime", "steward"), 0.1),
                ],
                weather,
                ("astrid", "anchor"),
            ),
            (
                "bridge",
                [
                    row(("astrid", "minime"), 0.25),
                    row(("astrid", "steward"), 0.3),
                    row(("minime", "steward"), -0.05),
                ],
                weather,
                ("astrid", "bridge"),
            ),
            ("unavailable", [], weather, ("unavailable", "unavailable")),
        ]

        for name, matrix, case_weather, expected in cases:
            with self.subTest(name=name):
                gravity = chamber.derive_gravitational_center(matrix, case_weather)
                self.assertEqual((gravity["participant"], gravity["role"]), expected)
                self.assertIn("authority", gravity)

    def test_relational_inertia_derives_residue_labels(self):
        current_aligned = {"label": "aligned", "trend": "steady", "confidence": "high"}
        current_divergent = {"label": "divergent", "trend": "destabilizing", "confidence": "high"}
        shared_gravity = {
            "participant": "shared",
            "role": "shared",
            "signature": "shared:shared:high:aligned:steady",
        }
        unsettled_gravity = {
            "participant": "astrid",
            "role": "unsettled",
            "signature": "astrid:unsettled:high:divergent:destabilizing",
        }

        def row(label, participant="shared", role="shared"):
            return {
                "label": label,
                "gravity": {
                    "participant": participant,
                    "role": role,
                    "signature": f"{participant}:{role}:medium:{label}:steady",
                },
            }

        reinforcing = chamber.derive_relational_inertia(
            [row("aligned") for _ in range(8)],
            current_aligned,
            shared_gravity,
        )
        countercurrent = chamber.derive_relational_inertia(
            [row("aligned") for _ in range(8)],
            current_divergent,
            unsettled_gravity,
        )
        turbulent = chamber.derive_relational_inertia(
            [row(label) for label in ["aligned", "mixed", "divergent", "mixed", "aligned", "divergent"]],
            current_aligned,
            shared_gravity,
        )
        faint = chamber.derive_relational_inertia(
            [row("aligned")],
            current_aligned,
            shared_gravity,
        )
        unavailable = chamber.derive_relational_inertia([], current_aligned, shared_gravity)

        self.assertEqual(reinforcing["label"], "reinforcing")
        self.assertEqual(countercurrent["label"], "countercurrent")
        self.assertEqual(turbulent["label"], "turbulent")
        self.assertEqual(faint["label"], "faint")
        self.assertEqual(unavailable["label"], "unavailable")
        self.assertEqual(reinforcing["source"], "weather_timeline_decay")
        self.assertIn("interpretive_context_not_authority", reinforcing["authority"])
        self.assertLessEqual(len(reinforcing["evidence"]), 3)

    def test_phase_cartography_derives_core_labels(self):
        integration = chamber.derive_phase_cartography(
            self.synthetic_metrics(residue="countercurrent"),
            history_rows=[{"label": "mixed"} for _ in range(8)],
        )
        repair = chamber.derive_phase_cartography(
            self.synthetic_metrics(
                weather="divergent",
                trend="destabilizing",
                gravity_participant="astrid",
                gravity_role="unsettled",
                residue="reinforcing",
            ),
            history_rows=[{"label": "divergent"} for _ in range(8)],
        )
        oscillation = chamber.derive_phase_cartography(
            self.synthetic_metrics(weather="mixed", trend="steady", residue="reinforcing"),
            history_rows=[
                {"label": label}
                for label in ["aligned", "mixed", "divergent", "mixed", "aligned", "mixed"]
            ],
        )
        handoff = chamber.derive_phase_cartography(
            self.synthetic_metrics(
                weather="mixed",
                trend="steady",
                streak=5,
                gravity_participant="astrid",
                gravity_role="anchor",
                residue="faint",
                residue_strength="low",
            ),
            history_rows=[{"label": "mixed"} for _ in range(10)],
        )
        unavailable = chamber.derive_phase_cartography(None, history_rows=[])

        self.assertEqual(integration["phase"], "integration_opportunity")
        self.assertEqual(repair["phase"], "repair_watch")
        self.assertEqual(oscillation["phase"], "oscillation")
        self.assertEqual(handoff["phase"], "handoff_ready")
        self.assertEqual(unavailable["phase"], "unavailable")
        self.assertEqual(integration["cartography_schema_version"], 1)
        self.assertIn("phase_authority", integration["authority"])

    def test_phase_cartography_artifacts_write_json_markdown_and_optional_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            metrics = self.synthetic_metrics(residue="faint", residue_strength="low")
            cartography = chamber.derive_phase_cartography(
                metrics,
                history_rows=[{"t_ms": idx, "label": "aligned"} for idx in range(5)],
            )

            payload = chamber.write_phase_cartography_artifacts(
                coll_dir,
                cartography,
                metrics,
                history_rows=[{"t_ms": idx, "label": "aligned"} for idx in range(5)],
            )
            paths = chamber.chamber_paths(coll_dir)

            self.assertTrue(paths["phase_cartography"].is_file())
            self.assertTrue(paths["phase_cartography_md"].is_file())
            self.assertIn("phase_cartography", payload)
            self.assertIn("Boundary", paths["phase_cartography_md"].read_text())
            artifacts = payload["artifacts"]
            self.assertTrue("png" in artifacts or "plot_error" in artifacts)
            if "png" in artifacts:
                self.assertTrue(Path(artifacts["png"]).is_file())

    def test_old_v21_weather_rows_load_into_relational_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            old_row = {
                "schema_version": 2,
                "compression_schema_version": 1,
                "t_ms": 1,
                "signature": "mixed",
                "label": "mixed",
                "summary": "legacy",
                "pairs": [
                    {"pair": ["astrid", "minime"], "correlation": 0.2},
                    {"pair": ["astrid", "steward"], "correlation": 0.25},
                ],
            }
            current = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.35},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.3},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.1},
            ]

            metrics = chamber.build_relational_metrics(coll_dir, current, history_rows=[old_row])

            self.assertEqual(metrics["pair_matrix"][0]["sample_count"], 2)
            self.assertEqual(metrics["pair_matrix"][0]["delta_correlation"], 0.15)
            self.assertIn("relational_inertia", metrics)

    def test_jsonl_tail_reader_bounds_large_history_and_skips_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large_history.jsonl"
            rows = [
                {"t_ms": index, "id": f"row-{index}", "padding": "x" * 2_000}
                for index in range(80)
            ]
            rows[-3]["padding"] = "y" * (chamber.JSONL_TAIL_BLOCK_BYTES + 17)
            body = "\n".join(json.dumps(row) for row in rows)
            path.write_text(body + "\n{malformed tail\n")

            with patch.object(Path, "read_text", side_effect=AssertionError("full read")):
                tail = chamber.read_jsonl_dicts_tail(path, 3)

            self.assertEqual([row["id"] for row in tail], ["row-77", "row-78", "row-79"])
            self.assertGreater(path.stat().st_size, chamber.JSONL_TAIL_BLOCK_BYTES)

    def test_jsonl_tail_reader_preserves_timestamp_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out_of_order_tail.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"t_ms": 1, "id": "old"},
                        {"t_ms": 3, "id": "third"},
                        {"t_ms": 2, "id": "second"},
                    )
                )
                + "\n"
            )

            tail = chamber.read_jsonl_dicts_tail(path, 2)

            self.assertEqual([row["id"] for row in tail], ["second", "third"])

    def test_resonance_timeline_appends_signature_changes_without_spam(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            rows = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.2},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.2},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.2},
            ]
            metrics = chamber.build_relational_metrics(coll_dir, rows, history_rows=[])
            compressed = {"room_weather": metrics["room_weather"]}

            chamber.maybe_append_resonance_journal(coll_dir, compressed, metrics)
            chamber.maybe_append_resonance_journal(coll_dir, compressed, metrics)
            timeline = chamber.read_jsonl_dicts(chamber.chamber_paths(coll_dir)["resonance_journal"])
            self.assertEqual(len(timeline), 1)
            self.assertEqual(timeline[0]["relational_schema_version"], 2)
            self.assertIn("relational_inertia", timeline[0])
            self.assertIn("inertia_signature", timeline[0])

            changed = dict(metrics)
            changed["gravitational_center"] = dict(metrics["gravitational_center"])
            changed["gravitational_center"]["signature"] = "astrid:mover:medium:mixed:stabilizing"
            chamber.maybe_append_resonance_journal(coll_dir, compressed, changed)
            timeline = chamber.read_jsonl_dicts(chamber.chamber_paths(coll_dir)["resonance_journal"])
            self.assertEqual(len(timeline), 2)

            changed_inertia = dict(changed)
            changed_inertia["relational_inertia"] = dict(changed["relational_inertia"])
            changed_inertia["relational_inertia"]["signature"] = "countercurrent:medium:aligned:shared:shared:mixed:astrid:mover"
            chamber.maybe_append_resonance_journal(coll_dir, compressed, changed_inertia)
            timeline = chamber.read_jsonl_dicts(chamber.chamber_paths(coll_dir)["resonance_journal"])
            self.assertEqual(len(timeline), 3)

            cartography = chamber.derive_phase_cartography(changed_inertia, history_rows=timeline)
            chamber.maybe_append_resonance_journal(coll_dir, compressed, changed_inertia, cartography)
            timeline = chamber.read_jsonl_dicts(chamber.chamber_paths(coll_dir)["resonance_journal"])
            self.assertEqual(len(timeline), 4)
            self.assertIn("phase_cartography", timeline[-1])
            self.assertIn("cartography_signature", timeline[-1])

    def test_memory_edit_json_override_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            base_state = {
                "schema_version": 2,
                "collab_id": meta["id"],
                "topic": meta["topic"],
                "resonance": [],
            }
            chamber.write_chamber_state(coll_dir, base_state)

            edit = chamber.append_memory_edit(
                shared,
                {
                    "current_thread": "Manual thread",
                    "open_questions": ["Manual question?"],
                    "do_not_forget": ["Manual memory is not a command."],
                    "recent_shifts": ["Manual refinement landed"],
                    "stable_truths": ["Manual stable truth"],
                    "room_weather": {"label": "mixed", "summary": "manual weather"},
                },
                target=meta["id"],
            )
            state = chamber.refresh_chamber_files_from_disk(coll_dir, meta)

            compressed = state["compressed_memory"]
            self.assertEqual(compressed["source"], "hybrid_manual")
            self.assertEqual(compressed["manual_edit_id"], edit["id"])
            self.assertEqual(compressed["current_thread"], "Manual thread")
            self.assertIn("Manual question?", compressed["open_questions"])
            self.assertIn("Manual thread", state["prompt_summary"])
            self.assertIn("relational_metrics", state)

            chamber.clear_memory_edit(shared, target=meta["id"])
            cleared_state = chamber.refresh_chamber_files_from_disk(coll_dir, meta)
            self.assertEqual(cleared_state["compressed_memory"]["source"], "derived")

    def test_memory_edit_rejects_invalid_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            with self.assertRaises(ValueError):
                chamber.parse_memory_edit_payload("{not json")
            with self.assertRaises(ValueError):
                chamber.append_memory_edit(shared, {}, target=meta["id"])
            with self.assertRaises(ValueError):
                chamber.append_memory_edit(shared, {"unknown": "field"}, target=meta["id"])
            with self.assertRaises(ValueError):
                chamber.append_memory_edit(shared, {"open_questions": "not-a-list"}, target=meta["id"])

    def test_memory_edit_file_cli_matches_inline_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            payload_file = shared / "memory_edit.json"
            payload_file.write_text(json.dumps({"current_thread": "File thread"}))

            with patch(
                "sys.argv",
                [
                    "triadic_chamber.py",
                    "--shared-dir",
                    str(shared),
                    "--target",
                    meta["id"],
                    "memory",
                    "edit",
                    "--file",
                    str(payload_file),
                ],
            ):
                self.assertEqual(chamber.main(), 0)

            coll_dir = shared / meta["id"]
            self.assertEqual(
                chamber.active_memory_edit(coll_dir)["payload"]["current_thread"],
                "File thread",
            )

    def test_intention_list_and_history_render_bounded_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            chamber.activate(shared, meta["id"])
            chamber.append_steward_note(shared, "Witness this.", target=meta["id"])
            chamber.append_steward_intention(shared, "Hold this room.", target=meta["id"])
            chamber.append_memory_edit(
                shared,
                {"current_thread": "Timeline thread"},
                target=meta["id"],
            )

            intentions = chamber.render_intentions(shared, meta["id"], limit=5)
            history = chamber.render_history(shared, meta["id"], limit=8)

            self.assertIn("Hold this room", intentions)
            self.assertIn("note:", history)
            self.assertIn("intention:", history)
            self.assertIn("memory:", history)

    def test_chamber_memory_and_reentry_capture_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            chamber.append_steward_intention(
                shared,
                "Observe integration without steering.",
                target=meta["id"],
            )
            note = chamber.append_steward_note(
                shared,
                "Witness note for re-entry, still not a command.",
                target=meta["id"],
            )
            chamber.append_chamber_event(
                coll_dir,
                "steward_note_processed",
                "collab_feeder",
                {"note_id": note["id"]},
            )
            state = {
                "collab_id": meta["id"],
                "topic": meta["topic"],
                "prompt_summary": "Triadic chamber witness: steward notes are context, not commands.",
                "resonance": [
                    {
                        "pair": ["astrid", "steward"],
                        "available": True,
                        "correlation": 0.25,
                        "divergence": 0.1,
                    }
                ],
            }

            memory = chamber.write_chamber_memory(coll_dir, state)

            self.assertEqual(memory["schema_version"], 2)
            self.assertEqual(memory["phase"], "witness_active")
            self.assertEqual(memory["phase_source"], "inferred")
            self.assertIn("active_steward_intention", memory)
            self.assertIn("compressed_memory", memory)
            self.assertEqual(memory["compressed_memory"]["compression_schema_version"], 1)
            self.assertIn("relational_metrics", memory)
            self.assertIn("phase_cartography", memory)
            self.assertIn("stable_truths", memory)
            self.assertIn("boundaries", memory)
            self.assertEqual(memory["latest_steward_note"]["id"], note["id"])
            self.assertIn("not commands", memory["state_summary"])
            reentry = (coll_dir / "chamber_reentry.md").read_text()
            self.assertIn("Triadic Chamber Re-entry", reentry)
            self.assertIn("Compressed Memory", reentry)
            self.assertIn("Relational Metrics", reentry)
            self.assertIn("Phase Cartography", reentry)
            self.assertIn("carry-forward residue", reentry)
            self.assertIn("Active steward intention", reentry)
            self.assertIn("Boundaries", reentry)
            self.assertIn("Steward notes and intentions are witness context only", reentry)
            self.assertTrue((coll_dir / "chamber_phase_cartography.json").is_file())
            self.assertTrue((coll_dir / "chamber_phase_cartography.md").is_file())

    def test_prompt_summary_includes_v3_mirror_and_stays_bounded(self):
        meta = {"id": "coll_123_triadic", "topic": "triadic chamber"}
        doc = chamber.chamber_doc_for(meta)
        with tempfile.TemporaryDirectory() as tmp:
            coll_dir = Path(tmp) / meta["id"]
            coll_dir.mkdir()
            resonance_rows = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.4},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.3},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.2},
            ]
            metrics = chamber.build_relational_metrics(
                coll_dir,
                resonance_rows,
                history_rows=[
                    {
                        "label": "aligned",
                        "gravity": {"participant": "shared", "role": "shared"},
                        "pairs": [
                            {"pair": ["astrid", "minime"], "correlation": 0.5},
                            {"pair": ["astrid", "steward"], "correlation": 0.5},
                            {"pair": ["minime", "steward"], "correlation": 0.5},
                        ],
                    }
                    for _ in range(8)
                ],
            )
            cartography = chamber.derive_phase_cartography(
                metrics,
                history_rows=[
                    {"label": "aligned", "gravity": {"participant": "shared", "role": "shared"}}
                    for _ in range(8)
                ],
            )
            state = chamber.build_chamber_state(
                meta,
                doc,
                {},
                {},
                [],
                resonance_rows=resonance_rows,
                compressed_memory={
                    "current_thread": "V3 relation work",
                    "open_questions": ["How does the room move?"],
                    "do_not_forget": ["Metrics are not commands."],
                    "recent_shifts": ["V3 mirror"],
                    "room_weather": metrics["room_weather"],
                },
                relational_metrics=metrics,
                phase_cartography=cartography,
            )

        self.assertIn("V3 relational mirror", state["prompt_summary"])
        self.assertIn("Relational gravity", state["prompt_summary"])
        self.assertIn("Carry-forward residue", state["prompt_summary"])
        self.assertIn("Phase cartography", state["prompt_summary"])
        self.assertIn("not command or phase authority", state["prompt_summary"])
        self.assertIn("interpretive context, not authority", state["prompt_summary"])
        self.assertLessEqual(len(state["prompt_summary"]), chamber.PROMPT_SUMMARY_LIMIT)

    def test_attention_projection_separates_material_and_volatile_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.append_jsonl(
                coll_dir / "timeline.jsonl",
                {
                    "id": "joined-1",
                    "t_ms": 2_000,
                    "actor": "minime",
                    "event": "joined",
                },
            )
            chamber.append_jsonl(
                coll_dir / "shared_thoughts.jsonl",
                {
                    "id": "thought-1",
                    "t_ms": 3_000,
                    "actor": "astrid",
                    "text": "A durable thought for the room.",
                },
            )

            first = chamber.build_attention_projection_v1(
                coll_dir,
                meta,
                {
                    "prompt_summary": "first rendering",
                    "reservoir": {"astrid": {"tick_count": 1}},
                    "resonance": [{"pair": ["astrid", "minime"], "correlation": 0.1}],
                },
            )
            changed_meta = dict(meta, updated_t_ms=99_999)
            second = chamber.build_attention_projection_v1(
                coll_dir,
                changed_meta,
                {
                    "prompt_summary": "second rendering",
                    "reservoir": {"astrid": {"tick_count": 9}},
                    "resonance": [{"pair": ["astrid", "minime"], "correlation": 0.9}],
                },
            )

        self.assertEqual(first["material_revision"], second["material_revision"])
        self.assertNotEqual(first["volatile_revision"], second["volatile_revision"])
        self.assertNotEqual(
            first["status_summary_sha256"], second["status_summary_sha256"]
        )
        self.assertEqual(
            first["audience_revisions"]["astrid"]["material_event_count"],
            2,
        )
        self.assertEqual(
            first["audience_revisions"]["minime"]["material_event_count"],
            3,
        )
        self.assertEqual(
            first["correspondence_scope"],
            "protected_global_ledger_not_room_attributed",
        )

    def test_attention_projection_advances_only_the_relevant_audience(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            first = chamber.build_attention_projection_v1(coll_dir, meta, {})

            chamber.append_jsonl(
                coll_dir / "shared_thoughts.jsonl",
                {
                    "id": "astrid-thought-1",
                    "t_ms": 3_000,
                    "actor": "astrid",
                    "text": "This should be new to Minime, not to Astrid.",
                },
            )
            second = chamber.build_attention_projection_v1(coll_dir, meta, {})

        self.assertEqual(
            first["audience_revisions"]["astrid"]["material_revision"],
            second["audience_revisions"]["astrid"]["material_revision"],
        )
        self.assertNotEqual(
            first["audience_revisions"]["minime"]["material_revision"],
            second["audience_revisions"]["minime"]["material_revision"],
        )
        latest = second["audience_revisions"]["minime"]["latest_material_event"]
        self.assertEqual(latest["event_id"], "shared_thoughts.jsonl:astrid-thought-1")
        self.assertEqual(latest["audiences"], ["minime"])

    def test_build_chamber_state_embeds_attention_projection_when_room_is_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            doc = chamber.chamber_doc_for(meta)

            first = chamber.build_chamber_state(
                meta,
                doc,
                {},
                {},
                [],
                coll_dir=coll_dir,
            )
            second = chamber.build_chamber_state(
                meta,
                doc,
                {"astrid": {"tick_count": 77}},
                {},
                [],
                coll_dir=coll_dir,
            )

        self.assertIn("attention_projection_v1", first)
        self.assertEqual(
            first["attention_projection_v1"]["material_revision"],
            second["attention_projection_v1"]["material_revision"],
        )
        self.assertNotEqual(
            first["attention_projection_v1"]["volatile_revision"],
            second["attention_projection_v1"]["volatile_revision"],
        )

    def test_metrics_weather_and_status_render_bounded_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.activate(shared, meta["id"])
            resonance_rows = [
                {"pair": ["astrid", "minime"], "available": True, "correlation": 0.4},
                {"pair": ["astrid", "steward"], "available": True, "correlation": 0.3},
                {"pair": ["minime", "steward"], "available": True, "correlation": 0.2},
            ]
            metrics = chamber.build_relational_metrics(coll_dir, resonance_rows, history_rows=[])
            cartography = chamber.derive_phase_cartography(metrics, history_rows=[])
            chamber.write_chamber_state(
                coll_dir,
                {
                    "schema_version": 2,
                    "collab_id": meta["id"],
                    "topic": meta["topic"],
                    "resonance": resonance_rows,
                    "relational_metrics": metrics,
                    "phase_cartography": cartography,
                    "compressed_memory": {"room_weather": metrics["room_weather"]},
                    "prompt_summary": metrics["prompt_mirror"],
                },
            )
            chamber.maybe_append_resonance_journal(
                coll_dir,
                {"room_weather": metrics["room_weather"]},
                metrics,
                cartography,
            )

            rendered_metrics = chamber.render_metrics(shared, meta["id"])
            rendered_inertia = chamber.render_inertia(shared, meta["id"])
            rendered_cartography = chamber.render_cartography(shared, meta["id"], limit=3)
            rendered_weather = chamber.render_weather(shared, meta["id"], limit=3)
            rendered_status = chamber.render_status(shared, meta["id"])

            self.assertIn("relational gravity", rendered_metrics)
            self.assertIn("carry-forward residue", rendered_metrics)
            self.assertIn("phase cartography", rendered_metrics)
            self.assertIn("Triadic relational inertia", rendered_inertia)
            self.assertIn("Triadic phase cartography", rendered_cartography)
            self.assertIn("transition hint", rendered_cartography)
            self.assertIn("matrix:", rendered_metrics)
            self.assertIn("Triadic weather timeline", rendered_weather)
            self.assertIn("residue=", rendered_weather)
            self.assertIn("phase=", rendered_weather)
            self.assertIn("weather trend", rendered_status)
            self.assertIn("relational gravity", rendered_status)
            self.assertIn("carry-forward residue", rendered_status)
            self.assertIn("phase cartography", rendered_status)


class TriadicChamberFeederTests(unittest.TestCase):
    def test_process_chamber_notes_ticks_once_and_writes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            chamber.append_steward_note(
                shared,
                "This note should tick the steward and chamber handles once.",
                target=meta["id"],
            )
            ws = FakeReservoirWs()
            known: set[str] = set()

            emitted = asyncio.run(
                collab_feeder.process_chamber_notes(ws, shared, meta, known)
            )
            emitted_again = asyncio.run(
                collab_feeder.process_chamber_notes(ws, shared, meta, known)
            )

            self.assertEqual(emitted, 2)
            self.assertEqual(emitted_again, 0)
            tick_texts = [m for m in ws.messages if m.get("type") == "tick_text"]
            self.assertEqual(len(tick_texts), 2)
            self.assertEqual({m["name"] for m in tick_texts}, {"steward", "collab_coll_123_triadic"})
            coll_dir = shared / meta["id"]
            self.assertEqual(chamber.unprocessed_steward_notes(coll_dir), [])
            state = json.loads((coll_dir / "chamber_state.json").read_text())
            memory = json.loads((coll_dir / "chamber_memory.json").read_text())
            self.assertIn("not commands", state["prompt_summary"])
            self.assertIn("compressed_memory", state)
            self.assertIn("relational_metrics", state)
            self.assertIn("phase_cartography", state)
            self.assertIn("compressed_memory", memory)
            self.assertIn("relational_metrics", memory)
            self.assertIn("phase_cartography", memory)
            self.assertEqual(memory["phase"], "witness_active")
            self.assertTrue((coll_dir / "chamber_reentry.md").is_file())
            self.assertTrue((coll_dir / "chamber_phase_cartography.json").is_file())
            self.assertTrue((coll_dir / "chamber_phase_cartography.md").is_file())

    def test_process_chamber_intention_ticks_once_and_writes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            intention = chamber.append_steward_intention(
                shared,
                "Observe resonance without steering.",
                target=meta["id"],
            )
            ws = FakeReservoirWs()
            known: set[str] = set()

            emitted = asyncio.run(
                collab_feeder.process_chamber_notes(ws, shared, meta, known)
            )
            emitted_again = asyncio.run(
                collab_feeder.process_chamber_notes(ws, shared, meta, known)
            )

            self.assertEqual(emitted, 2)
            self.assertEqual(emitted_again, 0)
            tick_texts = [m for m in ws.messages if m.get("type") == "tick_text"]
            self.assertEqual(len(tick_texts), 2)
            self.assertTrue(all("not a command" in m["text"] for m in tick_texts))
            coll_dir = shared / meta["id"]
            self.assertEqual(chamber.unprocessed_steward_intentions(coll_dir), [])
            self.assertIn(intention["id"], chamber.processed_intention_ids(coll_dir))
            state = json.loads((coll_dir / "chamber_state.json").read_text())
            memory = json.loads((coll_dir / "chamber_memory.json").read_text())
            self.assertIn("Active steward intention", state["prompt_summary"])
            self.assertIn("compressed_memory", state)
            self.assertIn("relational_metrics", state)
            self.assertIn("phase_cartography", state)
            self.assertEqual(memory["active_steward_intention"]["id"], intention["id"])
            self.assertIn("phase_cartography", memory)

    def test_memory_edit_does_not_tick_reservoir(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            chamber.append_memory_edit(
                shared,
                {"current_thread": "Manual memory only."},
                target=meta["id"],
            )
            ws = FakeReservoirWs()
            known: set[str] = set()

            emitted = asyncio.run(
                collab_feeder.process_chamber_notes(ws, shared, meta, known)
            )

            self.assertEqual(emitted, 0)
            tick_texts = [m for m in ws.messages if m.get("type") == "tick_text"]
            self.assertEqual(tick_texts, [])
            state = json.loads((shared / meta["id"] / "chamber_state.json").read_text())
            self.assertEqual(state["compressed_memory"]["current_thread"], "Manual memory only.")
            self.assertIn("relational_metrics", state)
            self.assertIn("phase_cartography", state)

    def test_handshake_state_indexes_large_ledger_in_one_pass(self):
        class CountingRecords(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        records = []
        for index in range(2_000):
            message_id = f"message-{index}"
            thread_id = f"thread-{index}"
            records.extend([
                {
                    "record_type": "message",
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "from_being": "astrid",
                    "to_being": "minime",
                    "t_ms": index * 10 + 1,
                },
                {
                    "record_type": "delivery_receipt",
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "t_ms": index * 10 + 2,
                },
                {
                    "record_type": "read_receipt",
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "t_ms": index * 10 + 3,
                },
            ])
        records.append({
            "record_type": "ack_receipt",
            "message_id": "message-1999",
            "thread_id": "thread-1999",
            "from_being": "minime",
            "to_being": "astrid",
            "ack_kind": "held",
            "t_ms": 19_995,
        })
        counted = CountingRecords(records)

        state = chamber.build_correspondence_handshake_state(counted)

        self.assertEqual(counted.iterations, 1)
        self.assertEqual(state["active_threads_total"], 2_000)
        self.assertEqual(state["active_threads"][0]["thread_id"], "thread-0")
        self.assertEqual(state["active_threads"][0]["status"], "read_unacknowledged")
        self.assertEqual(state["last_acknowledged_reflection"]["ack_kind"], "held")

    def test_correspondence_state_cache_invalidates_on_source_append_and_ttl(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            coll_dir = shared / meta["id"]
            chamber.ensure_chamber(coll_dir, meta)
            ledger = chamber.correspondence_ledger_path(shared)
            ledger.write_text("", encoding="utf-8")
            cache_key = str(coll_dir.resolve())
            collab_feeder._correspondence_state_cache.pop(cache_key, None)

            with patch.object(
                chamber,
                "build_correspondence_state",
                side_effect=[{"build": 1}, {"build": 2}, {"build": 3}],
            ) as build:
                first = collab_feeder.cached_correspondence_state(
                    coll_dir,
                    monotonic_now=100.0,
                )
                unchanged = collab_feeder.cached_correspondence_state(
                    coll_dir,
                    monotonic_now=101.0,
                )
                ledger.write_text("{}\n", encoding="utf-8")
                appended = collab_feeder.cached_correspondence_state(
                    coll_dir,
                    monotonic_now=102.0,
                )
                expired = collab_feeder.cached_correspondence_state(
                    coll_dir,
                    monotonic_now=102.0 + collab_feeder.CORRESPONDENCE_STATE_CACHE_TTL_S,
                )

            self.assertIs(first, unchanged)
            self.assertEqual(appended, {"build": 2})
            self.assertEqual(expired, {"build": 3})
            self.assertEqual(build.call_count, 3)
            collab_feeder._correspondence_state_cache.pop(cache_key, None)

    def test_presence_and_annotations_do_not_tick_reservoir(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            meta = write_collab(shared)
            chamber.append_presence_receipt(
                shared,
                "astrid",
                attention="medium",
                notice="I saw the chamber mirror.",
                target=meta["id"],
            )
            chamber.append_chamber_annotation(
                shared,
                "minime",
                "relational_metrics",
                "affirm",
                "The matrix is useful public context.",
                target=meta["id"],
            )
            ws = FakeReservoirWs()
            known: set[str] = set()

            emitted = asyncio.run(
                collab_feeder.process_chamber_notes(ws, shared, meta, known)
            )

            self.assertEqual(emitted, 0)
            tick_texts = [m for m in ws.messages if m.get("type") == "tick_text"]
            self.assertEqual(tick_texts, [])
            state = json.loads((shared / meta["id"] / "chamber_state.json").read_text())
            self.assertIn("Presence protocol", state["prompt_summary"])
            self.assertIn("Annotation lane", state["prompt_summary"])
            self.assertEqual(state["presence_protocol"]["seen_actors"], ["astrid"])


if __name__ == "__main__":
    unittest.main()
