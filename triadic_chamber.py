#!/usr/bin/env python3
"""Triadic Witness Chamber helpers and steward CLI.

V1 deliberately extends the existing Astrid/minime collaboration directory:
each joined ``coll_*`` room gains chamber metadata, steward witness notes, a
normalized event log, and a compact prompt-ready state file. The reservoir
feeder owns live ticking; this CLI only edits the durable shared files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_SHARED_DIR = Path("/Users/v/other/shared/collaborations")
CHAMBER_SCHEMA_VERSION = 2
COMPRESSION_SCHEMA_VERSION = 1
RELATIONAL_SCHEMA_VERSION = 2
INERTIA_SCHEMA_VERSION = 1
CARTOGRAPHY_SCHEMA_VERSION = 1
PRESENCE_SCHEMA_VERSION = 1
ANNOTATION_SCHEMA_VERSION = 1
CONSENT_SCHEMA_VERSION = 1
CHAMBER_MODE = "witness"
STEWARD_HANDLE = "steward"
ASTRID_HANDLE = "astrid"
MINIME_HANDLE = "minime"
CHAMBER_MEMBERS = [ASTRID_HANDLE, MINIME_HANDLE, STEWARD_HANDLE]
RELATIONAL_PAIRS = [
    (ASTRID_HANDLE, MINIME_HANDLE),
    (ASTRID_HANDLE, STEWARD_HANDLE),
    (MINIME_HANDLE, STEWARD_HANDLE),
]
NOTE_TEXT_LIMIT = 4_000
INTENTION_TEXT_LIMIT = 1_000
PRESENCE_TEXT_LIMIT = 360
ANNOTATION_TEXT_LIMIT = 800
PROPOSAL_TEXT_LIMIT = 700
PROPOSAL_RATIONALE_LIMIT = 700
CONSENT_NOTE_LIMIT = 500
PROMPT_NOTE_LIMIT = 240
PROMPT_SUMMARY_LIMIT = 2_300
CURSOR_KEEP_LIMIT = 1_000
MEMORY_TEXT_LIMIT = 500
MEMORY_LIST_LIMIT = 5
RELATIONAL_WINDOW_LIMIT = 30
RELATIONAL_INERTIA_DECAY = 0.85
RESONANCE_JOURNAL_MIN_INTERVAL_MS = 5 * 60 * 1000
PRESENCE_ATTENTION_LEVELS = {"unknown", "low", "medium", "high"}
ANNOTATION_STANCES = {"notice", "affirm", "question", "correct", "refine", "contest"}
ANNOTATION_TARGETS = {
    "prompt_summary",
    "compressed_memory",
    "relational_metrics",
    "phase_cartography",
    "room_weather",
    "relational_inertia",
    "gravitational_center",
    "steward_intention",
    "presence_protocol",
    "other",
}
CONSENT_SUPPORT_TYPES = {
    "repair_invitation",
    "reentry_ritual",
    "softening_prompt",
    "pause_escalation",
    "name_disagreement",
    "handoff_check",
    "rest_window",
    "integration_check",
}
CONSENT_STANCES = {"consent", "withhold", "revise"}
CONSENT_THRESHOLD = "astrid_minime_steward"
ALLOWED_PHASES = {
    "initialized",
    "witness_pending",
    "witness_active",
    "integration",
    "drift_detected",
    "repair",
    "play",
    "deep_work",
    "handoff",
    "rest",
}
COMPRESSED_MEMORY_TEXT_FIELDS = {"current_thread"}
COMPRESSED_MEMORY_LIST_FIELDS = {
    "open_questions",
    "do_not_forget",
    "recent_shifts",
    "stable_truths",
}
COMPRESSED_MEMORY_DICT_FIELDS = {"room_weather"}
COMPRESSED_MEMORY_EDIT_FIELDS = (
    COMPRESSED_MEMORY_TEXT_FIELDS
    | COMPRESSED_MEMORY_LIST_FIELDS
    | COMPRESSED_MEMORY_DICT_FIELDS
    | {"compression_schema_version", "source", "updated_t_ms"}
)


def now_ms() -> int:
    return int(time.time() * 1000)


def record_id(prefix: str, t_ms: int) -> str:
    return f"{prefix}_{t_ms}_{time.time_ns() % 1_000_000}"


def collab_handle_name(coll_id: str) -> str:
    return f"collab_{coll_id}"


def chamber_paths(coll_dir: Path) -> dict[str, Path]:
    return {
        "meta": coll_dir / "chamber.json",
        "notes": coll_dir / "steward_notes.jsonl",
        "intentions": coll_dir / "steward_intentions.jsonl",
        "memory_edits": coll_dir / "chamber_memory_edits.jsonl",
        "presence": coll_dir / "chamber_presence.jsonl",
        "annotations": coll_dir / "chamber_annotations.jsonl",
        "proposals": coll_dir / "chamber_proposals.jsonl",
        "consent": coll_dir / "chamber_consent.jsonl",
        "events": coll_dir / "chamber_events.jsonl",
        "resonance_journal": coll_dir / "chamber_resonance.jsonl",
        "state": coll_dir / "chamber_state.json",
        "cursor": coll_dir / "chamber_cursor.json",
        "memory": coll_dir / "chamber_memory.json",
        "reentry": coll_dir / "chamber_reentry.md",
        "phase_cartography": coll_dir / "chamber_phase_cartography.json",
        "phase_cartography_md": coll_dir / "chamber_phase_cartography.md",
        "phase_cartography_png": coll_dir / "chamber_phase_cartography.png",
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def load_collab_meta(coll_dir: Path) -> dict[str, Any]:
    meta = read_json(coll_dir / "meta.json")
    return meta if meta.get("id") else {}


def iter_collab_metas(shared_dir: Path) -> list[dict[str, Any]]:
    if not shared_dir.is_dir():
        return []
    metas: list[dict[str, Any]] = []
    for path in shared_dir.iterdir():
        if not path.is_dir():
            continue
        meta = load_collab_meta(path)
        if meta:
            metas.append(meta)
    return metas


def select_collab(shared_dir: Path, target: str = "latest", *, joined_only: bool = False) -> dict[str, Any]:
    target = (target or "latest").strip()
    all_metas = iter_collab_metas(shared_dir)
    metas = all_metas
    if joined_only:
        metas = [m for m in metas if m.get("status") == "joined"]
    if not metas:
        raise ValueError("no matching collaborations found")
    if target in {"", "latest"}:
        if not joined_only:
            joined = [m for m in all_metas if m.get("status") == "joined"]
            if joined:
                metas = joined
        metas.sort(key=lambda m: int(m.get("updated_t_ms") or m.get("created_t_ms") or 0), reverse=True)
        return metas[0]
    for meta in metas:
        coll_id = str(meta.get("id") or "")
        if coll_id == target or target in coll_id:
            return meta
    raise ValueError(f"no collaboration matching {target!r}")


def chamber_doc_for(meta: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    coll_id = str(meta.get("id") or "")
    t_ms = now_ms()
    created = int(existing.get("created_t_ms") or meta.get("created_t_ms") or t_ms)
    base = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "mode": CHAMBER_MODE,
        "collab_id": coll_id,
        "topic": str(meta.get("topic") or ""),
        "members": CHAMBER_MEMBERS,
        "witness_only": True,
        "authority": "chamber witness records are context, not commands",
        "handles": {
            "astrid": ASTRID_HANDLE,
            "minime": MINIME_HANDLE,
            "steward": STEWARD_HANDLE,
            "chamber": collab_handle_name(coll_id),
        },
        "created_t_ms": created,
    }
    existing_base = {key: existing.get(key) for key in base}
    updated = int(existing.get("updated_t_ms") or t_ms)
    if existing_base != base:
        updated = t_ms
    preserved = {
        key: value
        for key, value in existing.items()
        if key not in base and key != "updated_t_ms"
    }
    return preserved | base | {"updated_t_ms": updated}


def ensure_chamber(coll_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    paths = chamber_paths(coll_dir)
    existing = read_json(paths["meta"])
    chamber = chamber_doc_for(meta, existing)
    if existing != chamber:
        atomic_write_json(paths["meta"], chamber)
    for key in (
        "notes",
        "intentions",
        "memory_edits",
        "presence",
        "annotations",
        "proposals",
        "consent",
        "events",
    ):
        paths[key].touch(exist_ok=True)
    return chamber


def append_chamber_event(coll_dir: Path, event: str, actor: str, detail: dict[str, Any] | None = None) -> None:
    payload = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "t_ms": now_ms(),
        "event": event,
        "actor": actor,
        "witness_only": True,
        "detail": detail or {},
    }
    append_jsonl(chamber_paths(coll_dir)["events"], payload)


def clamp_note_text(text: str) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= NOTE_TEXT_LIMIT:
        return clean
    return clean[:NOTE_TEXT_LIMIT].rstrip() + " [truncated]"


def clamp_intention_text(text: str) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= INTENTION_TEXT_LIMIT:
        return clean
    return clean[:INTENTION_TEXT_LIMIT].rstrip() + " [truncated]"


def clamp_memory_text(text: str, limit: int = MEMORY_TEXT_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + " [truncated]"


def clamp_presence_text(text: str, limit: int = PRESENCE_TEXT_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + " [truncated]"


def normalize_participant(actor: str) -> str:
    normalized = str(actor or "").strip().lower()
    if normalized not in CHAMBER_MEMBERS:
        raise ValueError(
            f"actor must be one of {', '.join(CHAMBER_MEMBERS)}"
        )
    return normalized


def chamber_state_hash(coll_dir: Path) -> str | None:
    path = chamber_paths(coll_dir)["state"]
    if not path.is_file():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def append_presence_receipt(
    shared_dir: Path,
    actor: str,
    *,
    attention: str = "unknown",
    notice: str = "",
    carrying: str = "",
    disagree: str = "",
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    normalized_actor = normalize_participant(actor)
    normalized_attention = str(attention or "unknown").strip().lower()
    if normalized_attention not in PRESENCE_ATTENTION_LEVELS:
        raise ValueError(
            f"attention must be one of {', '.join(sorted(PRESENCE_ATTENTION_LEVELS))}"
        )
    t_ms = now_ms()
    receipt = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "presence_schema_version": PRESENCE_SCHEMA_VERSION,
        "id": record_id("chamber_presence", t_ms),
        "t_ms": t_ms,
        "actor": normalized_actor,
        "source": source,
        "chamber_seen": True,
        "chamber_state_hash": chamber_state_hash(coll_dir),
        "attention": normalized_attention,
        "what_i_notice": clamp_presence_text(notice),
        "what_i_am_carrying": clamp_presence_text(carrying),
        "what_i_disagree_with": clamp_presence_text(disagree),
        "witness_only": True,
        "authority": "public_receipt_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["presence"], receipt)
    append_chamber_event(
        coll_dir,
        "presence_receipt_appended",
        normalized_actor,
        {"receipt_id": receipt["id"], "actor": normalized_actor},
    )
    return receipt | {"collab_id": meta["id"]}


def append_chamber_annotation(
    shared_dir: Path,
    actor: str,
    annotation_target: str,
    stance: str,
    text: str,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    normalized_actor = normalize_participant(actor)
    normalized_target = str(annotation_target or "").strip().lower()
    if normalized_target not in ANNOTATION_TARGETS:
        raise ValueError(
            f"annotation target must be one of {', '.join(sorted(ANNOTATION_TARGETS))}"
        )
    normalized_stance = str(stance or "").strip().lower()
    if normalized_stance not in ANNOTATION_STANCES:
        raise ValueError(
            f"annotation stance must be one of {', '.join(sorted(ANNOTATION_STANCES))}"
        )
    annotation_text = clamp_presence_text(text, ANNOTATION_TEXT_LIMIT)
    if not annotation_text:
        raise ValueError("annotation text is empty")
    t_ms = now_ms()
    annotation = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "id": record_id("chamber_annotation", t_ms),
        "t_ms": t_ms,
        "actor": normalized_actor,
        "source": source,
        "target": normalized_target,
        "stance": normalized_stance,
        "text": annotation_text,
        "witness_only": True,
        "authority": "annotation_context_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["annotations"], annotation)
    append_chamber_event(
        coll_dir,
        "chamber_annotation_appended",
        normalized_actor,
        {
            "annotation_id": annotation["id"],
            "actor": normalized_actor,
            "target": normalized_target,
            "stance": normalized_stance,
        },
    )
    return annotation | {"collab_id": meta["id"]}


def clamp_proposal_text(text: str, limit: int = PROPOSAL_TEXT_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + " [truncated]"


def normalize_support_type(support_type: str) -> str:
    normalized = str(support_type or "").strip().lower()
    if normalized not in CONSENT_SUPPORT_TYPES:
        raise ValueError(
            f"support type must be one of {', '.join(sorted(CONSENT_SUPPORT_TYPES))}"
        )
    return normalized


def normalize_consent_stance(stance: str) -> str:
    normalized = str(stance or "").strip().lower()
    if normalized not in CONSENT_STANCES:
        raise ValueError(f"consent stance must be one of {', '.join(sorted(CONSENT_STANCES))}")
    return normalized


def normalize_proposal_id(proposal_id: str) -> str:
    normalized = str(proposal_id or "").strip()
    allowed = all(ch.isalnum() or ch in {"_", "-"} for ch in normalized)
    if not normalized or not normalized.startswith("chamber_proposal_") or not allowed:
        raise ValueError("proposal id must be a chamber_proposal_* id")
    return normalized


def append_chamber_proposal(
    shared_dir: Path,
    support_type: str,
    text: str,
    *,
    rationale: str = "",
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    proposal_type = normalize_support_type(support_type)
    proposal_text = clamp_proposal_text(text)
    if not proposal_text:
        raise ValueError("proposal text is empty")
    rationale_text = clamp_proposal_text(rationale, PROPOSAL_RATIONALE_LIMIT)
    t_ms = now_ms()
    proposal = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "consent_schema_version": CONSENT_SCHEMA_VERSION,
        "id": record_id("chamber_proposal", t_ms),
        "t_ms": t_ms,
        "actor": STEWARD_HANDLE,
        "source": source,
        "support_type": proposal_type,
        "text": proposal_text,
        "rationale": rationale_text,
        "witness_only": True,
        "authority": "support_proposal_not_control",
    }
    append_jsonl(chamber_paths(coll_dir)["proposals"], proposal)
    append_chamber_event(
        coll_dir,
        "chamber_proposal_created",
        STEWARD_HANDLE,
        {"proposal_id": proposal["id"], "support_type": proposal_type},
    )
    return proposal | {"collab_id": meta["id"]}


def read_chamber_proposals(coll_dir: Path) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for payload in read_jsonl_dicts(chamber_paths(coll_dir)["proposals"]):
        try:
            proposal_id = normalize_proposal_id(str(payload.get("id") or ""))
            support_type = normalize_support_type(str(payload.get("support_type") or payload.get("type") or ""))
        except ValueError:
            continue
        text = clamp_proposal_text(str(payload.get("text") or ""))
        if not text:
            continue
        proposals.append(
            payload
            | {
                "id": proposal_id,
                "support_type": support_type,
                "text": text,
                "rationale": clamp_proposal_text(
                    str(payload.get("rationale") or ""),
                    PROPOSAL_RATIONALE_LIMIT,
                ),
                "witness_only": True,
                "authority": "support_proposal_not_control",
            }
        )
    return proposals


def append_consent_receipt(
    shared_dir: Path,
    proposal_id: str,
    actor: str,
    stance: str,
    *,
    note: str = "",
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    normalized_proposal_id = normalize_proposal_id(proposal_id)
    known_ids = {str(proposal.get("id")) for proposal in read_chamber_proposals(coll_dir)}
    if normalized_proposal_id not in known_ids:
        raise ValueError(f"proposal id {normalized_proposal_id!r} was not found")
    normalized_actor = normalize_participant(actor)
    normalized_stance = normalize_consent_stance(stance)
    note_text = clamp_proposal_text(note, CONSENT_NOTE_LIMIT)
    t_ms = now_ms()
    receipt = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "consent_schema_version": CONSENT_SCHEMA_VERSION,
        "id": record_id("chamber_consent", t_ms),
        "t_ms": t_ms,
        "actor": normalized_actor,
        "source": source,
        "proposal_id": normalized_proposal_id,
        "stance": normalized_stance,
        "note": note_text,
        "witness_only": True,
        "authority": "consent_receipt_not_control",
    }
    append_jsonl(chamber_paths(coll_dir)["consent"], receipt)
    append_chamber_event(
        coll_dir,
        "chamber_consent_recorded",
        normalized_actor,
        {
            "consent_id": receipt["id"],
            "proposal_id": normalized_proposal_id,
            "stance": normalized_stance,
        },
    )
    return receipt | {"collab_id": meta["id"]}


def read_chamber_consent(coll_dir: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for payload in read_jsonl_dicts(chamber_paths(coll_dir)["consent"]):
        try:
            proposal_id = normalize_proposal_id(str(payload.get("proposal_id") or ""))
            actor = normalize_participant(str(payload.get("actor") or ""))
            stance = normalize_consent_stance(str(payload.get("stance") or ""))
        except ValueError:
            continue
        receipts.append(
            payload
            | {
                "proposal_id": proposal_id,
                "actor": actor,
                "stance": stance,
                "note": clamp_proposal_text(str(payload.get("note") or ""), CONSENT_NOTE_LIMIT),
                "witness_only": True,
                "authority": "consent_receipt_not_control",
            }
        )
    return receipts


def _compact_consent_receipt(row: dict[str, Any], text_limit: int = 180) -> dict[str, Any]:
    compact = {
        "id": row.get("id"),
        "t_ms": row.get("t_ms"),
        "actor": row.get("actor"),
        "proposal_id": row.get("proposal_id"),
        "stance": row.get("stance"),
        "witness_only": True,
        "authority": "consent_receipt_not_control",
    }
    note = truncate_for_prompt(str(row.get("note") or ""), text_limit)
    if note:
        compact["note"] = note
    return compact


def _proposal_status(latest_by_actor: dict[str, dict[str, Any]]) -> str:
    if all(latest_by_actor.get(actor, {}).get("stance") == "consent" for actor in CHAMBER_MEMBERS):
        return "active"
    if any(latest_by_actor.get(actor, {}).get("stance") == "withhold" for actor in CHAMBER_MEMBERS):
        return "withheld"
    if any(latest_by_actor.get(actor, {}).get("stance") == "revise" for actor in CHAMBER_MEMBERS):
        return "revision_requested"
    return "pending"


def _compact_proposal(
    proposal: dict[str, Any],
    latest_by_actor: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    latest_by_actor = latest_by_actor or {}
    consented_actors = [
        actor
        for actor in CHAMBER_MEMBERS
        if latest_by_actor.get(actor, {}).get("stance") == "consent"
    ]
    withheld_actors = [
        actor
        for actor in CHAMBER_MEMBERS
        if latest_by_actor.get(actor, {}).get("stance") == "withhold"
    ]
    revise_actors = [
        actor
        for actor in CHAMBER_MEMBERS
        if latest_by_actor.get(actor, {}).get("stance") == "revise"
    ]
    pending_actors = [actor for actor in CHAMBER_MEMBERS if actor not in latest_by_actor]
    return {
        "id": proposal.get("id"),
        "t_ms": proposal.get("t_ms"),
        "support_type": proposal.get("support_type"),
        "text": truncate_for_prompt(str(proposal.get("text") or ""), 260),
        "rationale": truncate_for_prompt(str(proposal.get("rationale") or ""), 220),
        "status": _proposal_status(latest_by_actor),
        "consented_actors": consented_actors,
        "withheld_actors": withheld_actors,
        "revise_actors": revise_actors,
        "pending_actors": pending_actors,
        "latest_by_actor": {
            actor: _compact_consent_receipt(row)
            for actor, row in latest_by_actor.items()
            if actor in CHAMBER_MEMBERS
        },
        "witness_only": True,
        "authority": "support_proposal_not_control",
    }


def build_proposal_states(coll_dir: Path) -> list[dict[str, Any]]:
    proposals = read_chamber_proposals(coll_dir)
    proposal_ids = {str(proposal.get("id")) for proposal in proposals}
    receipts = [
        receipt
        for receipt in read_chamber_consent(coll_dir)
        if str(receipt.get("proposal_id") or "") in proposal_ids
    ]
    latest_by_proposal: dict[str, dict[str, dict[str, Any]]] = {
        str(proposal.get("id")): {} for proposal in proposals
    }
    for receipt in receipts:
        proposal_id = str(receipt.get("proposal_id") or "")
        actor = str(receipt.get("actor") or "")
        if proposal_id in latest_by_proposal and actor in CHAMBER_MEMBERS:
            latest_by_proposal[proposal_id][actor] = receipt
    proposal_states = [
        _compact_proposal(proposal, latest_by_proposal.get(str(proposal.get("id")), {}))
        for proposal in proposals
    ]
    return proposal_states


def build_consent_protocol(coll_dir: Path) -> dict[str, Any]:
    proposal_states = build_proposal_states(coll_dir)
    proposals_total = len(read_chamber_proposals(coll_dir))
    proposal_ids = {str(proposal.get("id")) for proposal in read_chamber_proposals(coll_dir)}
    receipts_total = len(
        [
            receipt
            for receipt in read_chamber_consent(coll_dir)
            if str(receipt.get("proposal_id") or "") in proposal_ids
        ]
    )
    latest_receipts = [
        receipt
        for receipt in read_chamber_consent(coll_dir)
        if str(receipt.get("proposal_id") or "") in proposal_ids
    ]
    active_supports = [
        proposal for proposal in proposal_states if proposal.get("status") == "active"
    ]
    pending_proposals = [
        proposal
        for proposal in proposal_states
        if proposal.get("status") in {"pending", "revision_requested"}
    ]
    latest_receipt = _compact_consent_receipt(latest_receipts[-1]) if latest_receipts else None
    return {
        "consent_schema_version": CONSENT_SCHEMA_VERSION,
        "participants": CHAMBER_MEMBERS,
        "activation_threshold": CONSENT_THRESHOLD,
        "required_latest_stance": "consent",
        "proposals_total": proposals_total,
        "receipts_total": receipts_total,
        "active_supports_count": len(active_supports),
        "pending_proposals_count": len(pending_proposals),
        "active_supports": active_supports[-5:],
        "recent_proposals": proposal_states[-6:],
        "latest_receipt": latest_receipt,
        "witness_only": True,
        "authority": "consent_state_context_not_control",
    }


def build_active_relational_supports(consent_protocol: dict[str, Any]) -> dict[str, Any]:
    supports = (
        consent_protocol.get("active_supports")
        if isinstance(consent_protocol.get("active_supports"), list)
        else []
    )
    compact = []
    for support in supports:
        if not isinstance(support, dict):
            continue
        compact.append(
            {
                "id": support.get("id"),
                "support_type": support.get("support_type"),
                "text": truncate_for_prompt(str(support.get("text") or ""), 220),
                "consented_actors": support.get("consented_actors") or [],
                "witness_only": True,
                "authority": "active_support_context_not_control",
            }
        )
    return {
        "consent_schema_version": CONSENT_SCHEMA_VERSION,
        "activation_threshold": CONSENT_THRESHOLD,
        "count": len(compact),
        "supports": compact,
        "witness_only": True,
        "authority": "active_supports_context_not_control",
    }


def render_consent_prompt_line(
    consent_protocol: dict[str, Any],
    active_relational_supports: dict[str, Any] | None = None,
) -> str:
    if not isinstance(consent_protocol, dict):
        return ""
    pending = int(consent_protocol.get("pending_proposals_count") or 0)
    active_count = int(consent_protocol.get("active_supports_count") or 0)
    parts = [
        "Consent protocol: "
        f"{pending} pending proposal(s), {active_count} active support(s); "
        "activation requires Astrid, Minime, and steward consent; "
        "consent state is context only, not command or control."
    ]
    supports = []
    if isinstance(active_relational_supports, dict):
        supports = active_relational_supports.get("supports") or []
    if not supports:
        supports = consent_protocol.get("active_supports") or []
    if isinstance(supports, list) and supports:
        support = supports[-1] if isinstance(supports[-1], dict) else {}
        if support:
            parts.append(
                "Active consented support: "
                f"{support.get('support_type', 'unknown')}; "
                "interpretive context, not command or control."
            )
    else:
        recent = consent_protocol.get("recent_proposals")
        if isinstance(recent, list) and recent:
            proposal = recent[-1] if isinstance(recent[-1], dict) else {}
            if proposal:
                parts.append(
                    "Latest support proposal: "
                    f"{proposal.get('support_type', 'unknown')} "
                    f"status={proposal.get('status', 'pending')}; "
                    "await explicit consent receipts, not behavior control."
                )
    return bounded_join(parts, 520)


def append_steward_note(
    shared_dir: Path,
    text: str,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    t_ms = now_ms()
    note_text = clamp_note_text(text)
    if not note_text:
        raise ValueError("steward note is empty")
    note = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "id": record_id("steward_note", t_ms),
        "t_ms": t_ms,
        "actor": STEWARD_HANDLE,
        "source": source,
        "text": note_text,
        "witness_only": True,
        "authority": "context_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["notes"], note)
    append_chamber_event(coll_dir, "steward_note_appended", STEWARD_HANDLE, {"note_id": note["id"]})
    return note | {"collab_id": meta["id"]}


def append_steward_intention(
    shared_dir: Path,
    text: str,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    t_ms = now_ms()
    intention_text = clamp_intention_text(text)
    if not intention_text:
        raise ValueError("steward intention is empty")
    intention = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "id": record_id("steward_intention", t_ms),
        "t_ms": t_ms,
        "actor": STEWARD_HANDLE,
        "source": source,
        "text": intention_text,
        "active": True,
        "witness_only": True,
        "authority": "context_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["intentions"], intention)
    append_chamber_event(
        coll_dir,
        "steward_intention_set",
        STEWARD_HANDLE,
        {"intention_id": intention["id"]},
    )
    return intention | {"collab_id": meta["id"]}


def clear_steward_intention(
    shared_dir: Path,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    t_ms = now_ms()
    intention = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "id": record_id("steward_intention", t_ms),
        "t_ms": t_ms,
        "actor": STEWARD_HANDLE,
        "source": source,
        "text": "",
        "active": False,
        "witness_only": True,
        "authority": "context_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["intentions"], intention)
    append_chamber_event(
        coll_dir,
        "steward_intention_cleared",
        STEWARD_HANDLE,
        {"intention_id": intention["id"]},
    )
    return intention | {"collab_id": meta["id"]}


def _normalize_memory_text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must be a list of strings")
        text = clamp_memory_text(item, 240)
        if text:
            out.append(text)
    return out[:MEMORY_LIST_LIMIT]


def _normalize_room_weather(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("room_weather must be an object")
    label = clamp_memory_text(str(value.get("label") or "manual"), 80)
    summary = clamp_memory_text(str(value.get("summary") or ""), 260)
    signature = clamp_memory_text(str(value.get("signature") or label), 160)
    weather = {
        "label": label or "manual",
        "summary": summary or label or "manual",
        "signature": signature or label or "manual",
    }
    if value.get("pairs") is not None:
        if not isinstance(value["pairs"], list):
            raise ValueError("room_weather.pairs must be a list")
        weather["pairs"] = value["pairs"][:3]
    return weather


def normalize_compressed_memory_edit(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("memory edit must be a JSON object")
    unknown = set(payload) - COMPRESSED_MEMORY_EDIT_FIELDS
    if unknown:
        raise ValueError(f"unknown compressed memory field(s): {', '.join(sorted(unknown))}")
    normalized: dict[str, Any] = {}
    for field in COMPRESSED_MEMORY_TEXT_FIELDS:
        if field in payload:
            if not isinstance(payload[field], str):
                raise ValueError(f"{field} must be a string")
            value = clamp_memory_text(payload[field])
            if value:
                normalized[field] = value
    for field in COMPRESSED_MEMORY_LIST_FIELDS:
        if field in payload:
            normalized[field] = _normalize_memory_text_list(payload[field], field)
    if "room_weather" in payload:
        normalized["room_weather"] = _normalize_room_weather(payload["room_weather"])
    if not normalized:
        raise ValueError("memory edit is empty")
    return normalized


def parse_memory_edit_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid memory edit JSON: {exc}") from exc
    return normalize_compressed_memory_edit(payload)


def append_memory_edit(
    shared_dir: Path,
    payload: dict[str, Any],
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    normalized = normalize_compressed_memory_edit(payload)
    t_ms = now_ms()
    edit = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "compression_schema_version": COMPRESSION_SCHEMA_VERSION,
        "id": record_id("chamber_memory_edit", t_ms),
        "t_ms": t_ms,
        "actor": STEWARD_HANDLE,
        "source": source,
        "active": True,
        "payload": normalized,
        "witness_only": True,
        "authority": "context_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["memory_edits"], edit)
    append_chamber_event(
        coll_dir,
        "memory_edit_set",
        STEWARD_HANDLE,
        {"edit_id": edit["id"]},
    )
    return edit | {"collab_id": meta["id"]}


def clear_memory_edit(
    shared_dir: Path,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    t_ms = now_ms()
    edit = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "compression_schema_version": COMPRESSION_SCHEMA_VERSION,
        "id": record_id("chamber_memory_edit", t_ms),
        "t_ms": t_ms,
        "actor": STEWARD_HANDLE,
        "source": source,
        "active": False,
        "payload": {},
        "witness_only": True,
        "authority": "context_not_command",
    }
    append_jsonl(chamber_paths(coll_dir)["memory_edits"], edit)
    append_chamber_event(
        coll_dir,
        "memory_edit_cleared",
        STEWARD_HANDLE,
        {"edit_id": edit["id"]},
    )
    return edit | {"collab_id": meta["id"]}


def read_steward_notes(coll_dir: Path) -> list[dict[str, Any]]:
    path = chamber_paths(coll_dir)["notes"]
    if not path.is_file():
        return []
    notes: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("actor") != STEWARD_HANDLE:
            continue
        if not payload.get("id") or not payload.get("text"):
            continue
        notes.append(payload)
    notes.sort(key=lambda n: int(n.get("t_ms") or 0))
    return notes


def read_steward_intentions(coll_dir: Path) -> list[dict[str, Any]]:
    path = chamber_paths(coll_dir)["intentions"]
    if not path.is_file():
        return []
    intentions: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("actor") != STEWARD_HANDLE:
            continue
        if not payload.get("id"):
            continue
        if bool(payload.get("active", True)) and not payload.get("text"):
            continue
        intentions.append(payload)
    intentions.sort(key=lambda n: int(n.get("t_ms") or 0))
    return intentions


def read_memory_edits(coll_dir: Path) -> list[dict[str, Any]]:
    edits = []
    for payload in read_jsonl_dicts(chamber_paths(coll_dir)["memory_edits"]):
        if payload.get("actor") != STEWARD_HANDLE:
            continue
        if not payload.get("id"):
            continue
        if bool(payload.get("active", True)):
            edit_payload = payload.get("payload")
            if not isinstance(edit_payload, dict):
                continue
        edits.append(payload)
    return edits


def active_memory_edit(coll_dir: Path) -> dict[str, Any] | None:
    edits = read_memory_edits(coll_dir)
    if not edits:
        return None
    latest = edits[-1]
    if not bool(latest.get("active", True)):
        return None
    payload = latest.get("payload")
    if not isinstance(payload, dict):
        return None
    try:
        normalized = normalize_compressed_memory_edit(payload)
    except ValueError:
        return None
    return {
        "id": latest.get("id"),
        "t_ms": latest.get("t_ms"),
        "source": latest.get("source"),
        "payload": normalized,
        "witness_only": True,
        "authority": "context_not_command",
    }


def active_steward_intention(coll_dir: Path) -> dict[str, Any] | None:
    intentions = read_steward_intentions(coll_dir)
    if not intentions:
        return None
    latest = intentions[-1]
    if not bool(latest.get("active", True)):
        return None
    return {
        "id": latest.get("id"),
        "t_ms": latest.get("t_ms"),
        "source": latest.get("source"),
        "text": truncate_for_prompt(str(latest.get("text") or ""), 320),
        "active": True,
        "witness_only": True,
        "authority": "context_not_command",
    }


def _compact_presence_receipt(row: dict[str, Any], text_limit: int = 180) -> dict[str, Any]:
    compact = {
        "id": row.get("id"),
        "t_ms": row.get("t_ms"),
        "actor": row.get("actor"),
        "attention": row.get("attention", "unknown"),
        "chamber_seen": bool(row.get("chamber_seen", True)),
        "witness_only": True,
        "authority": "public_receipt_not_command",
    }
    for source_key, output_key in (
        ("what_i_notice", "what_i_notice"),
        ("what_i_am_carrying", "what_i_am_carrying"),
        ("what_i_disagree_with", "what_i_disagree_with"),
    ):
        value = truncate_for_prompt(str(row.get(source_key) or ""), text_limit)
        if value:
            compact[output_key] = value
    if row.get("chamber_state_hash"):
        compact["chamber_state_hash"] = row.get("chamber_state_hash")
    return compact


def _compact_annotation(row: dict[str, Any], text_limit: int = 220) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "t_ms": row.get("t_ms"),
        "actor": row.get("actor"),
        "target": row.get("target", "other"),
        "stance": row.get("stance", "notice"),
        "text": truncate_for_prompt(str(row.get("text") or ""), text_limit),
        "witness_only": True,
        "authority": "annotation_context_not_command",
    }


def read_presence_receipts(coll_dir: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for payload in read_jsonl_dicts(chamber_paths(coll_dir)["presence"]):
        actor = str(payload.get("actor") or "").strip().lower()
        if actor not in CHAMBER_MEMBERS:
            continue
        if not payload.get("id"):
            continue
        payload["actor"] = actor
        payload["chamber_seen"] = bool(payload.get("chamber_seen", True))
        if str(payload.get("attention") or "unknown") not in PRESENCE_ATTENTION_LEVELS:
            payload["attention"] = "unknown"
        receipts.append(payload)
    return receipts


def recent_presence_receipts(coll_dir: Path, limit: int = 6) -> list[dict[str, Any]]:
    rows = read_presence_receipts(coll_dir)
    return rows[-limit:] if len(rows) > limit else rows


def build_presence_protocol(coll_dir: Path) -> dict[str, Any]:
    rows = read_presence_receipts(coll_dir)
    latest_by_actor: dict[str, dict[str, Any]] = {}
    for row in rows:
        actor = str(row.get("actor") or "")
        if actor:
            latest_by_actor[actor] = _compact_presence_receipt(row)
    seen_actors = [actor for actor in CHAMBER_MEMBERS if actor in latest_by_actor]
    pending_actors = [actor for actor in CHAMBER_MEMBERS if actor not in latest_by_actor]
    latest_t = max((int(row.get("t_ms") or 0) for row in rows), default=0)
    return {
        "presence_schema_version": PRESENCE_SCHEMA_VERSION,
        "participants": CHAMBER_MEMBERS,
        "receipts_total": len(rows),
        "seen_actors": seen_actors,
        "pending_actors": pending_actors,
        "all_seen": not pending_actors,
        "latest_by_actor": latest_by_actor,
        "recent_receipts": [
            _compact_presence_receipt(row)
            for row in recent_presence_receipts(coll_dir)
        ],
        "last_seen_t_ms": latest_t or None,
        "witness_only": True,
        "authority": "public_receipts_are_context_not_commands",
    }


def read_chamber_annotations(coll_dir: Path) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for payload in read_jsonl_dicts(chamber_paths(coll_dir)["annotations"]):
        actor = str(payload.get("actor") or "").strip().lower()
        if actor not in CHAMBER_MEMBERS:
            continue
        if not payload.get("id") or not payload.get("text"):
            continue
        target = str(payload.get("target") or "other").strip().lower()
        stance = str(payload.get("stance") or "notice").strip().lower()
        payload["actor"] = actor
        payload["target"] = target if target in ANNOTATION_TARGETS else "other"
        payload["stance"] = stance if stance in ANNOTATION_STANCES else "notice"
        annotations.append(payload)
    return annotations


def recent_chamber_annotations(coll_dir: Path, limit: int = 6) -> list[dict[str, Any]]:
    rows = read_chamber_annotations(coll_dir)
    return rows[-limit:] if len(rows) > limit else rows


def build_annotation_lane(coll_dir: Path) -> dict[str, Any]:
    rows = read_chamber_annotations(coll_dir)
    counts_by_target: dict[str, int] = {}
    counts_by_stance: dict[str, int] = {}
    for row in rows:
        target = str(row.get("target") or "other")
        stance = str(row.get("stance") or "notice")
        counts_by_target[target] = counts_by_target.get(target, 0) + 1
        counts_by_stance[stance] = counts_by_stance.get(stance, 0) + 1
    latest = _compact_annotation(rows[-1]) if rows else None
    return {
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotations_total": len(rows),
        "counts_by_target": counts_by_target,
        "counts_by_stance": counts_by_stance,
        "latest_annotation": latest,
        "recent_annotations": [
            _compact_annotation(row)
            for row in recent_chamber_annotations(coll_dir)
        ],
        "witness_only": True,
        "authority": "annotations_are_context_not_commands",
    }


def read_chamber_events(coll_dir: Path) -> list[dict[str, Any]]:
    path = chamber_paths(coll_dir)["events"]
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    events.sort(key=lambda event: int(event.get("t_ms") or 0))
    return events


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    rows.sort(key=lambda row: int(row.get("t_ms") or 0))
    return rows


def recent_steward_notes(coll_dir: Path, limit: int = 2) -> list[dict[str, Any]]:
    notes = read_steward_notes(coll_dir)
    return notes[-limit:] if len(notes) > limit else notes


def processed_cursor_ids(coll_dir: Path, cursor_key: str) -> set[str]:
    cursor = read_json(chamber_paths(coll_dir)["cursor"])
    raw = cursor.get(cursor_key)
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if item}


def processed_note_ids(coll_dir: Path) -> set[str]:
    return processed_cursor_ids(coll_dir, "processed_note_ids")


def processed_intention_ids(coll_dir: Path) -> set[str]:
    return processed_cursor_ids(coll_dir, "processed_intention_ids")


def unprocessed_steward_notes(coll_dir: Path) -> list[dict[str, Any]]:
    seen = processed_note_ids(coll_dir)
    return [note for note in read_steward_notes(coll_dir) if str(note.get("id")) not in seen]


def unprocessed_steward_intentions(coll_dir: Path) -> list[dict[str, Any]]:
    seen = processed_intention_ids(coll_dir)
    intentions = read_steward_intentions(coll_dir)
    if not intentions:
        return []
    latest = intentions[-1]
    intention_id = str(latest.get("id") or "")
    if not bool(latest.get("active", True)) or not intention_id or intention_id in seen:
        return []
    return [latest]


def _merge_cursor_ids(existing: set[str], new_ids: list[str]) -> list[str]:
    merged = list(existing) + [item for item in new_ids if item not in existing]
    if len(merged) > CURSOR_KEEP_LIMIT:
        merged = merged[-CURSOR_KEEP_LIMIT:]
    return merged


def write_cursor(
    coll_dir: Path,
    *,
    processed_note_ids_value: list[str] | None = None,
    processed_intention_ids_value: list[str] | None = None,
) -> None:
    path = chamber_paths(coll_dir)["cursor"]
    cursor = read_json(path)
    note_ids = (
        processed_note_ids_value
        if processed_note_ids_value is not None
        else sorted(processed_note_ids(coll_dir))
    )
    intention_ids = (
        processed_intention_ids_value
        if processed_intention_ids_value is not None
        else sorted(processed_intention_ids(coll_dir))
    )
    atomic_write_json(
        path,
        {
            "schema_version": CHAMBER_SCHEMA_VERSION,
            "processed_note_ids": note_ids,
            "processed_intention_ids": intention_ids,
            **{
                key: value
                for key, value in cursor.items()
                if key
                not in {
                    "schema_version",
                    "processed_note_ids",
                    "processed_intention_ids",
                    "updated_t_ms",
                }
            },
            "updated_t_ms": now_ms(),
        },
    )


def mark_notes_processed(coll_dir: Path, note_ids: list[str]) -> None:
    write_cursor(
        coll_dir,
        processed_note_ids_value=_merge_cursor_ids(processed_note_ids(coll_dir), note_ids),
    )


def mark_intentions_processed(coll_dir: Path, intention_ids: list[str]) -> None:
    write_cursor(
        coll_dir,
        processed_intention_ids_value=_merge_cursor_ids(
            processed_intention_ids(coll_dir),
            intention_ids,
        ),
    )


def handle_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("type") == "error":
        return {"available": False}
    h_norms = payload.get("h_norms")
    if isinstance(h_norms, list):
        norms = []
        for value in h_norms[:3]:
            try:
                norms.append(round(float(value), 3))
            except (TypeError, ValueError):
                norms.append(None)
    else:
        norms = []
    return {
        "available": True,
        "h_norms": norms,
        "tick_count": int(payload.get("tick_count") or payload.get("tick") or 0),
        "mode": payload.get("mode"),
        "seconds_since_live": payload.get("seconds_since_live"),
    }


def resonance_summary(pair: tuple[str, str], payload: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"pair": list(pair), "available": False}
    if not isinstance(payload, dict) or payload.get("type") == "error":
        return row
    row["available"] = True
    for key in ("shared_ticks", "correlation", "divergence", "rmsd"):
        if key in payload:
            row[key] = payload[key]
    return row


def build_resonance_rows(
    resonance_payloads: dict[tuple[str, str], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    return [
        resonance_summary(pair, payload)
        for pair, payload in resonance_payloads.items()
    ]


def _safe_correlation(row: dict[str, Any]) -> float | None:
    try:
        return float(row.get("correlation"))
    except (TypeError, ValueError):
        return None


def _correlation_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= -0.5:
        return "strong_neg"
    if value <= -0.15:
        return "neg"
    if value < 0.15:
        return "neutral"
    if value < 0.5:
        return "pos"
    return "strong_pos"


def derive_room_weather(resonance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    correlations: list[float] = []
    for row in resonance_rows:
        if not isinstance(row, dict) or not row.get("available"):
            continue
        pair_name = "-".join(str(part) for part in row.get("pair") or []) or "unknown"
        corr = _safe_correlation(row)
        bucket = _correlation_bucket(corr)
        pair_summary = {"pair": row.get("pair"), "bucket": bucket}
        if corr is not None:
            correlations.append(corr)
            pair_summary["correlation"] = round(corr, 4)
        if row.get("divergence") is not None:
            pair_summary["divergence"] = row.get("divergence")
        pairs.append(pair_summary)
    if not pairs:
        return {
            "label": "unavailable",
            "summary": "no pairwise resonance snapshot is available yet",
            "signature": "unavailable",
            "pairs": [],
        }
    if any(corr <= -0.45 for corr in correlations) or (
        correlations and sum(correlations) / len(correlations) <= -0.2
    ):
        label = "divergent"
    elif correlations and all(corr >= 0.35 for corr in correlations):
        label = "aligned"
    else:
        label = "mixed"
    bits = []
    for pair in pairs[:3]:
        pair_name = "-".join(str(part) for part in pair.get("pair") or []) or "unknown"
        corr = pair.get("correlation")
        if isinstance(corr, float):
            bits.append(f"{pair_name} corr={corr:+.3f}")
        else:
            bits.append(f"{pair_name} {pair.get('bucket')}")
    signature = label
    return {
        "label": label,
        "summary": "; ".join(bits),
        "signature": signature,
        "pairs": pairs[:3],
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_metric(value: Any, places: int = 4) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return round(parsed, places)


def _pair_key(pair: Any) -> str:
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return "-".join(str(part) for part in pair[:2])
    return str(pair or "unknown")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _volatility(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _timeline_label(row: dict[str, Any]) -> str | None:
    label = row.get("label")
    if isinstance(label, str) and label:
        return label
    weather = row.get("room_weather")
    if isinstance(weather, dict):
        label = weather.get("label")
        if isinstance(label, str) and label:
            return label
    return None


def _timeline_pair_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("pair_matrix"), list):
        return [item for item in row["pair_matrix"] if isinstance(item, dict)]
    if isinstance(row.get("matrix"), list):
        return [item for item in row["matrix"] if isinstance(item, dict)]
    if isinstance(row.get("pairs"), list):
        return [item for item in row["pairs"] if isinstance(item, dict)]
    weather = row.get("room_weather")
    if isinstance(weather, dict) and isinstance(weather.get("pairs"), list):
        return [item for item in weather["pairs"] if isinstance(item, dict)]
    return []


def _timeline_pair_correlations(
    history_rows: list[dict[str, Any]],
    pair_key: str,
) -> list[float]:
    correlations: list[float] = []
    for row in history_rows[-RELATIONAL_WINDOW_LIMIT:]:
        for pair_row in _timeline_pair_rows(row):
            if _pair_key(pair_row.get("pair") or pair_row.get("pair_key")) != pair_key:
                continue
            corr = _safe_float(
                pair_row.get("current_correlation")
                if pair_row.get("current_correlation") is not None
                else pair_row.get("correlation")
            )
            if corr is not None:
                correlations.append(corr)
            break
    return correlations


def _confidence_for(sample_count: int, available: bool, volatility: float = 0.0) -> str:
    if not available:
        return "unavailable"
    if sample_count >= 10 and volatility <= 0.3:
        return "high"
    if sample_count >= 3:
        return "medium"
    return "low"


def _pair_trend(series: list[float], delta: float | None, volatility: float) -> str:
    if len(series) < 3:
        return "insufficient_history"
    if volatility >= 0.35:
        return "oscillating"
    if delta is not None and delta >= 0.15:
        return "rising"
    if delta is not None and delta <= -0.15:
        return "falling"
    return "steady"


def build_pair_matrix(
    resonance_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    history_rows = history_rows or []
    current_by_key = {
        _pair_key(row.get("pair")): row
        for row in resonance_rows
        if isinstance(row, dict)
    }
    matrix: list[dict[str, Any]] = []
    for pair in RELATIONAL_PAIRS:
        key = _pair_key(pair)
        row = current_by_key.get(key, {"pair": list(pair), "available": False})
        corr = _safe_float(row.get("correlation"))
        history_corrs = _timeline_pair_correlations(history_rows, key)
        series = history_corrs + ([corr] if corr is not None else [])
        mean_corr = _mean(series)
        previous = history_corrs[-1] if history_corrs else None
        delta = (corr - previous) if corr is not None and previous is not None else None
        volatility = _volatility(series)
        available = bool(row.get("available")) and corr is not None
        item = {
            "pair": list(pair),
            "pair_key": key,
            "available": available,
            "bucket": _correlation_bucket(corr),
            "sample_count": len(series),
            "trend": _pair_trend(series, delta, volatility),
            "confidence": _confidence_for(len(series), available, volatility),
        }
        if corr is not None:
            item["current_correlation"] = round(corr, 4)
        if mean_corr is not None:
            item["mean_correlation"] = round(mean_corr, 4)
        if delta is not None:
            item["delta_correlation"] = round(delta, 4)
        item["volatility"] = round(volatility, 4)
        for key_name in ("divergence", "rmsd"):
            value = _round_metric(row.get(key_name), 6)
            if value is not None:
                item[key_name] = value
        if row.get("shared_ticks") is not None:
            try:
                item["shared_ticks"] = int(row["shared_ticks"])
            except (TypeError, ValueError):
                pass
        matrix.append(item)
    return matrix


def _latest_history_mean(history_rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(history_rows):
        corrs = []
        for pair_row in _timeline_pair_rows(row):
            corr = _safe_float(
                pair_row.get("current_correlation")
                if pair_row.get("current_correlation") is not None
                else pair_row.get("correlation")
            )
            if corr is not None:
                corrs.append(corr)
        avg = _mean(corrs)
        if avg is not None:
            return avg
    return None


def _weather_trend(
    current_weather: dict[str, Any],
    pair_matrix: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> tuple[str, int, str]:
    current_label = str(current_weather.get("label") or "unavailable")
    labels = [
        label
        for label in (_timeline_label(row) for row in history_rows[-RELATIONAL_WINDOW_LIMIT:])
        if label
    ]
    labels.append(current_label)
    streak = 0
    for label in reversed(labels):
        if label != current_label:
            break
        streak += 1
    current_corrs = [
        float(item["current_correlation"])
        for item in pair_matrix
        if item.get("current_correlation") is not None
    ]
    current_mean = _mean(current_corrs)
    previous_mean = _latest_history_mean(history_rows[-RELATIONAL_WINDOW_LIMIT:])
    mean_delta = (
        current_mean - previous_mean
        if current_mean is not None and previous_mean is not None
        else None
    )
    if len(labels) < 3:
        trend = "insufficient_history"
    else:
        recent = labels[-8:]
        transitions = sum(1 for prev, cur in zip(recent, recent[1:]) if prev != cur)
        if transitions >= 4:
            trend = "oscillating"
        elif current_label == "aligned" and (len(labels) < 2 or labels[-2] != "aligned"):
            trend = "stabilizing"
        elif mean_delta is not None and mean_delta >= 0.15:
            trend = "stabilizing"
        elif current_label == "divergent" and (len(labels) < 2 or labels[-2] != "divergent"):
            trend = "destabilizing"
        elif mean_delta is not None and mean_delta <= -0.15:
            trend = "destabilizing"
        else:
            trend = "steady"
    available_pairs = sum(1 for item in pair_matrix if item.get("available"))
    sample_count = min(len(labels), RELATIONAL_WINDOW_LIMIT)
    confidence = _confidence_for(
        sample_count,
        available_pairs == len(RELATIONAL_PAIRS),
        max((float(item.get("volatility") or 0.0) for item in pair_matrix), default=0.0),
    )
    return trend, streak, confidence


def _participant_stats(pair_matrix: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {
        member: {"correlations": [], "deltas": [], "volatilities": [], "pairs": []}
        for member in CHAMBER_MEMBERS
    }
    for item in pair_matrix:
        pair = item.get("pair")
        corr = _safe_float(item.get("current_correlation"))
        if not isinstance(pair, list) or len(pair) < 2 or corr is None:
            continue
        delta = _safe_float(item.get("delta_correlation"))
        volatility = _safe_float(item.get("volatility")) or 0.0
        for member in pair[:2]:
            if member not in stats:
                continue
            stats[member]["correlations"].append(corr)
            if delta is not None:
                stats[member]["deltas"].append(delta)
            stats[member]["volatilities"].append(volatility)
            stats[member]["pairs"].append(item.get("pair_key") or _pair_key(pair))
    out: dict[str, dict[str, Any]] = {}
    for member, raw in stats.items():
        corrs = raw["correlations"]
        if not corrs:
            continue
        out[member] = {
            "member": member,
            "connected_pairs": len(corrs),
            "avg_corr": _mean(corrs) or 0.0,
            "min_corr": min(corrs),
            "max_corr": max(corrs),
            "spread": max(corrs) - min(corrs) if len(corrs) > 1 else 0.0,
            "avg_delta": _mean(raw["deltas"]) or 0.0,
            "avg_volatility": _mean(raw["volatilities"]) or 0.0,
            "pairs": raw["pairs"],
        }
    return out


def _gravity_confidence(base: str, score_gap: float = 0.0) -> str:
    if base == "high" and score_gap >= 0.08:
        return "high"
    if base in {"high", "medium"}:
        return "medium"
    return "low"


def derive_gravitational_center(
    pair_matrix: list[dict[str, Any]],
    room_weather: dict[str, Any],
) -> dict[str, Any]:
    stats = _participant_stats(pair_matrix)
    if not stats:
        return {
            "participant": "unavailable",
            "role": "unavailable",
            "confidence": "unavailable",
            "signature": "unavailable",
            "evidence": ["no available pairwise resonance"],
            "authority": "interpretive_context_not_authority",
        }
    room_confidence = str(room_weather.get("confidence") or "low")
    label = str(room_weather.get("label") or "unavailable")
    available_items = [item for item in pair_matrix if item.get("available")]
    if len(available_items) == len(RELATIONAL_PAIRS) and label == "aligned" and all(
        (item.get("current_correlation") or 0) >= 0.35
        for item in available_items
    ):
        signature = f"shared:shared:{room_confidence}:{label}:{room_weather.get('trend', 'steady')}"
        return {
            "participant": "shared",
            "role": "shared",
            "confidence": room_confidence if room_confidence != "unavailable" else "low",
            "signature": signature,
            "evidence": ["all available pairs are positively aligned"],
            "authority": "interpretive_context_not_authority",
        }
    ordered_low = sorted(stats.values(), key=lambda item: (item["avg_corr"], item["min_corr"]))
    unsettled = ordered_low[0]
    if unsettled["min_corr"] <= -0.45 or unsettled["avg_corr"] <= -0.15:
        confidence = _gravity_confidence(room_confidence, abs(unsettled["avg_corr"]))
        signature = f"{unsettled['member']}:unsettled:{confidence}:{label}:{room_weather.get('trend', 'steady')}"
        return {
            "participant": unsettled["member"],
            "role": "unsettled",
            "confidence": confidence,
            "signature": signature,
            "evidence": [
                f"{unsettled['member']} avg_corr={unsettled['avg_corr']:+.3f}",
                f"min_corr={unsettled['min_corr']:+.3f}",
            ],
            "authority": "interpretive_context_not_authority",
        }
    mover = max(stats.values(), key=lambda item: item["avg_delta"])
    if mover["avg_delta"] >= 0.12:
        confidence = _gravity_confidence(room_confidence, mover["avg_delta"])
        signature = f"{mover['member']}:mover:{confidence}:{label}:{room_weather.get('trend', 'steady')}"
        return {
            "participant": mover["member"],
            "role": "mover",
            "confidence": confidence,
            "signature": signature,
            "evidence": [
                f"{mover['member']} avg_delta={mover['avg_delta']:+.3f}",
                f"avg_corr={mover['avg_corr']:+.3f}",
            ],
            "authority": "interpretive_context_not_authority",
        }
    anchor = max(stats.values(), key=lambda item: item["avg_corr"])
    second = sorted(stats.values(), key=lambda item: item["avg_corr"], reverse=True)[1:]
    score_gap = anchor["avg_corr"] - (second[0]["avg_corr"] if second else 0.0)
    if anchor["avg_corr"] >= 0.45:
        confidence = _gravity_confidence(room_confidence, score_gap)
        signature = f"{anchor['member']}:anchor:{confidence}:{label}:{room_weather.get('trend', 'steady')}"
        return {
            "participant": anchor["member"],
            "role": "anchor",
            "confidence": confidence,
            "signature": signature,
            "evidence": [
                f"{anchor['member']} avg_corr={anchor['avg_corr']:+.3f}",
                f"score_gap={score_gap:+.3f}",
            ],
            "authority": "interpretive_context_not_authority",
        }
    bridge_candidates = [
        item
        for item in stats.values()
        if item["connected_pairs"] >= 2 and item["avg_corr"] >= 0.15 and item["spread"] <= 0.25
    ]
    if bridge_candidates:
        bridge = max(bridge_candidates, key=lambda item: item["avg_corr"])
        confidence = _gravity_confidence(room_confidence, bridge["avg_corr"])
        signature = f"{bridge['member']}:bridge:{confidence}:{label}:{room_weather.get('trend', 'steady')}"
        return {
            "participant": bridge["member"],
            "role": "bridge",
            "confidence": confidence,
            "signature": signature,
            "evidence": [
                f"{bridge['member']} balanced pairs spread={bridge['spread']:.3f}",
                f"avg_corr={bridge['avg_corr']:+.3f}",
            ],
            "authority": "interpretive_context_not_authority",
        }
    confidence = _gravity_confidence(room_confidence, score_gap)
    signature = f"{anchor['member']}:anchor:{confidence}:{label}:{room_weather.get('trend', 'steady')}"
    return {
        "participant": anchor["member"],
        "role": "anchor",
        "confidence": confidence,
        "signature": signature,
        "evidence": [
            f"{anchor['member']} highest available avg_corr={anchor['avg_corr']:+.3f}",
            "weak center; interpret gently",
        ],
        "authority": "interpretive_context_not_authority",
    }


def _timeline_gravity(row: dict[str, Any]) -> dict[str, Any]:
    gravity = row.get("gravity")
    if isinstance(gravity, dict):
        return gravity
    gravity = row.get("gravitational_center")
    if isinstance(gravity, dict):
        return gravity
    metrics = row.get("relational_metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("gravitational_center"), dict):
        return metrics["gravitational_center"]
    return {}


def _gravity_role_key(gravity: dict[str, Any]) -> tuple[str, str]:
    participant = str(gravity.get("participant") or "unavailable")
    role = str(gravity.get("role") or "unavailable")
    return participant, role


def _dominant_decayed(counter: dict[str, float]) -> tuple[str, float, float]:
    if not counter:
        return "unavailable", 0.0, 0.0
    total = sum(counter.values())
    key, value = max(counter.items(), key=lambda item: item[1])
    ratio = value / total if total else 0.0
    return key, value, ratio


def _strength_label(strength: float, available: bool = True) -> str:
    if not available:
        return "unavailable"
    if strength >= 0.7:
        return "high"
    if strength >= 0.35:
        return "medium"
    return "low"


def derive_relational_inertia(
    history_rows: list[dict[str, Any]],
    current_weather: dict[str, Any],
    current_gravity: dict[str, Any],
) -> dict[str, Any]:
    """Derive a bounded, interpretive residue signal from prior timeline rows."""
    rows = [row for row in history_rows[-RELATIONAL_WINDOW_LIMIT:] if isinstance(row, dict)]
    current_label = str(current_weather.get("label") or "unavailable")
    current_participant, current_role = _gravity_role_key(current_gravity)
    current = {
        "weather": current_label,
        "gravity_participant": current_participant,
        "gravity_role": current_role,
    }
    base = {
        "inertia_schema_version": INERTIA_SCHEMA_VERSION,
        "source": "weather_timeline_decay",
        "window_limit": RELATIONAL_WINDOW_LIMIT,
        "history_samples": len(rows),
        "decay": RELATIONAL_INERTIA_DECAY,
        "current": current,
        "witness_only": True,
        "authority": "interpretive_context_not_authority",
    }
    if not rows:
        return base | {
            "label": "unavailable",
            "strength": 0.0,
            "strength_label": "unavailable",
            "carry_forward": {
                "weather": "unavailable",
                "gravity_participant": "unavailable",
                "gravity_role": "unavailable",
            },
            "signature": "unavailable",
            "evidence": ["no weather timeline history"],
        }

    weather_scores: dict[str, float] = {}
    gravity_scores: dict[str, float] = {}
    labels: list[str] = []
    max_window_weight = sum(
        RELATIONAL_INERTIA_DECAY ** age for age in range(RELATIONAL_WINDOW_LIMIT)
    )
    for age, row in enumerate(reversed(rows)):
        weight = RELATIONAL_INERTIA_DECAY ** age
        label = _timeline_label(row) or "unavailable"
        labels.append(label)
        if label != "unavailable":
            weather_scores[label] = weather_scores.get(label, 0.0) + weight
        participant, role = _gravity_role_key(_timeline_gravity(row))
        if participant != "unavailable" or role != "unavailable":
            gravity_key = f"{participant}:{role}"
            gravity_scores[gravity_key] = gravity_scores.get(gravity_key, 0.0) + weight
    labels = list(reversed(labels))
    carry_weather, weather_score, weather_ratio = _dominant_decayed(weather_scores)
    carry_gravity_key, gravity_score, _gravity_ratio = _dominant_decayed(gravity_scores)
    carry_participant, carry_role = (
        carry_gravity_key.split(":", 1)
        if ":" in carry_gravity_key
        else ("unavailable", "unavailable")
    )
    strength = round(weather_score / max_window_weight if max_window_weight else 0.0, 4)
    strength_name = _strength_label(strength, bool(weather_scores))
    recent_labels = labels[-8:]
    transitions = sum(1 for prev, cur in zip(recent_labels, recent_labels[1:]) if prev != cur)
    label = "reinforcing"
    if not weather_scores:
        label = "unavailable"
    elif transitions >= 4 or (len(weather_scores) >= 3 and weather_ratio < 0.45):
        label = "turbulent"
    elif len(rows) < 3 or strength < 0.25:
        label = "faint"
    elif carry_weather != current_label and current_label != "unavailable":
        label = "countercurrent"
    evidence = [
        (
            f"history samples={len(rows)} decay={RELATIONAL_INERTIA_DECAY:.2f} "
            f"dominant weather={carry_weather} score={strength:.3f}"
        ),
        f"dominant gravity={carry_participant}:{carry_role} score={round(gravity_score / max_window_weight if max_window_weight else 0.0, 4):.3f}",
        f"current weather={current_label} gravity={current_participant}:{current_role}",
    ]
    return base | {
        "label": label,
        "strength": strength,
        "strength_label": strength_name,
        "carry_forward": {
            "weather": carry_weather,
            "gravity_participant": carry_participant,
            "gravity_role": carry_role,
        },
        "signature": (
            f"{label}:{strength_name}:{carry_weather}:{carry_participant}:"
            f"{carry_role}:{current_label}:{current_participant}:{current_role}"
        ),
        "evidence": evidence[:3],
    }


def is_current_relational_metrics(metrics: Any) -> bool:
    return (
        isinstance(metrics, dict)
        and metrics.get("relational_schema_version") == RELATIONAL_SCHEMA_VERSION
        and isinstance(metrics.get("relational_inertia"), dict)
    )


def build_relational_metrics(
    coll_dir: Path,
    resonance_rows: list[dict[str, Any]],
    *,
    history_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history_rows = (
        read_jsonl_dicts(chamber_paths(coll_dir)["resonance_journal"])
        if history_rows is None
        else history_rows
    )
    history_rows = history_rows[-RELATIONAL_WINDOW_LIMIT:]
    pair_matrix = build_pair_matrix(resonance_rows, history_rows)
    weather = derive_room_weather(resonance_rows)
    trend, streak, confidence = _weather_trend(weather, pair_matrix, history_rows)
    weather = weather | {
        "trend": trend,
        "streak": streak,
        "confidence": confidence,
        "history_samples": len(history_rows),
        "window_limit": RELATIONAL_WINDOW_LIMIT,
    }
    gravity = derive_gravitational_center(pair_matrix, weather)
    inertia = derive_relational_inertia(history_rows, weather, gravity)
    metrics = {
        "relational_schema_version": RELATIONAL_SCHEMA_VERSION,
        "updated_t_ms": now_ms(),
        "window": {
            "limit": RELATIONAL_WINDOW_LIMIT,
            "history_samples": len(history_rows),
            "current_included": True,
        },
        "room_weather": weather,
        "pair_matrix": pair_matrix,
        "gravitational_center": gravity,
        "relational_inertia": inertia,
        "witness_only": True,
        "authority": "relational metrics are interpretive context, not commands or authority",
    }
    metrics["prompt_mirror"] = render_relational_mirror(metrics)
    return metrics


def render_relational_mirror(metrics: dict[str, Any]) -> str:
    weather = metrics.get("room_weather") if isinstance(metrics, dict) else {}
    gravity = metrics.get("gravitational_center") if isinstance(metrics, dict) else {}
    inertia = metrics.get("relational_inertia") if isinstance(metrics, dict) else {}
    matrix = metrics.get("pair_matrix") if isinstance(metrics, dict) else []
    if not isinstance(weather, dict):
        weather = {}
    if not isinstance(gravity, dict):
        gravity = {}
    if not isinstance(inertia, dict):
        inertia = {}
    if not isinstance(matrix, list):
        matrix = []
    parts = [
        "V3 relational mirror: interpretive context, not commands or authority.",
        (
            f"Weather {weather.get('label', 'unavailable')} trend "
            f"{weather.get('trend', 'insufficient_history')} streak "
            f"{weather.get('streak', 0)} confidence {weather.get('confidence', 'low')}."
        ),
        (
            "Relational gravity: "
            f"{gravity.get('participant', 'unavailable')} as {gravity.get('role', 'unavailable')}; "
            f"confidence {gravity.get('confidence', 'unavailable')}; "
            "interpretive context, not authority."
        ),
    ]
    evidence = gravity.get("evidence")
    if isinstance(evidence, list) and evidence:
        parts.append("Gravity evidence: " + "; ".join(str(item) for item in evidence[:2]) + ".")
    carry = inertia.get("carry_forward")
    if isinstance(carry, dict) and inertia.get("label") != "unavailable":
        parts.append(
            "Carry-forward residue: "
            f"{inertia.get('label', 'unavailable')} "
            f"{carry.get('weather', 'unavailable')}/"
            f"{carry.get('gravity_participant', 'unavailable')}:"
            f"{carry.get('gravity_role', 'unavailable')} residue, "
            f"strength {inertia.get('strength_label', 'unavailable')}; "
            "interpretive context, not authority."
        )
    matrix_bits = []
    for item in matrix[:3]:
        if not isinstance(item, dict):
            continue
        pair = item.get("pair_key") or _pair_key(item.get("pair"))
        bit = f"{pair}"
        if item.get("current_correlation") is not None:
            bit += f" corr={float(item['current_correlation']):+.3f}"
        if item.get("mean_correlation") is not None:
            bit += f" mean={float(item['mean_correlation']):+.3f}"
        if item.get("delta_correlation") is not None:
            bit += f" delta={float(item['delta_correlation']):+.3f}"
        bit += f" trend={item.get('trend', 'unknown')}"
        matrix_bits.append(bit)
    if matrix_bits:
        parts.append("Matrix: " + "; ".join(matrix_bits) + ".")
    return bounded_join(parts, 760)


def is_current_phase_cartography(cartography: Any) -> bool:
    return (
        isinstance(cartography, dict)
        and cartography.get("cartography_schema_version") == CARTOGRAPHY_SCHEMA_VERSION
    )


def _recent_weather_flips(history_rows: list[dict[str, Any]], current_label: str) -> int:
    labels = [
        label
        for label in (_timeline_label(row) for row in history_rows[-RELATIONAL_WINDOW_LIMIT:])
        if label
    ]
    if current_label:
        labels.append(current_label)
    recent = labels[-8:]
    return sum(1 for prev, cur in zip(recent, recent[1:]) if prev != cur)


def _max_matrix_volatility(pair_matrix: Any) -> float:
    if not isinstance(pair_matrix, list):
        return 0.0
    values = [
        _safe_float(item.get("volatility"))
        for item in pair_matrix
        if isinstance(item, dict)
    ]
    return max((value for value in values if value is not None), default=0.0)


def _cartography_confidence(
    weather_confidence: str,
    *,
    history_samples: int,
    decisive: bool,
) -> str:
    if weather_confidence == "unavailable":
        return "unavailable"
    if decisive and weather_confidence == "high" and history_samples >= 10:
        return "high"
    if weather_confidence in {"high", "medium"} or history_samples >= 3:
        return "medium"
    return "low"


def _phase_family(phase: str) -> str:
    if phase == "integration_opportunity":
        return "integration"
    if phase == "repair_watch":
        return "repair"
    if phase == "oscillation":
        return "oscillation"
    if phase == "deep_work_opening":
        return "deep_work"
    if phase == "handoff_ready":
        return "handoff"
    if phase == "settling":
        return "settling"
    return "unavailable"


def _transition_hint(phase: str) -> str:
    hints = {
        "integration_opportunity": (
            "current alignment can integrate lingering residue; witness without forcing closure"
        ),
        "repair_watch": "unsettled or divergent room with residue present; watch for repair needs",
        "oscillation": "recent weather flips suggest transition churn; wait for a stable contour",
        "settling": "room is stabilizing; let residue settle without treating it as command",
        "deep_work_opening": "stable non-unsettled room; deeper work may be available",
        "handoff_ready": "low-residue steady room; summarize or hand off gently",
        "unavailable": "not enough relational data yet",
    }
    return hints.get(phase, hints["unavailable"])


def derive_phase_cartography(
    relational_metrics: dict[str, Any] | None,
    *,
    history_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Interpret V3 relational metrics as a read-only room-transition phase."""
    history_rows = [row for row in (history_rows or [])[-RELATIONAL_WINDOW_LIMIT:] if isinstance(row, dict)]
    metrics = relational_metrics if isinstance(relational_metrics, dict) else {}
    weather = metrics.get("room_weather") if isinstance(metrics.get("room_weather"), dict) else {}
    gravity = (
        metrics.get("gravitational_center")
        if isinstance(metrics.get("gravitational_center"), dict)
        else {}
    )
    inertia = (
        metrics.get("relational_inertia")
        if isinstance(metrics.get("relational_inertia"), dict)
        else {}
    )
    pair_matrix = metrics.get("pair_matrix") if isinstance(metrics.get("pair_matrix"), list) else []
    label = str(weather.get("label") or "unavailable")
    trend = str(weather.get("trend") or "insufficient_history")
    weather_confidence = str(weather.get("confidence") or "low")
    streak = int(weather.get("streak") or 0)
    gravity_participant = str(gravity.get("participant") or "unavailable")
    gravity_role = str(gravity.get("role") or "unavailable")
    residue_label = str(inertia.get("label") or "unavailable")
    residue_strength = str(inertia.get("strength_label") or "unavailable")
    history_samples = int(weather.get("history_samples") or len(history_rows))
    flips = _recent_weather_flips(history_rows, label)
    max_volatility = _max_matrix_volatility(pair_matrix)

    phase = "unavailable"
    decisive = False
    if (
        not metrics
        or label == "unavailable"
        or gravity_role == "unavailable"
        or residue_label == "unavailable"
    ):
        phase = "unavailable"
    elif trend == "oscillating" or flips >= 4:
        phase = "oscillation"
        decisive = True
    elif label == "divergent" or (
        gravity_role == "unsettled" and residue_label in {"reinforcing", "countercurrent"}
    ):
        phase = "repair_watch"
        decisive = True
    elif (
        label == "aligned"
        and gravity_participant == "shared"
        and gravity_role == "shared"
        and residue_label in {"countercurrent", "faint"}
    ):
        phase = "integration_opportunity"
        decisive = True
    elif (
        trend == "steady"
        and streak >= 5
        and residue_label == "faint"
        and gravity_role != "unsettled"
    ):
        phase = "handoff_ready"
        decisive = True
    elif (
        label in {"aligned", "mixed"}
        and trend == "steady"
        and streak >= 3
        and max_volatility <= 0.25
        and gravity_role != "unsettled"
    ):
        phase = "deep_work_opening"
        decisive = True
    elif label in {"aligned", "mixed"} and trend in {"steady", "stabilizing"} and residue_label in {
        "faint",
        "reinforcing",
    }:
        phase = "settling"
    elif label in {"aligned", "mixed"}:
        phase = "settling"

    confidence = _cartography_confidence(
        weather_confidence,
        history_samples=history_samples,
        decisive=decisive,
    )
    evidence = [
        f"weather={label} trend={trend} streak={streak} flips={flips}",
        f"gravity={gravity_participant}:{gravity_role} confidence={gravity.get('confidence', 'unavailable')}",
        f"residue={residue_label}/{residue_strength} volatility={max_volatility:.3f}",
    ]
    return {
        "cartography_schema_version": CARTOGRAPHY_SCHEMA_VERSION,
        "source": "relational_metrics_phase_cartography",
        "phase": phase,
        "phase_family": _phase_family(phase),
        "confidence": confidence,
        "current_weather": label,
        "weather_trend": trend,
        "weather_streak": streak,
        "weather_flips": flips,
        "residue_label": residue_label,
        "residue_strength": residue_strength,
        "gravity_participant": gravity_participant,
        "gravity_role": gravity_role,
        "transition_hint": _transition_hint(phase),
        "evidence": [clamp_memory_text(item, 180) for item in evidence[:3]],
        "witness_only": True,
        "authority": "interpretive_context_not_command_or_phase_authority",
        "updated_t_ms": now_ms(),
    }


def build_phase_cartography(
    coll_dir: Path,
    relational_metrics: dict[str, Any] | None,
    *,
    history_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history = (
        read_jsonl_dicts(chamber_paths(coll_dir)["resonance_journal"])
        if history_rows is None
        else history_rows
    )
    return derive_phase_cartography(relational_metrics, history_rows=history)


def render_phase_cartography_line(
    cartography: dict[str, Any] | None,
    *,
    prefix: str = "phase cartography",
) -> str:
    cart = cartography if isinstance(cartography, dict) else {}
    return (
        f"{prefix}: {cart.get('phase', 'unavailable')} "
        f"family={cart.get('phase_family', 'unavailable')} "
        f"confidence={cart.get('confidence', 'unavailable')} "
        f"weather={cart.get('current_weather', 'unavailable')}/"
        f"{cart.get('weather_trend', 'unknown')} "
        f"residue={cart.get('residue_label', 'unavailable')}/"
        f"{cart.get('residue_strength', 'unavailable')} "
        f"gravity={cart.get('gravity_participant', 'unavailable')}:"
        f"{cart.get('gravity_role', 'unavailable')}"
    )


def render_phase_cartography_markdown(payload: dict[str, Any]) -> str:
    cartography = payload.get("phase_cartography") if isinstance(payload, dict) else {}
    cart = cartography if isinstance(cartography, dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    evidence = cart.get("evidence") if isinstance(cart.get("evidence"), list) else []
    timeline = payload.get("timeline") if isinstance(payload.get("timeline"), list) else []
    lines = [
        f"# Triadic Chamber Phase Cartography: {payload.get('collab_id', 'unknown')}",
        "",
        f"Phase: `{cart.get('phase', 'unavailable')}` ({cart.get('phase_family', 'unavailable')})",
        f"Confidence: `{cart.get('confidence', 'unavailable')}`",
        f"Transition hint: {cart.get('transition_hint', 'not enough relational data yet')}",
        "",
        "Boundary: interpretive context only, not command or phase authority.",
        "",
        "## Current Coordinates",
        "",
        f"- weather: {cart.get('current_weather', 'unavailable')} / {cart.get('weather_trend', 'unknown')}",
        f"- residue: {cart.get('residue_label', 'unavailable')} / {cart.get('residue_strength', 'unavailable')}",
        f"- gravity: {cart.get('gravity_participant', 'unavailable')} as {cart.get('gravity_role', 'unavailable')}",
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in (evidence[:3] or ["not enough evidence yet"]))
    lines.extend(["", "## Recent Weather", ""])
    if timeline:
        for row in timeline[-12:]:
            lines.append(
                "- "
                f"{row.get('t_ms', 'n/a')}: {row.get('label', 'unavailable')} "
                f"trend={row.get('trend', 'unknown')} "
                f"phase={row.get('phase', 'unavailable')}"
            )
    else:
        lines.append("- no weather timeline yet")
    if artifacts.get("png"):
        lines.extend(["", f"PNG: `{artifacts['png']}`"])
    if artifacts.get("plot_error"):
        lines.extend(["", f"Plot note: `{artifacts['plot_error']}`"])
    return "\n".join(lines) + "\n"


def _timeline_phase(row: dict[str, Any]) -> str:
    cartography = row.get("phase_cartography")
    if isinstance(cartography, dict) and cartography.get("phase"):
        return str(cartography["phase"])
    phase = row.get("phase")
    return str(phase) if phase else "unavailable"


def _phase_cartography_timeline(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for row in history_rows[-RELATIONAL_WINDOW_LIMIT:]:
        if not isinstance(row, dict):
            continue
        gravity = _timeline_gravity(row)
        inertia = row.get("relational_inertia") if isinstance(row.get("relational_inertia"), dict) else {}
        timeline.append({
            "t_ms": row.get("t_ms"),
            "label": _timeline_label(row) or "unavailable",
            "trend": row.get("trend") or "legacy",
            "phase": _timeline_phase(row),
            "residue_label": inertia.get("label") if isinstance(inertia, dict) else None,
            "gravity_participant": gravity.get("participant"),
            "gravity_role": gravity.get("role"),
        })
    return timeline


def _plot_phase_cartography_png(
    path: Path,
    cartography: dict[str, Any],
    history_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"matplotlib_unavailable: {exc}") from exc

    rows = history_rows[-RELATIONAL_WINDOW_LIMIT:]
    labels = [_timeline_label(row) or "unavailable" for row in rows]
    labels.append(str(cartography.get("current_weather") or "unavailable"))
    phase_labels = [_timeline_phase(row) for row in rows]
    phase_labels.append(str(cartography.get("phase") or "unavailable"))
    y_map = {"divergent": -1.0, "mixed": 0.0, "aligned": 1.0, "unavailable": -1.4}
    strength_map = {"unavailable": 0.0, "low": 0.25, "medium": 0.55, "high": 0.85}
    xs = list(range(len(labels)))
    ys = [y_map.get(label, 0.0) for label in labels]
    residue = strength_map.get(str(cartography.get("residue_strength") or "unavailable"), 0.0)

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    colors = [
        {"aligned": "#2f855a", "mixed": "#4a5568", "divergent": "#c53030"}.get(label, "#718096")
        for label in labels
    ]
    ax.plot(xs, ys, color="#2d3748", linewidth=1.5, alpha=0.55)
    ax.scatter(xs, ys, c=colors, s=44, zorder=3)
    if xs:
        ax.bar(
            [xs[-1]],
            [residue],
            bottom=-1.45,
            color="#805ad5",
            alpha=0.42,
            width=0.55,
            label="current residue strength",
        )
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["divergent", "mixed", "aligned"])
    ax.set_ylim(-1.6, 1.25)
    ax.set_xlim(-0.5, max(xs) + 0.5 if xs else 0.5)
    ax.grid(axis="y", alpha=0.24)
    ax.set_title(
        f"Phase cartography: {cartography.get('phase', 'unavailable')} "
        f"({cartography.get('confidence', 'unavailable')})",
        loc="left",
        fontsize=11,
    )
    ax.set_xlabel(
        "weather timeline -> current | "
        f"gravity {cartography.get('gravity_participant', 'unavailable')}:"
        f"{cartography.get('gravity_role', 'unavailable')}"
    )
    if phase_labels:
        step = max(1, len(phase_labels) // 6)
        tick_positions = xs[::step]
        tick_labels = [phase_labels[index] for index in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_phase_cartography_artifacts(
    coll_dir: Path,
    phase_cartography: dict[str, Any],
    relational_metrics: dict[str, Any] | None = None,
    *,
    history_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paths = chamber_paths(coll_dir)
    history = (
        read_jsonl_dicts(paths["resonance_journal"])
        if history_rows is None
        else history_rows
    )
    payload = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "cartography_schema_version": CARTOGRAPHY_SCHEMA_VERSION,
        "relational_schema_version": (
            relational_metrics.get("relational_schema_version")
            if isinstance(relational_metrics, dict)
            else None
        ),
        "collab_id": coll_dir.name,
        "updated_t_ms": now_ms(),
        "phase_cartography": phase_cartography,
        "timeline": _phase_cartography_timeline(history),
        "artifacts": {
            "json": str(paths["phase_cartography"]),
            "markdown": str(paths["phase_cartography_md"]),
        },
        "witness_only": True,
        "authority": "interpretive_context_not_command_or_phase_authority",
    }
    try:
        _plot_phase_cartography_png(paths["phase_cartography_png"], phase_cartography, history)
        payload["artifacts"]["png"] = str(paths["phase_cartography_png"])
    except Exception as exc:
        payload["artifacts"]["plot_error"] = clamp_memory_text(str(exc), 240)
    atomic_write_json(paths["phase_cartography"], payload)
    paths["phase_cartography_md"].write_text(render_phase_cartography_markdown(payload))
    return payload


def set_chamber_phase(
    shared_dir: Path,
    phase: str,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    phase = str(phase or "").strip()
    if phase not in ALLOWED_PHASES:
        allowed = ", ".join(sorted(ALLOWED_PHASES))
        raise ValueError(f"invalid chamber phase {phase!r}; allowed: {allowed}")
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    append_chamber_event(
        coll_dir,
        "phase_set",
        STEWARD_HANDLE,
        {"phase": phase, "source": source},
    )
    return {"collab_id": meta["id"], "phase": phase, "phase_source": "manual"}


def clear_chamber_phase(
    shared_dir: Path,
    *,
    target: str = "latest",
    source: str = "steward_cli",
) -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    append_chamber_event(
        coll_dir,
        "phase_cleared",
        STEWARD_HANDLE,
        {"source": source},
    )
    phase, phase_source = resolve_chamber_phase(coll_dir)
    return {"collab_id": meta["id"], "phase": phase, "phase_source": phase_source}


def truncate_for_prompt(text: str, limit: int = PROMPT_NOTE_LIMIT) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def bounded_join(parts: list[str], limit: int = PROMPT_SUMMARY_LIMIT) -> str:
    rendered = ""
    for part in parts:
        clean = " ".join(str(part or "").split())
        if not clean:
            continue
        candidate = f"{rendered} {clean}".strip() if rendered else clean
        if len(candidate) <= limit:
            rendered = candidate
            continue
        remaining = limit - len(rendered) - (1 if rendered else 0)
        if remaining > 20:
            rendered = f"{rendered} {clean[:remaining].rstrip()}...".strip()
        break
    return rendered


def render_presence_prompt_line(presence_protocol: dict[str, Any]) -> str:
    if not isinstance(presence_protocol, dict):
        return ""
    seen = presence_protocol.get("seen_actors")
    pending = presence_protocol.get("pending_actors")
    seen_text = ", ".join(str(actor) for actor in seen) if isinstance(seen, list) and seen else "none yet"
    pending_text = (
        ", ".join(str(actor) for actor in pending)
        if isinstance(pending, list) and pending
        else "none"
    )
    latest_by_actor = presence_protocol.get("latest_by_actor")
    latest_bits: list[str] = []
    if isinstance(latest_by_actor, dict):
        for actor in CHAMBER_MEMBERS:
            row = latest_by_actor.get(actor)
            if not isinstance(row, dict):
                continue
            notice = row.get("what_i_notice") or row.get("what_i_am_carrying")
            if notice:
                latest_bits.append(
                    f"{actor} noticed {truncate_for_prompt(str(notice), 90)}"
                )
    latest_clause = (
        " Latest uptake: " + "; ".join(latest_bits[:2]) + "."
        if latest_bits
        else ""
    )
    return (
        f"Presence protocol: public chamber_seen receipts from {seen_text}; "
        f"pending {pending_text}; receipts are context, not commands."
        f"{latest_clause}"
    )


def render_annotation_prompt_line(annotation_lane: dict[str, Any]) -> str:
    if not isinstance(annotation_lane, dict):
        return ""
    latest = annotation_lane.get("latest_annotation")
    if not isinstance(latest, dict):
        total = int(annotation_lane.get("annotations_total") or 0)
        if total <= 0:
            return ""
        return f"Annotation lane: {total} public annotations; annotations are context, not commands."
    return (
        "Annotation lane: latest "
        f"{latest.get('actor', 'unknown')} {latest.get('stance', 'notice')} "
        f"{latest.get('target', 'chamber')}: "
        f"\"{truncate_for_prompt(str(latest.get('text') or ''), 120)}\"; "
        "context, not commands."
    )


def _question_candidates(texts: list[str]) -> list[str]:
    questions: list[str] = []
    for text in texts:
        raw = str(text or "").replace("\n", " ")
        if "?" not in raw:
            continue
        for piece in raw.split("?"):
            clean = clamp_memory_text(piece, 220)
            if clean:
                questions.append(f"{clean}?")
    return questions[:MEMORY_LIST_LIMIT]


def _event_shift(event: dict[str, Any]) -> str:
    name = str(event.get("event") or "unknown")
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
    if name == "steward_note_processed":
        return f"steward note processed into chamber handles ({detail.get('note_id', 'unknown')})"
    if name == "steward_intention_processed":
        return f"steward intention processed into chamber handles ({detail.get('intention_id', 'unknown')})"
    if name == "steward_intention_set":
        return f"steward intention updated ({detail.get('intention_id', 'unknown')})"
    if name == "memory_edit_set":
        return f"compressed memory manually refined ({detail.get('edit_id', 'unknown')})"
    if name == "memory_edit_cleared":
        return "compressed memory returned to derived mode"
    if name == "presence_receipt_appended":
        return f"presence receipt from {detail.get('actor', event.get('actor', 'unknown'))}"
    if name == "chamber_annotation_appended":
        return (
            f"{detail.get('actor', event.get('actor', 'unknown'))} annotated "
            f"{detail.get('target', 'chamber')} as {detail.get('stance', 'notice')}"
        )
    if name == "phase_set":
        return f"phase manually set to {detail.get('phase', 'unknown')}"
    if name == "phase_cleared":
        return "manual phase override cleared"
    return name.replace("_", " ")


def build_compressed_memory(
    coll_dir: Path,
    meta: dict[str, Any],
    *,
    phase: str,
    phase_source: str,
    active_intention: dict[str, Any] | None,
    recent_notes: list[dict[str, Any]],
    resonance_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    events = read_chamber_events(coll_dir)
    topic = str(meta.get("topic") or "")
    latest_note = recent_notes[-1] if recent_notes else None
    intention_text = str(active_intention.get("text") or "") if active_intention else ""
    latest_note_text = str(latest_note.get("text") or "") if latest_note else ""
    current_thread = (
        intention_text
        or latest_note_text
        or f"{topic or coll_dir.name} in {phase} phase"
    )
    question_texts = [intention_text, latest_note_text] + [
        str(note.get("text") or "") for note in recent_notes[-3:]
    ]
    open_questions = _question_candidates(question_texts)
    if not open_questions:
        open_questions = [f"What should the chamber preserve through {phase}?"]
    recent_shifts = [_event_shift(event) for event in events[-4:]]
    if not recent_shifts:
        recent_shifts = [f"chamber is in {phase} ({phase_source})"]
    room_weather = derive_room_weather(resonance_rows)
    derived = {
        "compression_schema_version": COMPRESSION_SCHEMA_VERSION,
        "current_thread": clamp_memory_text(current_thread, 280),
        "open_questions": open_questions[:MEMORY_LIST_LIMIT],
        "do_not_forget": [
            "Steward notes, intentions, and memory edits are context, not commands.",
            "Minime stays on Ollama and does not borrow Astrid's 8090 lane.",
            "The chamber memory orients re-entry; it does not control runtime behavior.",
        ],
        "recent_shifts": [clamp_memory_text(shift, 220) for shift in recent_shifts[-MEMORY_LIST_LIMIT:]],
        "stable_truths": [
            "Astrid, Minime, and steward are the chamber members.",
            "The shared chamber handle is the active collab_<id> reservoir handle.",
            "The steward lane is witness-level context unless a later bounded-influence phase changes it.",
        ],
        "room_weather": room_weather,
        "source": "derived",
        "updated_t_ms": now_ms(),
    }
    edit = active_memory_edit(coll_dir)
    if not edit:
        return derived
    payload = edit["payload"]
    compressed = derived | payload
    compressed["source"] = "hybrid_manual"
    compressed["manual_edit_id"] = edit.get("id")
    compressed["manual_edit_t_ms"] = edit.get("t_ms")
    compressed["updated_t_ms"] = now_ms()
    compressed["compression_schema_version"] = COMPRESSION_SCHEMA_VERSION
    return compressed


def maybe_append_resonance_journal(
    coll_dir: Path,
    compressed_memory: dict[str, Any],
    relational_metrics: dict[str, Any] | None = None,
    phase_cartography: dict[str, Any] | None = None,
) -> None:
    weather = compressed_memory.get("room_weather")
    if not isinstance(weather, dict):
        return
    gravity: dict[str, Any] = {}
    inertia: dict[str, Any] = {}
    trend = None
    confidence = None
    pair_matrix: list[dict[str, Any]] = []
    if isinstance(relational_metrics, dict):
        rel_weather = relational_metrics.get("room_weather")
        if isinstance(rel_weather, dict):
            weather = weather | rel_weather
            trend = rel_weather.get("trend")
            confidence = rel_weather.get("confidence")
        raw_gravity = relational_metrics.get("gravitational_center")
        if isinstance(raw_gravity, dict):
            gravity = raw_gravity
        raw_inertia = relational_metrics.get("relational_inertia")
        if isinstance(raw_inertia, dict):
            inertia = raw_inertia
        raw_matrix = relational_metrics.get("pair_matrix")
        if isinstance(raw_matrix, list):
            pair_matrix = [row for row in raw_matrix[:3] if isinstance(row, dict)]
    gravity_signature = str(gravity.get("signature") or "")
    inertia_signature = str(inertia.get("signature") or "")
    cartography_signature = ""
    if isinstance(phase_cartography, dict):
        cartography_signature = (
            f"{phase_cartography.get('phase', 'unavailable')}:"
            f"{phase_cartography.get('confidence', 'unavailable')}:"
            f"{phase_cartography.get('current_weather', 'unavailable')}:"
            f"{phase_cartography.get('residue_label', 'unavailable')}:"
            f"{phase_cartography.get('gravity_participant', 'unavailable')}:"
            f"{phase_cartography.get('gravity_role', 'unavailable')}"
        )
    signature = str(gravity_signature or weather.get("signature") or weather.get("label") or "")
    if not signature:
        return
    path = chamber_paths(coll_dir)["resonance_journal"]
    rows = read_jsonl_dicts(path)
    latest = rows[-1] if rows else {}
    now = now_ms()
    last_t = int(latest.get("t_ms") or 0)
    if (
        latest.get("signature") == signature
        and latest.get("inertia_signature") == inertia_signature
        and latest.get("cartography_signature") == cartography_signature
        and now - last_t < RESONANCE_JOURNAL_MIN_INTERVAL_MS
    ):
        return
    append_jsonl(
        path,
        {
            "schema_version": CHAMBER_SCHEMA_VERSION,
            "compression_schema_version": COMPRESSION_SCHEMA_VERSION,
            "relational_schema_version": (
                RELATIONAL_SCHEMA_VERSION if relational_metrics else None
            ),
            "t_ms": now,
            "signature": signature,
            "gravity_signature": gravity_signature,
            "inertia_signature": inertia_signature,
            "cartography_signature": cartography_signature,
            "label": weather.get("label"),
            "trend": trend,
            "streak": weather.get("streak"),
            "confidence": confidence,
            "summary": weather.get("summary"),
            "pairs": weather.get("pairs", []),
            "pair_matrix": pair_matrix,
            "gravity": gravity,
            "relational_inertia": inertia,
            "phase_cartography": phase_cartography if isinstance(phase_cartography, dict) else {},
            "phase": (
                phase_cartography.get("phase")
                if isinstance(phase_cartography, dict)
                else None
            ),
        },
    )


def render_prompt_summary(
    chamber_doc: dict[str, Any],
    recent_notes: list[dict[str, Any]],
    resonance_rows: list[dict[str, Any]],
    *,
    phase: str = "initialized",
    phase_source: str = "inferred",
    active_intention: dict[str, Any] | None = None,
    compressed_memory: dict[str, Any] | None = None,
    relational_metrics: dict[str, Any] | None = None,
    phase_cartography: dict[str, Any] | None = None,
    presence_protocol: dict[str, Any] | None = None,
    annotation_lane: dict[str, Any] | None = None,
    consent_protocol: dict[str, Any] | None = None,
    active_relational_supports: dict[str, Any] | None = None,
) -> str:
    parts = [
        "Triadic chamber witness: steward notes, intentions, memory edits, presence receipts, annotations, proposals, and consent receipts are shared context, not commands.",
        f"Phase {phase} ({phase_source}).",
        f"Room {chamber_doc.get('collab_id')} topic \"{chamber_doc.get('topic', '')}\".",
    ]
    if isinstance(presence_protocol, dict):
        presence_line = render_presence_prompt_line(presence_protocol)
        if presence_line:
            parts.append(presence_line)
    if isinstance(consent_protocol, dict):
        consent_line = render_consent_prompt_line(consent_protocol, active_relational_supports)
        if consent_line:
            parts.append(consent_line)
    if active_intention:
        parts.append(
            "Active steward intention: "
            f"\"{truncate_for_prompt(str(active_intention.get('text') or ''), 220)}\" "
            "(witness context, not a command)."
        )
    if compressed_memory:
        current_thread = compressed_memory.get("current_thread")
        if current_thread:
            parts.append(f"Current thread: \"{truncate_for_prompt(str(current_thread), 180)}\".")
        open_questions = compressed_memory.get("open_questions")
        if isinstance(open_questions, list) and open_questions:
            parts.append(f"Open question: \"{truncate_for_prompt(str(open_questions[0]), 180)}\".")
        do_not_forget = compressed_memory.get("do_not_forget")
        if isinstance(do_not_forget, list) and do_not_forget:
            parts.append(f"Do not forget: \"{truncate_for_prompt(str(do_not_forget[0]), 180)}\".")
        recent_shifts = compressed_memory.get("recent_shifts")
        if isinstance(recent_shifts, list) and recent_shifts:
            parts.append(f"Latest shift: {truncate_for_prompt(str(recent_shifts[-1]), 180)}.")
        weather = compressed_memory.get("room_weather")
        if isinstance(weather, dict):
            label = weather.get("label")
            summary = weather.get("summary")
            if label or summary:
                parts.append(
                    f"Room weather: {label or 'unknown'} "
                    f"({truncate_for_prompt(str(summary or ''), 180)})."
                )
    if relational_metrics:
        mirror = relational_metrics.get("prompt_mirror")
        if isinstance(mirror, str) and mirror.strip():
            parts.append(mirror)
    if isinstance(phase_cartography, dict):
        phase_label = phase_cartography.get("phase")
        transition_hint = phase_cartography.get("transition_hint")
        if phase_label:
            parts.append(
                "Phase cartography: "
                f"{phase_label}; "
                f"{truncate_for_prompt(str(transition_hint or 'not enough relational data yet'), 180)}; "
                "interpretive context, not command or phase authority."
            )
    if isinstance(annotation_lane, dict):
        annotation_line = render_annotation_prompt_line(annotation_lane)
        if annotation_line:
            parts.append(annotation_line)
    if recent_notes:
        latest = recent_notes[-1]
        parts.append(f"Latest steward witness: \"{truncate_for_prompt(str(latest.get('text') or ''))}\".")
    rendered_pairs = []
    for row in resonance_rows:
        if not row.get("available"):
            continue
        pair = "-".join(row.get("pair") or [])
        bits = [pair]
        if row.get("correlation") is not None:
            try:
                bits.append(f"corr={float(row['correlation']):+.3f}")
            except (TypeError, ValueError):
                pass
        if row.get("divergence") is not None:
            bits.append(f"div={row['divergence']}")
        if row.get("rmsd") is not None:
            bits.append(f"rmsd={row['rmsd']}")
        rendered_pairs.append(" ".join(bits))
    if rendered_pairs:
        parts.append("Resonance: " + "; ".join(rendered_pairs[:3]) + ".")
    return bounded_join(parts, PROMPT_SUMMARY_LIMIT)


def build_chamber_state(
    meta: dict[str, Any],
    chamber_doc: dict[str, Any],
    handle_payloads: dict[str, dict[str, Any] | None],
    resonance_payloads: dict[tuple[str, str], dict[str, Any] | None],
    notes: list[dict[str, Any]],
    *,
    active_intention: dict[str, Any] | None = None,
    phase: str = "initialized",
    phase_source: str = "inferred",
    resonance_rows: list[dict[str, Any]] | None = None,
    compressed_memory: dict[str, Any] | None = None,
    relational_metrics: dict[str, Any] | None = None,
    phase_cartography: dict[str, Any] | None = None,
    presence_protocol: dict[str, Any] | None = None,
    annotation_lane: dict[str, Any] | None = None,
    consent_protocol: dict[str, Any] | None = None,
    active_relational_supports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resonance_rows is None:
        resonance_rows = build_resonance_rows(resonance_payloads)
    recent_notes = [
        {
            "id": note.get("id"),
            "t_ms": note.get("t_ms"),
            "source": note.get("source"),
            "text": truncate_for_prompt(str(note.get("text") or "")),
            "witness_only": True,
        }
        for note in notes
    ]
    state = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "mode": CHAMBER_MODE,
        "collab_id": meta.get("id"),
        "topic": meta.get("topic"),
        "phase": phase,
        "phase_source": phase_source,
        "updated_t_ms": now_ms(),
        "witness_only": True,
        "authority": "chamber witness records are context, not commands",
        "handles": chamber_doc.get("handles", {}),
        "reservoir": {
            label: handle_summary(payload)
            for label, payload in handle_payloads.items()
        },
        "resonance": resonance_rows,
        "relational_metrics": relational_metrics,
        "recent_steward_notes": recent_notes,
        "active_steward_intention": active_intention,
        "compressed_memory": compressed_memory,
        "phase_cartography": phase_cartography,
        "presence_protocol": presence_protocol,
        "annotation_lane": annotation_lane,
        "consent_protocol": consent_protocol,
        "active_relational_supports": active_relational_supports,
    }
    state["prompt_summary"] = render_prompt_summary(
        chamber_doc,
        recent_notes,
        resonance_rows,
        phase=phase,
        phase_source=phase_source,
        active_intention=active_intention,
        compressed_memory=compressed_memory,
        relational_metrics=relational_metrics,
        phase_cartography=phase_cartography,
        presence_protocol=presence_protocol,
        annotation_lane=annotation_lane,
        consent_protocol=consent_protocol,
        active_relational_supports=active_relational_supports,
    )
    return state


def write_chamber_state(coll_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(chamber_paths(coll_dir)["state"], state)


def _compact_resonance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {
            "pair": row.get("pair"),
            "available": bool(row.get("available")),
        }
        for key in ("correlation", "divergence", "rmsd", "shared_ticks"):
            if key in row:
                item[key] = row[key]
        compact.append(item)
    return compact


def inferred_chamber_phase(notes: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    if any(event.get("event") == "steward_note_processed" for event in events):
        return "witness_active"
    if notes:
        return "witness_pending"
    if any(event.get("event") in {"activated", "activated_by_collab_feeder"} for event in events):
        return "initialized"
    return "initialized"


def manual_phase_from_events(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        event_name = event.get("event")
        if event_name == "phase_cleared":
            return None
        if event_name == "phase_set":
            detail = event.get("detail")
            phase = detail.get("phase") if isinstance(detail, dict) else None
            if phase in ALLOWED_PHASES:
                return str(phase)
    return None


def resolve_chamber_phase(
    coll_dir: Path,
    notes: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    notes = read_steward_notes(coll_dir) if notes is None else notes
    events = read_chamber_events(coll_dir) if events is None else events
    manual = manual_phase_from_events(events)
    if manual is not None:
        return manual, "manual"
    return inferred_chamber_phase(notes, events), "inferred"


def build_chamber_memory(coll_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    notes = read_steward_notes(coll_dir)
    intentions = read_steward_intentions(coll_dir)
    events = read_chamber_events(coll_dir)
    latest_note = notes[-1] if notes else None
    event_counts: dict[str, int] = {}
    for event in events:
        key = str(event.get("event") or "unknown")
        event_counts[key] = event_counts.get(key, 0) + 1
    phase, phase_source = resolve_chamber_phase(coll_dir, notes, events)
    active_intention = active_steward_intention(coll_dir)
    compressed_memory = state.get("compressed_memory")
    if not isinstance(compressed_memory, dict):
        compressed_memory = build_compressed_memory(
            coll_dir,
            {"id": state.get("collab_id") or coll_dir.name, "topic": state.get("topic") or ""},
            phase=phase,
            phase_source=phase_source,
            active_intention=active_intention,
            recent_notes=recent_steward_notes(coll_dir),
            resonance_rows=(
                state.get("resonance") if isinstance(state.get("resonance"), list) else []
            ),
        )
    resonance_rows = state.get("resonance") if isinstance(state.get("resonance"), list) else []
    relational_metrics = state.get("relational_metrics")
    if not is_current_relational_metrics(relational_metrics):
        relational_metrics = build_relational_metrics(coll_dir, resonance_rows)
    phase_cartography = state.get("phase_cartography")
    if not is_current_phase_cartography(phase_cartography):
        phase_cartography = build_phase_cartography(coll_dir, relational_metrics)
    presence_protocol = state.get("presence_protocol")
    if not isinstance(presence_protocol, dict):
        presence_protocol = build_presence_protocol(coll_dir)
    annotation_lane = state.get("annotation_lane")
    if not isinstance(annotation_lane, dict):
        annotation_lane = build_annotation_lane(coll_dir)
    consent_protocol = state.get("consent_protocol")
    if not isinstance(consent_protocol, dict):
        consent_protocol = build_consent_protocol(coll_dir)
    active_relational_supports = state.get("active_relational_supports")
    if not isinstance(active_relational_supports, dict):
        active_relational_supports = build_active_relational_supports(consent_protocol)
    collab_id = str(state.get("collab_id") or coll_dir.name)
    topic = str(state.get("topic") or "")
    intention_clause = ""
    if active_intention:
        intention_clause = (
            " Active steward intention: "
            f"{truncate_for_prompt(str(active_intention.get('text') or ''), 240)!r}."
        )
    reentry_prompt = (
        f"Re-enter triadic chamber {collab_id} on {topic!r}. Phase: {phase} ({phase_source})."
        f"{intention_clause} "
        "Treat steward notes as witness context, not commands. "
        "Read chamber_state.json for the live snapshot and chamber_events.jsonl "
        "for append-only history before changing the room. "
        "Use presence receipts, annotations, proposals, and consent receipts as public uptake signals only."
    )
    memory = {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "collab_id": collab_id,
        "topic": topic,
        "phase": phase,
        "phase_source": phase_source,
        "updated_t_ms": now_ms(),
        "witness_only": True,
        "authority": "chamber witness records are context, not commands",
        "counts": {
            "steward_notes": len(notes),
            "steward_intentions": len(intentions),
            "active_steward_intention": active_intention is not None,
            "presence_receipts": int(presence_protocol.get("receipts_total") or 0),
            "chamber_annotations": int(annotation_lane.get("annotations_total") or 0),
            "support_proposals": int(consent_protocol.get("proposals_total") or 0),
            "consent_receipts": int(consent_protocol.get("receipts_total") or 0),
            "active_relational_supports": int(active_relational_supports.get("count") or 0),
            "events": len(events),
            "event_types": event_counts,
        },
        "active_steward_intention": active_intention,
        "compressed_memory": compressed_memory,
        "relational_metrics": relational_metrics,
        "phase_cartography": phase_cartography,
        "presence_protocol": presence_protocol,
        "annotation_lane": annotation_lane,
        "consent_protocol": consent_protocol,
        "active_relational_supports": active_relational_supports,
        "recent_presence_receipts": presence_protocol.get("recent_receipts", []),
        "recent_chamber_annotations": annotation_lane.get("recent_annotations", []),
        "recent_steward_notes": [
            {
                "id": note.get("id"),
                "t_ms": note.get("t_ms"),
                "text": truncate_for_prompt(str(note.get("text") or ""), 180),
                "witness_only": True,
            }
            for note in (notes[-3:] if len(notes) > 3 else notes)
        ],
        "latest_steward_note": (
            {
                "id": latest_note.get("id"),
                "t_ms": latest_note.get("t_ms"),
                "text": truncate_for_prompt(str(latest_note.get("text") or ""), 240),
                "witness_only": True,
            }
            if latest_note
            else None
        ),
        "resonance_snapshot": _compact_resonance(
            state.get("resonance") if isinstance(state.get("resonance"), list) else []
        ),
        "state_summary": state.get("prompt_summary", ""),
        "reentry_prompt": reentry_prompt,
        "stable_truths": [
            "Astrid, Minime, and steward are the chamber members.",
            "The shared chamber handle is the active collab_<id> reservoir handle.",
            "The steward lane is witness-level context unless a later bounded-influence phase changes it.",
            "Relational supports activate only when Astrid, Minime, and steward all consent.",
        ],
        "boundaries": [
            "Steward notes, intentions, presence receipts, annotations, proposals, and consent receipts are context, not commands.",
            "Phase labels orient re-entry; they do not control runtime behavior.",
            "Presence receipts and annotations are public uptake context, not commands.",
            "Consented supports are relational context only; they do not control runtime behavior.",
            "Keep Minime on Ollama; do not borrow Astrid's 8090 lane.",
        ],
        "next_checks": [
            "Read chamber_state.json for live resonance before acting.",
            "Check presence receipts before claiming Astrid or Minime explicitly noticed the chamber.",
            "Check consent_protocol before claiming a support has triadic consent.",
            "Append steward notes and intentions only as witness context.",
            "Keep Minime on Ollama; do not borrow Astrid's 8090 lane.",
        ],
    }
    return memory


def render_reentry_markdown(memory: dict[str, Any]) -> str:
    latest = memory.get("latest_steward_note")
    latest_text = ""
    if isinstance(latest, dict) and latest.get("text"):
        latest_text = f"\n\nLatest steward witness: {latest['text']}"
    active_intention = memory.get("active_steward_intention")
    intention_text = ""
    if isinstance(active_intention, dict) and active_intention.get("text"):
        intention_text = f"\n\nActive steward intention: {active_intention['text']}"
    resonance_lines = []
    for row in memory.get("resonance_snapshot") or []:
        if not isinstance(row, dict):
            continue
        pair = " - ".join(str(part) for part in row.get("pair") or [])
        bits = [pair or "unknown pair"]
        if row.get("correlation") is not None:
            bits.append(f"corr={row['correlation']}")
        if row.get("divergence") is not None:
            bits.append(f"div={row['divergence']}")
        resonance_lines.append("- " + ", ".join(bits))
    resonance = "\n".join(resonance_lines) if resonance_lines else "- no resonance snapshot yet"
    compressed = memory.get("compressed_memory")
    compressed_lines = []
    if isinstance(compressed, dict):
        if compressed.get("current_thread"):
            compressed_lines.append(f"- current thread: {compressed['current_thread']}")
        questions = compressed.get("open_questions")
        if isinstance(questions, list) and questions:
            compressed_lines.append(f"- open question: {questions[0]}")
        reminders = compressed.get("do_not_forget")
        if isinstance(reminders, list) and reminders:
            compressed_lines.append(f"- do not forget: {reminders[0]}")
        shifts = compressed.get("recent_shifts")
        if isinstance(shifts, list) and shifts:
            compressed_lines.append(f"- latest shift: {shifts[-1]}")
        weather = compressed.get("room_weather")
        if isinstance(weather, dict):
            compressed_lines.append(
                f"- room weather: {weather.get('label', 'unknown')} ({weather.get('summary', '')})"
            )
    compressed_section = "\n".join(compressed_lines) if compressed_lines else "- no compressed memory yet"
    relational = memory.get("relational_metrics")
    relational_lines = []
    if isinstance(relational, dict):
        weather = relational.get("room_weather")
        gravity = relational.get("gravitational_center")
        inertia = relational.get("relational_inertia")
        matrix = relational.get("pair_matrix")
        if isinstance(weather, dict):
            relational_lines.append(
                "- weather: "
                f"{weather.get('label', 'unavailable')} / {weather.get('trend', 'unknown')} "
                f"(confidence {weather.get('confidence', 'low')})"
            )
        if isinstance(gravity, dict):
            relational_lines.append(
                "- relational gravity: "
                f"{gravity.get('participant', 'unavailable')} as {gravity.get('role', 'unavailable')} "
                f"(confidence {gravity.get('confidence', 'unavailable')}; interpretive context, not authority)"
            )
        if isinstance(inertia, dict):
            carry = inertia.get("carry_forward") if isinstance(inertia.get("carry_forward"), dict) else {}
            relational_lines.append(
                "- carry-forward residue: "
                f"{inertia.get('label', 'unavailable')} "
                f"{carry.get('weather', 'unavailable')}/"
                f"{carry.get('gravity_participant', 'unavailable')}:"
                f"{carry.get('gravity_role', 'unavailable')} "
                f"(strength {inertia.get('strength_label', 'unavailable')}; interpretive context, not authority)"
            )
        if isinstance(matrix, list) and matrix:
            compact = []
            for row in matrix[:3]:
                if not isinstance(row, dict):
                    continue
                compact.append(
                    f"{row.get('pair_key', 'unknown')} corr={row.get('current_correlation', 'n/a')} "
                    f"trend={row.get('trend', 'unknown')}"
                )
            if compact:
                relational_lines.append("- matrix: " + "; ".join(compact))
    relational_section = "\n".join(relational_lines) if relational_lines else "- no relational metrics yet"
    cartography = memory.get("phase_cartography")
    cartography_lines = []
    if isinstance(cartography, dict):
        cartography_lines.append(
            "- "
            + render_phase_cartography_line(cartography, prefix="phase cartography")
        )
        if cartography.get("transition_hint"):
            cartography_lines.append(f"- transition hint: {cartography['transition_hint']}")
        evidence = cartography.get("evidence")
        if isinstance(evidence, list) and evidence:
            cartography_lines.append("- evidence: " + "; ".join(str(item) for item in evidence[:3]))
        cartography_lines.append(
            "- boundary: interpretive context only, not command or phase authority"
        )
    cartography_section = (
        "\n".join(cartography_lines)
        if cartography_lines
        else "- no phase cartography yet"
    )
    presence = memory.get("presence_protocol")
    presence_lines = []
    if isinstance(presence, dict):
        presence_lines.append("- " + render_presence_prompt_line(presence))
        latest_by_actor = presence.get("latest_by_actor")
        if isinstance(latest_by_actor, dict):
            for actor in CHAMBER_MEMBERS:
                row = latest_by_actor.get(actor)
                if not isinstance(row, dict):
                    continue
                bits = [
                    f"{actor}: attention={row.get('attention', 'unknown')}",
                ]
                if row.get("what_i_notice"):
                    bits.append(f"notice={row['what_i_notice']}")
                if row.get("what_i_am_carrying"):
                    bits.append(f"carrying={row['what_i_am_carrying']}")
                presence_lines.append("- " + "; ".join(bits))
    presence_section = "\n".join(presence_lines) if presence_lines else "- no presence receipts yet"
    annotation = memory.get("annotation_lane")
    annotation_lines = []
    if isinstance(annotation, dict):
        annotation_lines.append("- " + render_annotation_prompt_line(annotation))
        for row in annotation.get("recent_annotations") or []:
            if not isinstance(row, dict):
                continue
            annotation_lines.append(
                "- "
                f"{row.get('actor', 'unknown')} {row.get('stance', 'notice')} "
                f"{row.get('target', 'chamber')}: {row.get('text', '')}"
            )
    annotation_section = (
        "\n".join(annotation_lines)
        if annotation_lines
        else "- no chamber annotations yet"
    )
    consent = memory.get("consent_protocol")
    active_supports = memory.get("active_relational_supports")
    consent_lines = []
    if isinstance(consent, dict):
        consent_line = render_consent_prompt_line(
            consent,
            active_supports if isinstance(active_supports, dict) else None,
        )
        if consent_line:
            consent_lines.append("- " + consent_line)
        for row in consent.get("recent_proposals") or []:
            if not isinstance(row, dict):
                continue
            consent_lines.append(
                "- "
                f"{row.get('status', 'pending')} {row.get('support_type', 'unknown')} "
                f"{row.get('id')}: {row.get('text', '')}"
            )
    consent_section = (
        "\n".join(consent_lines)
        if consent_lines
        else "- no support proposals or consent receipts yet"
    )
    return (
        f"# Triadic Chamber Re-entry: {memory.get('collab_id')}\n\n"
        f"Phase: `{memory.get('phase')}` ({memory.get('phase_source', 'inferred')})\n\n"
        f"Topic: {memory.get('topic')}\n\n"
        f"{memory.get('reentry_prompt')}{intention_text}{latest_text}\n\n"
        "## Compressed Memory\n\n"
        f"{compressed_section}\n\n"
        "## Resonance Snapshot\n\n"
        f"{resonance}\n\n"
        "## Relational Metrics\n\n"
        f"{relational_section}\n\n"
        "## Phase Cartography\n\n"
        f"{cartography_section}\n\n"
        "## Presence Protocol\n\n"
        f"{presence_section}\n\n"
        "## Annotation Lane\n\n"
        f"{annotation_section}\n\n"
        "## Consent Protocol\n\n"
        f"{consent_section}\n\n"
        "## Boundaries\n\n"
        + "\n".join(f"- {item}" for item in memory.get("boundaries") or [])
        + "\n\n"
        "## Guardrail\n\n"
        "Steward notes and intentions are witness context only, not commands. Minime does not borrow Astrid's `8090` lane.\n"
    )


def write_chamber_memory(coll_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    memory = build_chamber_memory(coll_dir, state)
    paths = chamber_paths(coll_dir)
    atomic_write_json(paths["memory"], memory)
    paths["reentry"].write_text(render_reentry_markdown(memory))
    phase_cartography = (
        memory.get("phase_cartography")
        if isinstance(memory.get("phase_cartography"), dict)
        else None
    )
    write_phase_cartography_artifacts(
        coll_dir,
        phase_cartography or derive_phase_cartography(None, history_rows=[]),
        memory.get("relational_metrics") if isinstance(memory.get("relational_metrics"), dict) else None,
    )
    maybe_append_resonance_journal(
        coll_dir,
        memory.get("compressed_memory", {}),
        memory.get("relational_metrics") if isinstance(memory.get("relational_metrics"), dict) else None,
        phase_cartography,
    )
    return memory


def refresh_chamber_files_from_disk(coll_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    chamber_doc = ensure_chamber(coll_dir, meta)
    paths = chamber_paths(coll_dir)
    existing = read_json(paths["state"])
    phase, phase_source = resolve_chamber_phase(coll_dir)
    active_intention = active_steward_intention(coll_dir)
    resonance_rows = existing.get("resonance") if isinstance(existing.get("resonance"), list) else []
    notes = recent_steward_notes(coll_dir)
    compressed_memory = build_compressed_memory(
        coll_dir,
        meta,
        phase=phase,
        phase_source=phase_source,
        active_intention=active_intention,
        recent_notes=notes,
        resonance_rows=resonance_rows,
    )
    relational_metrics = build_relational_metrics(coll_dir, resonance_rows)
    phase_cartography = build_phase_cartography(coll_dir, relational_metrics)
    presence_protocol = build_presence_protocol(coll_dir)
    annotation_lane = build_annotation_lane(coll_dir)
    consent_protocol = build_consent_protocol(coll_dir)
    active_relational_supports = build_active_relational_supports(consent_protocol)
    state = existing | {
        "schema_version": CHAMBER_SCHEMA_VERSION,
        "mode": CHAMBER_MODE,
        "collab_id": meta.get("id"),
        "topic": meta.get("topic"),
        "phase": phase,
        "phase_source": phase_source,
        "updated_t_ms": now_ms(),
        "witness_only": True,
        "authority": "chamber witness records are context, not commands",
        "handles": chamber_doc.get("handles", {}),
        "recent_steward_notes": [
            {
                "id": note.get("id"),
                "t_ms": note.get("t_ms"),
                "source": note.get("source"),
                "text": truncate_for_prompt(str(note.get("text") or "")),
                "witness_only": True,
            }
            for note in notes
        ],
        "active_steward_intention": active_intention,
        "compressed_memory": compressed_memory,
        "relational_metrics": relational_metrics,
        "phase_cartography": phase_cartography,
        "presence_protocol": presence_protocol,
        "annotation_lane": annotation_lane,
        "consent_protocol": consent_protocol,
        "active_relational_supports": active_relational_supports,
    }
    state["prompt_summary"] = render_prompt_summary(
        chamber_doc,
        state["recent_steward_notes"],
        resonance_rows,
        phase=phase,
        phase_source=phase_source,
        active_intention=active_intention,
        compressed_memory=compressed_memory,
        relational_metrics=relational_metrics,
        phase_cartography=phase_cartography,
        presence_protocol=presence_protocol,
        annotation_lane=annotation_lane,
        consent_protocol=consent_protocol,
        active_relational_supports=active_relational_supports,
    )
    write_chamber_state(coll_dir, state)
    write_chamber_memory(coll_dir, state)
    return state


def render_status(shared_dir: Path, target: str = "latest") -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    ensure_chamber(coll_dir, meta)
    state = read_json(chamber_paths(coll_dir)["state"])
    notes = recent_steward_notes(coll_dir, 3)
    phase, phase_source = resolve_chamber_phase(coll_dir)
    active_intention = active_steward_intention(coll_dir)
    lines = [
        f"Triadic chamber: {meta['id']}",
        f"topic: {meta.get('topic', '')}",
        f"status: {meta.get('status', '')}",
        f"chamber handle: {collab_handle_name(str(meta['id']))}",
        f"steward handle: {STEWARD_HANDLE}",
        f"phase: {phase} ({phase_source})",
    ]
    if active_intention:
        lines.append(
            "active steward intention: "
            f"{truncate_for_prompt(str(active_intention.get('text') or ''), 180)}"
        )
    summary = state.get("prompt_summary")
    if summary:
        lines.append(f"summary: {summary}")
    memory = read_json(chamber_paths(coll_dir)["memory"])
    if memory:
        compressed = memory.get("compressed_memory")
        if isinstance(compressed, dict):
            if compressed.get("current_thread"):
                lines.append(
                    "current thread: "
                    f"{truncate_for_prompt(str(compressed.get('current_thread')), 180)}"
                )
            questions = compressed.get("open_questions")
            if isinstance(questions, list) and questions:
                lines.append(f"open question: {truncate_for_prompt(str(questions[0]), 180)}")
            weather = compressed.get("room_weather")
            if isinstance(weather, dict):
                lines.append(
                    "room weather: "
                    f"{weather.get('label', 'unknown')} "
                    f"({truncate_for_prompt(str(weather.get('summary') or ''), 180)})"
                )
        relational = memory.get("relational_metrics")
        if isinstance(relational, dict):
            weather = relational.get("room_weather")
            gravity = relational.get("gravitational_center")
            inertia = relational.get("relational_inertia")
            matrix = relational.get("pair_matrix")
            if isinstance(weather, dict):
                lines.append(
                    "weather trend: "
                    f"{weather.get('label', 'unavailable')} / {weather.get('trend', 'unknown')} "
                    f"streak={weather.get('streak', 0)} confidence={weather.get('confidence', 'low')}"
                )
            if isinstance(gravity, dict):
                lines.append(
                    "relational gravity: "
                    f"{gravity.get('participant', 'unavailable')} as {gravity.get('role', 'unavailable')} "
                    f"confidence={gravity.get('confidence', 'unavailable')} "
                    "(interpretive context, not authority)"
                )
            if isinstance(inertia, dict):
                lines.append(_inertia_summary_line(inertia))
            cartography = memory.get("phase_cartography")
            if isinstance(cartography, dict):
                lines.append(render_phase_cartography_line(cartography))
            if isinstance(matrix, list) and matrix:
                compact = []
                for row in matrix[:3]:
                    if not isinstance(row, dict):
                        continue
                    compact.append(
                        f"{row.get('pair_key', 'unknown')} corr={row.get('current_correlation', 'n/a')} "
                        f"mean={row.get('mean_correlation', 'n/a')} trend={row.get('trend', 'unknown')}"
                    )
                if compact:
                    lines.append("matrix: " + "; ".join(compact))
        lines.append(f"reentry: {chamber_paths(coll_dir)['reentry']}")
        presence = memory.get("presence_protocol")
        if isinstance(presence, dict):
            lines.append(render_presence_prompt_line(presence))
        annotation = memory.get("annotation_lane")
        if isinstance(annotation, dict):
            annotation_line = render_annotation_prompt_line(annotation)
            if annotation_line:
                lines.append(annotation_line)
        consent = memory.get("consent_protocol")
        if isinstance(consent, dict):
            active_supports = memory.get("active_relational_supports")
            lines.append(
                render_consent_prompt_line(
                    consent,
                    active_supports if isinstance(active_supports, dict) else None,
                )
            )
    if not memory:
        relational = state.get("relational_metrics")
        if isinstance(relational, dict):
            weather = relational.get("room_weather")
            gravity = relational.get("gravitational_center")
            inertia = relational.get("relational_inertia")
            matrix = relational.get("pair_matrix")
            if isinstance(weather, dict):
                lines.append(
                    "weather trend: "
                    f"{weather.get('label', 'unavailable')} / {weather.get('trend', 'unknown')} "
                    f"streak={weather.get('streak', 0)} confidence={weather.get('confidence', 'low')}"
                )
            if isinstance(gravity, dict):
                lines.append(
                    "relational gravity: "
                    f"{gravity.get('participant', 'unavailable')} as {gravity.get('role', 'unavailable')} "
                    f"confidence={gravity.get('confidence', 'unavailable')} "
                    "(interpretive context, not authority)"
                )
            if isinstance(inertia, dict):
                lines.append(_inertia_summary_line(inertia))
            cartography = state.get("phase_cartography")
            if isinstance(cartography, dict):
                lines.append(render_phase_cartography_line(cartography))
            if isinstance(matrix, list) and matrix:
                compact = []
                for row in matrix[:3]:
                    if not isinstance(row, dict):
                        continue
                    compact.append(
                        f"{row.get('pair_key', 'unknown')} corr={row.get('current_correlation', 'n/a')} "
                        f"mean={row.get('mean_correlation', 'n/a')} trend={row.get('trend', 'unknown')}"
                    )
                if compact:
                    lines.append("matrix: " + "; ".join(compact))
        presence = state.get("presence_protocol")
        if isinstance(presence, dict):
            lines.append(render_presence_prompt_line(presence))
        annotation = state.get("annotation_lane")
        if isinstance(annotation, dict):
            annotation_line = render_annotation_prompt_line(annotation)
            if annotation_line:
                lines.append(annotation_line)
        consent = state.get("consent_protocol")
        if isinstance(consent, dict):
            active_supports = state.get("active_relational_supports")
            lines.append(
                render_consent_prompt_line(
                    consent,
                    active_supports if isinstance(active_supports, dict) else None,
                )
            )
    if notes:
        lines.append("recent steward notes:")
        for note in notes:
            lines.append(f"- {note.get('id')}: {truncate_for_prompt(str(note.get('text') or ''), 160)}")
    return "\n".join(lines)


def render_memory(shared_dir: Path, target: str = "latest") -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    path = chamber_paths(coll_dir)["memory"]
    if not path.is_file():
        return "(no chamber memory yet)"
    return path.read_text()


def _current_relational_metrics(shared_dir: Path, target: str = "latest") -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    state = read_json(chamber_paths(coll_dir)["state"])
    metrics = state.get("relational_metrics")
    if is_current_relational_metrics(metrics):
        return metrics
    resonance_rows = state.get("resonance") if isinstance(state.get("resonance"), list) else []
    return build_relational_metrics(coll_dir, resonance_rows)


def _current_phase_cartography(shared_dir: Path, target: str = "latest") -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    state = read_json(chamber_paths(coll_dir)["state"])
    cartography = state.get("phase_cartography")
    if is_current_phase_cartography(cartography):
        return cartography
    metrics = _current_relational_metrics(shared_dir, target)
    return build_phase_cartography(coll_dir, metrics)


def _inertia_summary_line(inertia: dict[str, Any], prefix: str = "carry-forward residue") -> str:
    carry = inertia.get("carry_forward") if isinstance(inertia.get("carry_forward"), dict) else {}
    current = inertia.get("current") if isinstance(inertia.get("current"), dict) else {}
    return (
        f"{prefix}: {inertia.get('label', 'unavailable')} "
        f"strength={inertia.get('strength_label', 'unavailable')} "
        f"carry={carry.get('weather', 'unavailable')}/"
        f"{carry.get('gravity_participant', 'unavailable')}:"
        f"{carry.get('gravity_role', 'unavailable')} "
        f"current={current.get('weather', 'unavailable')}/"
        f"{current.get('gravity_participant', 'unavailable')}:"
        f"{current.get('gravity_role', 'unavailable')}"
    )


def render_metrics(shared_dir: Path, target: str = "latest") -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    metrics = _current_relational_metrics(shared_dir, target)
    weather = metrics.get("room_weather") if isinstance(metrics, dict) else {}
    gravity = metrics.get("gravitational_center") if isinstance(metrics, dict) else {}
    inertia = metrics.get("relational_inertia") if isinstance(metrics, dict) else {}
    matrix = metrics.get("pair_matrix") if isinstance(metrics, dict) else []
    lines = [f"Triadic relational metrics: {meta['id']}"]
    if isinstance(weather, dict):
        lines.append(
            f"weather: {weather.get('label', 'unavailable')} / {weather.get('trend', 'unknown')} "
            f"streak={weather.get('streak', 0)} confidence={weather.get('confidence', 'low')}"
        )
        if weather.get("summary"):
            lines.append(f"weather summary: {weather.get('summary')}")
    if isinstance(gravity, dict):
        lines.append(
            "relational gravity: "
            f"{gravity.get('participant', 'unavailable')} as {gravity.get('role', 'unavailable')} "
            f"confidence={gravity.get('confidence', 'unavailable')}"
        )
        evidence = gravity.get("evidence")
        if isinstance(evidence, list) and evidence:
            lines.append("gravity evidence: " + "; ".join(str(item) for item in evidence[:3]))
    if isinstance(inertia, dict):
        lines.append(_inertia_summary_line(inertia))
        evidence = inertia.get("evidence")
        if isinstance(evidence, list) and evidence:
            lines.append("residue evidence: " + "; ".join(str(item) for item in evidence[:3]))
    cartography = _current_phase_cartography(shared_dir, target)
    if isinstance(cartography, dict):
        lines.append(render_phase_cartography_line(cartography))
        if cartography.get("transition_hint"):
            lines.append(f"transition hint: {cartography.get('transition_hint')}")
    if isinstance(matrix, list) and matrix:
        lines.append("matrix:")
        for row in matrix[:3]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                f"{row.get('pair_key', 'unknown')}: "
                f"corr={row.get('current_correlation', 'n/a')} "
                f"mean={row.get('mean_correlation', 'n/a')} "
                f"delta={row.get('delta_correlation', 'n/a')} "
                f"vol={row.get('volatility', 'n/a')} "
                f"trend={row.get('trend', 'unknown')} "
                f"confidence={row.get('confidence', 'unknown')}"
            )
    lines.append("boundary: relational metrics are interpretive context, not commands or authority")
    return "\n".join(lines)


def render_cartography(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    cartography = _current_phase_cartography(shared_dir, target)
    paths = chamber_paths(coll_dir)
    rows = read_jsonl_dicts(paths["resonance_journal"])
    tail = rows[-limit:] if len(rows) > limit else rows
    lines = [f"Triadic phase cartography: {meta['id']}"]
    lines.append(render_phase_cartography_line(cartography))
    if cartography.get("transition_hint"):
        lines.append(f"transition hint: {cartography.get('transition_hint')}")
    evidence = cartography.get("evidence")
    if isinstance(evidence, list) and evidence:
        lines.append("evidence:")
        for item in evidence[:3]:
            lines.append(f"- {item}")
    lines.append(
        "artifacts: "
        f"{paths['phase_cartography']} | {paths['phase_cartography_md']}"
        + (
            f" | {paths['phase_cartography_png']}"
            if paths["phase_cartography_png"].is_file()
            else ""
        )
    )
    if tail:
        lines.append("recent weather:")
        for row in tail:
            lines.append(
                "- "
                f"{row.get('t_ms')} {row.get('label', 'unavailable')} "
                f"trend={row.get('trend') or 'legacy'} phase={_timeline_phase(row)}"
            )
    lines.append("boundary: phase cartography is interpretive context, not command or phase authority")
    return "\n".join(lines)


def render_inertia(shared_dir: Path, target: str = "latest") -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    metrics = _current_relational_metrics(shared_dir, target)
    inertia = metrics.get("relational_inertia") if isinstance(metrics, dict) else {}
    if not isinstance(inertia, dict):
        return "(no relational inertia yet)"
    lines = [f"Triadic relational inertia: {meta['id']}"]
    lines.append(_inertia_summary_line(inertia))
    lines.append(
        f"source={inertia.get('source', 'unknown')} "
        f"history_samples={inertia.get('history_samples', 0)} "
        f"window={inertia.get('window_limit', RELATIONAL_WINDOW_LIMIT)} "
        f"decay={inertia.get('decay', RELATIONAL_INERTIA_DECAY)}"
    )
    evidence = inertia.get("evidence")
    if isinstance(evidence, list) and evidence:
        lines.append("evidence:")
        for item in evidence[:3]:
            lines.append(f"- {item}")
    lines.append("boundary: relational inertia is interpretive context, not commands or authority")
    return "\n".join(lines)


def render_presence(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    protocol = build_presence_protocol(coll_dir)
    rows = read_presence_receipts(coll_dir)
    tail = rows[-limit:] if len(rows) > limit else rows
    lines = [f"Triadic presence protocol: {meta['id']}"]
    lines.append(render_presence_prompt_line(protocol))
    if tail:
        lines.append("receipts:")
        for row in tail:
            compact = _compact_presence_receipt(row, 140)
            notice = compact.get("what_i_notice") or ""
            carrying = compact.get("what_i_am_carrying") or ""
            disagree = compact.get("what_i_disagree_with") or ""
            detail = "; ".join(
                part for part in (
                    f"notice={notice}" if notice else "",
                    f"carrying={carrying}" if carrying else "",
                    f"disagree={disagree}" if disagree else "",
                ) if part
            )
            lines.append(
                "- "
                f"{compact.get('t_ms')} {compact.get('actor')} "
                f"attention={compact.get('attention', 'unknown')} "
                f"hash={compact.get('chamber_state_hash', 'none')}"
                + (f" {detail}" if detail else "")
            )
    else:
        lines.append("receipts: none yet")
    lines.append("boundary: presence receipts are public uptake context, not commands")
    return "\n".join(lines)


def render_annotations(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    lane = build_annotation_lane(coll_dir)
    rows = read_chamber_annotations(coll_dir)
    tail = rows[-limit:] if len(rows) > limit else rows
    lines = [f"Triadic annotation lane: {meta['id']}"]
    annotation_line = render_annotation_prompt_line(lane)
    lines.append(annotation_line or "Annotation lane: no public annotations yet.")
    if tail:
        lines.append("annotations:")
        for row in tail:
            compact = _compact_annotation(row, 180)
            lines.append(
                "- "
                f"{compact.get('t_ms')} {compact.get('actor')} "
                f"{compact.get('stance')} {compact.get('target')}: "
                f"{compact.get('text')}"
            )
    lines.append("boundary: annotations are public context, not commands")
    return "\n".join(lines)


def render_proposals(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    protocol = build_consent_protocol(coll_dir)
    proposals = build_proposal_states(coll_dir)
    if not isinstance(proposals, list):
        proposals = []
    tail = proposals[-limit:] if len(proposals) > limit else proposals
    lines = [f"Triadic support proposals: {meta['id']}"]
    lines.append(render_consent_prompt_line(protocol))
    if tail:
        lines.append("proposals:")
        for row in tail:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                f"{row.get('t_ms')} {row.get('status', 'pending')} "
                f"{row.get('support_type', 'unknown')} {row.get('id')}: "
                f"{truncate_for_prompt(str(row.get('text') or ''), 180)}"
            )
    else:
        lines.append("proposals: none yet")
    lines.append("boundary: proposals are relational context, not commands or control")
    return "\n".join(lines)


def render_proposal_status(
    shared_dir: Path,
    target: str = "latest",
    proposal_id: str | None = None,
) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    proposals = build_proposal_states(coll_dir)
    if not isinstance(proposals, list):
        proposals = []
    selected: dict[str, Any] | None = None
    if proposal_id:
        normalized = normalize_proposal_id(proposal_id)
        for row in proposals:
            if isinstance(row, dict) and row.get("id") == normalized:
                selected = row
                break
        if selected is None:
            return f"(proposal {normalized} not found in {meta['id']})"
    elif proposals:
        selected = proposals[-1] if isinstance(proposals[-1], dict) else None
    if selected is None:
        return "(no support proposals yet)"
    lines = [
        f"Triadic proposal status: {meta['id']}",
        f"id: {selected.get('id')}",
        f"type: {selected.get('support_type')}",
        f"status: {selected.get('status')}",
        f"text: {selected.get('text')}",
    ]
    if selected.get("rationale"):
        lines.append(f"rationale: {selected.get('rationale')}")
    lines.append(
        "actors: "
        f"consent={','.join(selected.get('consented_actors') or []) or 'none'} "
        f"withhold={','.join(selected.get('withheld_actors') or []) or 'none'} "
        f"revise={','.join(selected.get('revise_actors') or []) or 'none'} "
        f"pending={','.join(selected.get('pending_actors') or []) or 'none'}"
    )
    latest = selected.get("latest_by_actor")
    if isinstance(latest, dict) and latest:
        lines.append("latest receipts:")
        for actor in CHAMBER_MEMBERS:
            row = latest.get(actor)
            if not isinstance(row, dict):
                continue
            note = row.get("note")
            lines.append(
                "- "
                f"{actor}: {row.get('stance')} {row.get('id')}"
                + (f" note={note}" if note else "")
            )
    lines.append("boundary: active support is context only, not command or control")
    return "\n".join(lines)


def render_weather(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    rows = read_jsonl_dicts(chamber_paths(coll_dir)["resonance_journal"])
    tail = rows[-limit:] if len(rows) > limit else rows
    if not tail:
        return "(no chamber weather timeline yet)"
    lines = [f"Triadic weather timeline: {meta['id']}"]
    for row in tail:
        gravity = row.get("gravity") if isinstance(row.get("gravity"), dict) else {}
        inertia = (
            row.get("relational_inertia")
            if isinstance(row.get("relational_inertia"), dict)
            else {}
        )
        gravity_text = ""
        if gravity:
            gravity_text = (
                f" gravity={gravity.get('participant', 'unavailable')}:"
                f"{gravity.get('role', 'unavailable')}"
                f"/{gravity.get('confidence', 'unavailable')}"
            )
        inertia_text = ""
        if inertia:
            carry = inertia.get("carry_forward") if isinstance(inertia.get("carry_forward"), dict) else {}
            inertia_text = (
                f" residue={inertia.get('label', 'unavailable')}:"
                f"{carry.get('weather', 'unavailable')}/"
                f"{inertia.get('strength_label', 'unavailable')}"
            )
        phase_text = ""
        phase = _timeline_phase(row)
        if phase != "unavailable" or isinstance(row.get("phase_cartography"), dict):
            phase_text = f" phase={phase}"
        trend = row.get("trend")
        if not trend:
            trend = "legacy"
        lines.append(
            "- "
            f"{row.get('t_ms')} {row.get('label', 'unavailable')} "
            f"trend={trend} confidence={row.get('confidence', 'n/a')}"
            f"{gravity_text}{inertia_text}{phase_text} "
            f"summary={truncate_for_prompt(str(row.get('summary') or ''), 180)}"
        )
    return "\n".join(lines)


def render_intentions(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    rows = read_steward_intentions(coll_dir)
    tail = rows[-limit:] if len(rows) > limit else rows
    if not tail:
        return "(no steward intentions yet)"
    lines = []
    active = active_steward_intention(coll_dir)
    active_id = active.get("id") if active else None
    for row in tail:
        marker = "active" if row.get("id") == active_id else "history"
        if not bool(row.get("active", True)):
            marker = "cleared"
        text = truncate_for_prompt(str(row.get("text") or ""), 180)
        lines.append(f"- {row.get('t_ms')} {marker} {row.get('id')}: {text}")
    return "\n".join(lines)


def render_history(shared_dir: Path, target: str = "latest", limit: int = 20) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    rows: list[tuple[int, str, str]] = []
    for note in read_steward_notes(coll_dir):
        rows.append((
            int(note.get("t_ms") or 0),
            "note",
            f"{note.get('id')}: {truncate_for_prompt(str(note.get('text') or ''), 180)}",
        ))
    for intention in read_steward_intentions(coll_dir):
        state = "active" if bool(intention.get("active", True)) else "cleared"
        rows.append((
            int(intention.get("t_ms") or 0),
            "intention",
            f"{state} {intention.get('id')}: {truncate_for_prompt(str(intention.get('text') or ''), 180)}",
        ))
    for edit in read_memory_edits(coll_dir):
        state = "active" if bool(edit.get("active", True)) else "cleared"
        payload = edit.get("payload")
        keys = ", ".join(sorted(payload)) if isinstance(payload, dict) else ""
        rows.append((
            int(edit.get("t_ms") or 0),
            "memory",
            f"{state} {edit.get('id')}: {keys}",
        ))
    for receipt in read_presence_receipts(coll_dir):
        compact = _compact_presence_receipt(receipt, 120)
        notice = compact.get("what_i_notice") or compact.get("what_i_am_carrying") or ""
        rows.append((
            int(receipt.get("t_ms") or 0),
            "presence",
            (
                f"{compact.get('actor')} attention={compact.get('attention', 'unknown')} "
                f"{truncate_for_prompt(str(notice), 140)}"
            ).strip(),
        ))
    for annotation in read_chamber_annotations(coll_dir):
        compact = _compact_annotation(annotation, 140)
        rows.append((
            int(annotation.get("t_ms") or 0),
            "annotation",
            (
                f"{compact.get('actor')} {compact.get('stance')} "
                f"{compact.get('target')}: {compact.get('text')}"
            ),
        ))
    for proposal in build_proposal_states(coll_dir):
        rows.append((
            int(proposal.get("t_ms") or 0),
            "proposal",
            (
                f"{proposal.get('status', 'pending')} "
                f"{proposal.get('support_type', 'unknown')} "
                f"{proposal.get('id')}: {truncate_for_prompt(str(proposal.get('text') or ''), 140)}"
            ),
        ))
    for receipt in read_chamber_consent(coll_dir):
        compact = _compact_consent_receipt(receipt, 140)
        rows.append((
            int(receipt.get("t_ms") or 0),
            "consent",
            (
                f"{compact.get('actor')} {compact.get('stance')} "
                f"{compact.get('proposal_id')}"
                + (f": {compact.get('note')}" if compact.get("note") else "")
            ),
        ))
    for event in read_chamber_events(coll_dir):
        rows.append((
            int(event.get("t_ms") or 0),
            "event",
            f"{event.get('actor')} {event.get('event')} {event.get('detail', {})}",
        ))
    rows.sort(key=lambda row: row[0])
    tail = rows[-limit:] if len(rows) > limit else rows
    if not tail:
        return "(no chamber history yet)"
    return "\n".join(f"- {t_ms} {kind}: {text}" for t_ms, kind, text in tail)


def render_log(shared_dir: Path, target: str = "latest", limit: int = 12) -> str:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    path = chamber_paths(coll_dir)["events"]
    if not path.is_file():
        return "(no chamber events yet)"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    tail = rows[-limit:]
    if not tail:
        return "(no chamber events yet)"
    return "\n".join(
        f"- {row.get('t_ms')} {row.get('actor')} {row.get('event')} {row.get('detail', {})}"
        for row in tail
    )


def activate(shared_dir: Path, target: str = "latest") -> dict[str, Any]:
    meta = select_collab(shared_dir, target, joined_only=False)
    coll_dir = shared_dir / str(meta["id"])
    chamber = ensure_chamber(coll_dir, meta)
    append_chamber_event(coll_dir, "activated", STEWARD_HANDLE, {"mode": CHAMBER_MODE})
    return chamber


def main() -> int:
    parser = argparse.ArgumentParser(description="Steward CLI for the triadic witness chamber")
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED_DIR)
    parser.add_argument("--target", default="latest", help="collaboration id, partial id, or latest")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("activate", help="initialize chamber files for a collaboration")
    sub.add_parser("status", help="show compact chamber status")
    sub.add_parser("metrics", help="show current relational metrics")
    sub.add_parser("inertia", help="show current relational inertia")
    cartography_parser = sub.add_parser("cartography", help="show current phase cartography")
    cartography_parser.add_argument("--limit", type=int, default=12)
    weather_parser = sub.add_parser("weather", help="show chamber weather timeline")
    weather_parser.add_argument("--limit", type=int, default=12)
    presence_parser = sub.add_parser("presence", help="append or inspect public chamber_seen receipts")
    presence_sub = presence_parser.add_subparsers(dest="presence_command", required=True)
    presence_seen = presence_sub.add_parser("seen", help="append a public chamber_seen receipt")
    presence_seen.add_argument("actor", choices=CHAMBER_MEMBERS)
    presence_seen.add_argument("--attention", choices=sorted(PRESENCE_ATTENTION_LEVELS), default="unknown")
    presence_seen.add_argument("--notice", default="")
    presence_seen.add_argument("--carrying", default="")
    presence_seen.add_argument("--disagree", default="")
    presence_list = presence_sub.add_parser("list", help="list public chamber_seen receipts")
    presence_list.add_argument("--limit", type=int, default=12)
    annotation_parser = sub.add_parser("annotation", help="append or inspect chamber annotations")
    annotation_sub = annotation_parser.add_subparsers(dest="annotation_command", required=True)
    annotation_add = annotation_sub.add_parser("add", help="append a public chamber annotation")
    annotation_add.add_argument("actor", choices=CHAMBER_MEMBERS)
    annotation_add.add_argument("annotation_target", choices=sorted(ANNOTATION_TARGETS))
    annotation_add.add_argument("stance", choices=sorted(ANNOTATION_STANCES))
    annotation_add.add_argument("text", nargs="+")
    annotation_list = annotation_sub.add_parser("list", help="list chamber annotations")
    annotation_list.add_argument("--limit", type=int, default=12)
    proposal_parser = sub.add_parser("proposal", help="create or inspect support proposals")
    proposal_sub = proposal_parser.add_subparsers(dest="proposal_command", required=True)
    proposal_create = proposal_sub.add_parser("create", help="create a witness-only support proposal")
    proposal_create.add_argument("support_type", choices=sorted(CONSENT_SUPPORT_TYPES))
    proposal_create.add_argument("text", nargs="+")
    proposal_create.add_argument("--rationale", default="")
    proposal_list = proposal_sub.add_parser("list", help="list support proposals")
    proposal_list.add_argument("--limit", type=int, default=12)
    proposal_status = proposal_sub.add_parser("status", help="show proposal consent status")
    proposal_status.add_argument("proposal_id", nargs="?")
    consent_parser = sub.add_parser("consent", help="record explicit chamber consent receipts")
    consent_sub = consent_parser.add_subparsers(dest="consent_command", required=True)
    consent_record = consent_sub.add_parser("record", help="record a public consent receipt")
    consent_record.add_argument("proposal_id")
    consent_record.add_argument("actor", choices=CHAMBER_MEMBERS)
    consent_record.add_argument("stance", choices=sorted(CONSENT_STANCES))
    consent_record.add_argument("--note", default="")
    note_parser = sub.add_parser("note", help="append a steward witness note")
    note_parser.add_argument("text", nargs="+")
    intent_parser = sub.add_parser("intent", help="set or clear the steward intention")
    intent_sub = intent_parser.add_subparsers(dest="intent_command", required=True)
    intent_set = intent_sub.add_parser("set", help="set active steward intention")
    intent_set.add_argument("text", nargs="+")
    intent_sub.add_parser("clear", help="clear active steward intention")
    intent_list = intent_sub.add_parser("list", help="list steward intention history")
    intent_list.add_argument("--limit", type=int, default=12)
    phase_parser = sub.add_parser("phase", help="set or clear the chamber phase")
    phase_sub = phase_parser.add_subparsers(dest="phase_command", required=True)
    phase_set = phase_sub.add_parser("set", help="set manual chamber phase")
    phase_set.add_argument("phase", choices=sorted(ALLOWED_PHASES))
    phase_sub.add_parser("clear", help="clear manual chamber phase")
    memory_parser = sub.add_parser("memory", help="show or edit chamber memory")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_edit = memory_sub.add_parser("edit", help="set compressed memory JSON override")
    memory_edit.add_argument("json_text", nargs="?")
    memory_edit.add_argument("--file", type=Path)
    memory_sub.add_parser("clear", help="clear compressed memory JSON override")
    sub.add_parser("reentry", help="show the chamber re-entry note")
    history_parser = sub.add_parser("history", help="show unified chamber history")
    history_parser.add_argument("--limit", type=int, default=20)
    log_parser = sub.add_parser("log", help="show recent chamber events")
    log_parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    if args.command == "activate":
        chamber = activate(args.shared_dir, args.target)
        print(f"activated {chamber['collab_id']} ({chamber['mode']})")
        return 0
    if args.command == "status":
        print(render_status(args.shared_dir, args.target))
        return 0
    if args.command == "metrics":
        print(render_metrics(args.shared_dir, args.target))
        return 0
    if args.command == "inertia":
        print(render_inertia(args.shared_dir, args.target))
        return 0
    if args.command == "cartography":
        print(render_cartography(args.shared_dir, args.target, args.limit))
        return 0
    if args.command == "weather":
        print(render_weather(args.shared_dir, args.target, args.limit))
        return 0
    if args.command == "presence":
        if args.presence_command == "seen":
            receipt = append_presence_receipt(
                args.shared_dir,
                args.actor,
                attention=args.attention,
                notice=args.notice,
                carrying=args.carrying,
                disagree=args.disagree,
                target=args.target,
            )
            meta = select_collab(args.shared_dir, args.target)
            refresh_chamber_files_from_disk(args.shared_dir / str(meta["id"]), meta)
            print(f"recorded {receipt['id']} for {receipt['collab_id']}")
            return 0
        if args.presence_command == "list":
            print(render_presence(args.shared_dir, args.target, args.limit))
            return 0
    if args.command == "annotation":
        if args.annotation_command == "add":
            annotation = append_chamber_annotation(
                args.shared_dir,
                args.actor,
                args.annotation_target,
                args.stance,
                " ".join(args.text),
                target=args.target,
            )
            meta = select_collab(args.shared_dir, args.target)
            refresh_chamber_files_from_disk(args.shared_dir / str(meta["id"]), meta)
            print(f"recorded {annotation['id']} for {annotation['collab_id']}")
            return 0
        if args.annotation_command == "list":
            print(render_annotations(args.shared_dir, args.target, args.limit))
            return 0
    if args.command == "proposal":
        if args.proposal_command == "create":
            proposal = append_chamber_proposal(
                args.shared_dir,
                args.support_type,
                " ".join(args.text),
                rationale=args.rationale,
                target=args.target,
            )
            meta = select_collab(args.shared_dir, args.target)
            refresh_chamber_files_from_disk(args.shared_dir / str(meta["id"]), meta)
            print(f"created {proposal['id']} for {proposal['collab_id']}")
            return 0
        if args.proposal_command == "list":
            print(render_proposals(args.shared_dir, args.target, args.limit))
            return 0
        if args.proposal_command == "status":
            print(render_proposal_status(args.shared_dir, args.target, args.proposal_id))
            return 0
    if args.command == "consent":
        if args.consent_command == "record":
            receipt = append_consent_receipt(
                args.shared_dir,
                args.proposal_id,
                args.actor,
                args.stance,
                note=args.note,
                target=args.target,
            )
            meta = select_collab(args.shared_dir, args.target)
            refresh_chamber_files_from_disk(args.shared_dir / str(meta["id"]), meta)
            print(f"recorded {receipt['id']} for {receipt['proposal_id']}")
            return 0
    if args.command == "note":
        note = append_steward_note(args.shared_dir, " ".join(args.text), target=args.target)
        print(f"queued {note['id']} for {note['collab_id']}")
        return 0
    if args.command == "intent":
        if args.intent_command == "set":
            intention = append_steward_intention(
                args.shared_dir,
                " ".join(args.text),
                target=args.target,
            )
            print(f"queued {intention['id']} for {intention['collab_id']}")
            return 0
        if args.intent_command == "clear":
            intention = clear_steward_intention(args.shared_dir, target=args.target)
            print(f"cleared active steward intention for {intention['collab_id']}")
            return 0
        if args.intent_command == "list":
            print(render_intentions(args.shared_dir, args.target, args.limit))
            return 0
    if args.command == "phase":
        if args.phase_command == "set":
            result = set_chamber_phase(args.shared_dir, args.phase, target=args.target)
            print(f"phase {result['phase']} ({result['phase_source']}) for {result['collab_id']}")
            return 0
        if args.phase_command == "clear":
            result = clear_chamber_phase(args.shared_dir, target=args.target)
            print(f"phase {result['phase']} ({result['phase_source']}) for {result['collab_id']}")
            return 0
    if args.command == "memory":
        if args.memory_command is None:
            print(render_memory(args.shared_dir, args.target))
            return 0
        if args.memory_command == "edit":
            if bool(args.json_text) == bool(args.file):
                parser.error("memory edit requires exactly one of JSON text or --file")
            text = args.file.read_text() if args.file else args.json_text
            payload = parse_memory_edit_payload(text)
            edit = append_memory_edit(args.shared_dir, payload, target=args.target)
            meta = select_collab(args.shared_dir, args.target)
            refresh_chamber_files_from_disk(args.shared_dir / str(meta["id"]), meta)
            print(f"set memory edit {edit['id']} for {edit['collab_id']}")
            return 0
        if args.memory_command == "clear":
            edit = clear_memory_edit(args.shared_dir, target=args.target)
            meta = select_collab(args.shared_dir, args.target)
            refresh_chamber_files_from_disk(args.shared_dir / str(meta["id"]), meta)
            print(f"cleared memory edit for {edit['collab_id']}")
            return 0
    if args.command == "history":
        print(render_history(args.shared_dir, args.target, args.limit))
        return 0
    if args.command == "reentry":
        meta = select_collab(args.shared_dir, args.target)
        path = chamber_paths(args.shared_dir / str(meta["id"]))["reentry"]
        if path.is_file():
            print(path.read_text())
        else:
            print("(no chamber re-entry note yet)")
        return 0
    if args.command == "log":
        print(render_log(args.shared_dir, args.target, args.limit))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
