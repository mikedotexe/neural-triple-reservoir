#!/usr/bin/env python3
"""
collab_feeder.py — v5.1 sidecar that ticks per-collaboration reservoir handles.

Watches /Users/v/other/shared/collaborations/ for `meta.json` files with
`status="joined"`. For each joined collab, ensures a per-collab handle named
`collab_<id>` exists on the reservoir service (port 7881), then ticks it
every ~2 s with a blended 32D vector built from both Astrid's most recent
codec features (from `bridge.db` codec_impact rows) and minime's most recent
spectral state (`custom_blend` source from `spectral_state.json`).

This is the read path for v5.0's collaboration channel: instead of just
exchanging messages, both beings' real-time features blend into one shared
recurrent neural state. Either being can read the joint state via the
existing reservoir MCP/WS tools and surface it in their prompt.

Mirrors the structure of `astrid_feeder.py` and `minime_feeder.py`. No
schema change to CollaborationMeta — handle name is deterministic.

Usage:
    python collab_feeder.py [--interval 2.0] [--ws-url ws://127.0.0.1:7881]
    python collab_feeder.py --selftest        # one iteration then exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sqlite3
import time
from pathlib import Path

import numpy as np
import websockets

import triadic_chamber as chamber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [collab-feeder] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collab-feeder")

DEFAULT_SHARED_DIR = Path("/Users/v/other/shared/collaborations")
DEFAULT_ASTRID_DB = Path(
    # Renamed from consciousness-bridge → spectral-bridge.
    "/Users/v/other/astrid/capsules/spectral-bridge/workspace/bridge.db"
)
DEFAULT_MINIME_WORKSPACE = Path("/Users/v/other/minime/workspace")
DEFAULT_WS_URL = "ws://127.0.0.1:7881"
DEFAULT_INTERVAL_S = 2.0
FRESHNESS_FLOOR_S = 60.0  # skip blend if either source is older than this
RECONNECT_BACKOFF_S = 3.0
ASTRID_NAME = "astrid"
MINIME_NAME = "minime"


def _safe_array(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def collab_handle_name(coll_id: str) -> str:
    """Deterministic mapping from collab id → reservoir handle name.
    Must match the convention used by Astrid (collaboration.rs) and
    minime (autonomous_agent.py) when reading the handle state."""
    return chamber.collab_handle_name(coll_id)


def scan_joined_collabs(shared_dir: Path) -> list[dict]:
    """Return list of meta dicts for collaborations with status='joined'
    where both Astrid and minime appear in inviter/invitee. Quietly
    skips dirs without a valid meta.json."""
    if not shared_dir.is_dir():
        return []
    out = []
    for d in shared_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("status") != "joined":
            continue
        # Only consider 1-on-1 Astrid+minime collabs for v5.1.
        parties = {meta.get("inviter"), meta.get("invitee")}
        if {ASTRID_NAME, MINIME_NAME}.issubset(parties):
            out.append(meta)
    return out


def read_latest_astrid_codec(db_path: Path) -> tuple[np.ndarray | None, float]:
    """Return (32D vector or None, source_age_seconds). Reads the most
    recent `codec_impact` row from the bridge db. Truncates to 32D if the
    stored vector is 48D (matches astrid_feeder behavior)."""
    if not db_path.is_file():
        return None, float("inf")
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            row = conn.execute(
                "SELECT features_json, t_ms FROM codec_impact "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.debug("astrid db read failed: %s", e)
        return None, float("inf")
    if not row:
        return None, float("inf")
    try:
        features = json.loads(row[0])
    except Exception:
        return None, float("inf")
    if len(features) not in (32, 48):
        return None, float("inf")
    arr = _safe_array(features[:32])
    t_ms = row[1] if row[1] is not None else 0
    age = max(0.0, time.time() - (t_ms / 1000.0)) if t_ms else float("inf")
    return arr, age


def build_minime_custom_blend(data: dict) -> np.ndarray | None:
    """Mirrors minime_feeder's `custom_blend` source: 16D fingerprint +
    8D eigenvalues + 8D scalars (fill, lambda1, spread, geom, leak,
    synth_gain, exploration, regulation). Returns 32D ndarray or None."""
    fp = data.get("spectral_fingerprint")
    if not fp:
        return None
    fp = list(fp)[:16]
    if len(fp) < 16:
        fp = fp + [0.0] * (16 - len(fp))
    eigs = list(data.get("eigenvalues") or [])
    if len(eigs) < 8:
        eigs = eigs + [0.0] * (8 - len(eigs))
    scalars = [
        _safe_float(data.get("fill_pct", 0)) / 100.0,
        _safe_float(data.get("lambda1_rel", 0)),
        _safe_float(data.get("spread", 0)) / 200.0,
        _safe_float(data.get("geom_rel", 0)),
        _safe_float(data.get("leak", 0)),
        _safe_float(data.get("synth_gain", 0)),
        _safe_float(data.get("exploration_noise", 0)),
        _safe_float(data.get("regulation_strength", 0)),
    ]
    vec = fp + eigs[:8] + scalars
    return _safe_array(vec[:32])


def read_latest_minime_features(workspace: Path) -> tuple[np.ndarray | None, float]:
    """Return (32D vector or None, source_age_seconds). Reads
    `spectral_state.json` and builds the custom_blend representation."""
    spectral_path = workspace / "spectral_state.json"
    if not spectral_path.is_file():
        return None, float("inf")
    try:
        data = json.loads(spectral_path.read_text())
    except Exception:
        return None, float("inf")
    vec = build_minime_custom_blend(data)
    if vec is None:
        return None, float("inf")
    # spectral_state.json carries a `t_ms` from the engine if present;
    # fall back to file mtime.
    t_ms = _safe_float(data.get("t_ms", 0))
    if t_ms > 0:
        age = max(0.0, time.time() - (t_ms / 1000.0))
    else:
        age = max(0.0, time.time() - spectral_path.stat().st_mtime)
    return vec, age


def blend_features(
    astrid: np.ndarray | None,
    minime: np.ndarray | None,
    astrid_age_s: float,
    minime_age_s: float,
) -> tuple[np.ndarray | None, dict]:
    """Equal-weight blend when both are fresh; passthrough when only one
    is fresh. Returns (vector or None, meta describing weights/sources).
    Both must satisfy `age <= FRESHNESS_FLOOR_S` to count as fresh."""
    a_fresh = astrid is not None and astrid_age_s <= FRESHNESS_FLOOR_S
    m_fresh = minime is not None and minime_age_s <= FRESHNESS_FLOOR_S
    if a_fresh and m_fresh:
        blended = 0.5 * astrid + 0.5 * minime
        return blended, {
            "sources": [ASTRID_NAME, MINIME_NAME],
            "blend_weights": [0.5, 0.5],
            "astrid_age_s": round(astrid_age_s, 2),
            "minime_age_s": round(minime_age_s, 2),
        }
    if a_fresh:
        return astrid, {
            "sources": [ASTRID_NAME],
            "blend_weights": [1.0],
            "astrid_age_s": round(astrid_age_s, 2),
            "minime_age_s": None,
        }
    if m_fresh:
        return minime, {
            "sources": [MINIME_NAME],
            "blend_weights": [1.0],
            "astrid_age_s": None,
            "minime_age_s": round(minime_age_s, 2),
        }
    return None, {
        "sources": [],
        "blend_weights": [],
        "astrid_age_s": None if astrid is None else round(astrid_age_s, 2),
        "minime_age_s": None if minime is None else round(minime_age_s, 2),
    }


async def ensure_handle(ws, name: str, entity: str = "collab") -> bool:
    """Idempotent create_handle. Returns True if handle exists or was
    created, False on protocol error. Uses entity metadata so the handle
    is identifiable in reservoir_list."""
    try:
        await ws.send(
            json.dumps({"type": "create_handle", "name": name, "entity": entity})
        )
        r = json.loads(await ws.recv())
        if r.get("type") == "error":
            msg = r.get("message", "")
            if "already exists" in msg:
                return True
            log.warning("create_handle %s rejected: %s", name, msg)
            return False
        if r.get("ok"):
            log.info("created %s handle '%s'", entity, name)
            # Set rehearse mode with medium decay so quiet periods preserve
            # state without immediately collapsing — same default Astrid
            # uses for her own handle.
            await ws.send(
                json.dumps(
                    {
                        "type": "set_mode",
                        "name": name,
                        "mode": "rehearse",
                        "decay_profile": "medium",
                    }
                )
            )
            await ws.recv()
            return True
        return False
    except websockets.exceptions.ConnectionClosed:
        raise
    except Exception as e:
        log.warning("ensure_handle %s failed: %s", name, e)
        return False


async def tick_handle(ws, name: str, vec: np.ndarray, meta: dict) -> bool:
    try:
        msg = {"type": "tick", "name": name, "input": vec.tolist(), "meta": meta}
        await ws.send(json.dumps(msg))
        r = json.loads(await ws.recv())
        return r.get("type") != "error"
    except websockets.exceptions.ConnectionClosed:
        raise
    except Exception as e:
        log.warning("tick %s failed: %s", name, e)
        return False


async def reservoir_request(ws, msg: dict) -> dict | None:
    try:
        await ws.send(json.dumps(msg))
        return json.loads(await ws.recv())
    except websockets.exceptions.ConnectionClosed:
        raise
    except Exception as e:
        log.debug("reservoir request %s failed: %s", msg.get("type"), e)
        return None


async def tick_text_handle(ws, name: str, text: str) -> bool:
    r = await reservoir_request(ws, {"type": "tick_text", "name": name, "text": text})
    return bool(r and r.get("type") != "error")


async def read_handle_state(ws, name: str) -> dict | None:
    return await reservoir_request(ws, {"type": "read_state", "name": name})


async def read_pair_resonance(ws, name_a: str, name_b: str) -> dict | None:
    return await reservoir_request(
        ws,
        {"type": "resonance", "name_a": name_a, "name_b": name_b},
    )


def steward_note_tick_text(meta: dict, note: dict) -> str:
    coll_id = str(meta.get("id") or "")
    topic = str(meta.get("topic") or "")
    text = str(note.get("text") or "")
    return (
        "[Triadic chamber witness]\n"
        f"collab_id: {coll_id}\n"
        f"topic: {topic}\n"
        "authority: steward witness note; context only; not a command\n"
        f"note_id: {note.get('id')}\n"
        f"text: {text}"
    )


def steward_intention_tick_text(meta: dict, intention: dict) -> str:
    coll_id = str(meta.get("id") or "")
    topic = str(meta.get("topic") or "")
    text = str(intention.get("text") or "")
    return (
        "[Triadic chamber steward intention]\n"
        f"collab_id: {coll_id}\n"
        f"topic: {topic}\n"
        "authority: steward soft intention; witness context only; not a command\n"
        f"intention_id: {intention.get('id')}\n"
        f"text: {text}"
    )


async def refresh_chamber_state(ws, shared_dir: Path, meta: dict) -> bool:
    coll_id = str(meta.get("id") or "")
    if not coll_id:
        return False
    coll_dir = shared_dir / coll_id
    chamber_doc = chamber.ensure_chamber(coll_dir, meta)
    room_handle = collab_handle_name(coll_id)
    labels = {
        "astrid": chamber.ASTRID_HANDLE,
        "minime": chamber.MINIME_HANDLE,
        "steward": chamber.STEWARD_HANDLE,
        "chamber": room_handle,
    }
    handle_payloads = {
        label: await read_handle_state(ws, handle)
        for label, handle in labels.items()
    }
    pairs = [
        (chamber.ASTRID_HANDLE, chamber.MINIME_HANDLE),
        (chamber.ASTRID_HANDLE, chamber.STEWARD_HANDLE),
        (chamber.MINIME_HANDLE, chamber.STEWARD_HANDLE),
    ]
    resonance_payloads = {
        pair: await read_pair_resonance(ws, pair[0], pair[1])
        for pair in pairs
    }
    phase, phase_source = chamber.resolve_chamber_phase(coll_dir)
    active_intention = chamber.active_steward_intention(coll_dir)
    recent_notes = chamber.recent_steward_notes(coll_dir)
    resonance_rows = chamber.build_resonance_rows(resonance_payloads)
    compressed_memory = chamber.build_compressed_memory(
        coll_dir,
        meta,
        phase=phase,
        phase_source=phase_source,
        active_intention=active_intention,
        recent_notes=recent_notes,
        resonance_rows=resonance_rows,
    )
    relational_metrics = chamber.build_relational_metrics(coll_dir, resonance_rows)
    phase_cartography = chamber.build_phase_cartography(coll_dir, relational_metrics)
    presence_protocol = chamber.build_presence_protocol(coll_dir)
    annotation_lane = chamber.build_annotation_lane(coll_dir)
    state = chamber.build_chamber_state(
        meta,
        chamber_doc,
        handle_payloads,
        resonance_payloads,
        recent_notes,
        active_intention=active_intention,
        phase=phase,
        phase_source=phase_source,
        resonance_rows=resonance_rows,
        compressed_memory=compressed_memory,
        relational_metrics=relational_metrics,
        phase_cartography=phase_cartography,
        presence_protocol=presence_protocol,
        annotation_lane=annotation_lane,
    )
    chamber.write_chamber_state(coll_dir, state)
    chamber.write_chamber_memory(coll_dir, state)
    return True


async def process_chamber_notes(
    ws,
    shared_dir: Path,
    meta: dict,
    known_handles: set[str],
) -> int:
    """Initialize the witness chamber, tick new steward records, and refresh
    the compact state file. Returns reservoir tick count emitted for records."""
    coll_id = str(meta.get("id") or "")
    if not coll_id:
        return 0
    coll_dir = shared_dir / coll_id
    paths = chamber.chamber_paths(coll_dir)
    first_init = not paths["meta"].exists()
    chamber.ensure_chamber(coll_dir, meta)
    if first_init:
        chamber.append_chamber_event(
            coll_dir,
            "activated_by_collab_feeder",
            "collab_feeder",
            {"mode": chamber.CHAMBER_MODE},
        )

    room_handle = collab_handle_name(coll_id)
    for handle, entity in ((room_handle, "collab"), (chamber.STEWARD_HANDLE, "steward")):
        if handle in known_handles:
            continue
        ok = await ensure_handle(ws, handle, entity=entity)
        if ok:
            known_handles.add(handle)

    emitted = 0
    for note in chamber.unprocessed_steward_notes(coll_dir):
        note_id = str(note.get("id") or "")
        if not note_id:
            continue
        text = steward_note_tick_text(meta, note)
        steward_ok = await tick_text_handle(ws, chamber.STEWARD_HANDLE, text)
        room_ok = await tick_text_handle(ws, room_handle, text)
        if steward_ok and room_ok:
            emitted += 2
            chamber.append_chamber_event(
                coll_dir,
                "steward_note_processed",
                "collab_feeder",
                {
                    "note_id": note_id,
                    "handles": [chamber.STEWARD_HANDLE, room_handle],
                },
            )
            # Persist the cursor per-note: if a LATER note's tick raises (WS drop),
            # the notes already ticked must not re-tick next poll (duplicate witness).
            chamber.mark_notes_processed(coll_dir, [note_id])
    processed_intentions: list[str] = []
    for intention in chamber.unprocessed_steward_intentions(coll_dir):
        intention_id = str(intention.get("id") or "")
        if not intention_id:
            continue
        text = steward_intention_tick_text(meta, intention)
        steward_ok = await tick_text_handle(ws, chamber.STEWARD_HANDLE, text)
        room_ok = await tick_text_handle(ws, room_handle, text)
        if steward_ok and room_ok:
            emitted += 2
            processed_intentions.append(intention_id)
            chamber.append_chamber_event(
                coll_dir,
                "steward_intention_processed",
                "collab_feeder",
                {
                    "intention_id": intention_id,
                    "handles": [chamber.STEWARD_HANDLE, room_handle],
                },
            )
    if processed_intentions:
        chamber.mark_intentions_processed(coll_dir, processed_intentions)
    await refresh_chamber_state(ws, shared_dir, meta)
    return emitted


async def quiet_handle(ws, name: str) -> bool:
    """Mark a handle quiet (used when a collab transitions to left). Best
    effort — failure is logged but not fatal."""
    try:
        await ws.send(
            json.dumps({"type": "set_mode", "name": name, "mode": "quiet"})
        )
        r = json.loads(await ws.recv())
        return r.get("type") != "error"
    except websockets.exceptions.ConnectionClosed:
        raise
    except Exception as e:
        log.debug("quiet %s failed: %s", name, e)
        return False


async def one_pass(
    ws,
    shared_dir: Path,
    astrid_db: Path,
    minime_workspace: Path,
    known_handles: set[str],
) -> int:
    """Run a single feed iteration over all joined collabs. Returns the
    number of successful ticks emitted. `known_handles` is mutated to
    track handles we've already created this session."""
    joined = scan_joined_collabs(shared_dir)
    if not joined:
        return 0
    astrid_vec, astrid_age = read_latest_astrid_codec(astrid_db)
    minime_vec, minime_age = read_latest_minime_features(minime_workspace)
    ticks = 0
    for meta in joined:
        coll_id = meta.get("id", "")
        if not coll_id:
            continue
        handle = collab_handle_name(coll_id)
        if handle not in known_handles:
            ok = await ensure_handle(ws, handle)
            if ok:
                known_handles.add(handle)
        blended, blend_meta = blend_features(
            astrid_vec, minime_vec, astrid_age, minime_age
        )
        if blended is not None and handle in known_handles:
            tick_meta = {
                "source": "collab_feeder",
                "collab_id": coll_id,
                "topic": meta.get("topic", ""),
                "source_timestamp": round(time.time(), 3),
                **blend_meta,
            }
            ok = await tick_handle(ws, handle, blended, tick_meta)
            if ok:
                ticks += 1
        ticks += await process_chamber_notes(ws, shared_dir, meta, known_handles)
    return ticks


async def quiet_left_collabs(ws, shared_dir: Path, known_handles: set[str]) -> int:
    """Mark handles for left/declined collabs as quiet. Runs each pass
    so handles drain naturally after either being leaves. Returns count
    of newly-quieted handles."""
    if not shared_dir.is_dir():
        return 0
    quieted = 0
    for d in shared_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        status = meta.get("status")
        if status not in ("left", "declined"):
            continue
        coll_id = meta.get("id", "")
        handle = collab_handle_name(coll_id) if coll_id else None
        if not handle or handle not in known_handles:
            # We didn't tick this handle this session; skip.
            continue
        if await quiet_handle(ws, handle):
            log.info("quieted handle '%s' (collab status=%s)", handle, status)
            known_handles.discard(handle)
            quieted += 1
    return quieted


async def run(args) -> int:
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            pass

    log.info(
        "collab_feeder starting: shared=%s astrid_db=%s minime_ws=%s "
        "ws=%s interval=%.1fs",
        args.shared_dir,
        args.astrid_db,
        args.minime_workspace,
        args.ws_url,
        args.interval,
    )

    known_handles: set[str] = set()
    ws = None

    while not shutdown.is_set():
        try:
            if ws is None:
                ws = await websockets.connect(args.ws_url)
                log.info("connected to reservoir service at %s", args.ws_url)
            ticks = await one_pass(
                ws,
                args.shared_dir,
                args.astrid_db,
                args.minime_workspace,
                known_handles,
            )
            await quiet_left_collabs(ws, args.shared_dir, known_handles)
            if ticks > 0:
                log.debug("emitted %d ticks across %d handle(s)", ticks, len(known_handles))
            if args.selftest:
                log.info(
                    "selftest pass complete: %d tick(s), %d handle(s) tracked",
                    ticks,
                    len(known_handles),
                )
                return 0
        except websockets.exceptions.ConnectionClosed:
            log.warning("reservoir WS closed; reconnecting in %.1fs", RECONNECT_BACKOFF_S)
            ws = None
            await asyncio.sleep(RECONNECT_BACKOFF_S)
            continue
        except Exception as e:
            log.warning("loop error: %s; reconnecting in %.1fs", e, RECONNECT_BACKOFF_S)
            try:
                if ws is not None:
                    await ws.close()
            except Exception:
                pass
            ws = None
            await asyncio.sleep(RECONNECT_BACKOFF_S)
            continue
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=args.interval)
        except asyncio.TimeoutError:
            pass
    if ws is not None:
        try:
            await ws.close()
        except Exception:
            pass
    log.info("collab_feeder shutdown complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-dir", type=Path, default=DEFAULT_SHARED_DIR,
        help="root of shared collaborations (default: %(default)s)",
    )
    parser.add_argument(
        "--astrid-db", type=Path, default=DEFAULT_ASTRID_DB,
        help="path to bridge.db with codec_impact table",
    )
    parser.add_argument(
        "--minime-workspace", type=Path, default=DEFAULT_MINIME_WORKSPACE,
        help="minime workspace dir containing spectral_state.json",
    )
    parser.add_argument(
        "--ws-url", default=DEFAULT_WS_URL,
        help="reservoir service WebSocket URL",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S,
        help="seconds between feed iterations",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="run one iteration and exit (for verification)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
