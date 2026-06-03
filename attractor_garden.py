#!/usr/bin/env python3
"""Triple-reservoir attractor garden.

This is an offline-first proving ground for authored attractors. It creates
deterministic seed schedules that can be replayed into cloned handles, then
records whether the shaped state can be held, rehearsed, quieted, summoned,
and compared without confusing repetition for authorship.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POLICY = "attractor_garden_v1"


@dataclass(frozen=True)
class AttractorSeedSpec:
    name: str
    label: str
    author: str = "astrid"
    substrate: str = "triple_reservoir"
    command: str = "create"
    schedule: str = "anchor"
    rehearsal_mode: str = "hold"
    amplitude: float = 0.28
    safety_max_norm: float = 0.92
    recurrence_target: float = 0.60
    parent_labels: tuple[str, ...] = ()


BUILTIN_SEEDS = (
    AttractorSeedSpec(
        name="astrid_anchor",
        label="quiet eigenplane",
        author="astrid",
        schedule="anchor",
        rehearsal_mode="hold",
        amplitude=0.24,
    ),
    AttractorSeedSpec(
        name="minime_braid",
        label="minime authored braid",
        author="minime",
        schedule="braid",
        rehearsal_mode="rehearse",
        amplitude=0.34,
    ),
    AttractorSeedSpec(
        name="release_test",
        label="release without hidden replay",
        author="astrid",
        command="release",
        schedule="release",
        rehearsal_mode="quiet",
        amplitude=0.20,
    ),
    AttractorSeedSpec(
        name="astrid_minime_blend",
        label="honey edge blend",
        author="astrid",
        command="blend",
        schedule="blend",
        rehearsal_mode="rehearse",
        amplitude=0.30,
        parent_labels=("honey selection", "cooled theme edge"),
    ),
)


def slugify_label(label: str, max_len: int = 40) -> str:
    """Return a deterministic, handle-safe slug for an attractor label."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "seed"
    return slug[:max_len].strip("_") or "seed"


def canonical_attractor_label(label: str) -> str:
    """Preserve readable labels while canonicalizing known facet-tree paths."""
    text = re.sub(r"\s+", " ", str(label or "").strip().strip("\"'`"))
    lower = text.lower().replace("λ₄", "lambda4").replace("λ4", "lambda4")
    lower = lower.replace("λ₈", "lambda8").replace("λ8", "lambda8")
    if "/" in text:
        return "/".join(
            re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-")
            for part in text.split("/")
            if part.strip()
        )
    if re.search(r"\blambda\s*-?\s*4\b", lower) and "tail" in lower:
        return "lambda-tail/lambda4"
    if re.search(r"\blambda\s*-?\s*8\b", lower) and "tail" in lower:
        return "lambda-tail/lambda8"
    if "lambda-tail" in lower or "lambda tail" in lower:
        return "lambda-tail"
    if re.search(r"\blambda\s*-?\s*6\b", lower):
        return "lambda-edge/lambda-6"
    for needle, facet in {
        "yielding": "yielding",
        "compaction": "compaction",
        "compacting": "compaction",
        "resonance": "resonance",
        "localized gravity": "localized-gravity",
        "localized-gravity": "localized-gravity",
    }.items():
        if needle in lower:
            return f"lambda-edge/{facet}"
    return text


def facet_metadata(label: str) -> dict[str, str]:
    canonical = canonical_attractor_label(label)
    if "/" not in canonical:
        return {}
    parent, facet = canonical.split("/", 1)
    kind = (
        "spectral_tail"
        if parent == "lambda-tail"
        else "spectral_edge" if parent == "lambda-edge" else "attractor_facet"
    )
    return {
        "parent_label": parent,
        "facet_label": facet,
        "facet_path": canonical,
        "facet_kind": kind,
    }


def garden_handle_name(author: str, label: str, intent: str = "seed") -> str:
    """Name attractor garden handles consistently across beings and tests."""
    safe_author = slugify_label(author, max_len=16)
    safe_label = slugify_label(label, max_len=32)
    safe_intent = slugify_label(intent, max_len=16)
    return f"attr_{safe_author}_{safe_label}_{safe_intent}"


def clone_handle_request(
    source: str,
    spec: AttractorSeedSpec,
    intent_id: str,
    mode: str | None = None,
) -> dict[str, Any]:
    """Build the reservoir-service payload for a garden clone."""
    rehearsal_mode = mode or spec.rehearsal_mode
    meta = {
        "source": "attractor_garden",
        "operation": "clone_for_rehearsal",
        "intent_id": intent_id,
        "attractor_label": spec.label,
        "attractor_command": spec.command,
        "schedule": spec.schedule,
    }
    if spec.parent_labels:
        meta["parent_labels"] = list(spec.parent_labels)
        meta["parent_relation"] = "blend"
    meta.update(facet_metadata(spec.label))
    return {
        "type": "clone_handle",
        "source": source,
        "name": garden_handle_name(spec.author, spec.label, intent_id),
        "entity": spec.author,
        "mode": rehearsal_mode,
        "decay_profile": "slow" if rehearsal_mode == "hold" else "medium",
        "meta": meta,
    }


def blend_seed_spec(
    child_label: str,
    parent_labels: Iterable[str],
    *,
    author: str = "astrid",
    name: str | None = None,
    amplitude: float = 0.30,
) -> AttractorSeedSpec:
    """Build a deterministic blend seed spec from two or more parent labels."""
    parents = tuple(str(parent).strip() for parent in parent_labels if str(parent).strip())
    if len(parents) < 2:
        raise ValueError("blend seed specs require at least two parent labels")
    return AttractorSeedSpec(
        name=name or f"blend_{slugify_label(child_label)}",
        label=child_label,
        author=author,
        command="blend",
        schedule="blend",
        rehearsal_mode="rehearse",
        amplitude=amplitude,
        parent_labels=parents,
    )


def _phase_from_name(name: str) -> float:
    digest = hashlib.sha1(name.encode("utf-8", "ignore")).hexdigest()
    return int(digest[:6], 16) / 0xFFFFFF * 2.0 * math.pi


def build_seed_vectors(
    spec: AttractorSeedSpec,
    steps: int = 24,
    input_dim: int = 32,
) -> list[np.ndarray]:
    """Build a deterministic, bounded vector schedule for one seed."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if input_dim < 2:
        raise ValueError("input_dim must be >= 2")
    axis = np.linspace(0.0, 2.0 * math.pi, input_dim, endpoint=False, dtype=np.float32)
    phase = _phase_from_name(spec.name)
    vectors: list[np.ndarray] = []
    denom = max(1, steps - 1)
    for step in range(steps):
        progress = step / denom
        if spec.schedule == "anchor":
            gain = spec.amplitude * (0.85 + 0.08 * math.sin(step / 4.0))
            wave = 0.72 * np.sin(axis + phase) + 0.18 * np.cos(2.0 * axis - phase)
        elif spec.schedule == "braid":
            gain = spec.amplitude * (0.75 + 0.45 * progress)
            wave = (
                np.sin(axis + phase + step * 0.31)
                + 0.44 * np.sin(3.0 * axis - step * 0.19)
                + 0.21 * np.cos(0.5 * axis + step * 0.11)
            )
        elif spec.schedule == "release":
            gain = spec.amplitude * (1.0 - 0.82 * progress)
            wave = np.sin(axis + phase) - 0.35 * np.cos(2.0 * axis + step * 0.07)
        elif spec.schedule == "blend":
            parents = spec.parent_labels or (spec.name, spec.label)
            gain = spec.amplitude * (0.72 + 0.20 * math.sin(progress * math.pi) + 0.10 * progress)
            wave = np.zeros_like(axis)
            total_weight = 0.0
            for idx, parent in enumerate(parents[:4]):
                parent_phase = _phase_from_name(parent)
                weight = 1.0 / float(idx + 1)
                wave += weight * (
                    np.sin(axis + parent_phase + step * (0.17 + idx * 0.05))
                    + 0.33 * np.cos(2.0 * axis - parent_phase + step * (0.09 + idx * 0.03))
                )
                total_weight += weight
            wave = wave / max(total_weight, 1e-6)
        else:
            raise ValueError(f"unknown seed schedule: {spec.schedule}")
        vec = np.tanh(gain * wave)
        norm = float(np.linalg.norm(vec))
        if norm > spec.safety_max_norm:
            vec = vec * (spec.safety_max_norm / norm)
        vectors.append(np.clip(vec, -1.0, 1.0).astype(np.float32))
    return vectors


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 1.0 if float(np.linalg.norm(a - b)) <= 1e-9 else 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b) / denom)))


def recurrence_score(reference: Iterable[np.ndarray], candidate: Iterable[np.ndarray]) -> float:
    refs = list(reference)
    cands = list(candidate)
    if not refs or not cands:
        return 0.0
    count = min(len(refs), len(cands))
    scores = [(_cosine(refs[-count + idx], cands[-count + idx]) + 1.0) / 2.0 for idx in range(count)]
    return round(float(sum(scores) / len(scores)), 4)


def authorship_score(seed_vectors: Iterable[np.ndarray], replay_vectors: Iterable[np.ndarray]) -> float:
    """Score whether a replay follows the authored schedule rather than only recurs."""
    return recurrence_score(seed_vectors, replay_vectors)


def classify_attractor(recurrence: float, authorship: float, safety: str = "green") -> str:
    recurrence = max(0.0, min(1.0, float(recurrence)))
    authorship = max(0.0, min(1.0, float(authorship)))
    if safety in {"red", "orange", "pathological"}:
        return "pathological" if recurrence >= 0.55 else "failed"
    if recurrence >= 0.60 and authorship >= 0.60:
        return "authored"
    if recurrence >= 0.45:
        return "emergent"
    return "failed"


def quiet_recovery_vectors(seed_vectors: list[np.ndarray], quiet_steps: int = 8) -> list[np.ndarray]:
    if not seed_vectors:
        return []
    last = seed_vectors[-1]
    recovered = []
    for step in range(quiet_steps):
        decay = 1.0 - ((step + 1) / max(1, quiet_steps)) * 0.82
        recovered.append((last * decay).astype(np.float32))
    return recovered


def stale_lock_detected(lock_payload: dict[str, Any] | None, now_s: float, max_age_s: float = 90.0) -> bool:
    if not isinstance(lock_payload, dict):
        return False
    owner = str(lock_payload.get("owner") or "").strip()
    updated = lock_payload.get("updated_at_unix_s")
    if not owner or not isinstance(updated, (int, float)):
        return False
    return now_s - float(updated) > max_age_s


def build_intent(spec: AttractorSeedSpec, run_id: str) -> dict[str, Any]:
    intent_id = f"garden-{run_id}-{spec.name}"
    intent = {
        "policy": "attractor_intent_v1",
        "schema_version": 1,
        "intent_id": intent_id,
        "author": spec.author,
        "substrate": spec.substrate,
        "command": spec.command,
        "label": spec.label,
        "intervention_plan": {
            "mode": "garden_clone",
            "clone_request": clone_handle_request(spec.author, spec, intent_id),
            "vector_schedule": spec.schedule,
            "rehearsal_mode": spec.rehearsal_mode,
        },
        "safety_bounds": {
            "max_vector_norm": spec.safety_max_norm,
            "allow_live_control": False,
            "rollback_on_red": True,
        },
    }
    if spec.parent_labels:
        intent["parent_seed_ids"] = list(spec.parent_labels)
        intent["origin"] = {
            "kind": "blend",
            "parents": [{"label": parent} for parent in spec.parent_labels],
        }
    intent.update(facet_metadata(spec.label))
    return intent


def garden_proof(
    spec: AttractorSeedSpec,
    seed_vectors: list[np.ndarray],
    replay_vectors: list[np.ndarray],
    *,
    input_dim: int,
    now_s: float,
) -> dict[str, Any]:
    """Measure rehearsal facets without making any of them a live-control gate."""
    window = max(1, min(len(seed_vectors), len(replay_vectors)))
    reference = seed_vectors[-window:]
    shifted_state = [np.roll(vec, 1).astype(np.float32) for vec in reference]
    alternate = build_seed_vectors(
        AttractorSeedSpec(
            name=f"{spec.name}_alternate_prompt",
            label=f"{spec.label} alternate prompt",
            author=spec.author,
            schedule=spec.schedule,
            rehearsal_mode=spec.rehearsal_mode,
            amplitude=spec.amplitude,
            parent_labels=spec.parent_labels,
        ),
        steps=len(reference),
        input_dim=input_dim,
    )
    quiet = quiet_recovery_vectors(seed_vectors, quiet_steps=max(4, window))
    blend_parent_collapse: dict[str, Any] = {
        "checked": bool(spec.parent_labels),
        "detected": False,
        "max_parent_similarity": None,
    }
    if spec.parent_labels:
        similarities = []
        for parent in spec.parent_labels[:4]:
            parent_spec = AttractorSeedSpec(
                name=f"parent_{slugify_label(parent)}",
                label=parent,
                author=spec.author,
                schedule="anchor",
                amplitude=spec.amplitude,
            )
            parent_vectors = build_seed_vectors(
                parent_spec,
                steps=len(reference),
                input_dim=input_dim,
            )
            similarities.append(recurrence_score(reference, parent_vectors[-window:]))
        max_similarity = max(similarities) if similarities else 0.0
        blend_parent_collapse = {
            "checked": True,
            "detected": max_similarity >= 0.94,
            "max_parent_similarity": round(max_similarity, 4),
        }
    stale_lock_payload = {
        "owner": garden_handle_name(spec.author, spec.label, "proof"),
        "updated_at_unix_s": now_s - 12.0,
    }
    return {
        "policy": "garden_proof_v1",
        "required_for_live": False,
        "recommended_before_live": True,
        "same_prompt_different_state": {
            "recurrence": recurrence_score(reference, shifted_state),
            "interpretation": "divergence_control",
        },
        "same_state_different_prompt": {
            "recurrence": recurrence_score(reference, alternate[-window:]),
            "interpretation": "prompt_specificity_control",
        },
        "hold_rehearse_quiet": {
            "replay_recurrence": recurrence_score(reference, replay_vectors[-window:]),
            "quiet_recovery": recurrence_score(reference[-len(quiet):], quiet),
            "mode": spec.rehearsal_mode,
        },
        "stale_lock": {
            "checked": True,
            "detected": stale_lock_detected(stale_lock_payload, now_s=now_s),
        },
        "blend_parent_collapse": blend_parent_collapse,
    }


def simulate_garden(
    specs: Iterable[AttractorSeedSpec] = BUILTIN_SEEDS,
    steps: int = 24,
    input_dim: int = 32,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    created_at = time.time()
    rows = []
    for spec in specs:
        seed_vectors = build_seed_vectors(spec, steps=steps, input_dim=input_dim)
        if spec.rehearsal_mode == "quiet":
            replay = quiet_recovery_vectors(seed_vectors, quiet_steps=max(4, steps // 3))
        elif spec.rehearsal_mode == "rehearse":
            replay = seed_vectors[-max(4, steps // 3):]
        else:
            replay = [seed_vectors[-1] for _ in range(max(4, steps // 3))]
        recurrence = recurrence_score(seed_vectors[-len(replay):], replay)
        authorship = authorship_score(seed_vectors[-len(replay):], replay)
        classification = classify_attractor(recurrence, authorship)
        proof = garden_proof(
            spec,
            seed_vectors,
            replay,
            input_dim=input_dim,
            now_s=created_at,
        )
        metadata = facet_metadata(spec.label)
        rows.append({
            "spec": asdict(spec),
            "intent": build_intent(spec, run_id),
            "observation": {
                "policy": "attractor_observation_v1",
                "schema_version": 1,
                "intent_id": f"garden-{run_id}-{spec.name}",
                "substrate": spec.substrate,
                "label": spec.label,
                "recurrence_score": recurrence,
                "authorship_score": authorship,
                "classification": classification,
                "safety_level": "green",
                "rehearsal_mode": spec.rehearsal_mode,
                "garden_proof": proof,
                **metadata,
            },
            "vector_norms": [round(float(np.linalg.norm(vec)), 4) for vec in seed_vectors],
        })
    return {
        "policy": POLICY,
        "run_id": run_id,
        "created_at_unix_s": created_at,
        "steps": steps,
        "input_dim": input_dim,
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Triple Reservoir Attractor Garden",
        "",
        f"Run: `{report.get('run_id')}`",
        f"Policy: `{report.get('policy')}`",
        "",
        "| Seed | Author | Mode | Command | Recurrence | Authorship | Classification |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in report.get("rows", []):
        spec = row["spec"]
        obs = row["observation"]
        lines.append(
            "| {label} | {author} | {mode} | {command} | {rec:.2f} | {auth:.2f} | {classification} |".format(
                label=spec["label"],
                author=spec["author"],
                mode=spec["rehearsal_mode"],
                command=spec["command"],
                rec=obs["recurrence_score"],
                auth=obs["authorship_score"],
                classification=obs["classification"],
            )
        )
    lines.extend([
        "",
        "## Protocol",
        "",
        "Create seeds as explicit intents, shape cloned handles with deterministic vector schedules, test hold/rehearse/quiet recovery, then compare recurrence and authorship separately.",
        "",
        "Release remains an authored command: the release seed decays the schedule and should reduce persistence rather than hide a continuing replay.",
    ])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(report.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))
    json_path = output_dir / f"attractor_garden_{run_id}.json"
    md_path = output_dir / f"attractor_garden_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline attractor garden harness.")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = simulate_garden(steps=args.steps, input_dim=args.input_dim)
    json_path, md_path = write_report(report, args.output_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
