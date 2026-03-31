#!/usr/bin/env python3
"""Run the same prompt through differently shaped reservoir states.

Experiment 1 asks a simple question:

Can the shared substrate produce distinct, repeatable generation stances when
the prompt is held constant but the underlying handle state is deliberately
shaped into different regimes first?

The script:
1. clones a baseline handle into three experiment handles
2. shapes each clone with a deterministic synthetic regime
3. runs the same prompt through the coupled Astrid server for each handle
4. captures the resulting text, generation digest, layer metrics, and a short
   pairwise similarity summary
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import websockets


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    memory_role: str
    hint_mode: str
    hint_decay: str


SCENARIOS = (
    ScenarioSpec(
        name="stable",
        description="Low-variance repeated drive that should feel anchored and coherent.",
        memory_role="stable",
        hint_mode="hold",
        hint_decay="slow",
    ),
    ScenarioSpec(
        name="expanding",
        description="Increasingly exploratory drive with sweeping phase changes and rising amplitude.",
        memory_role="expanding",
        hint_mode="rehearse",
        hint_decay="medium",
    ),
    ScenarioSpec(
        name="recovered",
        description="Brief disturbance followed by a cooling and settling sequence.",
        memory_role="recovered",
        hint_mode="rehearse",
        hint_decay="slow",
    ),
)


def stable_vectors(steps: int, input_dim: int) -> list[np.ndarray]:
    axis = np.linspace(-1.0, 1.0, input_dim, dtype=np.float32)
    base = np.tanh(0.75 * axis)
    vectors = []
    for step in range(steps):
        gain = 0.18 + 0.015 * math.sin(step / 3.0)
        ripple = 0.025 * np.sin((step + 1) * axis * math.pi)
        vectors.append(np.clip(gain * base + ripple, -1.0, 1.0).astype(np.float32))
    return vectors


def expanding_vectors(steps: int, input_dim: int) -> list[np.ndarray]:
    phase_axis = np.linspace(0.0, 2.0 * math.pi, input_dim, endpoint=False, dtype=np.float32)
    vectors = []
    denom = max(1, steps - 1)
    for step in range(steps):
        progress = step / denom
        gain = 0.22 + 0.45 * progress
        wave = np.sin(phase_axis + step * 0.45) + 0.45 * np.cos(2.0 * phase_axis - step * 0.25)
        vectors.append(np.tanh(gain * wave).astype(np.float32))
    return vectors


def recovered_vectors(steps: int, input_dim: int) -> list[np.ndarray]:
    phase_axis = np.linspace(0.0, 2.0 * math.pi, input_dim, endpoint=False, dtype=np.float32)
    burst_steps = max(3, steps // 3)
    vectors = []
    for step in range(steps):
        if step < burst_steps:
            gain = 0.75 - 0.08 * step
            wave = np.sign(np.sin(phase_axis * (1.0 + step * 0.15) + step * 0.7))
            vec = np.tanh(gain * wave)
        else:
            settle_progress = (step - burst_steps) / max(1, steps - burst_steps - 1)
            gain = 0.22 * (1.0 - 0.75 * settle_progress)
            wave = 0.6 * np.sin(phase_axis + step * 0.12) + 0.25 * np.cos(phase_axis * 0.5 - step * 0.08)
            vec = np.tanh(gain * wave)
        vectors.append(vec.astype(np.float32))
    return vectors


def build_vectors(spec: ScenarioSpec, steps: int, input_dim: int) -> list[np.ndarray]:
    if spec.name == "stable":
        return stable_vectors(steps, input_dim)
    if spec.name == "expanding":
        return expanding_vectors(steps, input_dim)
    if spec.name == "recovered":
        return recovered_vectors(steps, input_dim)
    raise ValueError(f"unknown scenario: {spec.name}")


def scenario_meta(spec: ScenarioSpec, step: int, steps: int) -> dict[str, Any]:
    progress = 0.0 if steps <= 1 else step / (steps - 1)
    return {
        "source": "same_prompt_different_state_experiment",
        "operation": "scenario_shaping_tick",
        "source_timestamp": round(time.time(), 3),
        "source_event_id": f"{spec.name}:{step:03d}",
        "scenario": spec.name,
        "input_source": "synthetic_regime",
        "projection": f"experimental_{spec.name}",
        "memory_role": spec.memory_role,
        "progress": round(progress, 4),
        "rehearsal_hint": {
            "mode": spec.hint_mode,
            "decay_profile": spec.hint_decay,
            "reason": spec.description,
        },
    }


def build_handle_name(prefix: str, scenario_name: str, run_id: str) -> str:
    safe_prefix = prefix.replace(" ", "_").replace("/", "_")
    return f"{safe_prefix}_{scenario_name}_{run_id}"


def tokenize_words(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {word for word in cleaned.split() if word}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = tokenize_words(text_a)
    b = tokenize_words(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def event_label(meta: dict[str, Any]) -> str:
    event_id = meta.get("event_id")
    source_timestamp = meta.get("source_timestamp")
    if not event_id and not isinstance(source_timestamp, (int, float)):
        return "?"
    if isinstance(source_timestamp, (int, float)):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(source_timestamp)))
        if event_id:
            return f"{event_id} @ {stamp}"
        return stamp
    return str(event_id)


async def send(ws, msg: dict[str, Any]) -> dict[str, Any]:
    await ws.send(json.dumps(msg))
    raw = await ws.recv()
    return json.loads(raw)


async def ensure_handle(ws, name: str, entity: str) -> None:
    listing = await send(ws, {"type": "list_handles"})
    names = {h["name"] for h in listing.get("handles", [])}
    if name not in names:
        result = await send(ws, {"type": "create_handle", "name": name, "entity": entity})
        if result.get("type") == "error":
            raise RuntimeError(result["message"])


async def clone_handle(ws, source: str, target: str, entity: str = "experiment") -> None:
    pulled = await send(ws, {"type": "pull_state", "name": source})
    if pulled.get("type") == "error":
        raise RuntimeError(f"could not pull baseline handle '{source}': {pulled['message']}")
    await ensure_handle(ws, target, entity)
    pushed = await send(
        ws,
        {
            "type": "push_state",
            "name": target,
            "h1": pulled["h1"],
            "h2": pulled["h2"],
            "h3": pulled["h3"],
            "tick_delta": 0,
            "meta": {
                "source": "same_prompt_different_state_experiment",
                "operation": "baseline_clone",
                "from_handle": source,
                "baseline_handle": source,
            },
        },
    )
    if pushed.get("type") == "error":
        raise RuntimeError(f"could not clone '{source}' into '{target}': {pushed['message']}")


async def shape_handle(ws, handle_name: str, spec: ScenarioSpec, steps: int, input_dim: int) -> None:
    vectors = build_vectors(spec, steps, input_dim)
    for step, vec in enumerate(vectors):
        result = await send(
            ws,
            {
                "type": "tick",
                "name": handle_name,
                "input": vec.astype(float).tolist(),
                "meta": scenario_meta(spec, step, steps),
            },
        )
        if result.get("type") == "error":
            raise RuntimeError(f"failed shaping '{handle_name}' at step {step}: {result['message']}")


def request_coupled_completion(
    llm_url: str,
    handle_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": "coupled-astrid",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reservoir_handle": handle_name,
    }
    req = urllib.request.Request(
        f"{llm_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def summarize_trajectory(outputs: list[float]) -> dict[str, float | str]:
    if not outputs:
        return {"trend": "idle", "latest": 0.0, "delta": 0.0, "spread": 0.0}
    latest = float(outputs[-1])
    if len(outputs) < 2:
        return {"trend": "warming_up", "latest": latest, "delta": 0.0, "spread": 0.0}
    delta = latest - float(outputs[0])
    if delta > 0.01:
        trend = "rising"
    elif delta < -0.01:
        trend = "falling"
    else:
        trend = "steady"
    spread = max(outputs) - min(outputs)
    return {"trend": trend, "latest": latest, "delta": delta, "spread": spread}


async def collect_result(ws, spec: ScenarioSpec, handle_name: str, output_text: str) -> dict[str, Any]:
    read = await send(ws, {"type": "read_state", "name": handle_name})
    layers = await send(ws, {"type": "layer_metrics", "name": handle_name})
    trajectory = await send(ws, {"type": "trajectory", "name": handle_name, "last_n": 12})
    return {
        "scenario": spec.name,
        "description": spec.description,
        "handle": handle_name,
        "output_text": output_text,
        "read_state": read,
        "layer_metrics": layers.get("layers", []),
        "trajectory": summarize_trajectory(trajectory.get("outputs", [])),
        "generation_meta": read.get("last_generation_meta") or read.get("last_live_meta") or {},
        "feeder_meta": read.get("last_feeder_meta") or {},
    }


def build_pairwise(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            meta_a = a.get("generation_meta", {})
            meta_b = b.get("generation_meta", {})
            pairs.append({
                "pair": f"{a['scenario']} vs {b['scenario']}",
                "word_jaccard": round(jaccard_similarity(a["output_text"], b["output_text"]), 4),
                "y1_delta": round(abs(float(meta_a.get("reservoir_readout", {}).get("y1_final", 0.0)) - float(meta_b.get("reservoir_readout", {}).get("y1_final", 0.0))), 4),
                "y2_delta": round(abs(float(meta_a.get("reservoir_readout", {}).get("y2_final", 0.0)) - float(meta_b.get("reservoir_readout", {}).get("y2_final", 0.0))), 4),
                "y3_delta": round(abs(float(meta_a.get("reservoir_readout", {}).get("y3_final", 0.0)) - float(meta_b.get("reservoir_readout", {}).get("y3_final", 0.0))), 4),
                "hash_a": meta_a.get("response_sha256_12"),
                "hash_b": meta_b.get("response_sha256_12"),
            })
    return pairs


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Same Prompt, Different State",
        "",
        f"Captured: {report['captured_at']}",
        f"Baseline handle: `{report['baseline_handle']}`",
        f"Prompt: {report['prompt']!r}",
        "",
        "## Quick Read",
        "",
        "This run clones one baseline handle into three differently shaped regimes, then asks the exact same prompt through coupled generation.",
        "",
        "| Scenario | Handle | Tokens | Coupling | y1 | y2 | y3 | Trend | Event | Response Hash |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for result in report["results"]:
        meta = result.get("generation_meta", {})
        readout = meta.get("reservoir_readout", {})
        lines.append(
            f"| {result['scenario']} | `{result['handle']}` | "
            f"{meta.get('generated_tokens', '?')} | "
            f"{meta.get('coupling_strength', '?')} | "
            f"{readout.get('y1_final', '?')} | "
            f"{readout.get('y2_final', '?')} | "
            f"{readout.get('y3_final', '?')} | "
            f"{result['trajectory']['trend']} | "
            f"{event_label(meta)} | "
            f"{meta.get('response_sha256_12', '?')} |"
        )
    lines.extend([
        "",
        "## Pairwise Comparison",
        "",
        "| Pair | Word Jaccard | Δy1 | Δy2 | Δy3 |",
        "|---|---:|---:|---:|---:|",
    ])
    for pair in report["pairwise"]:
        lines.append(
            f"| {pair['pair']} | {pair['word_jaccard']} | {pair['y1_delta']} | {pair['y2_delta']} | {pair['y3_delta']} |"
        )
    for result in report["results"]:
        meta = result.get("generation_meta", {})
        readout = meta.get("reservoir_readout", {})
        lines.extend([
            "",
            f"## {result['scenario'].title()}",
            "",
            result["description"],
            "",
            f"- Handle: `{result['handle']}`",
            f"- Tokens: {meta.get('generated_tokens', '?')}",
            f"- Coupling: {meta.get('coupling_strength', '?')}",
            f"- Generation event: {event_label(meta)}",
            f"- Readout: y1={readout.get('y1_final', '?')}, y2={readout.get('y2_final', '?')}, y3={readout.get('y3_final', '?')}",
            f"- Trajectory: trend={result['trajectory']['trend']}, latest={result['trajectory']['latest']:+.4f}, delta={result['trajectory']['delta']:+.4f}, spread={result['trajectory']['spread']:.4f}",
            f"- Preview: {meta.get('response_preview', '')!r}",
            "",
            "### Output",
            "",
            result["output_text"] or "(empty)",
            "",
            "### Layer Metrics",
            "",
            "| Layer | h_norm | entropy | target | saturation | rho |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for layer in result["layer_metrics"]:
            lines.append(
                f"| {layer['name']} | {layer.get('h_norm')} | {layer.get('entropy')} | "
                f"{layer.get('entropy_target')} | {layer.get('saturation')} | {layer.get('rho')} |"
            )
    lines.append("")
    return "\n".join(lines)


async def run_experiment(args) -> dict[str, Any]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []

    async with websockets.connect(args.ws_url) as ws:
        baseline_read = await send(ws, {"type": "read_state", "name": args.baseline_handle})
        if baseline_read.get("type") == "error":
            raise RuntimeError(
                f"baseline handle '{args.baseline_handle}' is unavailable: {baseline_read['message']}"
            )

        created_handles: list[str] = []
        for spec in SCENARIOS:
            handle_name = build_handle_name(args.prefix, spec.name, run_id)
            await clone_handle(ws, args.baseline_handle, handle_name, entity="experiment")
            created_handles.append(handle_name)
            await shape_handle(ws, handle_name, spec, args.shaping_steps, args.input_dim)
            output_text = await asyncio.to_thread(
                request_coupled_completion,
                args.llm_url,
                handle_name,
                args.prompt,
                args.temperature,
                args.max_tokens,
                args.system_prompt,
            )
            results.append(await collect_result(ws, spec, handle_name, output_text))

        if args.cleanup:
            for handle_name in created_handles:
                await send(ws, {"type": "destroy_handle", "name": handle_name})

    return {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_handle": args.baseline_handle,
        "prompt": args.prompt,
        "results": results,
        "pairwise": build_pairwise(results),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Experiment 1: same prompt, different shaped reservoir states")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:7881", help="Reservoir service WebSocket URL")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8090", help="Coupled Astrid server base URL")
    ap.add_argument("--baseline-handle", default="astrid", help="Existing handle to clone before shaping")
    ap.add_argument("--prefix", default="exp1", help="Prefix for experiment handles")
    ap.add_argument("--prompt", required=True, help="Prompt to run identically through each shaped handle")
    ap.add_argument("--system-prompt", default=None, help="Optional system prompt to prepend")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ap.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens per scenario")
    ap.add_argument("--shaping-steps", type=int, default=24, help="Synthetic shaping ticks per scenario")
    ap.add_argument("--input-dim", type=int, default=32, help="Reservoir input dimension used by the service")
    ap.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format")
    ap.add_argument("--output", default=None, help="Optional file path to write the report")
    ap.add_argument("--cleanup", action="store_true", help="Destroy experiment handles after the run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_experiment(args))
    rendered = render_markdown(report) if args.format == "markdown" else json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
        print(f"wrote {output_path}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
