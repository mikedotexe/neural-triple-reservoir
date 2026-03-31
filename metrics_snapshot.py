#!/usr/bin/env python3
"""Capture a newcomer-friendly snapshot of reservoir dynamics."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


def trend_label(outputs: list[float]) -> tuple[str, float, float]:
    if not outputs:
        return "idle", 0.0, 0.0
    latest = float(outputs[-1])
    if len(outputs) < 2:
        return "warming_up", latest, 0.0
    delta = latest - float(outputs[0])
    if delta > 0.01:
        return "rising", latest, delta
    if delta < -0.01:
        return "falling", latest, delta
    return "steady", latest, delta


def classify_layer(layer: dict) -> str:
    entropy = layer.get("entropy")
    target = layer.get("entropy_target")
    saturation = layer.get("saturation")
    if target is None or entropy is None:
        return "warming_up"
    if saturation is not None and saturation >= 0.20:
        return "high_saturation"
    delta = float(entropy) - float(target)
    if delta <= -0.08:
        return "below_target"
    if delta >= 0.08:
        return "above_target"
    return "near_target"


def resonance_label(corr: float | None) -> str:
    if corr is None:
        return "insufficient_data"
    if corr >= 0.50:
        return "aligned"
    if corr <= -0.30:
        return "divergent"
    return "independent"


def hint_status_label(status: str | None) -> str | None:
    mapping = {
        "none": None,
        "governing": "governing current mode",
        "present_not_adopted": "present, not governing",
        "present_not_governing": "present, not governing",
        "present_blocked_by_explicit_mode": "present, blocked by explicit mode",
    }
    return mapping.get(status or "none", status)


def preview_text(text: str | None, max_chars: int = 72) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def event_stamp(meta: dict | None) -> str | None:
    if not isinstance(meta, dict) or not meta:
        return None
    event_id = meta.get("event_id")
    source_timestamp = meta.get("source_timestamp")
    parts = []
    if event_id:
        parts.append(f"evt={event_id}")
    if isinstance(source_timestamp, (int, float)):
        parts.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(source_timestamp))))
    if not parts:
        return None
    if len(parts) == 2:
        return f"{parts[0]} @ {parts[1]}"
    return parts[0]


def generation_summary(meta: dict | None) -> str | None:
    if not isinstance(meta, dict) or meta.get("source") != "coupled_astrid_server":
        return None
    tokens = meta.get("generated_tokens")
    elapsed = meta.get("elapsed_s")
    tok_per_s = meta.get("tok_per_s")
    coupling = meta.get("coupling_strength")
    readout = meta.get("reservoir_readout")
    preview = preview_text(meta.get("response_preview"))
    parts = []
    if isinstance(tokens, int):
        generation = f"{tokens} tokens"
        if isinstance(elapsed, (int, float)):
            generation += f" in {float(elapsed):.1f}s"
        if isinstance(tok_per_s, (int, float)):
            generation += f" ({float(tok_per_s):.1f} tok/s)"
        parts.append(generation)
    if isinstance(coupling, (int, float)):
        parts.append(f"coupling={float(coupling):.3f}")
    if isinstance(readout, dict):
        y1 = readout.get("y1_final")
        y2 = readout.get("y2_final")
        y3 = readout.get("y3_final")
        if all(isinstance(v, (int, float)) for v in (y1, y2, y3)):
            parts.append(f"y=[{float(y1):+.3f}, {float(y2):+.3f}, {float(y3):+.3f}]")
    if preview:
        parts.append(f"preview={preview!r}")
    stamp = event_stamp(meta)
    if stamp:
        parts.append(stamp)
    return ", ".join(parts) if parts else "coupled generation check-in"


def feeder_summary(meta: dict | None) -> str | None:
    if not isinstance(meta, dict) or not meta:
        return None
    parts = []
    source = meta.get("source")
    if source:
        parts.append(str(source))
    memory_role = meta.get("memory_role")
    if memory_role:
        parts.append(f"memory={memory_role}")
    projection = meta.get("projection")
    if projection:
        parts.append(f"projection={projection}")
    conditioning = meta.get("conditioning")
    if conditioning:
        parts.append(f"conditioning={conditioning}")
    stamp = event_stamp(meta)
    if stamp:
        parts.append(stamp)
    return ", ".join(parts) if parts else None


async def send(ws, msg: dict) -> dict:
    await ws.send(json.dumps(msg))
    return json.loads(await ws.recv())


async def collect_snapshot(ws, last_n: int) -> dict:
    handles_resp = await send(ws, {"type": "list_handles"})
    handles = handles_resp.get("handles", [])
    enriched = []

    for handle in handles:
        name = handle["name"]
        trajectory = await send(ws, {"type": "trajectory", "name": name, "last_n": last_n})
        layer_metrics = await send(ws, {"type": "layer_metrics", "name": name})
        outputs = trajectory.get("outputs", [])
        trend, latest, delta = trend_label(outputs)
        spread = (max(outputs) - min(outputs)) if outputs else 0.0
        layers = layer_metrics.get("layers", [])
        layer_states = [classify_layer(layer) for layer in layers]
        enriched.append({
            **handle,
            "latest_output": latest,
            "trend": trend,
            "trend_delta": delta,
            "spread": spread,
            "layers": layers,
            "layer_states": layer_states,
        })

    resonances = []
    names = [handle["name"] for handle in handles]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            result = await send(ws, {"type": "resonance", "name_a": name_a, "name_b": name_b})
            result["alignment"] = resonance_label(result.get("correlation"))
            resonances.append(result)

    return {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "handles": enriched,
        "resonances": resonances,
    }


def render_markdown(snapshot: dict) -> str:
    handles = snapshot["handles"]
    resonances = snapshot["resonances"]
    mode_counts: dict[str, int] = {}
    warming_layers = 0
    for handle in handles:
        mode_counts[handle["mode"]] = mode_counts.get(handle["mode"], 0) + 1
        warming_layers += sum(1 for state in handle["layer_states"] if state == "warming_up")

    strongest = None
    with_corr = [res for res in resonances if res.get("correlation") is not None]
    if with_corr:
        strongest = max(with_corr, key=lambda res: abs(float(res["correlation"])))

    lines = [
        "# Reservoir Metrics Snapshot",
        "",
        f"Captured: {snapshot['captured_at']}",
        "",
        "## Quick Read",
        f"- Active handles: {len(handles)}",
        f"- Modes: {', '.join(f'{mode}={count}' for mode, count in sorted(mode_counts.items())) or 'none'}",
        f"- Layers still warming up: {warming_layers}",
    ]
    if strongest is not None:
        lines.append(
            "- Strongest resonance: "
            f"{strongest['name_a']} <-> {strongest['name_b']} "
            f"corr={strongest['correlation']:+.3f} ({strongest['alignment']})"
        )

    lines.extend([
        "",
        "## How To Read This",
        "",
        "- `trend` tells you whether a handle's recent output is rising, falling, or steady over the sampled window.",
        "- `near_target` means the layer's measured entropy is close to its learned thermostat target.",
        "- `warming_up` means the thermostat does not yet have enough history to judge that layer.",
        "- `high_saturation` means many neurons are near tanh saturation, which is usually worth watching.",
        "- `hint status` tells you whether a feeder hint is actively governing the handle or merely present in the latest metadata.",
        "",
        "## Handles",
        "",
    ])

    for handle in handles:
        ago = handle.get("last_tick_ago")
        ago_text = f"{ago}s ago" if ago is not None else "never"
        recent_sources = ", ".join(handle.get("recent_sources") or []) or "none"
        lines.extend([
            f"### {handle['name']} ({handle['entity']})",
            "",
            f"- Mode: `{handle['mode']}`",
            f"- Last live activity: {ago_text}",
            f"- Tick count: {handle['tick_count']}",
            f"- Output snapshot: latest={handle['latest_output']:+.4f}, trend={handle['trend']}, delta={handle['trend_delta']:+.4f}, spread={handle['spread']:.4f}",
            f"- Last live source: `{handle.get('last_source') or 'unknown'}`"
            + (f" ({event_stamp(handle.get('last_live_meta'))})" if event_stamp(handle.get("last_live_meta")) else ""),
            f"- Recent sources: {recent_sources}",
        ])
        feeder = feeder_summary(handle.get("last_feeder_meta"))
        if feeder:
            lines.append(f"- Feeder lane: {feeder}")
        generation = generation_summary(handle.get("last_live_meta"))
        if not generation:
            generation = generation_summary(handle.get("last_generation_meta"))
        if generation:
            lines.append(f"- Last generation: {generation}")
        hint = handle.get("rehearsal_hint")
        if isinstance(hint, dict):
            status_label = hint_status_label(handle.get("hint_status"))
            lines.append(
                f"- Soft hint: `{hint.get('mode', '?')}/{hint.get('decay_profile', '?')}`"
                + (f" because {hint.get('reason')}" if hint.get("reason") else "")
                + (f" [{status_label}]" if status_label else "")
            )
        lines.extend([
            "",
            "| Layer | h_norm | entropy | target | saturation | rho | state |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for layer, state in zip(handle["layers"], handle["layer_states"]):
            lines.append(
                f"| {layer['name']} | {layer.get('h_norm')} | "
                f"{layer.get('entropy')} | {layer.get('entropy_target')} | "
                f"{layer.get('saturation')} | {layer.get('rho')} | {state} |"
            )
        lines.append("")

    if resonances:
        lines.extend([
            "## Resonance",
            "",
            "| Pair | Shared Ticks | Correlation | Divergence | RMSD | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for res in resonances:
            lines.append(
                f"| {res['name_a']} <-> {res['name_b']} | {res.get('shared_ticks', 0)} | "
                f"{res.get('correlation')} | {res.get('divergence')} | {res.get('rmsd')} | "
                f"{res['alignment']} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_text(snapshot: dict) -> str:
    lines = [f"Reservoir metrics snapshot @ {snapshot['captured_at']}", ""]
    for handle in snapshot["handles"]:
        ago = handle.get("last_tick_ago")
        ago_text = f"{ago}s" if ago is not None else "never"
        lines.append(
            f"{handle['name']:15s} mode={handle['mode']:9s} last={ago_text:>6s} "
            f"out={handle['latest_output']:+.4f} trend={handle['trend']:>10s} "
            f"spread={handle['spread']:.4f} source={handle.get('last_source') or 'unknown'}"
        )
        live_stamp = event_stamp(handle.get("last_live_meta"))
        if live_stamp:
            lines.append(f"  - last event: {live_stamp}")
        feeder = feeder_summary(handle.get("last_feeder_meta"))
        if feeder:
            lines.append(f"  - feeder lane: {feeder}")
        generation = generation_summary(handle.get("last_live_meta"))
        if not generation:
            generation = generation_summary(handle.get("last_generation_meta"))
        if generation:
            lines.append(f"  - last generation: {generation}")
        for layer, state in zip(handle["layers"], handle["layer_states"]):
            lines.append(
                f"  - {layer['name']}: h_norm={layer.get('h_norm')} entropy={layer.get('entropy')} "
                f"target={layer.get('entropy_target')} sat={layer.get('saturation')} "
                f"rho={layer.get('rho')} state={state}"
            )
        hint = handle.get("rehearsal_hint")
        if isinstance(hint, dict):
            status_label = hint_status_label(handle.get("hint_status"))
            lines.append(
                f"  - hint: {hint.get('mode', '?')}/{hint.get('decay_profile', '?')} "
                f"{hint.get('reason', '')}".strip()
                + (f" [{status_label}]" if status_label else "")
            )
        lines.append("")

    if snapshot["resonances"]:
        lines.append("resonance:")
        for res in snapshot["resonances"]:
            lines.append(
                f"  - {res['name_a']} <-> {res['name_b']}: corr={res.get('correlation')} "
                f"div={res.get('divergence')} rmsd={res.get('rmsd')} {res['alignment']}"
            )
    return "\n".join(lines)


async def async_main(args):
    async with websockets.connect(args.ws_url) as ws:
        snapshot = await collect_snapshot(ws, args.last_n)
    if args.format == "markdown":
        print(render_markdown(snapshot))
    else:
        print(render_text(snapshot))


def main():
    ap = argparse.ArgumentParser(description="Capture a newcomer-friendly reservoir metrics snapshot")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:7881", help="Reservoir service WebSocket URL")
    ap.add_argument("--last-n", type=int, default=20, help="Trajectory window to summarize")
    ap.add_argument("--format", choices=["text", "markdown"], default="text", help="Output format")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
