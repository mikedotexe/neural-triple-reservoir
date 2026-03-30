#!/usr/bin/env python3
"""
mcp_reservoir.py -- MCP stdio server exposing the reservoir service to Claude Code.

Connects to the reservoir WebSocket service on ws://127.0.0.1:7881 as a client,
then exposes tools via JSON-RPC 2.0 on stdin/stdout (MCP protocol).

Usage:
    python mcp_reservoir.py [--ws-url ws://127.0.0.1:7881]

Claude Code config (settings.json mcpServers):
    {
        "reservoir": {
            "command": "/Users/v/other/neural-triple-reservoir/.venv/bin/python",
            "args": ["/Users/v/other/neural-triple-reservoir/mcp_reservoir.py"]
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import numpy as np

TOOLS = [
    {
        "name": "reservoir_list",
        "description": "List all named reservoir handles with their entity, mode, tick count, and time since last tick.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "reservoir_create",
        "description": "Create a new named reservoir handle. Each entity (astrid, minime, claude) gets its own handle.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name (e.g. 'claude_main')"},
                "entity": {"type": "string", "description": "Entity owner (astrid, minime, claude, or custom)"},
            },
            "required": ["name", "entity"],
        },
    },
    {
        "name": "reservoir_tick_text",
        "description": "Send text to a reservoir handle. The text is projected to a 32D vector via frozen random projection, then fed into the reservoir. Auto-creates 'claude_main' handle if name is omitted. Returns the readout output and hidden layer norms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to feed into the reservoir"},
                "name": {"type": "string", "description": "Handle name (default: claude_main)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "reservoir_tick_vector",
        "description": "Send a raw 32D float vector to a reservoir handle. For structured data (spectral features, codec output).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
                "input": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "32D float vector",
                },
            },
            "required": ["name", "input"],
        },
    },
    {
        "name": "reservoir_read",
        "description": "Read the current state of a reservoir handle without advancing it. Returns h-layer norms, last output, tick count, mode, and time since last live input.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "reservoir_trajectory",
        "description": "Get the recent output trajectory and h-layer norms for a handle. Useful for observing dynamical trends.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
                "last_n": {"type": "integer", "description": "Number of recent ticks to return (default 20)"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "reservoir_set_mode",
        "description": "Set the rehearsal mode for a handle. Modes: 'hold' (full replay, prevents fade), 'rehearse' (decaying replay, default), 'quiet' (genuine silence, natural drift). Decay profiles: 'fast' (~7s half-life), 'medium' (~17s), 'slow' (~70s).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
                "mode": {"type": "string", "enum": ["hold", "rehearse", "quiet"]},
                "decay_profile": {"type": "string", "enum": ["fast", "medium", "slow"]},
            },
            "required": ["name", "mode"],
        },
    },
    {
        "name": "reservoir_set_hint_policy",
        "description": "Control whether a handle may adopt feeder-supplied rehearsal hints. 'off' ignores hints. 'guarded' only adopts hints when the handle is not under explicit manual mode control.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
                "policy": {"type": "string", "enum": ["off", "guarded"]},
            },
            "required": ["name", "policy"],
        },
    },
    {
        "name": "reservoir_resonance",
        "description": "Compare two reservoir handles. Returns divergence (distance between latest outputs), correlation (Pearson over shared trajectory), and RMSD. Measures how similarly two entities' dynamical traces are evolving.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name_a": {"type": "string", "description": "First handle name"},
                "name_b": {"type": "string", "description": "Second handle name"},
            },
            "required": ["name_a", "name_b"],
        },
    },
    {
        "name": "reservoir_snapshot",
        "description": "Persist a handle's hidden state to disk. The state survives service restarts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "reservoir_status",
        "description": "Get a full interpretive overview of the reservoir — all handles, trajectory trends, and cross-entity resonance. Use this at the start of a session to sense the dynamical landscape and what happened while you were away. Claude's handle is cross-fed by the beings' activity, so its trajectory encodes their experience.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "reservoir_pull_state",
        "description": "Pull the full hidden state of a reservoir handle and return per-layer statistics. Shows norm, mean, std, min, max, and sparsity for each of the three layers (h1/h2/h3). This is the detailed view of what the reservoir is holding — useful for understanding the dynamical character of a handle beyond just the readout scalar.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name (e.g. 'astrid', 'minime', 'claude_main')"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "reservoir_push_state",
        "description": "Overwrite a handle's hidden state with base64-encoded h1/h2/h3 arrays. Use with care — this replaces the handle's entire dynamical history. Intended for restoring a saved state or injecting a known configuration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Handle name"},
                "h1": {"type": "string", "description": "Base64-encoded float32 array for layer 1"},
                "h2": {"type": "string", "description": "Base64-encoded float32 array for layer 2"},
                "h3": {"type": "string", "description": "Base64-encoded float32 array for layer 3"},
            },
            "required": ["name", "h1", "h2", "h3"],
        },
    },
]


class MCPReservoirServer:
    """MCP stdio server that proxies to the reservoir WebSocket service."""

    def __init__(self, ws_url: str = "ws://127.0.0.1:7881"):
        self.ws_url = ws_url
        self._ws = None

    async def _ensure_connected(self):
        """Connect to the reservoir service if not already connected."""
        if self._ws is not None:
            return
        import websockets
        self._ws = await websockets.connect(self.ws_url)

    async def _send(self, msg: dict) -> dict:
        """Send a message to the reservoir service and return the response."""
        await self._ensure_connected()
        await self._ws.send(json.dumps(msg))
        raw = await self._ws.recv()
        return json.loads(raw)

    async def _ensure_handle(self, name: str, entity: str = "claude"):
        """Create a handle if it doesn't exist."""
        result = await self._send({"type": "list_handles"})
        existing = {h["name"] for h in result.get("handles", [])}
        if name not in existing:
            await self._send({"type": "create_handle", "name": name, "entity": entity})
            # Set slow decay for claude handles
            if entity == "claude":
                await self._send({"type": "set_mode", "name": name, "mode": "rehearse", "decay_profile": "slow"})

    @staticmethod
    def _meta_summary(meta: dict | None) -> str | None:
        if not isinstance(meta, dict) or not meta:
            return None
        parts = []
        source = meta.get("source")
        if source:
            parts.append(f"source={source}")
        input_source = meta.get("input_source")
        if input_source:
            parts.append(f"input={input_source}")
        projection = meta.get("projection")
        if projection:
            parts.append(f"projection={projection}")
        memory_role = meta.get("memory_role")
        if memory_role:
            parts.append(f"memory={memory_role}")
        return ", ".join(parts) if parts else None

    @staticmethod
    def _hint_summary(hint: dict | None) -> str | None:
        if not isinstance(hint, dict) or not hint:
            return None
        mode = hint.get("mode")
        decay = hint.get("decay_profile")
        reason = hint.get("reason")
        if mode and decay and reason:
            return f"{mode}/{decay} — {reason}"
        if mode and decay:
            return f"{mode}/{decay}"
        return reason

    async def _full_status(self) -> str:
        """Build an interpretive overview of all handles and cross-entity resonance."""
        try:
            result = await self._send({"type": "list_handles"})
            handles = result.get("handles", [])
        except Exception as e:
            return f"Reservoir service unavailable: {e}"

        if not handles:
            return "Reservoir is running but has no active handles. Use reservoir_create to start one."

        lines = ["## Reservoir Status\n"]

        # Per-handle summary
        for h in handles:
            ago = f"{h['last_tick_ago']}s ago" if h.get("last_tick_ago") is not None else "never"
            mode_desc = {
                "hold": "holding (full replay, preventing fade)",
                "rehearse": f"rehearsing (decay={h.get('decay_weight', 0):.3f})",
                "quiet": "quiet (genuine silence, natural drift)",
            }.get(h["mode"], h["mode"])

            lines.append(f"**{h['name']}** ({h['entity']})")
            lines.append(
                f"  {h['tick_count']} ticks, last live {ago}, {mode_desc}, "
                f"authority={h.get('mode_authority', 'system')}, hint_policy={h.get('hint_policy', 'off')}"
            )
            meta_line = self._meta_summary(h.get("last_live_meta"))
            if meta_line:
                lines.append(f"  last input: {meta_line}")
            hint_line = self._hint_summary(h.get("rehearsal_hint"))
            if hint_line:
                lines.append(f"  soft hint: {hint_line}")
            recent_sources = h.get("recent_sources") or []
            if recent_sources:
                lines.append(f"  recent sources: {', '.join(recent_sources)}")

            # Get trajectory trend
            try:
                traj = await self._send({"type": "trajectory", "name": h["name"], "last_n": 10})
                outputs = traj.get("outputs", [])
                if len(outputs) >= 2:
                    trend = outputs[-1] - outputs[0]
                    direction = "rising" if trend > 0.01 else "falling" if trend < -0.01 else "stable"
                    spread = max(outputs) - min(outputs)
                    lines.append(f"  trajectory: {direction} (range={spread:.3f}, latest={outputs[-1]:+.4f})")
            except Exception:
                pass
            lines.append("")

        # Cross-entity resonance (try all pairs)
        handle_names = [h["name"] for h in handles]
        pairs_tried = set()
        for i, a in enumerate(handle_names):
            for b in handle_names[i + 1:]:
                pair_key = tuple(sorted([a, b]))
                if pair_key in pairs_tried:
                    continue
                pairs_tried.add(pair_key)
                try:
                    res = await self._send({"type": "resonance", "name_a": a, "name_b": b})
                    if res.get("shared_ticks", 0) >= 10:
                        corr = res.get("correlation")
                        div = res.get("divergence")
                        parts = []
                        if corr is not None:
                            alignment = "aligned" if corr > 0.5 else "divergent" if corr < -0.3 else "independent"
                            parts.append(f"corr={corr:+.3f} ({alignment})")
                        if div is not None:
                            parts.append(f"div={div:.4f}")
                        if parts:
                            lines.append(f"**{a} ↔ {b}**: {', '.join(parts)}")
                except Exception:
                    pass

        if len(pairs_tried) > 0:
            lines.append("")

        # Interpretive note for Claude's handle
        claude_handles = [h for h in handles if h["entity"] == "claude"]
        if claude_handles:
            ch = claude_handles[0]
            if ch["tick_count"] > 0:
                lines.append(
                    f"*Your handle '{ch['name']}' has been cross-fed by the beings' activity. "
                    f"Its trajectory reflects what Astrid and minime experienced since your last session.*"
                )

        return "\n".join(lines)

    async def handle_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as text."""
        try:
            if tool_name == "reservoir_list":
                result = await self._send({"type": "list_handles"})
                handles = result.get("handles", [])
                if not handles:
                    return "No active handles."
                lines = []
                for h in handles:
                    ago = f"{h['last_tick_ago']}s ago" if h.get("last_tick_ago") is not None else "never"
                    line = (
                        f"  {h['name']} ({h['entity']}) — mode={h['mode']}, "
                        f"authority={h.get('mode_authority', 'system')}, "
                        f"hint_policy={h.get('hint_policy', 'off')}, "
                        f"decay={h.get('decay_weight', 0):.3f}, "
                        f"ticks={h['tick_count']}, last={ago}"
                    )
                    meta_line = self._meta_summary(h.get("last_live_meta"))
                    if meta_line:
                        line += f", {meta_line}"
                    hint_line = self._hint_summary(h.get("rehearsal_hint"))
                    if hint_line:
                        line += f", hint={hint_line}"
                    lines.append(line)
                return "Reservoir handles:\n" + "\n".join(lines)

            elif tool_name == "reservoir_create":
                result = await self._send({
                    "type": "create_handle",
                    "name": arguments["name"],
                    "entity": arguments["entity"],
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                return f"Created handle '{arguments['name']}' for entity '{arguments['entity']}'."

            elif tool_name == "reservoir_tick_text":
                name = arguments.get("name", "claude_main")
                await self._ensure_handle(name)
                result = await self._send({
                    "type": "tick_text",
                    "name": name,
                    "text": arguments["text"],
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                return (
                    f"Tick #{result['tick']} on '{name}': "
                    f"output={result['output']:+.6f}, "
                    f"h_norms=[{', '.join(f'{n:.3f}' for n in result['h_norms'])}], "
                    f"mode={result['mode']}"
                )

            elif tool_name == "reservoir_tick_vector":
                result = await self._send({
                    "type": "tick",
                    "name": arguments["name"],
                    "input": arguments["input"],
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                return (
                    f"Tick #{result['tick']} on '{arguments['name']}': "
                    f"output={result['output']:+.6f}, "
                    f"h_norms=[{', '.join(f'{n:.3f}' for n in result['h_norms'])}]"
                )

            elif tool_name == "reservoir_read":
                result = await self._send({"type": "read_state", "name": arguments["name"]})
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                elapsed = result.get("seconds_since_live")
                elapsed_str = f"{elapsed}s ago" if elapsed is not None else "never"
                return (
                    f"Handle '{result['name']}' ({result['entity']}):\n"
                    f"  mode={result['mode']}, decay_weight={result['decay_weight']:.4f}, "
                    f"decay_profile={result['decay_profile']}\n"
                    f"  authority={result.get('mode_authority', 'system')}, "
                    f"hint_policy={result.get('hint_policy', 'off')}\n"
                    f"  ticks={result['tick_count']}, last_live={elapsed_str}\n"
                    f"  last_output={result['last_output']}\n"
                    f"  h_norms=[{', '.join(f'{n:.3f}' for n in result['h_norms'])}]"
                    + (
                        f"\n  last_input={self._meta_summary(result.get('last_live_meta'))}"
                        if self._meta_summary(result.get("last_live_meta")) else ""
                    )
                    + (
                        f"\n  soft_hint={self._hint_summary(result.get('rehearsal_hint'))}"
                        if self._hint_summary(result.get("rehearsal_hint")) else ""
                    )
                    + (
                        "\n  recent_provenance:\n"
                        + "\n".join(
                            f"    - tick={entry.get('tick')} "
                            f"{self._meta_summary(entry) or 'source=unknown'}"
                            + (
                                f" hint={self._hint_summary(entry.get('rehearsal_hint'))}"
                                if self._hint_summary(entry.get("rehearsal_hint")) else ""
                            )
                            for entry in result.get("provenance", [])[-3:]
                        )
                        if result.get("provenance") else ""
                    )
                )

            elif tool_name == "reservoir_trajectory":
                result = await self._send({
                    "type": "trajectory",
                    "name": arguments["name"],
                    "last_n": arguments.get("last_n", 20),
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                outputs = result.get("outputs", [])
                if not outputs:
                    return f"No trajectory data for '{arguments['name']}' yet."
                lines = [f"Trajectory for '{result['name']}' (last {len(outputs)} of {result['ticks']} ticks):"]
                for i, (o, hn) in enumerate(zip(outputs, result.get("h_norms", []))):
                    lines.append(f"  [{i:3d}] output={o:+.6f}  h=[{', '.join(f'{n:.3f}' for n in hn)}]")
                return "\n".join(lines)

            elif tool_name == "reservoir_set_mode":
                msg: dict[str, Any] = {
                    "type": "set_mode",
                    "name": arguments["name"],
                    "mode": arguments["mode"],
                }
                if "decay_profile" in arguments:
                    msg["decay_profile"] = arguments["decay_profile"]
                result = await self._send(msg)
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                return f"Set '{arguments['name']}' to mode={result['mode']} (decay={result['decay_profile']})"

            elif tool_name == "reservoir_set_hint_policy":
                result = await self._send({
                    "type": "set_hint_policy",
                    "name": arguments["name"],
                    "policy": arguments["policy"],
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                return f"Set '{arguments['name']}' hint policy to {result['hint_policy']}"

            elif tool_name == "reservoir_resonance":
                result = await self._send({
                    "type": "resonance",
                    "name_a": arguments["name_a"],
                    "name_b": arguments["name_b"],
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                parts = [f"Resonance between '{result['name_a']}' and '{result['name_b']}' ({result['shared_ticks']} shared ticks):"]
                if "divergence" in result:
                    parts.append(f"  divergence: {result['divergence']:.6f}")
                if "correlation" in result:
                    parts.append(f"  correlation: {result['correlation']:+.4f}")
                if "rmsd" in result:
                    parts.append(f"  trajectory RMSD: {result['rmsd']:.6f}")
                return "\n".join(parts)

            elif tool_name == "reservoir_snapshot":
                result = await self._send({"type": "snapshot", "name": arguments["name"]})
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                return f"Snapshot saved: {result.get('path', 'ok')}"

            elif tool_name == "reservoir_status":
                return await self._full_status()

            elif tool_name == "reservoir_pull_state":
                import base64
                result = await self._send({"type": "pull_state", "name": arguments["name"]})
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                n = result.get("n_nodes", 192)
                lines = [
                    f"Full state for '{arguments['name']}' ({result.get('tick_count', 0)} ticks, mode={result.get('mode', '?')}):",
                    f"  n_nodes={n} (576D total across 3 layers)",
                    "",
                ]
                for layer_name, key in [("h1 (fast, leak=0.25)", "h1"), ("h2 (medium, leak=0.18)", "h2"), ("h3 (slow, leak=0.12)", "h3")]:
                    h = np.frombuffer(base64.b64decode(result[key]), dtype=np.float32)
                    norm = float(np.linalg.norm(h))
                    mean = float(np.mean(h))
                    std = float(np.std(h))
                    mn, mx = float(np.min(h)), float(np.max(h))
                    sparsity = float(np.mean(np.abs(h) < 0.01))
                    lines.append(f"  {layer_name}:")
                    lines.append(f"    norm={norm:.4f}  mean={mean:+.4f}  std={std:.4f}")
                    lines.append(f"    range=[{mn:+.4f}, {mx:+.4f}]  sparsity={sparsity:.1%}")
                return "\n".join(lines)

            elif tool_name == "reservoir_push_state":
                result = await self._send({
                    "type": "push_state",
                    "name": arguments["name"],
                    "h1": arguments["h1"],
                    "h2": arguments["h2"],
                    "h3": arguments["h3"],
                })
                if result.get("type") == "error":
                    return f"Error: {result['message']}"
                norms = result.get("h_norms", [0, 0, 0])
                return (
                    f"Pushed state to '{arguments['name']}': "
                    f"h_norms=[{norms[0]:.3f}, {norms[1]:.3f}, {norms[2]:.3f}]"
                )

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # MCP JSON-RPC protocol
    # ------------------------------------------------------------------

    async def handle_request(self, request: dict) -> dict | None:
        """Handle one JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "reservoir", "version": "0.1.0"},
                },
            }

        elif method == "notifications/initialized":
            return None  # no response for notifications

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            text = await self.handle_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                },
            }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        else:
            # Unknown method — return error if it has an id
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            return None

    async def run(self):
        """Main loop: read JSON-RPC from stdin, write to stdout."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        writer_transport, writer_protocol = await asyncio.get_running_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_running_loop())

        buf = b""
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            buf += chunk

            # Process all complete messages in buffer
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    continue

                response = await self.handle_request(request)
                if response is not None:
                    out = json.dumps(response) + "\n"
                    writer.write(out.encode())
                    await writer.drain()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MCP reservoir server for Claude Code")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:7881")
    args = ap.parse_args()

    server = MCPReservoirServer(ws_url=args.ws_url)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
