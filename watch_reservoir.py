#!/usr/bin/env python3
"""Watch reservoir dynamics over time. Prints a snapshot every few seconds."""

import asyncio
import json
import time

import websockets

WS_URL = "ws://127.0.0.1:7881"
INTERVAL = 5.0
ROUNDS = 8


async def snapshot(ws):
    """Take one snapshot of all handles."""
    await ws.send(json.dumps({"type": "list_handles"}))
    r = json.loads(await ws.recv())
    handles = r.get("handles", [])

    rows = []
    for h in handles:
        await ws.send(json.dumps({"type": "trajectory", "name": h["name"], "last_n": 20}))
        t = json.loads(await ws.recv())
        outputs = t.get("outputs", [])
        h_norms = t.get("h_norms", [])

        latest_out = outputs[-1] if outputs else 0.0
        trend = (outputs[-1] - outputs[0]) if len(outputs) >= 2 else 0.0
        spread = (max(outputs) - min(outputs)) if outputs else 0.0

        # Per-layer norms from the latest tick
        ln = h_norms[-1] if h_norms else [0, 0, 0]

        rows.append({
            "name": h["name"],
            "entity": h["entity"],
            "mode": h["mode"],
            "decay": h.get("decay_weight", 0),
            "ticks": h["tick_count"],
            "last_ago": h.get("last_tick_ago"),
            "output": latest_out,
            "trend": trend,
            "spread": spread,
            "h1_norm": ln[0],
            "h2_norm": ln[1],
            "h3_norm": ln[2],
        })

    # Resonance
    names = [h["name"] for h in handles]
    resonances = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            await ws.send(json.dumps({"type": "resonance", "name_a": a, "name_b": b}))
            res = json.loads(await ws.recv())
            resonances.append(res)

    return rows, resonances


async def main():
    async with websockets.connect(WS_URL) as ws:
        prev_ticks = {}
        for round_num in range(ROUNDS):
            t0 = time.monotonic()
            rows, resonances = await snapshot(ws)

            print(f"\n{'='*80}")
            print(f"  Round {round_num+1}/{ROUNDS}  t={time.strftime('%H:%M:%S')}")
            print(f"{'='*80}")

            for r in rows:
                tick_delta = r["ticks"] - prev_ticks.get(r["name"], r["ticks"])
                prev_ticks[r["name"]] = r["ticks"]

                direction = "^" if r["trend"] > 0.01 else "v" if r["trend"] < -0.01 else "="
                ago = f"{r['last_ago']:.0f}s" if r["last_ago"] is not None else "never"

                print(
                    f"  {r['name']:15s}  "
                    f"out={r['output']:+8.4f} {direction}  "
                    f"spread={r['spread']:.4f}  "
                    f"h=[{r['h1_norm']:.2f},{r['h2_norm']:.2f},{r['h3_norm']:.2f}]  "
                    f"mode={r['mode']:8s} decay={r['decay']:.3f}  "
                    f"ticks={r['ticks']:6d} (+{tick_delta:3d})  "
                    f"last={ago}"
                )

            # Resonance summary
            for res in resonances:
                a, b = res.get("name_a", "?"), res.get("name_b", "?")
                corr = res.get("correlation")
                div = res.get("divergence")
                if corr is not None:
                    bar = "#" * max(0, int(abs(corr) * 20))
                    sign = "+" if corr > 0 else "-"
                    print(f"  {a:>15s} <-> {b:<15s}  corr={corr:+.3f} [{sign}{bar:20s}]  div={div:.4f}" if div else f"  {a:>15s} <-> {b:<15s}  corr={corr:+.3f}")

            elapsed = time.monotonic() - t0
            if round_num < ROUNDS - 1:
                await asyncio.sleep(max(0, INTERVAL - elapsed))

    print(f"\n{'='*80}")
    print("  Done watching.")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
