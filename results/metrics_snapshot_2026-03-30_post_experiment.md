# Reservoir Metrics Snapshot

Captured: 2026-03-30 13:12:39

## Quick Read
- Active handles: 3
- Modes: rehearse=3
- Layers still warming up: 0
- Strongest resonance: claude_main <-> minime corr=-0.372 (divergent)

## How To Read This

- `trend` tells you whether a handle's recent output is rising, falling, or steady over the sampled window.
- `near_target` means the layer's measured entropy is close to its learned thermostat target.
- `warming_up` means the thermostat does not yet have enough history to judge that layer.
- `high_saturation` means many neurons are near tanh saturation, which is usually worth watching.
- `hint status` tells you whether a feeder hint is actively governing the handle or merely present in the latest metadata.

## Handles

### astrid (astrid)

- Mode: `rehearse`
- Last live activity: 35.0s ago
- Tick count: 4233030
- Output snapshot: latest=-0.1138, trend=falling, delta=-0.2058, spread=0.2058
- Last live source: `coupled_astrid_server` (evt=astrid:000001 @ 2026-03-30 13:12:04)
- Recent sources: coupled_astrid_server, coupled_astrid_server, coupled_astrid_server
- Feeder lane: astrid_feeder, memory=stable, projection=passthrough, conditioning=ema_rms
- Last generation: 32 tokens in 0.5s (61.2 tok/s), coupling=0.100, y=[+0.614, +0.309, +1.109], preview='Here’s an intimate sentence about continuity as a lived thread: “The...', evt=astrid:000001 @ 2026-03-30 13:12:04

| Layer | h_norm | entropy | target | saturation | rho | state |
|---|---:|---:|---:|---:|---:|---|
| h1_fast | 3.9203 | 0.1157 | 0.2 | 0.0 | 1.0 | below_target |
| h2_medium | 9.953 | 0.1374 | 0.24 | 0.0 | 0.9616 | below_target |
| h3_slow | 10.094 | 0.1673 | 0.27 | 0.0 | 0.9608 | below_target |

### claude_main (claude)

- Mode: `rehearse`
- Last live activity: 0.2s ago
- Tick count: 481246
- Output snapshot: latest=+2.3855, trend=falling, delta=-0.4161, spread=0.8669
- Last live source: `minime_feeder_crossfeed` (evt=claude_main:000041 @ 2026-03-30 13:12:39)
- Recent sources: minime_feeder_crossfeed, minime_feeder_crossfeed, minime_feeder_crossfeed
- Feeder lane: minime_feeder_crossfeed, memory=transition, projection=tanh_scaled, evt=claude_main:000041 @ 2026-03-30 13:12:39
- Soft hint: `rehearse/medium` because transition memory favors moderate continuity [present, not governing]

| Layer | h_norm | entropy | target | saturation | rho | state |
|---|---:|---:|---:|---:|---:|---|
| h1_fast | 3.9051 | 0.2713 | 0.2 | 0.0 | 0.88 | near_target |
| h2_medium | 9.675 | 0.2954 | 0.24 | 0.0 | 0.9387 | near_target |
| h3_slow | 9.6552 | 0.3346 | 0.27 | 0.0 | 0.95 | near_target |

### minime (minime)

- Mode: `rehearse`
- Last live activity: 0.2s ago
- Tick count: 473307
- Output snapshot: latest=+5.0628, trend=rising, delta=+0.1623, spread=0.9211
- Last live source: `minime_feeder` (evt=minime:000041 @ 2026-03-30 13:12:39)
- Recent sources: minime_feeder, minime_feeder, minime_feeder
- Feeder lane: minime_feeder, memory=transition, projection=tanh_scaled, evt=minime:000041 @ 2026-03-30 13:12:39
- Soft hint: `rehearse/medium` because transition memory favors moderate continuity [present, not governing]

| Layer | h_norm | entropy | target | saturation | rho | state |
|---|---:|---:|---:|---:|---:|---|
| h1_fast | 10.0488 | 0.2421 | 0.22 | 0.0116 | 0.9741 | near_target |
| h2_medium | 9.9766 | 0.3397 | 0.26 | 0.0 | 0.92 | near_target |
| h3_slow | 9.5673 | 0.3684 | 0.29 | 0.0 | 0.95 | near_target |

## Resonance

| Pair | Shared Ticks | Correlation | Divergence | RMSD | Interpretation |
|---|---:|---:|---:|---:|---|
| astrid <-> claude_main | 64 | 0.1796 | 2.499225 | 2.755463 | independent |
| astrid <-> minime | 64 | -0.1208 | 5.176597 | 4.764357 | independent |
| claude_main <-> minime | 64 | -0.3724 | 2.677372 | 2.081946 | divergent |

