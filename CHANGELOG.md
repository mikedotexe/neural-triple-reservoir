# Changelog

## [Unreleased]

- Added sampled shadow metrics rollups and rotation. Reservoir shadow comparisons now keep all live ticks, throttle rehearsal metrics to a per-handle interval, update `shadow_metrics_rollup.json`, and rotate `shadow_metrics.jsonl` at the configured cap so shadow experiments cannot silently consume the machine.
- Added Volitional Attractor Shaping V4 garden proof metadata. The attractor garden now records facet-tree label metadata and rehearsal proof controls for same-prompt/different-state, same-state/different-prompt, hold/rehearse/quiet recovery, stale-lock detection, and blend parent-collapse checks without making rehearsal a hard live-control gate.
- Added blend support to the attractor garden. Garden seed specs can now carry parent labels, deterministic blend vector schedules, and parent metadata on clone-handle requests and generated attractor intents so blend rehearsals remain traceable and repeatable.
