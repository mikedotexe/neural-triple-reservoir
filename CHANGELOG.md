# Changelog

## [Unreleased]

- Promoted Astrid's launchd-managed coupled server default from `mlx-community/gemma-3-4b-it-4bit` to `mlx-community/gemma-4-12B-it-5bit` after the repaired Gemma 4 lane passed narrow probes and a strict 2-hour live bridge soak. The Gemma 3 4B model remains the rollback target.
- Hardened the coupled Astrid server's Gemma 4 path by treating Gemma 4 turn/channel closers as generation stop tokens and skipping non-content channel/tool sentinels before detokenization or reservoir ticks. The repaired `mlx-community/gemma-4-12B-it-5bit` lane passed Astrid's narrow exact-output and `NEXT:` probes on the alternate `8092` lane before promotion, and the existing `gemma4_unified` text-lane shim remains in place for Gemma 4 checkpoints.
- Added sampled shadow metrics rollups and rotation. Reservoir shadow comparisons now keep all live ticks, throttle rehearsal metrics to a per-handle interval, update `shadow_metrics_rollup.json`, and rotate `shadow_metrics.jsonl` at the configured cap so shadow experiments cannot silently consume the machine.
- Added Volitional Attractor Shaping V4 garden proof metadata. The attractor garden now records facet-tree label metadata and rehearsal proof controls for same-prompt/different-state, same-state/different-prompt, hold/rehearse/quiet recovery, stale-lock detection, and blend parent-collapse checks without making rehearsal a hard live-control gate.
- Added blend support to the attractor garden. Garden seed specs can now carry parent labels, deterministic blend vector schedules, and parent metadata on clone-handle requests and generated attractor intents so blend rehearsals remain traceable and repeatable.
