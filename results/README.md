# Results

This directory is the repo's small longitudinal evidence trail.

The goal is simple: keep a few timestamped artifacts that show what the
system actually did, not just what we believe it should do.

## What Goes Here

- canonical experiment reports from `experiment_same_prompt_different_state.py`
- notable metrics snapshots that help explain a runtime state
- short markdown memos when a run changes our understanding

## Naming

Prefer timestamped, descriptive filenames:

- `same_prompt_different_state_2026-03-30_canonical.md`
- `metrics_snapshot_2026-03-30_post_restart.md`

Keep the corpus small and high-signal. This folder is for evidence we want
newcomers and future-us to read, not for every transient debug artifact.

## Current Artifacts

- [same_prompt_different_state_2026-03-30_canonical.md](/Users/v/other/neural-triple-reservoir/results/same_prompt_different_state_2026-03-30_canonical.md) — canonical run of experiment 1 using the same prompt across `stable`, `expanding`, and `recovered` shaped handles.
- [metrics_snapshot_2026-03-30_post_experiment.md](/Users/v/other/neural-triple-reservoir/results/metrics_snapshot_2026-03-30_post_experiment.md) — live newcomer-facing snapshot taken after the canonical experiment run.
