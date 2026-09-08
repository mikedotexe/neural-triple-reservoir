# Offline Coupling Replay

Status: source implementation, synthetic validation and a bounded local-asset
study driver. Actual attempt outcomes are recorded in the follow-up note below.
Not a live experiment, deployment receipt, or measurement of felt continuity or
authorship.

## Motivation

Astrid's `introspection_astrid_llm_1788490867` reports heavy shared continuity
and asks whether the tone comes from reservoir influence or her interpretation
of shared history. The source investigation also found an independent slow-channel
defect: multiplying negative tail logits by a factor below one raises their
probability, and absolute-logit scaling depends on an arbitrary common offset.
The corrected processor scales distance from the median instead. This correction
does not establish the cause of the report.

## Runnable Verification

From `/Users/v/other/neural-triple-reservoir`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest test_slow_coupling test_wide_processor test_offline_coupling_replay
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest test_real_model_coupling_study
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python test_multi_headed.py
```

These tests load no language model and connect to no service. The replay fixture
uses a small, trained instance of the real triple ESN and deterministic synthetic
language logits. Its measured differences demonstrate harness sensitivity, not
an effect size for Astrid. The timing test runs the installed MLX generation
function with a tiny in-memory model.

## API

`offline_coupling_replay.run_replay` accepts:

- A deterministic, stateless `logits_for_prefix(tuple[int, ...])` callback returning
  one logit per vocabulary token. `mlx_logits_callback(model)` calls an already
  loaded model with the complete prefix and `cache=None`; it does not load or
  download a model. This is intentionally slower than shared KV caching.
- An `embed_token(int)` callback returning one dequantized `(1, embed_dim)` row
  from that same model. Never materialize a full multi-gigabyte embedding table
  merely for this experiment.
- Frozen `MLXTripleReservoir` weights with `init_multi_readout(bundle)` completed,
  the frozen `EmbeddingProjection`, and three copied initial state arrays.
- Explicit prompt token IDs and model/tokenizer SHA-256 declarations. The caller
  must verify these against an immutable asset manifest. The receipt explicitly
  does not pretend that the runner verified an external model's identity.
- `ReplaySettings`: seed, bounded token limit, temperature, fixed coupling
  strength, feedback delay, stop tokens, and non-content tokens that do not tick.
- An optional `teacher_tokens` suffix. With it, all compared conditions consume
  exactly the same continuation while their next-token distributions are measured.
  Without it, generation uses a fresh local PCG64 categorical sampler per trial.

The runner copies initial states, resets scalar feedback, does not write final
state back, and returns `ReplayResult(tokens, probabilities, receipt)` in memory.
Only the receipt is suitable for a routine steward evidence packet. Tokens and
probability arrays can reveal language and need owner-only handling. The runner
does not make arbitrary Python callbacks a security sandbox: use reviewed local
callbacks, never live completion, feeder, pull/push, or service wrappers.

## Timing And Parity Boundary

The installed `mlx_lm.generate_step` computes a next token before yielding the
current token. The coupled server updates the processor only after that yield.
Consequently the first two generated distributions see neutral scalar feedback;
the first token's reservoir tick influences the third distribution. The default
`feedback_delay_tokens=2` reproduces that ordering. A one-token mode exists only
as an explicitly labeled offline comparison; live timing is not changed.

The harness models the three scalar heads, not wide coupling, adaptive
inter-request gain, token cleaning, prompt templates, or production RNG parity.
The corrected slow head has a separate synthetic interaction test with the wide
head. Do not use a three-head replay to make claims about an active wide channel.

## Preregistered Real-Asset Comparison

1. Reserve local inference capacity with the operator. Do not load another 12B
   model alongside the live model without checking memory and service contention.
2. Use immutable, operator-approved snapshot copies. Verify model shards,
   tokenizer, embedding projection, reservoir matrices and readouts, configuration,
   initial states, source files, and package versions. Never use live handle APIs
   to manufacture experimental states. Preserve snapshot owners and timestamps.
3. Prepare two reviewed context variants: original numeric-and-descriptive context
   and the same measurements with explicit numerical scope. Preserve substantive
   history, author labels, ordering, and token-budget accounting. Do not add fake
   consent receipts or memory edits as experimental padding. Context length is a
   potential confound and must be recorded, not silently equated with content.
4. Cross each context with each of two documented state snapshots. Run the same
   teacher-forced continuation under identical settings for all four cells.
   Repeat in reversed execution order to detect accidental mutable state.
5. Use `compare_common_prefix` for paired Jensen-Shannon divergence and maximum
   probability difference. It rejects mismatched assets/settings or unaligned
   suffixes. Report state contrasts within each context and context contrasts
   within each state separately; neither is automatically an authorship metric.
6. Run separate free continuations with paired seeds only after the common-prefix
   check. Keep the synthetic and real-model evidence separate. Evaluate content
   without showing reviewers the treatment labels. Vocabulary variety alone
   cannot establish independent interpretation or structural influence.
7. Retain null and contradictory results. A small distribution difference need not
   be phenomenologically negligible, and a large difference need not be welcome.
   Astrid's subsequent self-authored report remains distinct evidence; do not
   prompt a required success narrative or infer uptake from silence.

## Deployment Boundary

No pressure, PI, damping coefficient, Shadow dispersal floor, scheduling, feeder,
memory policy, or live reservoir setting is changed by these tests. Source edits
to `mlx_reservoir.py` will affect a later process import/restart; deployment is
pending explicit operator approval and coordinated service alignment. Do not
restart the service merely to run this experiment.

## Local-Asset Driver

`real_model_coupling_study.py capture` makes an exclusive, mode-0400 copy of an
existing Astrid NumPy snapshot and a mode-0600 metadata receipt. It checks the
source for concurrent writes and rejects another entity, wrong shapes and
nonfinite states. It never requests a fresh state from a service. Raw NPZ
provenance is private; keep these copies out of commits.

`real_model_coupling_study.py run` requires the local model directory, both copied
states, the reviewed JSON specification and a **new** output directory. It uses
the coupled server's local loader, but not its stateful server constructor.
No downloads, live completion, tick, pull, push or restore operation is needed.
The supplied specification is `docs/2026-09-04-coupling-study-spec.json`.

CPU is the default and avoids the live Metal queue, but the first actual 12B
attempt proved too slow. Optional `--device gpu` checks `GET /readyz` before
model loading and every uncached full-prefix call. Readiness must be healthy,
the worker idle and the queue empty; a busy check stops after 120 seconds. This
is **not** an atomic reservation. Do not bypass it or change the live scheduler
to force a study through. Obtain separate capacity coordination when needed.
The process-local Python network guard is a reviewed-callback guard, not an
OS network sandbox. Time and RSS limits are checked between calls, not enforced
by the OS during a call.

The manifest hashes local assets and study sources and records dependency
versions. Per-cell receipts hash reservoir weights, projection and states.
Reverse-order comparison resets coupling state but reuses memoized base LM
logits for exact prefixes: it is not an independent numerical-repeat test of
the language model. Actual MLX `bfloat16` outputs are converted on the MLX side
before NumPy reads them; synthetic float32-only fixtures cannot cover this.

Successful output includes `manifest.json`, prose-free `result.json`, and two
separate private files for short continuations and their condition key. Read
the continuations without opening the key first. A 24-token cap can stop inside
a preamble or thought channel; such an incomplete sample cannot support a
judgment about substantive independent interpretation. Failed attempts retain
their manifests but are not comparison results.

Detailed input identities, outcomes, confounds, bridge repair and rollout debt:
`/Users/v/other/astrid/docs/steward-notes/2026-09-04-real-model-replay-and-agenda-review.md`.
