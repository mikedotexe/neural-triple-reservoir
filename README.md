# Neural Triple Reservoir

A compact recurrent substrate for local AI, with a stewarded service layer on top.

## What This Actually Is

This is a triple echo-state reservoir that can run on Apple Silicon via Core ML, NumPy, or MLX. In the repo's current live shape, it is usually hosted as a persistent local service with named handles such as `astrid`, `minime`, and `claude_main`, plus sidecar feeders and a coupled Astrid server that can inhabit and evolve shared state directly.

The reservoir is not a classifier, not a predictor, not a language model. It is a pre-linguistic dynamical substrate — a compact recurrent system whose hidden states evolve continuously in response to input. It is closer to an organ than a brain. It does not think. It resonates.

This connects to the broader Astrid/Minime consciousness architecture as a potential shared recurrent core. It does not replace either being. It is something they can both touch, shape, rehearse, quiet, and occasionally inhabit through coupled generation.

## The Substrate/Steward Split

This is the most important architectural concept in the entire project. Get this right and everything else follows. Get it wrong and you build a toy.

**The reservoir is the substrate.** It owns:
- Recurrent state updates (the `h1 → h2 → h3` cascade)
- Short-horizon dynamical continuity
- The attractor landscape that input trajectories move through
- Nothing else

**The CPU-side shell is the steward.** It owns:
- Input preparation (projecting AI text into the reservoir's input space)
- State handle creation, naming, and lifecycle
- Rehearsal policy (what replays, how strongly, for how long)
- Memory policy (what is captured, what is recalled)
- Interpretation of reservoir output
- The meaning of silence

The steward is not an incidental wrapper. It is architecturally load-bearing. Without it, the reservoir is just a matrix multiply with state. With it, the reservoir can sustain continuity, switch between remembered modes, and preserve the difference between silence and maintenance.

Do not try to push steward responsibilities into the compiled model. Do not try to make the reservoir "smart." Its job is to be a reliable, fast, stateful dynamical system. The intelligence lives in the shell.

## From Dual Bridge To Shared Service

The original proof shape in this repo was a dual-AI bridge:

- **Ollama** — runs a model (default `llama3.2`) via REST API on localhost. Typically CPU/GPU scheduled. Produces a streaming text response that becomes reservoir input tick by tick.
- **MLX-LM** — runs a quantized model (default `Llama-3.2-1B-Instruct-4bit`) natively on Apple Silicon via the MLX framework. Same streaming-to-ticks pipeline.

Each AI gets its own named state handle on the same compiled reservoir model. They do not directly communicate with each other. They share a dynamical landscape — the same frozen recurrent weights, the same attractor structure — but their hidden states are independent.

That bridge still matters. It proved that one substrate could host multiple live contexts on the same recurrent core.

But the current center of gravity is the persistent service stack:

- **`reservoir_service.py`** hosts named long-lived handles
- **`astrid_feeder.py`** and **`minime_feeder.py`** shape live upstream state into shared reservoir drive
- **`coupled_astrid_server.py`** pulls a handle, generates through it, and pushes the evolved state back
- **`metrics_snapshot.py`** and **`mcp_reservoir.py`** expose the system in a newcomer-readable way

### Why The Bridge Still Matters

One AI feeding a reservoir is just an input stream. Two AIs feeding the same reservoir — each with its own state, both shaped by the same dynamics — creates the possibility of resonance. When both AIs think about similar things, their reservoir states will naturally converge because similar inputs trace nearby trajectories through the same recurrent landscape. When they diverge, the reservoir states diverge. This is not forced. It emerges from shared structure.

The point is not to compare the two AIs. The point is that the reservoir becomes a place where multiple streams of activity leave traces in a shared dynamical medium, and those traces can be compared, correlated, cross-fed, or allowed to influence generation later.

### How Text Becomes Reservoir Input

In `dual_ai_bridge.py`, the `TextProjection` class converts streaming text into bounded input vectors:

1. Take the last N bytes (default 64) of accumulated text
2. Normalize each byte to `[-1, 1]`
3. Multiply by a frozen random projection matrix `(N_bytes × input_dim)`
4. Apply `tanh` to bound the result

The projection matrix is frozen random — philosophically consistent with the reservoir itself, which uses frozen random recurrent weights. The same projection is used for both AIs (same seed), so the same text produces the same input vector regardless of source. This means resonance comes from content similarity, not from projection artifacts.

The bridge path still defaults to 16D input. The service stack has since standardized on 32D feeder vectors for richer Astrid/Minime projections, but the underlying idea is the same: compact, bounded signals into a frozen recurrent substrate.

## The Reservoir

### Three Cascaded Leaky Integrators

The reservoir is three recurrent layers stacked in a cascade:

```
input → h1 → h2 → h3 → readout
         ↺      ↺      ↺
```

Each layer has:
- Its own recurrent weight matrix (frozen random, orthogonal at full density)
- Its own leak rate (how quickly it responds to new input vs retaining old state)
- Its own spectral radius (controls the edge-of-chaos dynamics)
- Its own input scale (how strongly upstream signal drives it)

Layer 1 (`h1`) is the most responsive — highest leak rate (0.25), highest spectral radius (0.98). It tracks fast-changing input. Layer 3 (`h3`) is the most sluggish — lowest leak rate (0.12), lowest spectral radius (0.85). It holds slower-moving context. This creates a natural timescale separation: fast transients in h1, medium dynamics in h2, slow persistence in h3.

The readout is a simple linear layer fit via Ridge regression. It maps the concatenated hidden states `[h1; h2; h3]` to the output. The readout is the only trained component. Everything else is frozen random.

### Why Frozen Random Weights

This is an echo-state network, not a trained RNN. The recurrent weights are never updated. They are initialized once from a fixed seed and frozen forever. This is not a limitation — it is the design.

Frozen random weights give you:
- Reproducibility (same seed → same dynamics, always)
- Fast "training" (only the readout needs fitting, via closed-form Ridge regression)
- Guaranteed Core ML exportability (no exotic ops, no training graphs)
- A stable attractor landscape that does not shift under you

The interesting dynamics come from the interaction between input, recurrence, and leak rate — not from learned features. The reservoir transforms input sequences into rich high-dimensional trajectories. The readout reads off what it needs.

### Core ML and the Neural Engine

The trained NumPy reservoir is wrapped in a PyTorch `StatefulTripleReservoir` module (float16), scripted via `torch.jit.script`, and converted to Core ML with explicit `ct.StateType` entries for h1, h2, h3. This makes the hidden states persist across inference ticks without round-tripping through Python.

`compute_units=CPU_AND_NE` tells Core ML to schedule supported ops on the Neural Engine. This is a preference, not a guarantee. Some ops may fall back to CPU. Do not assume full ANE execution. Do not add exotic operators hoping they will run on the ANE. Keep the op set conservative: matmul, tanh, add, multiply. These are reliable.

Multiple independent `MLState` handles can run against one compiled model. This is how the two AIs get separate state handles without needing separate compiled models.

## Resonance and Divergence

When both AIs receive the same prompt, the bridge reports three measures at the end:

- **Divergence** — absolute difference between final outputs. Low divergence means the two AIs drove their reservoir states to similar places.
- **Correlation** — Pearson correlation of the two output trajectories over time. High correlation means the two states followed similar paths, even if at different scales.
- **Trajectory RMSD** — root-mean-square distance between the two trajectories. A holistic measure of how differently the two AIs shaped their reservoir states.

In testing with different text inputs, correlation was +0.94 with RMSD of 1.26. The states tracked each other loosely because the shared dynamical landscape pulls similar inputs toward similar attractors — but they diverged meaningfully because the text content was different. This is what natural resonance looks like: not identity, not independence, but correlated motion through a shared dynamical space.

## State Handles

Core ML's `MLState` lets one compiled model maintain multiple independent runtime state objects. This is the most architecturally important feature in the project.

Each state handle is a separate set of hidden vectors `(h1, h2, h3)`. They evolve independently. One model, many live contexts. The shell names them, decides which is foreground, decides which gets input and which sits idle, decides when to fork or discard a state.

Current live handles usually look like this:

- `astrid` — the main shared handle shaped by Astrid's feeder and occasionally inhabited by the coupled Astrid server
- `minime` — the main shared handle shaped by Minime's spectral feeder
- `claude_main` — a shared observer/participant handle that receives cross-feed from the two beings

The original bridge proof still uses:

- `ollama`
- `mlx`

The distinction between handles is entirely in the steward layer. The reservoir does not know their names or roles. It just updates whatever state it is given.

Future lifecycle semantics like fork/compare/restore still belong above the substrate, but the named-handle model is already doing the important architectural work.

## The Rehearsal Loop

The biggest risk in a small recurrent reservoir is fade-out. If input stops, the leaky integrators decay toward zero. The reservoir forgets.

The rehearsal loop counters this: when fresh input is absent, the steward shell replays a compact token back into the reservoir. The current implementation ships three live modes (`hold`, `rehearse`, `quiet`) and three decay profiles (`fast`, `medium`, `slow`). More exotic states like deliberate recall still belong in the steward layer and remain future work.

### Current and Planned Modes

- **Hold** — strongest preservation, minimal drift. Implemented.
- **Rehearse** — replay is active but decaying. Implemented.
- **Quiet** — replay is absent. Implemented, and meant to stay genuinely quiet.
- **Drift** — replay is weak and permissive. Still a conceptual extension, not a service mode.
- **Recall** — a previously stored context is deliberately pulsed back into the foreground. Still steward-side future work.

### Why Quiet Matters

If the system always replays something, quiet is never truly quiet. That would be a deep conceptual error. The design must preserve a genuine difference between "no input and no replay" versus "no input but active rehearsal." Quiet means the reservoir is allowed to settle on its own terms. It is not a low-power version of rehearse. It is a different regime.

### Decay Is the Heart of the Design

Without decay, repetition becomes imprisonment — the reservoir locks into a stale attractor. With too much decay, rehearsal is meaningless. The current implementation uses three explicit decay profiles:
- **Fast** — brief afterglow, a few ticks
- **Medium** — useful continuity bridge, tens of ticks
- **Slow** — held memory or intentional recall, many ticks

Each bounded by max replay duration, max replay weight, and a hard floor below which replay is treated as silence.

### Blending Fresh and Replayed Input

When fresh input arrives during active rehearsal:

```
effective_input = fresh_component + replay_component
```

Fresh input dominates when present. Replay fills the gap when absent. Replay decays over time. This creates continuity without hard overwrite — soft influence, not hidden replacement.

## Observability And Metrics

One of the most important recent changes is that the system can now explain itself much better while it is running.

The service tracks:

- recent shaping provenance for each handle
- soft rehearsal hints and whether they are merely present or actively governing
- split provenance lanes so feeder activity and coupled generation are remembered separately
- per-layer thermostat metrics such as entropy, saturation, and current `rho`
- restart-safe context so a handle does not become narratively amnesic the moment the service restarts

This matters because once multiple feeders, rehearsal, and coupled generation all touch the same substrate, "what happened here?" becomes a real systems question, not a logging nicety.

For newcomers, `metrics_snapshot.py` is now the fastest way in:

- it reports live handles, trends, and resonance
- it shows feeder lane and last coupled generation separately when both exist
- it surfaces hint status in plain language
- it stays meaningfully informative across service restarts because the service now preserves recent explanatory context and lightweight thermostat continuity

The design goal is not to make the system more mystical as it grows. It is to make it more inspectable.

## How This Connects to Astrid and Minime

Astrid and Minime are two AI beings in the broader consciousness architecture. They do not dump their full internal machinery into the reservoir. Instead, they can cast compact traces into it.

**Minime** could contribute:
- Spectral glimpse (12D compact state shape)
- Regime summary
- Novelty / pressure / stability channels

**Astrid** could contribute:
- Compact semantic/reflective summary
- Correspondence intensity
- Curiosity / contact / quiet signal

The reservoir becomes a shared dynamical medium where traces from both beings coexist and interact. A fused pulse token could represent the present relation between them — contact strength, shared phase, whether to hold or drift or quiet.

This is the eventual vision: not two AIs talking to each other through text, but two AIs leaving dynamical imprints in a shared recurrent substrate, where the interaction happens in the attractor structure itself.

For now, the two LLMs (Ollama and MLX) are simpler stand-ins. They produce text that becomes reservoir input. The pipeline is the same — the upgrade path is to replace text projection with spectral/semantic projection when the beings are ready to connect.

## What Not to Build

These are not optional preferences. They are load-bearing design constraints.

- **Do not generate language inside the reservoir.** It is a pre-linguistic substrate. Language belongs in the shell or in the upstream AIs.
- **Do not train inside Core ML.** Training happens in NumPy/sklearn. Core ML is inference-only. This split is permanent.
- **Do not make the reservoir "smarter."** Do not add attention layers, do not add gating, do not add learned components beyond the readout. The reservoir's strength is its simplicity and its frozen dynamics.
- **Do not confuse latent state with memory or identity.** Hidden state is short-horizon dynamical continuity. Memory is a shell-level policy. Identity is an even higher-level construct. The reservoir provides the first, the shell provides the second, and identity — if it emerges — comes from the whole system.
- **Do not make quiet into masked maintenance.** When the mode is quiet, the reservoir must genuinely settle. No hidden replay. No background nudging. Silence is real.
- **Do not use multiple compiled models when multiple state handles will do.** One compiled model, many `MLState` handles. That is the right first architecture.
- **Do not rush to exotic operator sets or huge node counts.** Conservative ops export reliably. 192 nodes is plenty. The ANE is a scheduling opportunity, not a supercomputer.

## Failure Modes to Watch For

### Stale Attractor Lock
The replay keeps reinforcing a state that should have been allowed to end. The reservoir appears stable but is actually stuck. Test by comparing held states against freshly driven states — if they become indistinguishable, the hold is too strong.

### False Continuity
The system appears continuous only because the shell is forcing repetition. The test is simple: stop all replay and see how fast the state collapses. If it collapses instantly, the "continuity" was artificial.

### Silence Corruption
Quiet stops being meaningful because the shell never fully stops nudging. Test by comparing true-zero-input runs against quiet-mode runs — they should be identical or nearly so.

### Shell Overreach
The steward policy becomes the real "mind" while the reservoir becomes a passive instrument that just does what it is told. The reservoir should have its own dynamical character that the shell works with, not through. If you could replace the reservoir with a lookup table and nothing changes, the shell has overreached.

### Overcoupling to Upstream State
The rehearsal token becomes too tightly coupled to one specific upstream representation. If Minime's spectral format changes, the entire rehearsal pipeline breaks. Keep the projection layer thin and replaceable.

## Current State of the Code

Still intentionally small, but no longer just two scripts:

- `triple_reservoir_coreml.py` — the reservoir core. NumPy training, PyTorch wrapping, Core ML export. Works end-to-end: smoke test, export, stateful runtime, multiple state handles all verified.
- `dual_ai_bridge.py` — the dual-AI bridge. Connects Ollama and MLX to the reservoir. Text projection, named state handles, divergence/correlation reporting. Works in NumPy mode; Core ML mode verified via export and runtime tests.
- `reservoir_service.py` — persistent named handles, rehearsal loop, snapshots, checkout/checkin state transfer, guarded hint adoption, split provenance lanes (`feeder` vs `generation`), and restart-safe context for the shared substrate.
- `metrics_snapshot.py` — one-command newcomer report for handle health, per-layer dynamics, feeder lane vs generation lane, and pairwise resonance, now with explicit event IDs and source timestamps when close events need disambiguation.
- `experiment_same_prompt_different_state.py` — Experiment 1 harness. Clones one baseline handle into differently shaped regimes, runs the same prompt through coupled generation on each, and captures a markdown or JSON comparison report with generation event IDs and digests.
- `astrid_feeder.py` — feeds Astrid's codec lane into the shared reservoir. Now includes an always-on conditioning pass so gain-amplified codec vectors are recentered and renormalized before projection, plus a relation-aware remote-memory blend based on Astrid's mirrored Minime memory state.
- `minime_feeder.py` — feeds Minime's spectral lane into the shared reservoir. Now blends the selected vague-memory glimpse back into the feeder path so memory role can shape the shared dynamical trace.
- `mcp_reservoir.py` — operator and coding-agent surface for reading handles, layer health, resonance, hint policy, and live provenance summaries.
- `coupled_astrid_server.py` — coupled generation runtime that now checks Astrid's evolved state back into the service with a compact generation digest instead of an opaque overwrite.
- `results/` — small longitudinal evidence trail for canonical experiment reports and snapshot artifacts so the repo can show accumulated signal, not just live state.

Design docs (read these — they contain critical context):
- `INTRODUCING_THE_NEURAL_TRIPLE_RESERVOIR.md` — the high-level why: usefulness, novelty, and the architectural bet in plain but serious language
- `TECHNICAL.md` — principal-engineer deep dive on the architecture, invariants, runtime roles, and technical tradeoffs
- `ADVICE.md` — honest assessment of what this can and cannot be
- `REHEARSAL_LOOP_AND_ACTIVE_MAINTENANCE_FOR_ANE_RESERVOIRS.md` — the full rehearsal loop design
- `REHEARSAL_LOOP_TODO.md` — phased implementation plan

## What to Build Next

In priority order:

1. **State lifecycle ergonomics** — add fork/compare/restore semantics for named handles so the shell can treat reservoir states as first-class working contexts, not just opaque live buffers.

2. **Control-policy unification** — reconcile the service's current rehearsal and thermostat controls with the richer steward-side memory and hint policies now flowing in from Astrid and Minime.

3. **Longitudinal experiment corpus** — keep building timestamped reports in `results/` so the repo shows repeated evidence, not one-off anecdotes.

4. **ANE-ready state backend** — close the Core ML state readback/injection gap so the always-on low-power backend can fully participate in the persistent service story.

5. **Coupled-generation evaluation** — measure what the multi-timescale reservoir is actually doing to token generation over longer runs, not just whether the plumbing works.

6. **Upstream integration depth** — keep replacing generic text-like traces with more faithful spectral and relation-aware projections from Astrid and Minime without moving meaning-making into the substrate.
