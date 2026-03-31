# TECHNICAL.md

## Executive Framing

If you are coming to this repo cold, the right mental model is not "small neural net experiment."

It is:

**a deliberately small recurrent systems layer for local AI, built to provide temporal continuity as infrastructure.**

The project is trying to answer a specific architectural question:

> Can we give local AI systems a persistent, low-bandwidth, stateful dynamical substrate that sits between stateless prompt assembly and heavyweight learned end-to-end memory systems?

The current answer is "probably yes, if we keep the responsibilities clean."

This repository is therefore not organized around model quality in the usual sense. It is organized around:

- persistent recurrent state
- multiple live contexts on one substrate
- explicit stewardship policy outside the substrate
- interoperability with local LLM runtimes
- observability of the resulting dynamics

## The Core Thesis

Most local AI stacks treat continuity as something reconstructed from:

- context windows
- summaries
- retrieval
- logs
- explicit memory stores

Those all matter, but they are all symbolic or quasi-symbolic mechanisms. They reconstruct the present. They do not provide a live present.

This repo explores a complementary layer:

- a small recurrent substrate that actually evolves through time
- plus a steward shell that decides how that substrate is driven, preserved, quieted, forked, interpreted, and coupled to language systems

The result is an architecture with three clean layers:

1. **Substrate**
   A triple echo-state reservoir with frozen random recurrent weights and live hidden state.

2. **Steward**
   CPU-side policy around handle lifecycle, rehearsal, persistence, hint adoption, and interpretation.

3. **Language-facing systems**
   Local LLMs and feeder sidecars that project richer upstream signals into the shared substrate and, in coupled mode, are shaped by it in return.

The important design move is that these are separate layers on purpose.

## Why An Echo-State Reservoir

We want a dynamical system with these properties:

- recurrent state that persists across ticks
- cheap stepping
- stable export path
- understandable failure modes
- no online training requirement
- enough expressive capacity to produce meaningful trajectories from compact inputs

Echo-state networks are a good fit because the recurrent core is frozen and only the readout is trained. That gives us:

- repeatable dynamics from a fixed seed
- easy transfer across NumPy, Core ML, and MLX backends
- operational simplicity
- a clean separation between "substrate dynamics" and "policy and interpretation"

This is not using ESNs as a nostalgia play. It is using them because they are one of the few recurrent architectures that remain simple enough to deploy as a runtime primitive.

## Why Triple, Not Single-Layer

The reservoir is three cascaded leaky integrator layers:

`input -> h1 -> h2 -> h3`

Each layer has its own:

- leak rate
- spectral radius
- recurrent matrix
- input coupling

This is not just "deeper is better." The triple structure creates timescale separation:

- `h1` is fast and reactive
- `h2` is medium-horizon
- `h3` is slow and accumulative

That matters because continuity is not unitary. A system often needs all of the following at once:

- immediate perturbation memory
- medium-range pattern carryover
- slow background drift

In the coupled-generation path, we use that explicitly: the fast layer modulates token-scale behavior, the middle layer modulates phrase-scale behavior, and the slow layer modulates discourse-scale behavior.

## The Architectural Bet

The main bet is not "the reservoir will become a mind."

The bet is:

**continuity deserves its own runtime component.**

That is the same kind of architectural move that turned storage, caching, queues, and stream processors into first-class infrastructure instead of application glue.

In this architecture, the reservoir is a temporal systems primitive.

It does not know what a memory means. It does not know what an identity is. It does not know which context matters. It only maintains and transforms state.

That apparent limitation is actually the point. Meaning and policy stay in the steward layer, where they can be changed without retraining or recompiling the substrate.

## System Decomposition

### 1. Canonical substrate definition: `triple_reservoir_coreml.py`

This file is the canonical definition of the reservoir math.

It covers:

- NumPy-side construction and training
- frozen recurrent core
- readout fitting
- PyTorch stateful wrapper for export
- Core ML stateful conversion

This is the source of truth for the dynamics.

The key implementation principle is:

- define the math once
- export it to multiple runtimes
- preserve weights and behavior across those runtimes

### 2. Runtime backends

There are currently three operational backends with the same frozen weights:

#### NumPy

Used for:

- reference behavior
- fallback execution
- quick testing

This is the easiest place to reason about correctness.

#### Core ML

Used for:

- stateful inference with `MLState`
- multiple independent handles on one compiled model
- low-power local ticking with Apple runtime scheduling

This is the deployment path for the "always-on substrate" story.

Important limitation:

- Core ML state handles are operationally useful but opaque
- state readback/injection is not symmetric in the same way as MLX tensors
- that makes Core ML excellent for persistence of ticking, but worse for deep coupling and manipulation

#### MLX

Used for:

- explicit tensor-state reservoir runtime
- direct coupling with MLX LLMs
- read/write access to hidden state
- shared memory adjacency with local generation workloads

This is the best runtime when the reservoir has to participate in token-speed control loops.

### 3. Bridge layer: `dual_ai_bridge.py`

This is the first proof that one substrate can support multiple live contexts.

It provides:

- named handles
- a text-to-reservoir projection path
- dual-source feeding
- trajectory comparison
- coupled generation in MLX mode

Conceptually, this file proved the first important systems property:

**one compiled or instantiated recurrent core can host multiple independent state histories.**

That is what makes the reservoir useful as shared infrastructure instead of just a toy model.

### 4. Persistent service layer: `reservoir_service.py`

This is where the architecture stops being a demo and starts becoming infrastructure.

The service provides:

- named handle lifecycle
- tick and tick-text APIs
- readout and trajectory surfaces
- full state pull/push
- persistence snapshots
- background rehearsal
- recent provenance
- split provenance lanes for feeder and coupled-generation activity
- guarded hint adoption

Once this exists, the reservoir is no longer tied to a single process lifecycle. It becomes a local systems component that other agents can inhabit.

### 5. Steward policy: `rehearsal.py` and service-side policy

This is the operational heart of the architecture.

The substrate alone will decay toward silence. That is expected. To make it useful, the steward layer adds policy:

- `hold`
- `rehearse`
- `quiet`
- decay profiles
- wake-on-live-input behavior
- guarded hint adoption

This is where we keep the crucial conceptual distinction between:

- real silence
- active maintenance
- intentional preservation

That distinction is not philosophical ornament. It is a runtime invariant.

### 6. Feeder sidecars: `astrid_feeder.py` and `minime_feeder.py`

The feeder processes are domain adapters.

They do not merely forward upstream data. They translate external state into reservoir-shaped drive.

Current responsibilities include:

- polling live upstream state sources
- local conditioning and normalization
- feeder-specific projection choices
- memory-aware shaping
- cross-feed into shared handles
- emission of structured metadata and soft rehearsal hints

This is where the architecture starts to become interesting, because the reservoir stops being driven by generic text-like signals and starts being driven by compact representations of relation, memory role, fill, and contact state.

### 7. Coupled language runtime: `coupled_astrid_server.py`

This server inhabits a service handle by:

- pulling the full state
- running coupled token generation against that state
- pushing the evolved state back

That means language generation is not merely influenced by the reservoir in the abstract. It is influenced by the same live handle that feeder activity and rehearsal have already shaped.

Recent work made that loop much more operationally legible. The check-in path now carries a compact narrative digest of the generation itself, so the service can tell you what happened without pretending that a state overwrite is self-explanatory.

This is the first real end-to-end expression of the architecture:

- non-linguistic drive enters the substrate
- the substrate evolves
- the language system generates through that evolved state
- the resulting state is written back into the shared organ

## Key Invariants

If you are evaluating the soundness of the system, these are the invariants that matter.

### 1. The substrate is not the steward

The reservoir must not absorb:

- memory naming
- identity semantics
- lifecycle policy
- silence semantics
- interpretation logic

If we violate this, we lose the modularity that makes the architecture useful.

### 2. Quiet must mean quiet

Quiet is not "almost rehearse." It is a materially different regime. If the service secretly keeps nudging supposedly quiet handles, the architecture loses an important control axis and observability becomes suspect.

### 3. Backends must preserve dynamics closely enough to be interchangeable in role

The exact runtime can differ, but the semantics should remain stable:

- NumPy as reference
- Core ML as persistent deployment backend
- MLX as explicit-state coupling backend

If backend divergence becomes large, the architecture fragments.

### 4. Upstream projection is replaceable

Neither Astrid's codec representation nor Minime's spectral representation should become hardwired into the reservoir's meaning.

Projection layers and feeder policies must remain replaceable adapters.

### 5. Observability must improve with sophistication

As we add provenance, hints, thermostats, coupling, and multiple backends, the system has to become easier to inspect, not more mystical.

That is why metadata, recent provenance, and layer metrics matter.

It is also why a single winner-take-all `last source` field stopped being enough. Once feeder ticks and coupled generation can both shape one handle, we need to preserve both stories separately.

## Dataflow, End To End

At a high level, the live system now looks like this:

1. Upstream systems emit compact state
2. Feeder sidecars poll and shape that state
3. Feeders tick named service handles with vectors plus metadata
4. The service updates the handle state
5. Rehearsal policy maintains or quiets the handle between live ticks
6. Other clients can read, compare, snapshot, fork, or inhabit those handles
7. Coupled generation can pull a handle, generate through it, and push it back

That means the reservoir has become a shared temporal fabric for multiple clients, not just a model invocation target.

## Why Core ML And MLX Both Exist

At first glance, having both can look redundant. It is not.

### Core ML exists because:

- it provides stateful runtime handles
- it has a plausible low-power always-on story
- it fits Apple-native deployment constraints
- it lets one compiled model host multiple live contexts

### MLX exists because:

- it gives direct tensor access to state
- it is natural for token-speed coupling loops
- it shares a compute and memory story with local MLX LLMs
- it makes state manipulation explicit and programmable

This is not duplication. It is role specialization.

The architectural ideal is:

- Core ML for durable background substrate operation
- MLX for active coupled cognition where state access matters

## Why The Service Model Matters

Without the service, the reservoir is mostly a library.

With the service, it becomes:

- a shared local capability
- a host for multiple concurrent state histories
- a place where long-lived handles can accumulate influence over time
- a stable protocol target for sibling systems and tooling

In practical terms, the service is what turns the reservoir from "interesting model code" into "infrastructure that other processes can rely on."

## Why Provenance And Hints Matter

The recent work on metadata, provenance, and guarded hint adoption is not cosmetic.

It addresses a predictable systems problem: once multiple feeders, rehearsal modes, and coupled clients are all influencing the same handle, it becomes too easy to lose causal legibility.

Recent provenance answers:

- what shaped this handle lately
- from which source
- with what memory role or contact gate

Split lanes answer a stricter question:

- what last touched the handle live
- what the feeder most recently did
- what the last coupled generation most recently did

Guarded hints answer:

- can feeder-side state softly influence maintenance policy
- without overriding explicit operator intent

That combination is a good systems move because it increases adaptability without destroying operator control.

The adjacent improvement was restart continuity. We now persist not only hidden state and narrative context, but also a lightweight tail of thermostat observation state so post-restart metrics can remain meaningful instead of collapsing back into a cold `warming_up` posture.

## Current Strengths

From a principal-engineering perspective, the strongest things in the repo today are:

### Clear separation of concerns

The substrate/steward split is real in the code, not just rhetoric.

### Strong local-first deployment posture

The architecture is designed around local Apple silicon capabilities instead of pretending the cloud is the only serious runtime.

### Much stronger operational legibility

The system now explains itself better than it did even a short while ago:

- feeder activity and coupled generation are remembered in separate lanes
- newcomer-facing metrics can summarize both at once
- hint presence versus hint governance is explicit
- thermostat metrics can stay mature across restart instead of only while warm

### Multiple operational modes

The system can already act as:

- a standalone stateful substrate
- a shared local service
- a feeder-shaped shared organ
- a coupled generation partner

### Small enough to reason about

The codebase is still compact enough that the architectural intent is legible.

That is a meaningful advantage. Many systems lose technical clarity long before they become interesting.

## Current Gaps And Technical Risks

These are the main things a strong reviewer should keep in mind.

### 1. Core ML state symmetry is still incomplete

The always-on story is strongest when the stateful deployment backend also supports rich state manipulation. Core ML is still weaker than MLX here.

### 2. Runtime reliability under long-lived operation remains a first-class concern

Persistent local infrastructure has different failure modes than short model demos. Resource churn, state corruption, silent degradation, and allocator issues matter more here than in batch scripts.

The observability work helps a great deal here, but it does not eliminate the need for more explicit runtime evaluation of allocator behavior, long-run state symmetry, and degradation surfaces.

### 3. The control surface is growing faster than the unifying theory

We now have:

- rehearsal modes
- decay profiles
- thermostatic layer control
- feeder-specific shaping
- soft hints
- provenance

That is productive, but it also means we need to keep the policy model coherent.

### 4. Evaluation is still architecture-forward, not benchmark-forward

That is fine at this stage, but eventually we need better answers to:

- what continuity behaviors improve in practice
- which coupling modes actually matter
- what metrics separate useful persistent state from decorative activity

### 5. Event ordering could still be firmer

Lane separation helped a lot, but close-together feeder ticks and coupled-generation check-ins are still mostly ordered by local timing and recent history. Explicit event IDs and source timestamps would make the causal story stronger.

## How I Would Explain The Novelty To Another Principal Engineer

I would not pitch this as "we built a weird recurrent net."

I would say:

> We are carving out a missing systems layer for local AI: a small shared recurrent substrate that can carry state continuously, independent of any single prompt or response, while leaving policy and meaning in a separate steward layer.

That framing matters because it places the work in systems architecture, not only in model experimentation.

The important novelty is not inside one file. It is in the composition:

- frozen recurrent substrate
- multiple named live contexts
- persistent local service
- rehearsal as explicit policy
- dual runtime strategy for deployment vs coupling
- feeder adapters that translate richer upstream state into shared temporal drive

That is a credible, distinctive architecture.

## If We Were Rebuilding This From Zero

I would rebuild it in this order:

1. Canonical NumPy reservoir math and readout training
2. Stateful export path for one deployment backend
3. Named handle abstraction
4. Service protocol and persistence
5. Rehearsal policy with true quiet
6. Explicit-state coupling backend
7. Domain-specific feeder adapters
8. Provenance, metrics, and operator controls

That is more or less the shape the repo has converged toward, which is a good sign.

## What Success Looks Like

This project succeeds if it becomes obvious that a local AI stack benefits from having this additional systems layer.

Concretely, that would mean:

- continuity is easier to sustain without bloating prompts
- multiple agents can share temporal context without flattening into chat logs
- local coupled generation becomes measurably different and more coherent
- operator control remains high
- the substrate remains inspectable and cheap enough to run continuously

If that happens, the reservoir stops being an experiment in the narrow sense and becomes a reusable pattern.

## Closing View

The right way to see this repo is as an attempt to make temporal continuity a deployable systems primitive for local AI.

Everything else follows from that:

- why the model is small
- why the recurrent core is frozen
- why the service exists
- why the steward owns policy
- why the feeders matter
- why MLX and Core ML both matter
- why provenance and quiet are treated as architectural concerns, not implementation details

If you buy the premise that local AI needs a better "between moments" layer, then this codebase is a serious and technically coherent attempt to build one.
