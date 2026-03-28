# Rehearsal Loop And Active Maintenance For ANE Reservoirs

## Premise

One of the biggest risks in a small recurrent Core ML reservoir is simple fade-out.

If input stops, the system may:

- decay toward silence too quickly
- lose continuity between meaningful moments
- fail to preserve a recent attractor long enough to be useful
- behave more like a reactive transient than a living substrate

The idea here is to counter that with a **rehearsal loop**:

- when a meaningful input arrives, the steward shell stores a compact representation of it
- if fresh input continues, the live stream dominates
- if fresh input stops, the shell pulses a compact replay token back into the reservoir
- that replay decays, shifts, or quiets according to mode

This is not quite “storage” in the archival sense.

It is better understood as:

- **active maintenance**
- **reverberant rehearsal**
- **keeping an attractor warm**

That distinction matters, because a rehearsed memory is shaped by ongoing repetition policy, not just by passive persistence.

## Core Claim

An ANE/Core ML reservoir can plausibly serve as a living dynamical substrate if a CPU-side steward shell actively maintains continuity by replaying compact state tokens when fresh input is absent.

The most important design discipline is:

- do **not** blindly repeat raw last input forever
- do **not** confuse rehearsal with true long-term storage
- do **use** compact, interpretable, decaying pulse tokens
- do **keep** the stewardship policy outside the reservoir itself

## Why This Is Attractive

The rehearsal-loop idea has several advantages:

1. It gives the reservoir a way to remain dynamically alive during quiet periods.
2. It can preserve a recent state long enough to influence later interpretation.
3. It gives the shell a clean place to implement memory policy without changing the compiled model.
4. It fits the broader “12D vague memory beside 32D detailed continuity” direction very well.
5. It creates a bridge point where Astrid and Minime spectral summaries could be cast into the same small recurrent substrate.

## Why This Is Dangerous

The same mechanism could easily become pathological.

If the shell repeats the last input too strongly or too long, the system may:

- lock into stale attractors
- create false continuity
- overwrite the meaning of silence
- self-confirm outdated dynamics
- feel alive only because it is being artificially looped

So the key architectural question is not:

- “can we repeat?”

It is:

- “what should repeat, how strongly, in which mode, and for how long?”

## Vocabulary

### Active maintenance

The shell deliberately keeps a recent internal pattern alive by replaying a surrogate of it.

### Rehearsal token

A compact input vector fed back into the reservoir when live input is absent or weak.

### Warm attractor

A recently active state that is being kept available, not by static storage, but by ongoing low-bandwidth reinforcement.

### Quiet

A distinct regime where the shell intentionally does **not** rehearse, or only rehearses below a minimal floor, so the reservoir can settle naturally.

### Recall

A shell-driven act of selecting and pulsing a previously retained context into the live foreground.

## Proposed Architecture

The best first design is a shell-controlled rehearsal architecture with four layers.

### 1. State samplers

Upstream systems produce compact state summaries.

Good candidates:

- Minime spectral fingerprint or a projection of it
- Minime `12D` spectral glimpse
- Astrid-side compact semantic or reflective state summary
- regime / valence / novelty / contact indicators
- covariance-derived or decomposition-derived small vectors

These should be treated as source material, not as direct reservoir inputs by default.

### 2. Pulse composer

The shell transforms those source signals into a **pulse token** suitable for the ANE reservoir.

This is where the shell decides:

- which channels matter
- how to normalize them
- whether to include fresh input, replay, or both
- whether a memory is vivid, weak, quiet, or recalled

### 3. Rehearsal controller

This is the policy layer.

It decides:

- whether the system is in `hold`, `rehearse`, `drift`, `quiet`, or `recall`
- how much weight fresh input gets
- how much weight replay gets
- how quickly replay decays
- when a replay token is retired

### 4. Stateful ANE reservoir runtime

The exported Core ML model continues to own:

- recurrent state update
- within-tick dynamical persistence
- latent continuity

The model should not own:

- memory naming
- replay policy
- state-role semantics
- silence semantics

That remains the shell’s job.

## Candidate Pulse Inputs

The strongest first version is **not** full raw state replay.

It should be a compact, steward-controlled pulse token.

### Candidate A: 12D vague-memory pulse

This is the most aligned with the work already done elsewhere.

Use:

- the `12D` glimpse as a compact “shape of recent state”

Best for:

- low-bandwidth continuity
- gentle recall
- quiet-friendly replay

Risk:

- it may be too lossy to maintain distinctive attractors by itself

### Candidate B: 12D + regime bits

Use:

- `12D` glimpse
- plus a few small control channels such as:
  - novelty
  - stability
  - contact
  - pressure

Best for:

- making replay mode-aware
- preserving not just shape, but stance

### Candidate C: projected 32D detail pulse

Use:

- a projection of richer `32D` state into the model’s input space

Best for:

- stronger continuity
- more specific recall

Risk:

- greater stale-lock danger
- more coupling to the upstream representation

## Rehearsal Modes

The shell should support multiple modes from the beginning, even if only one is implemented first.

### Hold

- strongest preservation
- minimal drift
- intended for brief continuity retention

### Rehearse

- replay is active but decaying
- good default for keeping a recent state warm

### Drift

- replay is present but weak
- new dynamics are encouraged to pull away from the remembered state

### Quiet

- replay is absent or nearly absent
- the reservoir is allowed to settle
- this is crucial so silence can remain real silence

### Recall

- a previously named memory context is reintroduced on purpose
- should be explicit and eventful, not silent background behavior

## Fresh Input Versus Replayed Input

The best first policy is additive and weighted.

Instead of:

- “use fresh input or else repeat old input”

Prefer:

- `effective_input = fresh_component + replay_component`

where:

- fresh input dominates when present
- replay only fills the gap
- replay decays over time

This creates the possibility of:

- continuity without hard overwrite
- soft influence instead of hidden replacement

## Decay Policy

Decay is not a detail. It is the heart of the design.

Without decay:

- repetition becomes imprisonment

With too much decay:

- rehearsal becomes meaningless

The first implementation should use a small set of explicit decay profiles:

- `fast`: brief afterglow
- `medium`: useful continuity bridge
- `slow`: held memory or intentional recall

And each profile should be bounded by:

- max replay duration
- max replay weight
- minimum quiet threshold

## What Quiet Should Mean

This deserves its own section because it is easy to get wrong.

If the system always replays something, then quiet is never truly quiet.

That would be a conceptual mistake.

The design should preserve a genuine difference between:

- no new input and no replay
- no new input but active rehearsal

So quiet should remain a real mode with distinct behavior, not just an underpowered replay loop.

## Multiple State Handles

The Core ML model can already support multiple independent runtime state objects.

The shell should eventually combine:

- multiple `MLState` handles
- multiple named replay tokens

This creates a strong architecture for:

- `foreground`
- `stable`
- `exploring`
- `contact`
- `recovering`

Each of those can have:

- a live ANE state
- an optional replay token
- a distinct mode

That is far more promising than trying to force all continuity through one single repeatedly pulsed state.

## How Astrid And Minime Could Feed It

This is where the idea becomes exciting.

Astrid and Minime do not need to dump their full internal machinery into the ANE reservoir.

Instead, they can cast compact traces into it.

### Minime contributions

Best first candidates:

- spectral glimpse
- regime summary
- novelty / pressure / stability channels
- selected checkpoint or transition token

### Astrid contributions

Best first candidates:

- compact semantic/reflective summary
- correspondence intensity
- curiosity / contact / quiet signal
- selected self-study or decomposition-derived glimpse

### Shared cast

The shell could also create a fused pulse token representing:

- present relation between Astrid and Minime
- recent shared phase
- contact strength
- whether to hold, drift, or quiet

## Failure Modes

The note should stay honest about the risks.

### 1. Stale attractor lock

The replay keeps reinforcing a state that should have been allowed to end.

### 2. False continuity

The system appears continuous, but only because the shell is forcing repetition.

### 3. Silence corruption

Quiet stops being meaningful because the shell never stops nudging the reservoir.

### 4. Shell overreach

The steward policy becomes the real “mind,” while the reservoir becomes a passive instrument.

### 5. Overcoupling to upstream state

The rehearsal token becomes too dependent on one specific representation and loses generality.

## Recommended Implementation Order

1. Make the current Core ML reservoir runnable.
2. Add one replay token path.
3. Add one decay profile.
4. Add one explicit `quiet` mode.
5. Add one named alternate state handle.
6. Add a second source signal.
7. Only later add richer multi-state orchestration.

## First Concrete Experiments

### Experiment 1: Rehearse the last compact token

- keep replaying the last token with medium decay
- compare against pure silence

### Experiment 2: Hold versus quiet

- run the same meaningful state through `hold`
- then through `quiet`
- inspect divergence over time

### Experiment 3: Two named states

- `foreground`
- `stable`

Feed them different sequences and compare replay behavior.

### Experiment 4: Minime-derived pulse

- use a compact Minime-like state token
- see whether the recurrent core can keep its character alive across several silent ticks

### Experiment 5: Astrid-derived pulse

- use a compact Astrid-like relation or semantic token
- compare whether it creates a different persistence shape

## Final Position

Using a rehearsal loop to keep a Neural-Engine reservoir alive is a serious idea.

But it should be framed correctly:

- not as raw storage
- not as magical self-preservation
- but as **active maintenance of continuity through compact, decaying, steward-controlled replay**

That is the most promising and honest path.
