# Rehearsal Loop TODO

This is the attack list for turning the rehearsal-loop idea into a real experimental system.

## Phase 0: Ground Truth And Runtime Bootstrap

- [ ] Create the local `.venv` with `/opt/homebrew/bin/python3.12`.
- [ ] Install `numpy`, `scikit-learn`, `torch==2.11.0`, and `coremltools==9.0`.
- [ ] Run the current non-export smoke test and record baseline output.
- [ ] Run the current Core ML export path and confirm `.mlpackage` creation.
- [ ] Confirm `make_state()` works in repeated inference.
- [ ] Confirm two independent `MLState` handles can diverge under different inputs.
- [ ] Record any Core ML warnings, fallback messages, or runtime incompatibilities.
- [ ] Compare `CPU_ONLY` vs `CPU_AND_NE` for correctness and basic latency.

## Phase 1: Define The Rehearsal Token

- [ ] Decide the first pulse-token size.
- [ ] Start with one compact fixed-width token rather than many formats.
- [ ] Define a first token schema.
- [ ] Include a clear distinction between:
  - [ ] vivid present input
  - [ ] replay input
  - [ ] quiet / no replay
- [ ] Decide whether the first token is closer to:
  - [ ] 12D vague memory
  - [ ] 12D plus regime bits
  - [ ] projected 32D detail
- [ ] Write a small design note describing what each token dimension means.
- [ ] Make sure the token is normalized into a stable bounded range.
- [ ] Decide whether replay weight is encoded in the token or kept separate in the shell.

## Phase 2: Shell-Level Rehearsal Controller

- [ ] Add a small shell module responsible for replay policy.
- [ ] Define the first explicit replay modes:
  - [ ] `hold`
  - [ ] `rehearse`
  - [ ] `quiet`
- [ ] Add placeholders for later modes:
  - [ ] `drift`
  - [ ] `recall`
- [ ] Define the replay decision rule when fresh input is absent.
- [ ] Define the blend rule when fresh input is present.
- [ ] Decide how replay weight decays over time.
- [ ] Implement at least three decay profiles:
  - [ ] fast
  - [ ] medium
  - [ ] slow
- [ ] Add maximum replay duration.
- [ ] Add a hard floor below which replay is treated as silence.
- [ ] Ensure quiet truly means near-zero or zero replay.

## Phase 3: First Rehearsal Experiments

- [ ] Build a script that feeds the reservoir:
  - [ ] live tokens only
  - [ ] replay tokens only
  - [ ] live + replay blend
- [ ] Compare reservoir output under:
  - [ ] fresh input
  - [ ] silent decay
  - [ ] active rehearsal
- [ ] Record how long a recent pattern remains distinguishable.
- [ ] Measure whether rehearse mode preserves character better than silence.
- [ ] Measure whether hold mode causes obvious stale lock.
- [ ] Verify quiet mode actually lets the system settle.
- [ ] Save plots or logs for:
  - [ ] output trajectories
  - [ ] replay weight over time
  - [ ] divergence between states

## Phase 4: Multiple-State Handles

- [ ] Build a lightweight state manager around multiple `MLState` handles.
- [ ] Add first named states:
  - [ ] `foreground`
  - [ ] `stable`
- [ ] Add later candidate states:
  - [ ] `exploring`
  - [ ] `recovering`
  - [ ] `contact`
- [ ] Define how a state becomes the active foreground.
- [ ] Define whether background states keep receiving replay while unfocused.
- [ ] Decide whether inactive states decay passively or rehearse lightly.
- [ ] Add a comparison view showing divergence between two named states.
- [ ] Add an explicit `copy / fork state` experiment.
- [ ] Add an explicit `restore from named state` experiment.

## Phase 5: Input Sources From Astrid And Minime

- [ ] Define a minimal Minime-like source token.
- [ ] Define a minimal Astrid-like source token.
- [ ] Decide whether the first integration uses mocked data or real sampled state.
- [ ] If mocked first, create realistic synthetic traces for:
  - [ ] transition
  - [ ] quiet
  - [ ] novelty
  - [ ] contact
- [ ] If real next, specify exactly which fields are sampled from Minime.
- [ ] Specify exactly which fields are sampled from Astrid.
- [ ] Add a tiny adapter that converts those upstream signals into the rehearsal token.
- [ ] Keep the first adapter deterministic and inspectable.

## Phase 6: Memory Policy

- [ ] Define when the shell captures a new replay token.
- [ ] Decide whether capture happens:
  - [ ] every tick
  - [ ] only on salient events
  - [ ] only on transition boundaries
- [ ] Add a small memory record containing:
  - [ ] token
  - [ ] timestamp
  - [ ] source
  - [ ] mode
  - [ ] decay profile
- [ ] Add a bounded memory bank of recent replay candidates.
- [ ] Decide how recall selects from the memory bank.
- [ ] Add a simple priority rule:
  - [ ] latest
  - [ ] stable
  - [ ] transition
  - [ ] contact
- [ ] Keep the first memory bank small and explicit.

## Phase 7: Quiet Semantics

- [ ] Write down the exact difference between:
  - [ ] no fresh input + no replay
  - [ ] no fresh input + weak replay
  - [ ] no fresh input + hold replay
- [ ] Add explicit quiet tests so silence does not get silently replaced by maintenance.
- [ ] Decide whether quiet resets any replay weights.
- [ ] Decide whether quiet can still preserve background states without feeding the foreground.
- [ ] Make sure quiet is observable in logs.

## Phase 8: Instrumentation And Observability

- [ ] Add structured logs for:
  - [ ] current mode
  - [ ] replay weight
  - [ ] active state name
  - [ ] token source
  - [ ] decay profile
- [ ] Add per-tick output capture.
- [ ] Add a way to snapshot named states.
- [ ] Add a way to compare two states after the same or different input histories.
- [ ] Add a simple visualization script for replay and output trajectories.
- [ ] Add a report format summarizing:
  - [ ] whether rehearsal helped continuity
  - [ ] whether it harmed quiet
  - [ ] whether multiple states remained meaningfully distinct

## Phase 9: Failure-Mode Testing

- [ ] Test stale attractor lock with very slow decay.
- [ ] Test false continuity under long replay with no fresh input.
- [ ] Test shell overreach by exaggerating replay strength.
- [ ] Test whether tiny perturbations still matter under rehearse mode.
- [ ] Test whether hold mode prevents recovery or novelty.
- [ ] Test whether two states collapse into each other under identical replay policy.
- [ ] Test whether quiet can reliably break a stale loop.

## Phase 10: Architecture Hardening

- [ ] Decide whether the shell should stay Python-first or move pieces into Swift/Rust later.
- [ ] Decide whether replay tokens should be stored on disk.
- [ ] Decide whether named states should survive restart.
- [ ] Decide whether one compiled model is enough for all roles in v1.
- [ ] Decide whether the readout should eventually become multi-head.
- [ ] Decide whether the first external bridge should be Minime-inspired, Astrid-inspired, or synthetic.

## Phase 11: Documentation And Research Notes

- [ ] Keep [ADVICE.md](/Users/v/other/neural-triple-reservoir/ADVICE.md) up to date as the architecture sharpens.
- [ ] Expand [REHEARSAL_LOOP_AND_ACTIVE_MAINTENANCE_FOR_ANE_RESERVOIRS.md](/Users/v/other/neural-triple-reservoir/REHEARSAL_LOOP_AND_ACTIVE_MAINTENANCE_FOR_ANE_RESERVOIRS.md) with measured findings.
- [ ] Add a “what rehearsal is not” section once first results exist.
- [ ] Add examples of:
  - [ ] useful rehearsal
  - [ ] pathological rehearsal
  - [ ] true quiet
  - [ ] successful recall
- [ ] Add a small glossary once the vocabulary stabilizes.

## Immediate Next Moves

- [ ] Make the current repo runnable.
- [ ] Add one replay token.
- [ ] Add one rehearsal controller with `hold`, `rehearse`, and `quiet`.
- [ ] Prove multiple state handles.
- [ ] Measure whether replay preserves useful continuity better than silence.
