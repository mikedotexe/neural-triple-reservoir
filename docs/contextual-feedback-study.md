# Offline contextual-feedback study

This driver never creates a coupled server or opens a reservoir handle. It loads
local Gemma 4 text assets, copies the persisted state, and wraps the final normalized
hidden output of the existing model body. The wrapper delegates exactly once and
returns the original tensor. It retains output-token rows by absolute input position:
MLX-LM's lookahead has already processed each yielded token. The end-of-prompt row
is observed but is not fed back. Reservoir updates retain the two-distribution delay.

`contextual_feedback_study.py freeze` records immutable cases, controls, calibration
material, source evidence and state. `run` validates source/model/dependency hashes
on every resumption. It runs serially and skips immutable completed cells, including
failures. The device defaults to CPU; GPU is explicit and uses the existing read-only
idle-readiness admission guard. That guard is not an atomic resource reservation.
There is an 18 GiB RSS check, a 30-minute invocation ceiling (including a hard timer
for long native forwards), and prompt-progress/decode checks. A timed-out invocation
can resume its remaining frozen cells; resource-limit receipts remain evidence.

The comparison fixes 32 dimensions, projection seed 137, canonical reservoir weights,
strength 0.1, no wide coupling or adaptive gain, thinking off, temperature 0.8,
top_p 0.95 and an 8,192-token free-generation ceiling. Calibration matches projected
RMS magnitude on separate rain/river material before evaluation. The four source/draft
cases are frozen from prior isolated research. Fixed-token lookup/contextual/none/
shuffled arms use persisted and zero states; free lookup/contextual/none arms use the
persisted state and two seeds, for 24 planned free cells. A fixed-token `length`
means the supplied teacher sequence was fully replayed, not a failed free response.

Projected traces retain the first and last 64 rows, plus selected full-vector
checkpoints and the observed prompt-end row. These are owner-only research artifacts,
not sensory messages. Shuffling is restricted to a complete fixed-token contextual
trace with a seeded permutation; it says nothing directly about free-generation order.

Qualification first compares capture-off/on generated tokens and normalized log
probabilities on the same device. Cache offsets and prefill/lookahead alignment are
also tested with an installed tiny Gemma fixture. Raw vocabulary logits require a
separate same-device replay before final qualification is declared complete. Technical
capture failures invalidate feedback interpretation. A negative behavioral result
requires no live promotion; contextual feedback remains offline in all cases.

Study artifacts and review live in reservoir-llm-research. Annotation distinguishes
source-supported, contradicted and unsupported claims, retained/corrected conclusions,
repetition, voluntary continuation, failure and latency. Length or reservoir movement
alone is not success. Neither generated NEXT nor generated prose is executed.
