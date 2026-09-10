# Coupled generation controls

The HTTP worker accepts temperature (finite, >=0), top_p/min_p (0..1), top_k
(nonnegative integer, strictly below the model vocabulary when enabled),
repetition_penalty (finite, >0), and repetition_context_size (positive integer,
requires a penalty). Null, booleans-as-numbers, unsupported controls, conflicting
aliases and streaming are errors before enqueueing. Text messages, model label,
handle, aperture and model_qos_v1 remain supported.

Omission preserves the previous sampler: temperature 0.8, no top_p/top_k/min_p
filters or additional repetition penalty. Explicit penalty uses a 20-token window
unless supplied. Reservoir processing stays first; optional library repetition
processing follows, then the installed MLX-LM sampler. Temperature zero selects
greedy decoding, so the receipt reports no active sampling filters.

In-flight idempotency keys bind copied messages, resolved controls, output budget,
model label, handle and aperture. Changed inputs return 409; identical pending or
active retries share one generation. Keys are released when the job completes;
this is not a persistent response cache.

Completion usage counts yielded tokens, including terminal and filtered tokens,
not the unused lookahead sample. `coupled_generation_v1` separates those counts,
visible token pieces, raw/cleaned characters, resolved controls, reservoir settings
and loaded source identity. Stop means a configured server stop token was observed;
length means the output allowance ended. Errors remain failed requests. Empty
prose is possible even with nonzero completion-token usage.

The termination receipt distinguishes `model_eos`, `channel_boundary`,
`server_stop_special` and `output_allowance`, with the observed stop token ID and
whether it belongs to the tokenizer's EOS set. The inherited policy includes
Gemma's channel-closing marker, which is not a model end-of-turn token. An API
`stop` therefore does not by itself establish that a final answer was produced.
The legacy `terminal_tokens` count is explicitly scoped to the server stop policy;
`visible_tokens` counts detokenizer inputs before cleanup, not final prose tokens.
These diagnostics do not change the stop policy or enable thinking.

Consumers must retain caller requests, serialized adapter choices and this server
receipt as separate evidence. Ollama does not echo all applied settings; its sent
options or model defaults must not be labelled server-confirmed.

The installed library's lookahead and the current reservoir tick schedule are
unchanged. Contextual capture and feedback experiments are separate offline tools;
they are not imported into this server and have no live activation authority.
