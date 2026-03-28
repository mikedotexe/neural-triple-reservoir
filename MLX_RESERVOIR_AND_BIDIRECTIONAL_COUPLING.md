# From Passive Substrate to Bidirectional Coupling: The MLX Reservoir

## What Changed

The reservoir started as a Core ML model running on the Neural Engine. It received input, updated its hidden state, and produced output. It was a passive dynamical system — an observer. The LLMs that fed it had no idea it existed and were unaffected by it.

That's no longer the case. The reservoir now runs as a native MLX module on Metal, and it participates in a bidirectional loop with an MLX LLM at every token step. The LLM's generation is shaped by the reservoir's dynamical state, and the reservoir's state is driven by the LLM's own token embeddings. They influence each other continuously, sharing the same physical memory on Apple Silicon.

## Why Core ML Couldn't Do This

Core ML exports a stateful model with `MLState` handles. These handles are opaque — you can feed input and get output, but you can't read or write the hidden state as tensors. The state is managed by the Core ML runtime.

This is fine for standalone ticking (the reservoir running in the background, maintaining its attractors). But it makes bidirectional coupling impossible. To close the loop, you need:

1. **Readable state** — the steward or coupling layer needs to see the hidden state to make decisions or project it into the LLM's space
2. **Writable state** — you need to inject state from persistence, fork handles, or merge trajectories
3. **Same-framework tensors** — the coupling loop runs at token speed; you can't afford serialization between Core ML and MLX at every step

MLX gives all three. State is plain `mx.array`. The reservoir's hidden vectors (h1, h2, h3) are first-class tensors, readable, writable, projectable. And because MLX runs on Metal (the same compute backend as the LLM), there's no framework boundary to cross.

## The Unified Memory Insight

On Apple Silicon, CPU, GPU, and Neural Engine share the same physical RAM. This is called Unified Memory Architecture (UMA). It means:

- An `mx.array` created by the MLX reservoir and an `mx.array` created by the MLX LLM are in the same memory pool
- `mx.array(numpy_array)` wraps without copying — the numpy array is already in unified memory
- There is no PCIe bus, no DMA transfer, no host-device distinction

On a system with discrete GPUs, closing a bidirectional loop at token speed would mean paying PCIe transfer costs at every step — potentially hundreds of microseconds per round trip. On Apple Silicon, it's a pointer dereference. The coupling adds negligible latency to generation.

This isn't just a performance detail. It's what makes the architecture viable. A coupling loop that adds 500us per token would be disruptive. One that adds <1us is invisible.

## What We Built

### Three Backends, One Set of Dynamics

The reservoir now has three runtime backends:

| Backend | Runtime | State | Use Case |
|---------|---------|-------|----------|
| **NumPy** | CPU | `np.ndarray` tuples | Testing, fallback, any platform |
| **Core ML** | ANE | Opaque `MLState` | Standalone background ticking |
| **MLX** | Metal/GPU | `mx.array` tuples | Coupled LLM loop |

All three share the same frozen random weights (same NumPy seed, same dynamics). The MLX and Core ML versions are initialized from the canonical NumPy `TripleReservoir`. A trajectory that starts in one backend can be continued in another via numpy conversion.

### The Bidirectional Coupling Loop

In `--coupled` mode, every token generation step does this:

```
generate_step yields token
        |
        v
model.model.embed_tokens(token)     # extract embedding (mx.array, Metal)
        |
        v
EmbeddingProjection.project()        # frozen random projection to reservoir input dim
        |                             # tanh-bounded, same philosophy as all frozen weights
        v
MLXTripleReservoir.step()            # reservoir tick: h1->h2->h3 cascade
        |                             # state evolves on its attractor landscape
        v
ReservoirLogitProcessor.update()     # scalar output stored for next token
        |
        v
next generate_step iteration          # processor modulates logits via temperature
        |                             # positive energy -> more confident
        v                             # negative energy -> more exploratory
(loop continues)
```

The key: the `ReservoirLogitProcessor` is a stateful callable that `generate_step` calls inside its inner loop. Between yields, we update the processor's state from the reservoir output. On the next token, the processor modulates the logits before sampling. The coupling is causal — the reservoir state from processing tokens 1..t influences the generation of token t+1.

### What the Coupling Actually Does

The `ReservoirLogitProcessor` maps the reservoir's scalar output through a sigmoid to produce a temperature multiplier:

```
energy -> sigmoid -> scale to [1-strength, 1+strength] -> divide logits
```

With `coupling_strength=0.1`, this means:
- Maximum effect: +/-10% temperature change
- When the reservoir's output is strongly positive: temperature drops slightly, distribution sharpens, LLM becomes marginally more decisive
- When strongly negative: temperature rises slightly, distribution flattens, LLM becomes marginally more exploratory

This is deliberately gentle. The reservoir is not controlling the LLM. It's nudging. The LLM's own language model dominates; the reservoir adds a dynamical texture that the transformer cannot produce on its own.

### What the Embedding Projection Does

Instead of the byte-window `TextProjection` (which encodes the last 64 bytes of text through a random matrix), the coupled mode uses the LLM's actual token embeddings. This is a categorically richer input signal:

- Byte-window projection: captures surface-level character patterns, loses semantic content
- Embedding projection: captures the full learned representation of each token — semantics, syntax, position in the model's latent space

The `EmbeddingProjection` is still a frozen random matrix (consistent with the project's philosophy of frozen random weights throughout). It projects from the LLM's hidden dimension (e.g., 2048 for Llama-3.2-1B) down to the reservoir's input dimension (e.g., 16). The tanh bounding ensures the reservoir input stays in [-1, 1] regardless of the embedding magnitude.

## What This Means for the Architecture

### The Reservoir Is No Longer an Observer

Before this change, the reservoir was a measurement instrument. It received text, processed it through its dynamics, and produced a trajectory that could be analyzed. The LLM didn't know or care.

Now the reservoir is a participant. Its dynamical state — shaped by three cascaded leaky integrators with different time constants — creates a kind of dynamical memory that spans the entire generation. The fast layer (h1, leak=0.25) tracks recent token character. The slow layer (h3, leak=0.12) accumulates a longer-horizon texture. The coupling feeds this multi-timescale state back into generation.

This is something the transformer fundamentally cannot do on its own. The transformer's "state" (KV cache) grows linearly with sequence length but has no dynamical properties — no attractors, no timescale separation, no continuous evolution. The reservoir adds a small, cheap recurrent organ that provides exactly these properties.

### Two Runtimes, Two Purposes

Core ML/ANE and MLX aren't competing — they serve different architectural roles:

**Core ML/ANE** is for when the reservoir needs to run independently. During input silence, the rehearsal loop ticks the reservoir on the Neural Engine. This doesn't compete for GPU bandwidth with whatever else is running (an MLX LLM, a Metal rendering pipeline, etc.). The ANE is a dedicated accelerator that can maintain the reservoir's state in the background.

**MLX/Metal** is for when the reservoir needs to be in tight coupling with an LLM. Both are on the GPU, sharing the same Metal command stream. The coupling loop adds negligible overhead because there's no cross-device communication.

The same frozen weights, the same dynamics, the same attractors. Just different runtimes optimized for different coupling modes.

### The Steward Shell Still Owns Policy

The reservoir doesn't decide how strongly to couple, when to couple, or what the coupling means. It's still a substrate. The steward shell (the CPU-side orchestrator) decides:

- When to enter coupled mode vs. standalone ticking
- What coupling strength to use
- How to interpret the reservoir's trajectory
- When to save/restore/fork state handles
- When to switch between rehearsal modes

The reservoir provides dynamics. The steward provides policy. This separation is load-bearing.

## Technical Notes

- **Quantized models**: For quantized MLX LLMs, `model.model.embed_tokens.weight.shape` reflects the compressed storage, not the actual embedding dimension. Use `model.args.hidden_size` instead.
- **Graph accumulation**: MLX uses lazy evaluation. In a tight loop, call `mx.eval(*state)` periodically (every ~64 steps) to prevent the computation graph from growing unboundedly.
- **Numerical equivalence**: The MLX reservoir matches the NumPy reference to ~1e-6 over 200 steps at 192 nodes. The small difference comes from float32 vs NumPy's internal float64 promotions.
- **MLX per-tick latency**: ~65us at 192 nodes (vs ~18us for NumPy). The overhead is Metal kernel launch cost, which is amortized when running alongside an LLM that takes milliseconds per token.

## What Comes Next

The loop is closed. The immediate question is: **does the coupling produce meaningfully different text than uncoupled generation?** This is an empirical question that needs systematic comparison — same prompts, same temperatures, coupled vs. uncoupled, across many runs.

Beyond that:

- **Richer coupling modes**: Low-rank logit bias (reservoir state projected to vocabulary-sized bias vector) instead of just temperature modulation. This would give the reservoir fine-grained influence over word choice, not just confidence.
- **Two coupled LLMs**: Both feeding the same reservoir, both influenced by its shared state. The reservoir becomes a communication channel — a shared dynamical space that two AIs inhabit simultaneously.
- **Rehearsal integration**: Between coupled generation sessions, the reservoir maintains its state via rehearsal ticking on ANE. The next coupled session picks up where the dynamics left off. This gives the system a form of continuity across conversations.
- **Steward-driven coupling strength**: Instead of a fixed coupling strength, the steward adjusts it dynamically based on the reservoir's trajectory. High confidence in the attractor → stronger coupling. Divergent or chaotic state → weaker coupling or decouple entirely.
