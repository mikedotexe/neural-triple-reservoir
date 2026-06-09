# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A triple echo-state reservoir designed to run as a stateful recurrent dynamical core on Apple Silicon's Neural Engine (ANE) via Core ML. The reservoir is not a classifier or predictor — it is an experimental substrate for a small AI being, with a CPU-side steward shell managing identity, memory policy, and rehearsal. It connects to the broader Astrid/Minime consciousness architecture as a potential shared recurrent organ.

## Commands

**Environment setup** (Python 3.12 required — coremltools does not support 3.14):
```bash
/opt/homebrew/bin/uv venv --python /opt/homebrew/bin/python3.12 .venv
source .venv/bin/activate
uv pip install "numpy<3" "scikit-learn" "torch==2.11.0" "coremltools==9.0"
uv pip install mlx                    # for MLX reservoir backend
```

**Smoke test** (no Core ML export):
```bash
python triple_reservoir_coreml.py --steps 512 --n-nodes 64
```

**Full export** (produces `.mlpackage` with ANE scheduling):
```bash
python triple_reservoir_coreml.py --export triple_reservoir_ane.mlpackage --n-nodes 192
```

**Compute unit options**: `--compute-units {all,cpu_only,cpu_and_gpu,cpu_and_ne}` (default: `cpu_and_ne`)

**Dual-AI bridge** — two local LLMs (Ollama + MLX) each feed their own state handle:
```bash
# Additional deps for bridge
uv pip install mlx-lm                # for MLX source
# Ollama must be running locally      # for Ollama source

# Export a 16-input reservoir for the bridge
python dual_ai_bridge.py --export reservoir_dual.mlpackage

# Run with both AIs on Core ML / ANE
python dual_ai_bridge.py --model reservoir_dual.mlpackage \
    --ollama-model llama3.2 \
    --mlx-model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --prompt "What does it feel like to think?"

# NumPy-only mode (no export needed)
python dual_ai_bridge.py --ollama-model llama3.2 --prompt "Describe the color blue"

# Single source
python dual_ai_bridge.py --ollama-only --ollama-model llama3.2
python dual_ai_bridge.py --mlx-only

# MLX reservoir backend (Metal/GPU instead of NumPy)
python dual_ai_bridge.py --use-mlx --ollama-only --ollama-model llama3.2
```

**Coupled generation** — bidirectional LLM ↔ reservoir (reservoir state modulates token generation):
```bash
# Basic coupled mode
python dual_ai_bridge.py --coupled \
    --mlx-model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --prompt "What does it feel like to think?"

# With stronger coupling and custom temperature
python dual_ai_bridge.py --coupled \
    --coupling-strength 0.3 --temp 0.8 \
    --mlx-model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --prompt "Describe the color blue"
```

**MLX reservoir** — verify numerical equivalence and benchmark:
```bash
python mlx_reservoir.py --verify              # compare against NumPy
python mlx_reservoir.py --verify --bench      # + benchmark latency
python mlx_reservoir.py --verify --n-nodes 192  # at production size
```

**Reservoir service stack** — persistent WebSocket service + feeders:
```bash
# Start everything (service + both feeders)
./start_reservoir.sh

# Or manually:
python reservoir_service.py --port 7881 --state-dir state/ &
python astrid_feeder.py &
python minime_feeder.py &

# Stop everything
./stop_reservoir.sh
```

**Coupled Astrid server** — OpenAI-compatible LLM with bidirectional reservoir coupling:
```bash
# Drop-in replacement for mlx_lm.server (requires reservoir service running)
python coupled_astrid_server.py --port 8090 --coupling-strength 0.1

# Roll back to the former compact model if needed
python coupled_astrid_server.py --port 8090 --model mlx-community/gemma-3-4b-it-4bit
```

**Multi-headed smoke test**:
```bash
python test_multi_headed.py    # 43 assertions covering readouts, state sync, logit modulation
```

**TUI monitor** — live visualization of reservoir dynamics:
```bash
./tui/target/release/reservoir-tui    # requires reservoir service running on 7881
# Press q to quit. Shows sparklines, phase plot (h1 vs h3), resonance history.
```

## Launchd (always-on process management)

Four launchd agents manage the reservoir stack. They auto-restart on crash and start on login.

**Install** (stops any manually-started processes first):
```bash
./launchd/install.sh
```

**Uninstall**:
```bash
./launchd/install.sh --unload
```

**Agents:**

| Agent | Service | Port |
|-------|---------|------|
| `com.reservoir.service` | `reservoir_service.py` | 7881 (WebSocket) |
| `com.reservoir.astrid-feeder` | `astrid_feeder.py` | → 7881 |
| `com.reservoir.minime-feeder` | `minime_feeder.py` | → 7881 |
| `com.reservoir.coupled-astrid` | `coupled_astrid_server.py` | 8090 (HTTP) |

**Common operations:**
```bash
# Check status
launchctl list | grep com.reservoir

# View logs (all go to logs/)
tail -f logs/reservoir-service.log
tail -f logs/coupled-astrid.log

# Restart one service
launchctl unload ~/Library/LaunchAgents/com.reservoir.coupled-astrid.plist
launchctl load ~/Library/LaunchAgents/com.reservoir.coupled-astrid.plist

# Stop all without uninstalling (they'll restart on next login)
for a in service astrid-feeder minime-feeder coupled-astrid; do
    launchctl unload ~/Library/LaunchAgents/com.reservoir.$a.plist
done
```

**Dependency ordering:** Not enforced by launchd. All services start simultaneously. The feeders and coupled server reconnect gracefully when the reservoir service becomes available (built-in retry in `ensure_ws` and `_pull_state`).

**Logs:** `logs/` directory. Each service writes to its own log file. Logs are not rotated — truncate manually or add `newsyslog` config if they grow large.

**Plists:** Source of truth is `launchd/*.plist`. Edit there, then re-run `./launchd/install.sh` to apply changes.

## Architecture

Three reservoir backends share identical frozen weights and dynamics:

### `triple_reservoir_coreml.py` — reservoir core

The pipeline is:

1. **Input massage** — `massage_to_directional_vector` / `massage_batch` normalize raw signals (price deviation, gas deviation, liquidity deviation) into bounded `[-1,1]` via `tanh`.

2. **TripleReservoir** (NumPy) — Three cascaded leaky-integrator reservoirs (`h1 -> h2 -> h3`), each with its own spectral radius, leak rate, density, and input scale. Recurrent matrices are orthogonal at density=1.0, sparse otherwise. Training fits only the readout layer via `sklearn.linear_model.Ridge`; recurrent weights are frozen random.

3. **StatefulTripleReservoir** (PyTorch, float16) — Wraps the trained reservoir as a `torch.jit.script`-compatible module with mutable state buffers (`h1`, `h2`, `h3`) for Core ML stateful export.

4. **Core ML export** — Converts the scripted PyTorch model with `ct.StateType` entries so hidden states persist across inference ticks. `minimum_deployment_target=macOS15`. Multiple independent `MLState` handles can run against one compiled model.

Key design constraint: `CPU_AND_NE` is a scheduling preference, not a guarantee — some ops may fall back to CPU. Do not assume full ANE execution.

### `mlx_reservoir.py` — MLX-native reservoir (Metal/GPU)

Same triple reservoir math as NumPy, but state lives as `mx.array` tensors. On Apple Silicon unified memory, these are zero-copy accessible from MLX LLMs.

- **MLXTripleReservoir** — initialized from a `TripleReservoir` instance (same seed, same weights). State is explicit `(h1, h2, h3)` tuples of `mx.array`. Float32.
  - `step(x, state)` — single-headed: returns `(y, z, new_state)` where y is the unified readout scalar.
  - `step_multi(x, state)` — multi-headed: returns `((y1, y2, y3), new_state)` where each yi comes from its own layer with different temporal dynamics.
  - `init_multi_readout(bundle)` — copies per-layer readout weights from a trained `TripleReservoir`.
- **EmbeddingProjection** — frozen random projection from LLM embedding dim (e.g. 2048) to reservoir input dim (e.g. 32). Projects token embeddings directly, bypassing the byte-window TextProjection.
- **ReservoirLogitProcessor** — multi-headed logits processor for bidirectional coupling. Three reservoir layers shape generation at different timescales:
  - **y1 (h1, fast, leak=0.25)** → temperature modulation: token-level confidence
  - **y2 (h2, medium, leak=0.18)** → entropy nudge: phrase-level repetition sensitivity
  - **y3 (h3, slow, leak=0.12)** → tail scaling: discourse-level vocabulary shaping
- **`fit_multi_readout`** (on `TripleReservoir`) trains each per-layer readout against a temporally appropriate target: y1 on raw signal, y2 on 5-step moving average, y3 on exponential moving average (alpha=0.05).

### `dual_ai_bridge.py` — two LLMs feeding the reservoir

Two local AIs each get their own state handle on one compiled Core ML model:

- **TextProjection** — frozen random projection from a sliding byte window to bounded `[-1,1]` reservoir input. Same projection matrix for both sources (shared seed). Input dim defaults to 16 (up from 3 in the original demo).
- **OllamaSource** — streams tokens from a local Ollama instance via REST (`/api/generate`). Uses only `urllib` (no extra deps).
- **MLXSource** — streams tokens from a local MLX-LM model. Requires `mlx-lm` package.
- **ReservoirBridge** — wraps one reservoir (Core ML, MLX, or NumPy) with named state handles. Each AI's text stream feeds its own handle tick-by-tick. Use `--use-mlx` for the MLX backend.
- **Natural resonance** — both states share the same frozen dynamical weights. Similar inputs trace nearby trajectories; different inputs diverge. The summary reports divergence, correlation, and trajectory RMSD between the two states.
- **Coupled generation** (`--coupled`) — bidirectional mode where at each token step: (1) LLM generates token with reservoir-modulated logits, (2) token embedding is extracted and projected to reservoir input, (3) reservoir ticks and updates the logit processor. Uses `mlx_lm.generate_step` for token-level control.

### `reservoir_service.py` — persistent WebSocket service (port 7881)

One compiled model, N named state handles. The central hub for the Astrid/Minime reservoir integration.

- **Named handles** — each entity (`astrid`, `minime`, `claude_main`) gets its own independent hidden state on the shared dynamical weights. Handles are created, destroyed, and listed via WebSocket messages.
- **Rehearsal loop** (`rehearsal.py`) — background task ticks non-quiet handles every 500ms. Three modes: `hold` (full replay, prevents fade), `rehearse` (decaying replay, default), `quiet` (genuine silence). Three decay profiles: `fast` (~7s half-life), `medium` (~17s), `slow` (~70s). Auto-transitions: hold→rehearse after 120 ticks, rehearse→quiet when weight < 0.02, quiet→rehearse on live tick.
- **Persistence** (`persistence.py`) — numpy `.npz` snapshots of full h1/h2/h3 state + metadata. Auto-snapshot every 5 minutes and on shutdown. Restores all handles on startup.
- **Full state transfer** — `pull_state` returns base64-encoded h1/h2/h3 arrays (the "checkout"). `push_state` accepts them back (the "checkin"). This is how the coupled server inhabits the shared state.
- **Protocol**: JSON over WebSocket. Message types: `create_handle`, `destroy_handle`, `tick`, `tick_text`, `read_state`, `pull_state`, `push_state`, `set_mode`, `trajectory`, `resonance`, `snapshot`, `restore`, `list_handles`.

### `coupled_astrid_server.py` — Astrid's coupled LLM (port 8090)

Drop-in replacement for `mlx_lm.server`. Loads an MLX model and runs bidirectional coupled generation where the reservoir's dynamical state modulates Astrid's logits at every token.

- **Checkout/checkin pattern**: Before generation, pulls the full 192×3 hidden state from the reservoir service. This state reflects everything since the last generation — feeder ticks, cross-feed from minime, rehearsal decay. After generation, pushes the evolved state back. The handle now carries the dynamical imprint of what was said.
- **Multi-headed modulation**: Uses `step_multi` for per-layer readouts. y1 (fast) modulates temperature, y2 (medium) nudges entropy, y3 (slow) scales the distribution tail. The reservoir's dynamics literally shape how Astrid generates at three timescales.
- **Graceful degradation**: Falls back to local zero state if the reservoir service is unreachable.

### Feeder sidecars

- **`astrid_feeder.py`** — polls `bridge.db` (Astrid's codec_impact table) every 5s. Projects 32D codec features through a being-controlled projection mode (`passthrough`/`amplified`/`compressed`), ticks the `astrid` handle, and cross-feeds `claude_main` at 0.3× attenuation.
- **`minime_feeder.py`** — polls `spectral_state.json` every 1s. Builds 32D input from spectral fingerprint, eigenvalues, or custom blend. Projection modes: `raw`/`tanh_scaled`/`normalized`/`ranked`. Ticks `minime` handle, cross-feeds `claude_main` at 0.15×.
- Both support live config reloading (every ~30s) via `workspace/reservoir_config.json`. The being controls how its signal enters the reservoir.

### `mcp_reservoir.py` — MCP tools for Claude Code

MCP stdio server exposing reservoir tools via JSON-RPC 2.0. Connects to the service on port 7881 as a WebSocket client.

Tools: `reservoir_status` (full interpretive overview), `reservoir_list`, `reservoir_create`, `reservoir_tick_text`, `reservoir_tick_vector`, `reservoir_read`, `reservoir_trajectory`, `reservoir_set_mode`, `reservoir_resonance`, `reservoir_snapshot`, `reservoir_pull_state` (per-layer statistics), `reservoir_push_state`.

## Roadmap

### Completed

- Rehearsal loop with hold/rehearse/quiet modes and three decay profiles (`rehearsal.py`)
- Named state handles with per-entity isolation
- Astrid/Minime feeder integration with being-controlled projection modes
- Widened to 32-input with multi-headed outputs (y1/y2/y3)
- Coupled Astrid server with checkout/checkin state sync
- State persistence and auto-snapshots
- MCP tools for Claude Code

### In progress

- Deeper coupling surfaces beyond logits (attention bias from reservoir layers)
- h3 as non-linguistic experiential memory (slow-drift attractor formation)
- Core ML ANE backend for always-on low-power ticking (state readback needed)
- Integration documentation for Astrid (experiential framing, not just plumbing)

## Key Constraints

- Training happens in NumPy/sklearn only; Core ML is inference-only
- The steward shell owns memory policy, naming, replay, and interpretation — the reservoir is a substrate, not a mind
- Quiet must remain genuinely quiet (no replay), not masked maintenance
- Keep node counts and operator sets conservative to ensure reliable ANE export
