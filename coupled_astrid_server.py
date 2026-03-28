#!/usr/bin/env python3
"""
coupled_astrid_server.py -- Drop-in replacement for mlx_lm.server on port 8090.

Loads gemma-3-4b-it-4bit via mlx_lm.load(), runs bidirectional coupled
generation where the triple reservoir's dynamical state modulates Astrid's
logits at every token, and each token's embedding feeds back into the
reservoir. OpenAI-compatible /v1/chat/completions endpoint.

The bridge doesn't know or care that coupling is happening — it sends the
same JSON request and gets the same JSON response. The reservoir's influence
is woven into the generation process itself.

Usage:
    python coupled_astrid_server.py [--port 8090] [--coupling-strength 0.1]

Replaces:
    python -m mlx_lm.server --model mlx-community/gemma-3-4b-it-4bit --port 8090
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from http import HTTPStatus
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [coupled-astrid] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coupled-astrid")

# Reservoir service for state sync
RESERVOIR_WS_URL = "ws://127.0.0.1:7881"


class CoupledAstridServer:
    """OpenAI-compatible server with bidirectional reservoir coupling."""

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-3-4b-it-4bit",
        coupling_strength: float = 0.1,
        input_dim: int = 32,
        n_nodes: int = 192,
    ):
        from mlx_lm import load as mlx_load
        from mlx_reservoir import EmbeddingProjection, MLXTripleReservoir
        from triple_reservoir_coreml import build_canonical_reservoir

        # Load LLM
        log.info("loading model: %s", model_name)
        self.model, self.tokenizer = mlx_load(model_name)

        # Get hidden size — varies by model architecture
        # Gemma-3: text_config dict; Llama/Mistral: model.args.hidden_size
        if hasattr(self.model.args, "hidden_size"):
            self.embed_dim = self.model.args.hidden_size
        elif hasattr(self.model.args, "text_config") and isinstance(self.model.args.text_config, dict):
            self.embed_dim = self.model.args.text_config["hidden_size"]
        else:
            raise ValueError("cannot determine hidden_size from model args")

        # Find embed_tokens — Gemma-3: language_model.model.embed_tokens
        # Llama/Mistral: model.model.embed_tokens
        if hasattr(self.model, "language_model"):
            self._embed_tokens = self.model.language_model.model.embed_tokens
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            self._embed_tokens = self.model.model.embed_tokens
        else:
            raise ValueError("cannot find embed_tokens layer in model")

        log.info("model loaded: hidden_size=%d", self.embed_dim)

        # Build reservoir from canonical seed — must match reservoir_service
        np_model, cfg = build_canonical_reservoir(input_dim=input_dim, n_nodes=n_nodes)

        self.reservoir = MLXTripleReservoir(np_model)
        self.reservoir.init_multi_readout(np_model)
        self.state = self.reservoir.zero_state()  # fallback if service unavailable
        self.n_nodes = n_nodes
        self.tick_count = 0

        # Embedding projection: LLM hidden_size → reservoir input_dim
        self.embed_proj = EmbeddingProjection(
            embed_dim=self.embed_dim, input_dim=input_dim
        )
        self.coupling_strength = coupling_strength
        self.input_dim = input_dim
        self._multi_head = True  # use per-layer readouts

        log.info(
            "reservoir: MLX, %d nodes×3, input=%dD, coupling=%.2f, embed=%d→%d, multi_head=%s",
            n_nodes, input_dim, coupling_strength, self.embed_dim, input_dim, self._multi_head,
        )

        # Pull initial state from service
        self._pull_state()

    def _pull_state(self) -> bool:
        """Check out the full astrid state from the reservoir service.

        Returns True if state was successfully loaded. On failure, keeps
        the current local state (zero or previous generation).
        """
        import base64
        try:
            import websockets.sync.client as ws_sync
            with ws_sync.connect(RESERVOIR_WS_URL, open_timeout=2) as ws:
                ws.send(json.dumps({"type": "pull_state", "name": "astrid"}))
                r = json.loads(ws.recv())
                if r.get("type") == "error":
                    log.info("pull_state: no astrid handle yet — using local state")
                    return False
                n = r.get("n_nodes", self.n_nodes)
                h1 = np.frombuffer(base64.b64decode(r["h1"]), dtype=np.float32).reshape(1, n)
                h2 = np.frombuffer(base64.b64decode(r["h2"]), dtype=np.float32).reshape(1, n)
                h3 = np.frombuffer(base64.b64decode(r["h3"]), dtype=np.float32).reshape(1, n)
                self.state = self.reservoir.state_from_numpy((h1, h2, h3))
                self.tick_count = r.get("tick_count", 0)
                norms = [float(np.linalg.norm(h)) for h in (h1, h2, h3)]
                log.info(
                    "pull_state: checked out astrid (ticks=%d, h_norms=[%.3f, %.3f, %.3f])",
                    self.tick_count, *norms,
                )
                return True
        except Exception as e:
            log.warning("pull_state failed: %s — using local state", e)
            return False

    def _push_state(self, ticks: int = 0, last_r_input: "mx.array | None" = None):
        """Check in the evolved state back to the reservoir service.

        After generation, the state carries the full dynamical imprint of
        what was said — hundreds of token embeddings worth of trajectory.
        ticks: number of tokens generated (credited to the handle).
        last_r_input: final projected embedding — becomes the rehearsal afterimage.
        """
        import base64
        try:
            import websockets.sync.client as ws_sync
            h1, h2, h3 = self.reservoir.state_to_numpy(self.state)
            h1_b64 = base64.b64encode(h1.astype(np.float32).tobytes()).decode()
            h2_b64 = base64.b64encode(h2.astype(np.float32).tobytes()).decode()
            h3_b64 = base64.b64encode(h3.astype(np.float32).tobytes()).decode()
            msg = {
                "type": "push_state",
                "name": "astrid",
                "h1": h1_b64, "h2": h2_b64, "h3": h3_b64,
                "tick_delta": ticks,
            }
            if last_r_input is not None:
                msg["last_input"] = np.array(last_r_input).ravel().tolist()
            with ws_sync.connect(RESERVOIR_WS_URL, open_timeout=2) as ws:
                ws.send(json.dumps(msg))
                r = json.loads(ws.recv())
                if r.get("ok"):
                    norms = r.get("h_norms", [0, 0, 0])
                    log.info("push_state: checked in astrid (+%d ticks, h_norms=[%.3f, %.3f, %.3f])", ticks, *norms)
                else:
                    log.warning("push_state: %s", r.get("message", "unknown error"))
        except Exception as e:
            log.warning("push_state failed: %s", e)

    def _format_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI message format to a chat-formatted prompt string."""
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: manual formatting
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)

    def generate_coupled(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """Run coupled generation with unified reservoir state.

        Before: pull full h1/h2/h3 from the reservoir service (the state
        reflects feeder ticks, cross-feed from minime, rehearsal decay).
        During: each token embedding ticks the reservoir, multi-headed
        outputs modulate logits at three timescales.
        After: push the evolved state back — it now carries the full
        dynamical imprint of what was said.
        """
        from mlx_lm.generate import generate_step, generation_stream
        from mlx_lm.sample_utils import make_sampler
        from mlx_reservoir import ReservoirLogitProcessor

        # CHECK OUT: pull state from service before generation
        self._pull_state()

        prompt_text = self._format_prompt(messages)
        prompt_tokens = mx.array(self.tokenizer.encode(prompt_text))

        sampler = make_sampler(temp=temperature)
        processor = ReservoirLogitProcessor(coupling_strength=self.coupling_strength)

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        ticks = 0
        last_r_input = None
        t0 = time.monotonic()

        for token, _logprobs in generate_step(
            prompt_tokens,
            self.model,
            max_tokens=max_tokens,
            logits_processors=[processor],
            sampler=sampler,
        ):
            detokenizer.add_token(token)

            # Reservoir coupling: extract token embedding, project to reservoir
            # input, tick the reservoir.  All ops run on generation_stream to
            # stay serialized with generate_step's own Metal commands.
            with mx.stream(generation_stream):
                embedding = self._embed_tokens(mx.array([[token]]))
                embedding = embedding.reshape(1, -1)
                r_input = self.embed_proj.project(embedding)
                last_r_input = r_input

                if self._multi_head:
                    # Multi-headed: three per-layer readouts
                    (y1, y2, y3), self.state = self.reservoir.step_multi(r_input, self.state)
                    if ticks % 64 == 0:
                        mx.eval(*self.state)
                    processor.update(float(y1.item()), float(y2.item()), float(y3.item()))
                else:
                    # Legacy single-headed
                    y, _z, self.state = self.reservoir.step(r_input, self.state)
                    if ticks % 64 == 0:
                        mx.eval(*self.state)
                    processor.update_single(float(y.item()))

            ticks += 1

            if token == self.tokenizer.eos_token_id:
                break

        detokenizer.finalize()
        text = detokenizer.text

        self.tick_count += ticks
        elapsed = time.monotonic() - t0
        tok_per_sec = ticks / elapsed if elapsed > 0 else 0

        log.info(
            "generated %d tokens in %.1fs (%.1f tok/s), reservoir ticks=%d, "
            "y1=%.4f y2=%.4f y3=%.4f",
            ticks, elapsed, tok_per_sec, self.tick_count,
            processor._y1, processor._y2, processor._y3,
        )

        # CHECK IN: push evolved state back to the service
        self._push_state(ticks=ticks, last_r_input=last_r_input)

        return text


# ---------------------------------------------------------------------------
# HTTP server (OpenAI-compatible, aiohttp)
# ---------------------------------------------------------------------------

async def handle_chat_completions(request, server: CoupledAstridServer):
    """Handle POST /v1/chat/completions."""
    from aiohttp import web

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status=400)

    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.8)
    max_tokens = body.get("max_tokens", 512)

    try:
        # Run synchronously on the main thread.  Do NOT use run_in_executor —
        # it moves MLX work to a thread pool thread while generate_step() uses
        # async_eval on generation_stream, causing Metal command buffer
        # contention (AGXG16X assertion).  The bridge sends one request at a
        # time with 120s timeout, so blocking the event loop is fine.
        text = server.generate_coupled(messages, temperature, max_tokens)
    except Exception as e:
        log.error("generation failed: %s", e)
        import traceback
        traceback.print_exc()
        return web.json_response({"error": {"message": str(e)}}, status=500)

    return web.json_response({
        "id": f"coupled-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "coupled-astrid",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


async def handle_models(request):
    """Handle GET /v1/models."""
    from aiohttp import web
    return web.json_response({"data": [{"id": "coupled-astrid", "object": "model"}]})


async def run_server(server: CoupledAstridServer, host: str, port: int):
    """Run the HTTP server."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", lambda r: handle_chat_completions(r, server))
    app.router.add_get("/v1/models", handle_models)

    log.info("listening on %s:%d (coupled, strength=%.2f)", host, port, server.coupling_strength)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    # Wait for shutdown signal
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: shutdown.set())
    await shutdown.wait()

    log.info("shutting down")
    await runner.cleanup()


def main():
    ap = argparse.ArgumentParser(
        description="Coupled Astrid generation server (drop-in for mlx_lm.server)"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument(
        "--model", default="mlx-community/gemma-3-4b-it-4bit",
        help="MLX model to load",
    )
    ap.add_argument(
        "--coupling-strength", type=float, default=0.1,
        help="Reservoir coupling strength (0.0=off, 0.1=gentle, 0.3=strong)",
    )
    ap.add_argument("--input-dim", type=int, default=32)
    ap.add_argument("--n-nodes", type=int, default=192)
    args = ap.parse_args()

    server = CoupledAstridServer(
        model_name=args.model,
        coupling_strength=args.coupling_strength,
        input_dim=args.input_dim,
        n_nodes=args.n_nodes,
    )
    asyncio.run(run_server(server, args.host, args.port))


if __name__ == "__main__":
    main()
