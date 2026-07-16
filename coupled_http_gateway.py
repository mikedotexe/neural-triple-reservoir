"""Responsive HTTP gateway for the main-thread coupled MLX worker."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger("coupled-astrid")


class GenerationServer(Protocol):
    """The model surface consumed by the main-thread worker."""

    coupling_strength: float

    def generate_coupled(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        handle_name: str,
        aperture: float,
    ) -> str: ...

    def health_snapshot(self) -> dict[str, Any]: ...


class RuntimeNotReady(RuntimeError):
    """Raised when generation is requested before the model worker is ready."""


class RuntimeQueueFull(RuntimeError):
    """Raised when the bounded generation queue has no capacity."""


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict[str, Any]]
    temperature: float
    max_tokens: int
    handle_name: str
    aperture: float


@dataclass
class GenerationJob:
    request: GenerationRequest
    future: concurrent.futures.Future[str]


class ModelRuntimeCoordinator:
    """Thread-safe state and queue between aiohttp and the MLX main thread."""

    def __init__(
        self,
        *,
        queue_capacity: int = 1,
        stale_after_s: float = 5.0,
        enforce_main_thread: bool = True,
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self._jobs: queue.Queue[GenerationJob] = queue.Queue(maxsize=queue_capacity)
        self._stale_after_s = stale_after_s
        self._enforce_main_thread = enforce_main_thread
        self._lock = threading.Lock()
        self._phase = "starting"
        self._server: GenerationServer | None = None
        self._failure: str | None = None
        self._last_generation_error: str | None = None
        self._heartbeat_monotonic: float | None = None

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def attach_server(self, server: GenerationServer) -> None:
        """Publish a fully loaded model to the HTTP gateway."""
        with self._lock:
            if self._phase != "starting":
                raise RuntimeError(f"cannot attach model while phase={self._phase}")
            self._server = server
            self._phase = "ready"
            self._heartbeat_monotonic = time.monotonic()

    def mark_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._failure = str(exc)
            self._phase = "failed"

    def mark_stopping(self) -> None:
        with self._lock:
            self._phase = "stopping"
        self._fail_pending(RuntimeNotReady("model server is stopping"))

    def submit(self, request: GenerationRequest) -> concurrent.futures.Future[str]:
        with self._lock:
            phase = self._phase
        if phase not in {"ready", "generating"}:
            raise RuntimeNotReady(f"model is {phase}")
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        try:
            self._jobs.put_nowait(GenerationJob(request=request, future=future))
        except queue.Full as exc:
            raise RuntimeQueueFull("generation queue is full") from exc
        return future

    def run_worker(self, stop_event: threading.Event) -> None:
        """Run all model work on the calling thread (the process main thread)."""
        if self._enforce_main_thread and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("MLX worker must run on the process main thread")
        while not stop_event.is_set():
            self.worker_once(timeout=0.1)

    def worker_once(self, *, timeout: float = 0.1) -> bool:
        with self._lock:
            self._heartbeat_monotonic = time.monotonic()
            server = self._server
            phase = self._phase
        if server is None or phase in {"starting", "failed", "stopping"}:
            time.sleep(min(timeout, 0.1))
            return False
        try:
            job = self._jobs.get(timeout=timeout)
        except queue.Empty:
            return False

        with self._lock:
            self._phase = "generating"
            self._heartbeat_monotonic = time.monotonic()
        request = job.request
        try:
            text = server.generate_coupled(
                request.messages,
                request.temperature,
                request.max_tokens,
                handle_name=request.handle_name,
                aperture=request.aperture,
            )
        except BaseException as exc:  # propagate model failures to the request
            log.exception("generation failed")
            self._last_generation_error = str(exc)
            if not job.future.cancelled():
                job.future.set_exception(exc)
        else:
            self._last_generation_error = None
            if not job.future.cancelled():
                job.future.set_result(text)
        finally:
            self._jobs.task_done()
            with self._lock:
                if self._phase == "generating":
                    self._phase = "ready"
                self._heartbeat_monotonic = time.monotonic()
        return True

    def liveness_snapshot(self) -> dict[str, Any]:
        with self._lock:
            phase = self._phase
        return {
            "live": True,
            "status": "live",
            "phase": phase,
            "model": "coupled-astrid",
        }

    def readiness_snapshot(self) -> tuple[int, dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            phase = self._phase
            failure = self._failure
            heartbeat = self._heartbeat_monotonic
            generation_error = self._last_generation_error
            server = self._server
        heartbeat_age_s = None if heartbeat is None else max(0.0, now - heartbeat)
        stale = (
            phase == "ready"
            and heartbeat_age_s is not None
            and heartbeat_age_s > self._stale_after_s
        )
        ready = phase in {"ready", "generating"} and not stale
        status = "stale" if stale else phase
        reservoir: dict[str, Any] = {"status": "unknown"}
        if server is not None:
            try:
                reservoir = dict(server.health_snapshot())
            except Exception as exc:  # readiness must never call a blocking probe
                reservoir = {"status": "degraded", "detail": str(exc)}
        payload = {
            "ready": ready,
            "status": status,
            "model": "coupled-astrid",
            "worker": {
                "phase": phase,
                "heartbeat_age_ms": (
                    None if heartbeat_age_s is None else round(heartbeat_age_s * 1000, 3)
                ),
                "queue_depth": self._jobs.qsize(),
                "queue_capacity": self._jobs.maxsize,
                "last_generation_error": generation_error,
            },
            "reservoir": reservoir,
        }
        if failure:
            payload["error"] = failure
        return (200 if ready else 503), payload

    def _fail_pending(self, exc: BaseException) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            if not job.future.cancelled():
                job.future.set_exception(exc)
            self._jobs.task_done()


async def _handle_chat(request, runtime: ModelRuntimeCoordinator):
    from aiohttp import web

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status=400)

    generation_request = GenerationRequest(
        messages=body.get("messages", []),
        temperature=body.get("temperature", 0.8),
        max_tokens=body.get("max_tokens", 512),
        handle_name=body.get("reservoir_handle", body.get("handle_name", "astrid")),
        aperture=body.get("aperture", body.get("wide_coupling_strength", 1.0)),
    )
    try:
        future = runtime.submit(generation_request)
    except RuntimeNotReady as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=503)
    except RuntimeQueueFull as exc:
        return web.json_response(
            {"error": {"message": str(exc)}},
            status=429,
            headers={"Retry-After": "1"},
        )

    try:
        text = await asyncio.wrap_future(future)
    except Exception as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=500)

    created = int(time.time())
    return web.json_response(
        {
            "id": f"coupled-{created}",
            "object": "chat.completion",
            "created": created,
            "model": "coupled-astrid",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def create_http_app(runtime: ModelRuntimeCoordinator):
    from aiohttp import web

    async def chat(request):
        return await _handle_chat(request, runtime)

    async def models(_request):
        return web.json_response(
            {"data": [{"id": "coupled-astrid", "object": "model"}]}
        )

    async def live(_request):
        return web.json_response(runtime.liveness_snapshot())

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/livez", live)

    async def ready(_request):
        status, payload = runtime.readiness_snapshot()
        return web.json_response(payload, status=status)

    app.router.add_get("/readyz", ready)
    return app


class AiohttpGateway:
    """Own aiohttp in a dedicated thread so health remains responsive."""

    def __init__(self, runtime: ModelRuntimeCoordinator, host: str, port: int):
        self.runtime = runtime
        self.host = host
        self.port = port
        self.bound_port: int | None = None
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="coupled-aiohttp-gateway",
            daemon=True,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._startup_error: BaseException | None = None

    def start(self, *, timeout: float = 10.0) -> None:
        self._thread.start()
        if not self._started.wait(timeout):
            raise TimeoutError("HTTP gateway did not start in time")
        if self._startup_error is not None:
            raise RuntimeError("HTTP gateway failed to start") from self._startup_error

    def stop(self, *, timeout: float = 10.0) -> None:
        loop = self._loop
        shutdown = self._shutdown
        if loop is not None and shutdown is not None and loop.is_running():
            loop.call_soon_threadsafe(shutdown.set)
        if self._thread.is_alive():
            self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("HTTP gateway did not stop in time")

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
            log.exception("HTTP gateway failed")

    async def _serve(self) -> None:
        from aiohttp import web

        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        runner = web.AppRunner(create_http_app(self.runtime))
        try:
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            sockets = getattr(site, "_server", None)
            bound_sockets = [] if sockets is None else list(sockets.sockets or [])
            self.bound_port = (
                int(bound_sockets[0].getsockname()[1]) if bound_sockets else self.port
            )
            log.info("HTTP gateway listening on %s:%d", self.host, self.bound_port)
            self._started.set()
            await self._shutdown.wait()
        finally:
            await runner.cleanup()
