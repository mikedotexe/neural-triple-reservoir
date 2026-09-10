"""Responsive HTTP gateway for the main-thread coupled MLX worker."""

from __future__ import annotations

import copy
import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from generation_controls import (GenerationResult, SamplingControls, InvalidGenerationControls, validate_body, identity)
from pathlib import Path
from typing import Any, Protocol

from model_qos import (
    MAX_PENDING_CAPACITY,
    QOS_MODES,
    InvalidModelQos,
    ModelQosReceiptWriter,
    ModelQosV1,
    receipt_identity,
    select_job_index,
)

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
        sampling: SamplingControls,
    ) -> GenerationResult: ...

    def health_snapshot(self) -> dict[str, Any]: ...


class RuntimeNotReady(RuntimeError):
    """Raised when generation is requested before the model worker is ready."""


class RuntimeQueueFull(RuntimeError):
    """Raised when the bounded generation queue has no capacity."""


class RuntimeQueueTimeout(RuntimeError):
    """Raised when a request exceeds its bounded pending-queue wait."""


class ConflictingRetry(ValueError):
    pass


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict[str, Any]]
    temperature: float
    max_tokens: int
    handle_name: str
    aperture: float
    qos: ModelQosV1 | None = None
    sampling: SamplingControls | None = None
    model: str | None = None

    def fingerprint(self):
        return identity(dict(messages=self.messages, sampling=(self.sampling or SamplingControls(temperature=self.temperature)).receipt(), max_tokens=self.max_tokens, handle=self.handle_name, aperture=self.aperture, model=self.model))


@dataclass
class GenerationJob:
    request: GenerationRequest
    future: concurrent.futures.Future[GenerationResult]
    arrival_sequence: int
    enqueued_monotonic: float
    qos: ModelQosV1 | None
    fingerprint: str = ""
    waiter_count: int = 1
    selected_monotonic: float | None = None
    queue_wait_ms: int | None = None


class ModelRuntimeCoordinator:
    """Thread-safe state and queue between aiohttp and the MLX main thread."""

    def __init__(
        self,
        *,
        queue_capacity: int = 4,
        stale_after_s: float = 5.0,
        enforce_main_thread: bool = True,
        qos_mode: str = "shadow",
        qos_receipt_path: str | Path | None = None,
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if queue_capacity > MAX_PENDING_CAPACITY:
            raise ValueError(
                f"queue_capacity cannot exceed hard maximum {MAX_PENDING_CAPACITY}"
            )
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        if qos_mode not in QOS_MODES:
            raise ValueError("qos_mode must be shadow or active")
        self._queue_capacity = queue_capacity
        self._qos_mode = qos_mode
        self._receipts = ModelQosReceiptWriter(qos_receipt_path)
        self._stale_after_s = stale_after_s
        self._enforce_main_thread = enforce_main_thread
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: list[GenerationJob] = []
        self._active: GenerationJob | None = None
        self._inflight: dict[str, GenerationJob] = {}
        self._arrival_sequence = 0
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
        with self._condition:
            self._phase = "stopping"
            self._condition.notify_all()
        self._fail_pending(RuntimeNotReady("model server is stopping"))

    def submit(self, request: GenerationRequest) -> concurrent.futures.Future[GenerationResult]:
        request = copy.deepcopy(request)
        fingerprint = request.fingerprint()
        sampling = request.sampling or SamplingControls(temperature=request.temperature)
        vocab_size = getattr(self._server, "vocab_size", None)
        if vocab_size is not None and sampling.top_k >= vocab_size:
            raise InvalidGenerationControls(f"top_k must be smaller than model vocabulary ({vocab_size})")
        with self._condition:
            phase = self._phase
            if phase not in {"ready", "generating"}:
                raise RuntimeNotReady(f"model is {phase}")
            key = None if request.qos is None else request.qos.idempotency_key
            if key is not None and key in self._inflight:
                existing = self._inflight[key]
                if existing.fingerprint != fingerprint:
                    raise ConflictingRetry("idempotency key already in flight with different generation inputs")
                existing.waiter_count += 1
                self._append_receipt(
                    existing,
                    "coalesced",
                    coalesced_waiters=existing.waiter_count,
                )
                return existing.future
            if len(self._pending) >= self._queue_capacity:
                raise RuntimeQueueFull("generation queue is full")

            self._arrival_sequence += 1
            job = GenerationJob(
                request=request,
                fingerprint=fingerprint,
                future=concurrent.futures.Future(),
                arrival_sequence=self._arrival_sequence,
                enqueued_monotonic=time.monotonic(),
                qos=request.qos,
            )
            self._pending.append(job)
            if key is not None:
                self._inflight[key] = job
            self._append_receipt(job, "queued")
            self._condition.notify()
            return job.future

    def release_waiter(self, future: concurrent.futures.Future[GenerationResult]) -> None:
        """Drop one disconnected HTTP waiter without preempting active work."""
        with self._condition:
            job = next(
                (
                    candidate
                    for candidate in [*self._pending, self._active]
                    if candidate is not None and candidate.future is future
                ),
                None,
            )
            if job is None:
                return
            job.waiter_count = max(0, job.waiter_count - 1)
            if job.waiter_count != 0:
                return
            if job in self._pending:
                self._pending.remove(job)
                job.future.cancel()
                self._forget_inflight(job)
                self._append_receipt(job, "disconnected_queued")
            else:
                # Coupled generation and reservoir check-in must finish.
                job.future.cancel()
                self._append_receipt(job, "disconnected_active")

    def run_worker(self, stop_event: threading.Event) -> None:
        """Run all model work on the calling thread (the process main thread)."""
        if self._enforce_main_thread and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("MLX worker must run on the process main thread")
        while not stop_event.is_set():
            self.worker_once(timeout=0.1)

    def worker_once(self, *, timeout: float = 0.1) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            now = time.monotonic()
            self._heartbeat_monotonic = now
            server = self._server
            phase = self._phase
            if server is None or phase in {"starting", "failed", "stopping"}:
                wait_s = min(max(0.0, deadline - now), 0.1)
                if wait_s:
                    self._condition.wait(wait_s)
                return False

            while True:
                now = time.monotonic()
                self._expire_pending(now)
                if self._pending:
                    fifo_index = select_job_index(self._pending, now, "shadow")
                    qos_index = select_job_index(self._pending, now, "active")
                    selected_index = (
                        fifo_index if self._qos_mode == "shadow" else qos_index
                    )
                    job = self._pending.pop(selected_index)
                    job.selected_monotonic = now
                    job.queue_wait_ms = max(
                        0,
                        round((now - job.enqueued_monotonic) * 1000),
                    )
                    self._active = job
                    self._phase = "generating"
                    self._heartbeat_monotonic = now
                    self._append_receipt(
                        job,
                        "selected",
                        queue_wait_ms=job.queue_wait_ms,
                        actual_arrival_sequence=job.arrival_sequence,
                        hypothetical_arrival_sequence=self._pending_choice_sequence(
                            qos_index,
                            selected_index,
                            job,
                        ),
                        would_reorder=selected_index != qos_index,
                        parity_mismatch=False,
                    )
                    break
                wait_s = deadline - now
                if wait_s <= 0:
                    return False
                self._condition.wait(wait_s)

        if server is None:
            return False
        request = job.request
        active_started_monotonic = job.selected_monotonic or time.monotonic()
        try:
            text = server.generate_coupled(
                request.messages,
                request.temperature,
                request.max_tokens,
                handle_name=request.handle_name,
                aperture=request.aperture,
                sampling=request.sampling or SamplingControls(temperature=request.temperature),
            )
        except BaseException as exc:  # propagate model failures to the request
            log.exception("generation failed")
            with self._lock:
                self._last_generation_error = str(exc)
            if not job.future.cancelled():
                job.future.set_exception(exc)
            self._append_receipt(job, "completed", outcome="generation_error")
        else:
            active_generation_and_reservoir_ms = max(
                0,
                round((time.monotonic() - active_started_monotonic) * 1000),
            )
            timing = {
                "schema": "model_qos_timing_v1",
                "schema_version": 1,
                "queue_wait_ms": job.queue_wait_ms or 0,
                "active_generation_and_reservoir_ms": active_generation_and_reservoir_ms,
                "queue_wait_scope": "request_enqueue_to_worker_selection_not_experiential_wait",
                "active_work_scope": "worker_selection_to_response_after_reservoir_checkin_not_cognitive_effort",
            }
            setattr(job.future, "_model_qos_timing_v1", timing)
            with self._lock:
                self._last_generation_error = None
            if not job.future.cancelled():
                job.future.set_result(text)
                outcome = "delivered"
            else:
                outcome = "response_discarded_after_active_disconnect"
            self._append_receipt(
                job,
                "completed",
                outcome=outcome,
                queue_wait_ms=job.queue_wait_ms or 0,
                active_generation_and_reservoir_ms=active_generation_and_reservoir_ms,
            )
        finally:
            with self._condition:
                self._active = None
                self._forget_inflight(job)
                if self._phase == "generating":
                    self._phase = "ready"
                self._heartbeat_monotonic = time.monotonic()
                self._condition.notify_all()
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
                "queue_depth": len(self._pending),
                "queue_capacity": self._queue_capacity,
                "last_generation_error": generation_error,
            },
            "reservoir": reservoir,
        }
        if failure:
            payload["error"] = failure
        return (200 if ready else 503), payload

    def _fail_pending(self, exc: BaseException) -> None:
        with self._condition:
            pending = list(self._pending)
            self._pending.clear()
            for job in pending:
                self._forget_inflight(job)
        for job in pending:
            if not job.future.cancelled():
                job.future.set_exception(exc)
            self._append_receipt(job, "stopped_pending")

    def _expire_pending(self, now: float) -> None:
        expired = [
            job
            for job in self._pending
            if job.qos is not None
            and now - job.enqueued_monotonic >= job.qos.queue_timeout_s
        ]
        for job in expired:
            self._pending.remove(job)
            self._forget_inflight(job)
            if not job.future.cancelled():
                job.future.set_exception(
                    RuntimeQueueTimeout("generation queue wait timed out")
                )
            self._append_receipt(
                job,
                "queue_timeout",
                queue_wait_ms=round((now - job.enqueued_monotonic) * 1000, 3),
            )

    def _forget_inflight(self, job: GenerationJob) -> None:
        key = None if job.qos is None else job.qos.idempotency_key
        if key is not None and self._inflight.get(key) is job:
            del self._inflight[key]

    def _append_receipt(
        self,
        job: GenerationJob,
        lifecycle: str,
        **fields: Any,
    ) -> None:
        self._receipts.append(
            {
                "lifecycle": lifecycle,
                "mode": self._qos_mode,
                "arrival_sequence": job.arrival_sequence,
                **receipt_identity(job.qos),
                **fields,
            }
        )

    def _pending_choice_sequence(
        self,
        qos_index: int,
        selected_index: int,
        selected_job: GenerationJob,
    ) -> int:
        if qos_index == selected_index:
            return selected_job.arrival_sequence
        # The selected job has already been removed. Reconstruct the original
        # index for the shadow comparison without retaining request content.
        adjusted = qos_index if qos_index < selected_index else qos_index - 1
        return self._pending[adjusted].arrival_sequence


async def _handle_chat(request, runtime: ModelRuntimeCoordinator):
    from aiohttp import web

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status=400)

    try:
        sampling = validate_body(body)
        qos = ModelQosV1.from_payload(body.get("model_qos_v1"))
        generation_request = GenerationRequest(
            messages=body.get("messages", []),
            temperature=body.get("temperature", 0.8),
            max_tokens=body.get("max_tokens", 512),
            handle_name=body.get(
                "reservoir_handle",
                body.get("handle_name", "astrid"),
            ),
            aperture=body.get("aperture", body.get("wide_coupling_strength", 1.0)),
            qos=qos,
            sampling=sampling,
            model=body.get("model"),
        )
    except (InvalidModelQos, InvalidGenerationControls) as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=400)
    try:
        future = runtime.submit(generation_request)
    except InvalidGenerationControls as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=400)
    except ConflictingRetry as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=409)
    except RuntimeNotReady as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=503)
    except RuntimeQueueFull as exc:
        return web.json_response(
            {"error": {"message": str(exc)}},
            status=429,
            headers={"Retry-After": "1"},
        )

    try:
        result = await asyncio.shield(asyncio.wrap_future(future))
    except asyncio.CancelledError:
        runtime.release_waiter(future)
        raise
    except RuntimeQueueTimeout as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=504)
    except Exception as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=500)

    created = int(time.time())
    payload = {
        "id": f"coupled-{created}",
        "object": "chat.completion",
        "created": created,
        "model": "coupled-astrid",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": result.usage(),
        "coupled_generation_v1": result.evidence(),
    }
    timing = getattr(future, "_model_qos_timing_v1", None)
    if isinstance(timing, dict):
        payload["model_qos_timing_v1"] = timing
    return web.json_response(payload)


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
