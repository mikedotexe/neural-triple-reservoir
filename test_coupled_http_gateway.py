"""Concurrency and compatibility tests for the coupled HTTP gateway."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from coupled_http_gateway import (
    AiohttpGateway,
    GenerationRequest,
    ModelRuntimeCoordinator,
    RuntimeQueueFull,
    RuntimeQueueTimeout,
)
from model_qos import (
    MAX_PENDING_CAPACITY,
    ModelQosV1,
    effective_rank,
    select_job_index,
)


class BlockingFakeServer:
    coupling_strength = 0.1

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_coupled(
        self,
        messages,
        temperature,
        max_tokens,
        *,
        handle_name,
        aperture,
    ):
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("test generation was not released")
        return "gateway-compatible response"

    def health_snapshot(self):
        return {
            "status": "degraded",
            "operation": "pull_state",
            "detail": "test reservoir unavailable",
        }


class RecordingFakeServer:
    coupling_strength = 0.1

    def __init__(self):
        self.handles = []

    def generate_coupled(
        self,
        messages,
        temperature,
        max_tokens,
        *,
        handle_name,
        aperture,
    ):
        self.handles.append(handle_name)
        return f"response-{handle_name}"

    def health_snapshot(self):
        return {"status": "ok"}


def qos(
    qos_class: str,
    identifier: str,
    *,
    timeout_s: float = 60.0,
) -> ModelQosV1:
    return ModelQosV1(
        request_id=f"request-{identifier}",
        idempotency_key=f"idempotency-{identifier}",
        qos_class=qos_class,
        queue_timeout_s=timeout_s,
    )


def generation_request(
    handle: str,
    qos_value: ModelQosV1 | None = None,
) -> GenerationRequest:
    return GenerationRequest([], 0.8, 1, handle, 1.0, qos_value)


def request_json(url: str, *, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            status = response.status
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = json.loads(exc.read())
    return status, payload, time.perf_counter() - started


class GatewayTests(unittest.TestCase):
    def test_worker_stop_waits_for_active_generation_return(self):
        runtime = ModelRuntimeCoordinator(enforce_main_thread=False)
        server = BlockingFakeServer()
        runtime.attach_server(server)
        stop_event = threading.Event()
        future = runtime.submit(generation_request("astrid"))
        worker = threading.Thread(target=runtime.run_worker, args=(stop_event,))
        worker.start()
        self.addCleanup(lambda: (stop_event.set(), worker.join(2)))
        self.addCleanup(server.release.set)
        self.assertTrue(server.started.wait(2))
        stop_event.set()
        worker.join(0.05)
        self.assertTrue(worker.is_alive(), "stop must not preempt active generation")
        self.assertFalse(future.done())
        server.release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(future.result(timeout=1), "gateway-compatible response")

    def start_gateway(self, runtime):
        gateway = AiohttpGateway(runtime, "127.0.0.1", 0)
        gateway.start()
        self.addCleanup(gateway.stop)
        return gateway, f"http://127.0.0.1:{gateway.bound_port}"

    def test_livez_starts_before_model_and_readyz_reports_starting(self):
        runtime = ModelRuntimeCoordinator(enforce_main_thread=False)
        _gateway, base_url = self.start_gateway(runtime)

        live_status, live, _elapsed = request_json(f"{base_url}/livez")
        ready_status, ready, _elapsed = request_json(f"{base_url}/readyz")

        self.assertEqual(live_status, 200)
        self.assertTrue(live["live"])
        self.assertEqual(live["phase"], "starting")
        self.assertEqual(ready_status, 503)
        self.assertFalse(ready["ready"])
        self.assertEqual(ready["status"], "starting")

    def test_health_stays_below_100ms_while_main_worker_generates(self):
        runtime = ModelRuntimeCoordinator(enforce_main_thread=False)
        gateway, base_url = self.start_gateway(runtime)
        server = BlockingFakeServer()
        runtime.attach_server(server)
        stop_event = threading.Event()
        worker = threading.Thread(target=runtime.run_worker, args=(stop_event,))
        worker.start()
        self.addCleanup(lambda: (stop_event.set(), worker.join(2)))
        self.addCleanup(server.release.set)

        response_holder = {}

        def send_chat():
            response_holder["response"] = request_json(
                f"{base_url}/v1/chat/completions",
                body={
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.4,
                    "max_tokens": 12,
                    "reservoir_handle": "astrid",
                    "aperture": 0.75,
                },
            )

        chat = threading.Thread(target=send_chat)
        chat.start()
        self.addCleanup(chat.join, 2)
        self.assertTrue(server.started.wait(2), "fake generation never started")

        live_status, _live, live_elapsed = request_json(f"{base_url}/livez")
        ready_status, ready, ready_elapsed = request_json(f"{base_url}/readyz")

        self.assertEqual(live_status, 200)
        self.assertLess(live_elapsed, 0.1)
        self.assertEqual(ready_status, 200)
        self.assertLess(ready_elapsed, 0.1)
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["status"], "generating")
        self.assertEqual(ready["reservoir"]["status"], "degraded")

        server.release.set()
        chat.join(2)
        self.assertFalse(chat.is_alive())
        status, payload, _elapsed = response_holder["response"]
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "coupled-astrid")
        self.assertEqual(
            payload["choices"][0]["message"],
            {"role": "assistant", "content": "gateway-compatible response"},
        )
        self.assertEqual(
            payload["usage"],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        timing = payload["model_qos_timing_v1"]
        self.assertEqual(timing["schema"], "model_qos_timing_v1")
        self.assertEqual(timing["schema_version"], 1)
        self.assertGreaterEqual(timing["queue_wait_ms"], 0)
        self.assertGreater(timing["active_generation_and_reservoir_ms"], 0)
        self.assertEqual(
            timing["queue_wait_scope"],
            "request_enqueue_to_worker_selection_not_experiential_wait",
        )
        self.assertEqual(
            timing["active_work_scope"],
            "worker_selection_to_response_after_reservoir_checkin_not_cognitive_effort",
        )
        self.assertEqual(gateway.runtime.phase, "ready")

    def test_readyz_reports_stale_worker(self):
        runtime = ModelRuntimeCoordinator(
            stale_after_s=0.02,
            enforce_main_thread=False,
        )
        _gateway, base_url = self.start_gateway(runtime)
        runtime.attach_server(BlockingFakeServer())
        time.sleep(0.04)

        status, payload, _elapsed = request_json(f"{base_url}/readyz")

        self.assertEqual(status, 503)
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["status"], "stale")

    def test_models_shape_and_bounded_queue_are_preserved(self):
        runtime = ModelRuntimeCoordinator(
            queue_capacity=1,
            enforce_main_thread=False,
        )
        _gateway, base_url = self.start_gateway(runtime)
        runtime.attach_server(BlockingFakeServer())

        status, payload, _elapsed = request_json(f"{base_url}/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {"data": [{"id": "coupled-astrid", "object": "model"}]},
        )

        request = GenerationRequest([], 0.8, 1, "astrid", 1.0)
        runtime.submit(request)
        with self.assertRaises(RuntimeQueueFull):
            runtime.submit(request)

    def test_shadow_mode_preserves_fifo_and_active_mode_selects_interactive(self):
        for mode, expected in (
            ("shadow", ["background", "interactive"]),
            ("active", ["interactive", "background"]),
        ):
            with self.subTest(mode=mode):
                server = RecordingFakeServer()
                runtime = ModelRuntimeCoordinator(
                    qos_mode=mode,
                    enforce_main_thread=False,
                )
                runtime.attach_server(server)
                background = runtime.submit(
                    generation_request("background", qos("background", f"{mode}-b"))
                )
                interactive = runtime.submit(
                    generation_request(
                        "interactive",
                        qos("interactive", f"{mode}-i"),
                    )
                )

                self.assertTrue(runtime.worker_once(timeout=0))
                self.assertTrue(runtime.worker_once(timeout=0))

                self.assertEqual(server.handles, expected)
                self.assertEqual(background.result(), "response-background")
                self.assertEqual(interactive.result(), "response-interactive")

    def test_aging_promotes_background_and_ties_break_by_arrival(self):
        now = 100.0
        jobs = [
            SimpleNamespace(
                qos=qos("background", "old"),
                enqueued_monotonic=39.0,
                arrival_sequence=1,
            ),
            SimpleNamespace(
                qos=qos("interactive", "new"),
                enqueued_monotonic=99.0,
                arrival_sequence=2,
            ),
        ]
        self.assertEqual(effective_rank(jobs[0].qos, 61.0), 0)
        self.assertEqual(select_job_index(jobs, now, "active"), 0)

    def test_duplicate_idempotency_coalesces_only_while_inflight(self):
        server = RecordingFakeServer()
        runtime = ModelRuntimeCoordinator(enforce_main_thread=False)
        runtime.attach_server(server)
        request = generation_request("shared", qos("reflective", "shared"))

        first = runtime.submit(request)
        second = runtime.submit(request)
        self.assertIs(first, second)
        self.assertTrue(runtime.worker_once(timeout=0))
        self.assertEqual(server.handles, ["shared"])

        third = runtime.submit(request)
        self.assertIsNot(first, third)
        self.assertTrue(runtime.worker_once(timeout=0))
        self.assertEqual(server.handles, ["shared", "shared"])

    def test_queue_timeout_expires_without_generation(self):
        server = RecordingFakeServer()
        runtime = ModelRuntimeCoordinator(enforce_main_thread=False)
        runtime.attach_server(server)
        future = runtime.submit(
            generation_request(
                "expired",
                qos("interactive", "expired", timeout_s=0.001),
            )
        )
        time.sleep(0.005)

        self.assertFalse(runtime.worker_once(timeout=0))
        with self.assertRaises(RuntimeQueueTimeout):
            future.result()
        self.assertEqual(server.handles, [])

    def test_disconnected_pending_job_is_dropped_before_generation(self):
        server = RecordingFakeServer()
        runtime = ModelRuntimeCoordinator(enforce_main_thread=False)
        runtime.attach_server(server)
        future = runtime.submit(
            generation_request("disconnected", qos("normal", "disconnected"))
        )

        runtime.release_waiter(future)

        self.assertTrue(future.cancelled())
        self.assertFalse(runtime.worker_once(timeout=0))
        self.assertEqual(server.handles, [])

    def test_capacity_hard_limit_and_qos_validation(self):
        with self.assertRaises(ValueError):
            ModelRuntimeCoordinator(queue_capacity=MAX_PENDING_CAPACITY + 1)
        parsed = ModelQosV1.from_payload(
            {
                "schema_version": 1,
                "request_id": "request",
                "idempotency_key": "key",
                "class": "interactive",
                "queue_timeout_ms": 999_999,
            }
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.queue_timeout_s, 120.0)

    def test_receipts_are_owner_only_and_exclude_prompt_and_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "private" / "receipts.jsonl"
            runtime = ModelRuntimeCoordinator(
                enforce_main_thread=False,
                qos_receipt_path=receipt_path,
            )
            server = RecordingFakeServer()
            runtime.attach_server(server)
            request = GenerationRequest(
                [{"role": "user", "content": "private prompt marker"}],
                0.8,
                1,
                "receipt",
                1.0,
                qos("reflective", "receipt"),
            )
            runtime.submit(request)
            self.assertTrue(runtime.worker_once(timeout=0))

            content = receipt_path.read_text()
            self.assertNotIn("private prompt marker", content)
            self.assertNotIn("response-receipt", content)
            self.assertNotIn("idempotency-receipt", content)
            self.assertEqual(
                stat.S_IMODE(os.stat(receipt_path).st_mode),
                0o600,
            )


if __name__ == "__main__":
    unittest.main()
