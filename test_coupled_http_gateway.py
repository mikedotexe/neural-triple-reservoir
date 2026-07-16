"""Concurrency and compatibility tests for the coupled HTTP gateway."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from coupled_http_gateway import (
    AiohttpGateway,
    GenerationRequest,
    ModelRuntimeCoordinator,
    RuntimeQueueFull,
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


if __name__ == "__main__":
    unittest.main()
