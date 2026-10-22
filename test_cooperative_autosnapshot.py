"""Stdlib-only tests of AST-extracted snapshot scheduling; no service imports."""
from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().with_name("reservoir_service.py")
METHODS = {"snapshot_all", "snapshot_all_cooperative", "auto_snapshot_loop", "run"}


def extracted_service(namespace):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    service = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "ReservoirService")
    methods = [node for node in service.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in METHODS]
    if {node.name for node in methods} != METHODS or any(node.decorator_list for node in methods):
        raise AssertionError("expected undecorated production snapshot methods")

    def refuse_import(name, *args, **kwargs):
        raise AssertionError("extracted method attempted an import: " + name)

    scope = {"__builtins__": {**vars(builtins), "__import__": refuse_import}, **namespace}
    # Compile only selected function definitions, never the module or constructor.
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), scope)
    return type("ExtractedService", (), {name: scope[name] for name in METHODS})()


class CooperativeAutosnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.saved = []
        self.failures = {}
        self.on_yield = None
        self.log = Mock()
        self.interval = object()
        self.runtime = SimpleNamespace(sleep=self.checkpoint, wait_for=self.wait_for,
                                       TimeoutError=asyncio.TimeoutError)
        self.service = extracted_service({"asyncio": self.runtime, "log": self.log,
                                          "AUTO_SNAPSHOT_INTERVAL": self.interval})
        self.service.handles = {"first": 1, "second": 2, "third": 3}
        self.service._shutdown = asyncio.Event()
        self.service.snapshot_handle = self.snapshot_handle
        self.waits = 0

    def snapshot_handle(self, name):
        self.events.append(("snapshot", name))
        if name in self.failures:
            raise self.failures[name]
        if name not in self.service.handles:
            raise ValueError("deleted synthetic handle: " + name)
        self.saved.append((name, self.service.handles[name]))

    async def checkpoint(self, delay):
        self.assertEqual(delay, 0)
        self.events.append(("yield", delay))
        if self.on_yield is not None:
            callback, self.on_yield = self.on_yield, None
            callback()
        await asyncio.sleep(0)

    async def wait_for(self, waiter, *, timeout):
        self.assertIs(timeout, self.interval)
        waiter.close()
        self.waits += 1
        self.events.append(("interval", self.waits))
        if self.waits == 1:
            raise asyncio.TimeoutError
        self.assertEqual(self.waits, 2, "autosnapshot loop failed to stop")
        self.service._shutdown.set()

    async def test_yields_between_handles_and_allows_another_task_to_run(self):
        observed = []

        async def observer():
            for _ in range(2):
                observed.append(list(self.saved))
                await asyncio.sleep(0)

        task = asyncio.create_task(observer())
        try:
            await self.service.snapshot_all_cooperative()
            await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.events, [("snapshot", "first"), ("yield", 0),
                                      ("snapshot", "second"), ("yield", 0),
                                      ("snapshot", "third")])
        self.assertEqual(observed, [[("first", 1)], [("first", 1), ("second", 2)]])
        self.assertEqual(self.saved, [("first", 1), ("second", 2), ("third", 3)])
        self.log.error.assert_not_called()

    async def test_empty_batch_does_not_save_or_yield(self):
        self.service.handles.clear()
        await self.service.snapshot_all_cooperative()
        self.assertEqual(self.events, [])
        self.log.error.assert_not_called()

    async def test_single_handle_does_not_add_a_trailing_yield(self):
        self.service.handles = {"only": 1}
        await self.service.snapshot_all_cooperative()
        self.assertEqual(self.events, [("snapshot", "only")])
        self.assertEqual(self.saved, [("only", 1)])

    async def test_failed_handle_is_logged_and_remaining_handles_continue(self):
        error = OSError("synthetic save failure")
        self.failures["second"] = error
        await self.service.snapshot_all_cooperative()
        self.assertEqual(self.saved, [("first", 1), ("third", 3)])
        self.assertEqual(self.events, [("snapshot", "first"), ("yield", 0),
                                      ("snapshot", "second"), ("yield", 0),
                                      ("snapshot", "third")])
        self.log.error.assert_called_once_with("snapshot_all failed for '%s': %s", "second", error)

    async def test_new_handle_added_at_yield_is_deferred_until_next_batch(self):
        self.on_yield = lambda: self.service.handles.update(late=4)
        await self.service.snapshot_all_cooperative()
        self.assertEqual(self.saved, [("first", 1), ("second", 2), ("third", 3)])
        self.events.clear()
        self.saved.clear()
        await self.service.snapshot_all_cooperative()
        self.assertEqual(self.saved, [("first", 1), ("second", 2), ("third", 3), ("late", 4)])
        self.assertEqual(self.events.count(("yield", 0)), 3)

    async def test_deleted_handle_is_logged_and_later_handle_uses_current_value(self):
        def change_handles():
            del self.service.handles["second"]
            self.service.handles["third"] = 30

        self.on_yield = change_handles
        await self.service.snapshot_all_cooperative()
        self.assertEqual(self.saved, [("first", 1), ("third", 30)])
        self.assertEqual(self.events, [("snapshot", "first"), ("yield", 0),
                                      ("snapshot", "second"), ("yield", 0),
                                      ("snapshot", "third")])
        self.log.error.assert_called_once()
        message, name, error = self.log.error.call_args.args
        self.assertEqual((message, name), ("snapshot_all failed for '%s': %s", "second"))
        self.assertIsInstance(error, ValueError)

    async def test_autosnapshot_loop_awaits_cooperative_batch_after_interval(self):
        self.service.snapshot_all = Mock(side_effect=AssertionError("periodic save must cooperate"))
        await self.service.auto_snapshot_loop()
        self.assertEqual(self.events, [("interval", 1), ("snapshot", "first"), ("yield", 0),
                                      ("snapshot", "second"), ("yield", 0),
                                      ("snapshot", "third"), ("interval", 2)])
        self.assertEqual(self.saved, [("first", 1), ("second", 2), ("third", 3)])
        self.service.snapshot_all.assert_not_called()

    async def test_autosnapshot_loop_skips_empty_batch(self):
        self.service.handles.clear()
        self.service.snapshot_all_cooperative = Mock(side_effect=AssertionError("empty batch invoked"))
        await self.service.auto_snapshot_loop()
        self.assertEqual(self.events, [("interval", 1), ("interval", 2)])
        self.service.snapshot_all_cooperative.assert_not_called()

    async def test_autosnapshot_loop_does_nothing_after_shutdown(self):
        self.service._shutdown.set()
        await self.service.auto_snapshot_loop()
        self.assertEqual(self.events, [])

    def test_synchronous_snapshot_all_keeps_failure_isolation_without_yields(self):
        self.failures["second"] = OSError("synthetic shutdown save failure")
        self.assertFalse(inspect.iscoroutinefunction(self.service.snapshot_all))
        self.assertIsNone(self.service.snapshot_all())
        self.assertEqual(self.events, [("snapshot", "first"), ("snapshot", "second"),
                                      ("snapshot", "third")])
        self.assertEqual(self.saved, [("first", 1), ("third", 3)])
        self.log.error.assert_called_once()

    async def test_run_cancels_background_tasks_before_synchronous_final_snapshot(self):
        self.service._shutdown.set()
        signal_handlers = []

        async def rehearsal_loop():
            raise AssertionError("synthetic rehearsal task must not start")

        def create_task(coroutine):
            name = coroutine.cr_code.co_name
            coroutine.close()  # Do not start real background tasks in this fixture.
            self.events.append(("task", name))
            return SimpleNamespace(cancel=lambda: self.events.append(("cancel", name)))

        class Listener:
            async def __aenter__(inner):
                self.events.append(("listener", "enter"))

            async def __aexit__(inner, *args):
                self.events.append(("listener", "exit"))

        def serve(handler, host, port):
            self.assertIs(handler, self.service.ws_handler)
            self.assertEqual((host, port), ("synthetic.invalid", 1))
            return Listener()

        self.runtime.create_task = create_task
        self.runtime.get_running_loop = lambda: SimpleNamespace(
            add_signal_handler=lambda sig, callback: signal_handlers.append(sig))
        scope = self.service.run.__func__.__globals__
        scope.update(signal=SimpleNamespace(SIGTERM="TERM", SIGINT="INT"), serve=serve)
        self.service.restore_all = lambda: self.events.append(("restore", "all"))
        self.service.rehearsal_loop = rehearsal_loop
        self.service.bridge = SimpleNamespace(backend_label=lambda: "synthetic")
        self.service.config = SimpleNamespace(input_dim=1, n_nodes=1)
        self.service.ws_handler = object()
        self.service.snapshot_all_cooperative = Mock(side_effect=AssertionError("final save must be synchronous"))
        await self.service.run(host="synthetic.invalid", port=1)
        self.assertEqual(signal_handlers, ["TERM", "INT"])
        self.assertEqual(self.events, [
            ("restore", "all"), ("task", "rehearsal_loop"), ("task", "auto_snapshot_loop"),
            ("listener", "enter"), ("listener", "exit"),
            ("cancel", "rehearsal_loop"), ("cancel", "auto_snapshot_loop"),
            ("snapshot", "first"), ("snapshot", "second"), ("snapshot", "third"),
        ])
        self.assertEqual(self.saved, [("first", 1), ("second", 2), ("third", 3)])
        self.service.snapshot_all_cooperative.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
