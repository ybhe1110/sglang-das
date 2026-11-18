"""Utility cancellation retains physical work; slow sends retain response budgets."""

import asyncio
import json
import threading
import unittest
from dataclasses import fields, replace
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.utility_executor import (
    UtilityError,
    UtilityLimits,
    UtilityService,
)
from sglang.srt.entrypoints.openai.utility_http import (
    UtilityHTTPMiddleware,
    utility_error_response,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from starlette.responses import Response

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUtilityExecution(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def asyncSetUp(self):
        self.services = []
        self.gates = []

    async def asyncTearDown(self):
        for gate in self.gates:
            gate.set()
        for service in self.services:
            for ticket in list(service.tickets):
                ticket.close()
            await asyncio.wait_for(service.aclose(), 5)

    async def service(self, backend=None, **limits):
        service = UtilityService(replace(UtilityLimits(), **limits))
        self.services.append(service)
        await service.start(lambda: backend or (lambda *_: Response(b"ok")))
        return service

    def blocking_backend(self, *, fail=False):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        gate = threading.Event()
        self.gates.append(gate)
        calls = []

        def backend(kind, request):
            calls.append((kind, request, threading.get_ident()))
            loop.call_soon_threadsafe(started.set)
            if not gate.wait(5):
                raise RuntimeError("Test did not release backend")
            if fail:
                raise ValueError("backend failed")
            return Response(b"ok")

        return backend, started, gate, calls

    async def settle(self, service):
        async def wait():
            while service.work_used:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), 5)

    def assert_empty(self, service):
        self.assertFalse(service.tickets)
        self.assertTrue(all(value == 0 for value in service.snapshot().values()))

    async def test_cancelled_queue_entries_still_count_until_dequeued(self):
        backend, started, gate, calls = self.blocking_backend()
        service = await self.service(backend, max_admitted=2)
        first = service.admit()
        running = asyncio.create_task(service.run(first, "tokenize", "first"))
        await asyncio.wait_for(started.wait(), 2)
        self.assertNotEqual(calls[0][2], threading.get_ident())
        second = service.admit()
        queued = asyncio.create_task(service.run(second, "tokenize", "second"))
        await asyncio.sleep(0)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        second.close()
        for _ in range(20):
            with self.assertRaises(UtilityError) as rejected:
                service.admit()
            self.assertEqual(rejected.exception.status, 429)
        self.assertEqual(service.work_used, 2)
        self.assertEqual(service.snapshot()["cancelled_work"], 1)
        gate.set()
        await running
        first.close()
        await self.settle(service)
        self.assertEqual(len(calls), 1)
        self.assert_empty(service)

    async def test_running_cancel_retains_input_and_response_until_exception_returns(
        self,
    ):
        backend, started, gate, _ = self.blocking_backend(fail=True)
        service = await self.service(backend, max_admitted=1)
        ticket = service.admit()
        task = asyncio.create_task(service.run(ticket, "detokenize", [1]))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        ticket.close()
        self.assertEqual(service.work_used, 1)
        self.assertEqual(service.output_bytes, service.limits.max_output_bytes)
        self.assertEqual(service.input_bytes, service.limits.max_input_bytes)
        gate.set()
        await self.settle(service)
        self.assert_empty(service)

    async def test_cancel_after_result_ready_releases_once(self):
        service = await self.service(max_admitted=1)
        ticket = service.admit()
        await service.run(ticket, "tokenize", "x")
        self.assertEqual(service.work_used, 0)
        self.assertEqual(service.output_bytes, 256)
        self.assertEqual(service.response_used, 1)
        with self.assertRaises(UtilityError):
            service.admit()
        ticket.close()
        ticket.close()
        self.assert_empty(service)

    async def test_completion_and_cancellation_in_same_loop_turn(self):
        backend, started, gate, _ = self.blocking_backend()
        service = await self.service(backend)
        ticket = service.admit()
        task = asyncio.create_task(service.run(ticket, "tokenize", "x"))
        await asyncio.wait_for(started.wait(), 2)
        ticket.reply.add_done_callback(lambda _: task.cancel())
        gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        ticket.close()
        self.assert_empty(service)

    async def test_initializer_failure_joins_workers_and_refuses_requests(self):
        service = UtilityService(UtilityLimits())
        self.services.append(service)

        def fail():
            raise ValueError("tokenizer cannot be copied")

        with self.assertRaisesRegex(ValueError, "cannot be copied"):
            await service.start(fail)
        self.assertFalse(service._workers)
        with self.assertRaises(UtilityError) as unavailable:
            service.admit()
        self.assertEqual(unavailable.exception.status, 503)
        self.assert_empty(service)

    async def test_total_input_and_output_reservations_gate_before_work(self):
        for budget in ("max_total_input_bytes", "max_total_output_bytes"):
            with self.subTest(budget=budget):
                service = await self.service(
                    **{budget: 256, "max_input_bytes": 256, "max_output_bytes": 256}
                )
                ticket = service.admit()
                with self.assertRaises(UtilityError):
                    service.admit()
                ticket.close()
                self.assert_empty(service)

    async def test_shutdown_waits_for_physical_work_and_stops_admission(self):
        backend, started, gate, _ = self.blocking_backend()
        service = await self.service(backend)
        ticket = service.admit()

        async def request():
            try:
                await service.run(ticket, "tokenize", "x")
            finally:
                ticket.close()

        task = asyncio.create_task(request())
        await asyncio.wait_for(started.wait(), 2)
        closing = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        with self.assertRaises(UtilityError) as rejected:
            service.admit()
        self.assertEqual(rejected.exception.status, 503)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(closing.done())
        gate.set()
        await asyncio.wait_for(closing, 5)
        self.assert_empty(service)

    async def http(
        self, service, *, path="/v1/tokenize", chunks=None, send=None, app=None
    ):
        incoming = asyncio.Queue()
        for chunk, more in chunks or [(b"{}", False)]:
            incoming.put_nowait(
                {"type": "http.request", "body": chunk, "more_body": more}
            )
        sent = []

        async def downstream(scope, receive, send):
            if scope["path"] == "/health":
                return await Response(b"healthy")(scope, receive, send)
            body = await receive()
            try:
                response = await service.run(
                    scope["utility_ticket"], "tokenize", body["body"]
                )
            except UtilityError as error:
                response = utility_error_response(error)
            await response(scope, receive, send)

        async def record(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(utility_service=service)),
        }
        task = asyncio.create_task(
            UtilityHTTPMiddleware(app or downstream)(
                scope, incoming.get, send or record
            )
        )
        return task, incoming, sent

    async def test_disconnect_drops_result_without_releasing_running_work(self):
        backend, started, gate, calls = self.blocking_backend()
        service = await self.service(backend, max_admitted=1)
        task, incoming, sent = await self.http(service)
        await asyncio.wait_for(started.wait(), 2)
        incoming.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(task, 2)
        self.assertFalse(sent)
        self.assertEqual(service.work_used, 1)
        with self.assertRaises(UtilityError):
            service.admit()
        gate.set()
        await self.settle(service)
        self.assertEqual(len(calls), 1)
        self.assert_empty(service)

    async def test_slow_send_holds_budget_but_not_event_loop(self):
        service = await self.service(max_admitted=1)
        sending, release = asyncio.Event(), asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                sending.set()
                await release.wait()

        task, _, _ = await self.http(service, send=send)
        await asyncio.wait_for(sending.wait(), 2)
        self.assertEqual(service.work_used, 0)
        self.assertEqual(service.output_bytes, 256)
        with self.assertRaises(UtilityError):
            service.admit()
        health, _, sent = await self.http(service, path="/health")
        await asyncio.wait_for(health, 2)
        self.assertEqual(sent[-1]["body"], b"healthy")
        closing = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        release.set()
        await task
        await closing
        self.assert_empty(service)

    async def test_deadline_preserves_running_capacity_and_returns_flat_504(self):
        backend, started, gate, _ = self.blocking_backend()
        service = await self.service(backend, timeout=0.1, max_admitted=1)
        task, _, sent = await self.http(service)
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(task, 2)
        self.assertEqual(sent[0]["status"], 504)
        self.assertEqual(json.loads(sent[1]["body"])["code"], 504)
        self.assertEqual(service.work_used, 1)
        gate.set()
        await self.settle(service)
        self.assert_empty(service)

    async def test_chunked_body_is_bounded_without_content_length(self):
        service = await self.service(max_input_bytes=4)
        task, _, sent = await self.http(
            service, chunks=[(b"123", True), (b"45", False)]
        )
        await task
        self.assertEqual(sent[0]["status"], 413)
        self.assert_empty(service)

    async def test_incomplete_body_deadline_and_disconnect_do_not_submit(self):
        service = await self.service(body_timeout=0.02)
        task, _, sent = await self.http(service, chunks=[(b"{", True)])
        await task
        self.assertEqual(sent[0]["status"], 504)
        task, incoming, sent = await self.http(service, chunks=[(b"{", True)])
        incoming.put_nowait({"type": "http.disconnect"})
        await task
        self.assertFalse(sent)
        self.assert_empty(service)

    async def test_all_aliases_reject_oversize_results_and_release_budgets(self):
        service = await self.service(
            lambda *_: Response(b"x" * 512), max_output_bytes=256
        )
        for path in ("/tokenize", "/v1/tokenize", "/detokenize", "/v1/detokenize"):
            task, _, sent = await self.http(service, path=path)
            await task
            self.assertEqual(sent[0]["status"], 413)
            self.assert_empty(service)

    async def test_send_failure_releases_result(self):
        service = await self.service()

        async def fail_send(message):
            raise OSError("peer closed")

        task, _, _ = await self.http(service, send=fail_send)
        with self.assertRaises(OSError):
            await task
        self.assert_empty(service)

    async def test_deadline_during_send_does_not_start_a_second_response(self):
        service = await self.service(timeout=0.05)
        headers = []
        cancelled = asyncio.Event()

        async def slow_send(message):
            if message["type"] == "http.response.start":
                headers.append(message["status"])
            else:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        task, _, _ = await self.http(service, send=slow_send)
        await asyncio.wait_for(task, 2)
        self.assertEqual(headers, [200])
        self.assertTrue(cancelled.is_set())
        self.assert_empty(service)

    async def test_framework_response_limit_applies_without_submitting_work(self):
        service = await self.service(max_output_bytes=256)
        received = asyncio.Queue()
        received.put_nowait({"type": "http.request", "body": b"{}"})
        sent = []

        async def downstream(scope, receive, send):
            await Response(b"x" * 512)(scope, receive, send)

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/tokenize",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(utility_service=service)),
        }
        await UtilityHTTPMiddleware(downstream)(scope, received.get, send)
        self.assertEqual(sent[0]["status"], 413)
        self.assert_empty(service)

    async def test_shutdown_deadline_retains_native_work_until_late_completion(self):
        """A stuck native call must not make the shutdown coroutine wait forever."""
        backend, started, gate, _ = self.blocking_backend()
        service = await self.service(backend, shutdown_timeout=0.03)
        task, _, _ = await self.http(service)
        await asyncio.wait_for(started.wait(), 2)
        with self.assertRaisesRegex(TimeoutError, "grace period expired"):
            await asyncio.wait_for(service.aclose(), 1)
        self.assertEqual(service.work_used, 1)
        self.assertEqual(service.response_used, 1)
        with self.assertRaises(UtilityError):
            service.admit()
        await asyncio.gather(task, return_exceptions=True)
        gate.set()
        await self.settle(service)
        await service.aclose()
        self.assert_empty(service)

    async def test_shutdown_deadline_does_not_clear_uncooperative_sender(self):
        """A sender delaying cancellation keeps its lease after shutdown failure."""
        service = await self.service(shutdown_timeout=0.03)
        sending, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                sending.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()

        task, _, _ = await self.http(service, send=send)
        try:
            await asyncio.wait_for(sending.wait(), 2)
            with self.assertRaisesRegex(TimeoutError, "grace period expired"):
                await asyncio.wait_for(service.aclose(), 1)
            await asyncio.wait_for(cancelled.wait(), 2)
            self.assertEqual(service.response_used, 1)
            self.assertGreaterEqual(service.output_bytes, 256)
            self.assertFalse(task.done())
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        await service.aclose()
        self.assert_empty(service)

    async def test_preheader_timeout_error_fits_shrunken_reservation(self):
        """Replacing a tiny completed result with 504 must stay within its lease."""
        service = await self.service(timeout=0.03, max_output_bytes=256)
        sent = []

        async def downstream(scope, receive, send):
            await service.run(scope["utility_ticket"], "tokenize", b"{}")
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                self.assertLessEqual(len(message["body"]), service.output_bytes)
            sent.append(message)

        task, _, _ = await self.http(service, app=downstream, send=send)
        await asyncio.wait_for(task, 2)
        self.assertEqual(sent[0]["status"], 504)
        self.assert_empty(service)
        error = utility_error_response(UtilityError("x" * 4096, 413))
        self.assertLessEqual(len(error.body), 256)
        self.assertEqual(error.status_code, 413)

    async def test_duplicate_response_start_is_rejected_before_and_after_headers(self):
        """A second start must never silently replace or repeat response headers."""
        for first_body in (False, True):
            with self.subTest(first_body=first_body):
                service = await self.service()
                sent = []

                async def downstream(scope, receive, send):
                    await send(
                        {"type": "http.response.start", "status": 200, "headers": []}
                    )
                    if first_body:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": b"x",
                                "more_body": True,
                            }
                        )
                    await send(
                        {"type": "http.response.start", "status": 201, "headers": []}
                    )

                async def send(message):
                    sent.append(message)

                task, _, _ = await self.http(service, app=downstream, send=send)
                with self.assertRaisesRegex(RuntimeError, "Duplicate utility"):
                    await asyncio.wait_for(task, 2)
                starts = [
                    m["status"] for m in sent if m["type"] == "http.response.start"
                ]
                self.assertEqual(starts, [200] if first_body else [])
                self.assert_empty(service)

    def test_invalid_limits_fail_before_startup(self):
        for values in (
            {"workers": 0},
            {"timeout": float("nan")},
            {"max_admitted": 1, "workers": 2},
            {"max_total_input_bytes": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                UtilityLimits(**values)

    def test_limit_types_reject_bool_and_fractional_integer_settings(self):
        """Programmatic configuration must obey the same types as CLI fields."""
        durations = {"timeout", "body_timeout", "shutdown_timeout"}
        for field in fields(UtilityLimits):
            bad_values = (True, "2", None)
            if field.name not in durations:
                bad_values += (1.5, 2.0)
            else:
                bad_values += (float("inf"), float("nan"))
            for value in bad_values:
                with (
                    self.subTest(field=field.name, value=value),
                    self.assertRaises(ValueError),
                ):
                    UtilityLimits(**{field.name: value})


if __name__ == "__main__":
    unittest.main()
