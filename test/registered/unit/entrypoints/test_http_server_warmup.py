import asyncio
import gc
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


class TestHealthProbeCleanup(CustomTestCase):
    """Health responses must not leave probe waiters or unobserved cancellations.

    Busy DP serving used to cancel probe children and return 200 without draining
    their aggregate, logging an unhandled CancelledError during normal inference.
    """

    def run_case(
        self, scenario, *, dp_size=2, is_generation=True, cancel_during_cleanup=False
    ):
        async def run():
            started = asyncio.Event()
            release = asyncio.Event()
            cleanup_started = asyncio.Event()
            finish_cleanup = asyncio.Event()
            closed = []
            requests = []
            errors = []
            business_state = object()
            manager = SimpleNamespace(
                gracefully_exit=False,
                server_status=http_server.ServerStatus.Up,
                elastic_worker_count=dp_size,
                is_generation=is_generation,
                last_receive_tstamp=0.0,
                rid_to_state={"business-request": business_state},
            )

            async def generate_request(gri, request):
                rids = gri.rid if isinstance(gri.rid, list) else [gri.rid]
                requests.append(gri)
                manager.rid_to_state.update({rid: object() for rid in rids})
                started.set()
                try:
                    if scenario == "complete":
                        yield {}
                        return
                    if scenario == "failure" and len(requests) == 1:
                        # Let every peer enter before one probe fails.
                        await asyncio.sleep(0)
                        raise RuntimeError("probe failed")
                    if scenario == "probe_cancel":
                        raise asyncio.CancelledError()
                    if scenario == "busy":
                        manager.last_receive_tstamp = time.time() + 1
                    await release.wait()
                    yield {}
                finally:
                    if cancel_during_cleanup:
                        cleanup_started.set()
                        await finish_cleanup.wait()
                    # Make cleanup asynchronous so cancel-without-await fails.
                    await asyncio.sleep(0)
                    closed.extend(rids)

            manager.generate_request = generate_request
            loop = asyncio.get_running_loop()
            previous_handler = loop.get_exception_handler()
            loop.set_exception_handler(lambda loop, context: errors.append(context))
            endpoint = None
            try:
                with (
                    get_context().override_server_args(
                        tp_size=dp_size,
                        attn_dp_size=dp_size,
                        disaggregation_mode="decode",
                    ),
                    patch.object(
                        http_server,
                        "_global_state",
                        SimpleNamespace(tokenizer_manager=manager),
                    ),
                    patch.object(http_server, "HEALTH_CHECK_TIMEOUT", 0.01),
                ):
                    request = http_server.Request(
                        {"type": "http", "path": "/health_generate"}
                    )
                    endpoint = asyncio.create_task(http_server.health_generate(request))
                    if scenario == "cancel":
                        await asyncio.wait_for(started.wait(), timeout=5)
                        endpoint.cancel()
                    if cancel_during_cleanup:
                        # The endpoint has already cancelled its probe. A later
                        # HTTP cancellation must not cancel the probe a second time.
                        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
                        for _ in range(2):
                            endpoint.cancel()
                            await asyncio.sleep(0)
                            await asyncio.sleep(0)
                            self.assertFalse(endpoint.done())
                            self.assertEqual(closed, [])
                            self.assertGreater(len(manager.rid_to_state), 1)
                        finish_cleanup.set()
                    if scenario == "cancel" or cancel_during_cleanup:
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(endpoint, timeout=5)
                    else:
                        response = await asyncio.wait_for(endpoint, timeout=5)
                        self.assertEqual(
                            response.status_code,
                            200 if scenario in ("busy", "complete") else 503,
                        )
                    # Check at endpoint completion, before test teardown can mask leaks.
                    expected_rids = [
                        rid
                        for gri in requests
                        for rid in (gri.rid if isinstance(gri.rid, list) else [gri.rid])
                    ]
                    self.assertTrue(expected_rids)
                    self.assertCountEqual(closed, expected_rids)
                    self.assertEqual(
                        manager.rid_to_state, {"business-request": business_state}
                    )
                    if scenario != "cancel":
                        self.assertEqual(
                            manager.server_status,
                            http_server.ServerStatus.Up
                            if scenario in ("busy", "complete")
                            else http_server.ServerStatus.UnHealthy,
                        )
                    endpoint = None
                    await asyncio.sleep(0)
                    gc.collect()
                    await asyncio.sleep(0)
                    self.assertEqual(errors, [])
            finally:
                release.set()
                finish_cleanup.set()
                if endpoint is not None:
                    endpoint.cancel()
                    await asyncio.gather(endpoint, return_exceptions=True)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                gc.collect()
                loop.set_exception_handler(previous_handler)

        asyncio.run(run())

    def test_busy_dp_probe_is_drained_without_asyncio_error(self):
        self.run_case("busy")

    def test_timed_out_dp_probe_is_drained(self):
        self.run_case("timeout")

    def test_cancelled_http_request_drains_probes_and_propagates(self):
        for dp_size in (1, 2):
            with self.subTest(dp_size=dp_size):
                self.run_case("cancel", dp_size=dp_size)

    def test_http_cancellation_during_cleanup_preserves_probe_lifetime(self):
        """Repeated cancellation cannot interrupt cleanup or retire states early."""
        for dp_size in (1, 2):
            for scenario in ("busy", "timeout", "cancel"):
                with self.subTest(dp_size=dp_size, scenario=scenario):
                    self.run_case(scenario, dp_size=dp_size, cancel_during_cleanup=True)

    def test_failed_probe_drains_other_dp_probe(self):
        self.run_case("failure", is_generation=False)

    def check_cancel_during_aclose(self, dp_size):
        async def run():
            baseline_tasks = asyncio.all_tasks()
            all_closing = asyncio.Event()
            gates = [asyncio.Event() for _ in range(dp_size)]
            closing = []
            closed = []
            business_state = object()
            states = {"business-request": business_state}

            async def response(rid, gate):
                states[rid] = object()
                try:
                    yield {}
                finally:
                    closing.append(rid)
                    if len(closing) == dp_size:
                        all_closing.set()
                    await gate.wait()
                    closed.append(rid)

            async def batch_response(rids):
                yield await http_server.TokenizerManager._collect_batch_responses(
                    manager, [response(rid, gate) for rid, gate in zip(rids, gates)]
                )

            def generate_request(gri, request):
                if dp_size == 1:
                    self.assertIsInstance(gri.rid, str)
                    return response(gri.rid, gates[0])
                self.assertIsInstance(gri.rid, list)
                self.assertEqual(len(gri.rid), dp_size)
                return batch_response(gri.rid)

            manager = SimpleNamespace(
                gracefully_exit=False,
                server_status=http_server.ServerStatus.Up,
                elastic_worker_count=dp_size,
                is_generation=True,
                last_receive_tstamp=0.0,
                rid_to_state=states,
                generate_request=generate_request,
            )
            with (
                get_context().override_server_args(
                    tp_size=dp_size, attn_dp_size=dp_size, disaggregation_mode="decode"
                ),
                patch.object(
                    http_server,
                    "_global_state",
                    SimpleNamespace(tokenizer_manager=manager),
                ),
            ):
                request = http_server.Request(
                    {"type": "http", "path": "/health_generate"}
                )
                endpoint = asyncio.create_task(http_server.health_generate(request))
                try:
                    await asyncio.wait_for(all_closing.wait(), timeout=5)
                    expected_states = states.copy()
                    self.assertEqual(
                        set(expected_states), {"business-request", *closing}
                    )
                    for gate in gates:
                        for _ in range(2):
                            endpoint.cancel()
                            await asyncio.sleep(0)
                            await asyncio.sleep(0)
                            self.assertFalse(endpoint.done())
                            self.assertEqual(states, expected_states)
                        gate.set()
                        await asyncio.sleep(0)
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(endpoint, timeout=5)
                    self.assertCountEqual(closed, closing)
                    self.assertEqual(states, {"business-request": business_state})
                    self.assertEqual(asyncio.all_tasks(), baseline_tasks)
                finally:
                    for gate in gates:
                        gate.set()
                    endpoint.cancel()
                    await asyncio.gather(endpoint, return_exceptions=True)

        asyncio.run(run())

    def test_single_probe_aclose_survives_http_cancellation(self):
        """A probe that yielded must finish closing before its RID is retired."""
        self.check_cancel_during_aclose(1)

    def test_native_batch_aclose_survives_http_cancellation(self):
        """Keep every batch RID until the last rank's response generator closes."""
        self.check_cancel_during_aclose(2)

    def test_cancelled_probe_reports_unhealthy(self):
        self.run_case("probe_cancel")

    def test_success_closes_generator_and_releases_probe_states(self):
        self.run_case("complete")

    def test_single_rank_probe_cleanup_on_busy_and_timeout(self):
        for scenario in ("busy", "timeout"):
            with self.subTest(scenario=scenario):
                self.run_case(scenario, dp_size=1)


if __name__ == "__main__":
    unittest.main()
