import asyncio
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

script = (
    Path(__file__).resolve().parents[4]
    / "benchmark/http_utility/detokenize_phase_probe.py"
)
spec = importlib.util.spec_from_file_location("detokenize_phase_probe", script)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class TestDetokenizePhaseProbe(CustomTestCase):
    def test_probe_rejects_reconnected_socket_and_worker_change(self):
        header = {
            "instance": "worker-1",
            "pid": 123,
            "ns": dict.fromkeys(probe.GET_MARKS, 1),
        }
        sock = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 32000))
        stream = SimpleNamespace(get_extra_info=lambda key: sock)

        async def check():
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={probe.HEADER: json.dumps(header)},
                    extensions={"network_stream": stream},
                )
            )
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                original = await probe.request(client, probe.GETS[0])
                for key, value in (("pid", 456), ("instance", "worker-2")):
                    previous = header[key]
                    header[key] = value
                    with self.assertRaisesRegex(
                        RuntimeError, "connection/worker changed"
                    ):
                        await probe.request(
                            client, probe.GETS[0], pinned=original["worker"]
                        )
                    header[key] = previous
                sock.getsockname = lambda: ("127.0.0.1", 32001)
                with self.assertRaisesRegex(RuntimeError, "connection/worker changed"):
                    await probe.request(
                        client, probe.GETS[0], pinned=original["worker"]
                    )

        asyncio.run(check())

    def test_clock_bounds_and_phase_attribution(self):
        # Different monotonic clock origins. Idle GET takes 4 ms round-trip.
        ms = probe.MS
        origin = 400_000 * ms

        def event(start, end, entered, ready):
            return dict(
                start=start * ms,
                end=end * ms,
                path=probe.GETS[0],
                ns={
                    "asgi_enter": origin + entered * ms,
                    "headers_ready": origin + ready * ms,
                    "handler_begin": origin + entered * ms,
                    "handler_end": origin + ready * ms,
                },
            )

        idle = event(-10, -6, -9, -7)
        delayed = event(20, 85, 80, 84)
        # With headers already prepared, response delivery can also overlap decode.
        delivery = event(20, 85, 21, 24)
        tool = {
            "start": 0,
            "ns": {"decode_begin": origin + 19 * ms, "decode_end": origin + 80 * ms},
        }
        for measurement in (delayed, delivery):
            bounds = probe.clock_bounds([idle, measurement])
            self.assertEqual(bounds, (origin - ms, origin + ms))
            row = probe.analyze(tool, measurement, [idle], bounds)
            self.assertEqual(row["excess_in_decode_ms"], [55, 61])
            self.assertEqual(
                row["pre_handler_decode_ms"][0], 59 if measurement is delayed else 0
            )
            rows = [dict(row, round=n, offset_ms=20) for n in (1, 2, 3)]

            def verdict(samples):
                return probe.summarize(samples, 10)[probe.GETS[0]]["verdict"]

            self.assertEqual(verdict(rows), "majority_overlaps_decode")
            self.assertEqual(verdict(rows[:2]), "inconclusive")
            self.assertEqual(
                verdict([dict(r, excess_in_decode_ms=[0, 0]) for r in rows]),
                "majority_outside_decode",
            )
            self.assertEqual(
                verdict([dict(r, excess_in_decode_ms=[10, 50]) for r in rows]),
                "inconclusive",
            )
        with self.assertRaisesRegex(RuntimeError, "clock"):
            probe.clock_bounds([event(0, 2, 1, 100)])


if __name__ == "__main__":
    unittest.main()
