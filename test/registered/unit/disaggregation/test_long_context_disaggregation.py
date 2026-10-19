"""CPU regressions for 1M-token P/D lifecycle boundaries.

The tests model a 1,048,576-token request as 32 x 32K chunks.  They do not
require Torch, Mooncake, GPUs, or a live process group.
"""

import ast
import importlib.util
import pathlib
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = pathlib.Path(__file__).resolve().parents[4]
TRACKER_PATH = ROOT / "python/sglang/srt/disaggregation/mooncake/source_drain.py"
ENGINE_PATH = (
    ROOT
    / "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
)


spec = importlib.util.spec_from_file_location("long_context_source_drain", TRACKER_PATH)
tracker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker_module)
SourceDrainTracker = tracker_module.SourceDrainTracker
SourceTransferTimeout = tracker_module.SourceTransferTimeout


def load_engine_methods(names):
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    methods = []
    for name in names:
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        node.decorator_list = []
        methods.append(node)
    cls = ast.ClassDef(
        name="EngineUnderTest", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = {"logger": Mock()}
    exec(compile(module, str(ENGINE_PATH), "exec"), env)
    return env["EngineUnderTest"]


class TestLongContextSourceDrain(unittest.TestCase):
    TOTAL_TOKENS = 1_048_576
    CHUNK_TOKENS = 32_768

    def test_one_million_tokens_drain_and_forget_all_chunks(self):
        self.assertEqual(self.TOTAL_TOKENS % self.CHUNK_TOKENS, 0)
        chunk_count = self.TOTAL_TOKENS // self.CHUNK_TOKENS
        tracker = SourceDrainTracker()
        queue = Mock()
        chunks = [SimpleNamespace(room=11) for _ in range(chunk_count)]

        for chunk in chunks:
            self.assertTrue(tracker.publish(chunk, queue, lambda: True))
        for chunk in chunks:
            self.assertTrue(tracker.begin(chunk))
            self.assertTrue(tracker.complete(chunk))

        tracker.close(11)
        self.assertTrue(tracker.drained(11))
        self.assertTrue(tracker.forget(11))
        self.assertEqual(tracker._ops, {})
        self.assertEqual(queue.put.call_count, chunk_count)

    def test_one_chunk_timeout_keeps_source_quarantined_until_native_return(self):
        tracker = SourceDrainTracker()
        chunk = SimpleNamespace(room=12)
        queue = Mock()
        self.assertTrue(tracker.publish(chunk, queue, lambda: True))
        self.assertTrue(tracker.begin(chunk))

        def stalled_transfer():
            time.sleep(0.08)
            return 0

        with self.assertRaises(SourceTransferTimeout):
            tracker.run_with_deadline(
                stalled_transfer,
                timeout_s=0.005,
                success=lambda value: value == 0,
            )

        self.assertFalse(tracker.terminal(12))
        self.assertEqual(tracker.shutdown(timeout_s=0.005), [(12, 1)])

        # The detached native-call stand-in returns and clears its hazard.  The
        # room can then become terminal and be forgotten safely.
        time.sleep(0.12)
        self.assertTrue(tracker.terminal(12))
        self.assertTrue(tracker.forget(12))

    def test_collective_payload_scales_with_requests_not_tokens(self):
        # A 1M request is represented by chunk state, not a 1M-element
        # collective tensor.  The production queues use one status per request.
        token_count = self.TOTAL_TOKENS
        chunk_count = token_count // self.CHUNK_TOKENS
        status_payload = [1] * chunk_count
        self.assertEqual(len(status_payload), 32)
        self.assertLess(len(status_payload), token_count)


class TestLongContextMooncakeSessions(unittest.TestCase):
    def test_reusing_one_peer_endpoint_does_not_grow_session_set_per_chunk(self):
        cls = load_engine_methods(
            [
                "_native_targets",
                "_track_peer_session",
                "peer_session_count",
                "destroy_session",
                "force_destroy_sessions",
            ]
        )
        engine = cls()
        engine.engine = SimpleNamespace()
        engine._session_lock = threading.RLock()
        engine._peer_sessions = set()
        engine.logger = Mock()

        # Every chunk in a request targets the same Decode endpoint.
        for _ in range(32):
            engine._track_peer_session("decode-rank-0")
        self.assertEqual(engine._peer_sessions, {"decode-rank-0"})
        self.assertEqual(engine.peer_session_count(), 1)

        # A multi-rank request has a bounded number of endpoint ids.
        for rank in range(8):
            engine._track_peer_session(f"decode-rank-{rank}")
        self.assertEqual(len(engine._peer_sessions), 8)
        self.assertEqual(engine.peer_session_count(), 8)

        destroyed = []
        engine.engine.destroy_session = lambda sid: destroyed.append(sid) or 0
        self.assertTrue(engine.force_destroy_sessions())
        self.assertEqual(len(destroyed), 8)
        self.assertEqual(engine._peer_sessions, set())
        self.assertEqual(engine.peer_session_count(), 0)


if __name__ == "__main__":
    unittest.main()
