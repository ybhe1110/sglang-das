"""Retired senders stay terminal without touching a reused room (CPU only)."""

import ast
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import test_external_kv_abort_drain as abort_drain_tests
from test_external_kv_abort_drain import (
    CONN,
    KVPoll,
    PREFILL,
    ROOT,
    SCHED,
    SourceDrainTracker,
    load_methods,
)

register_cpu_ci = abort_drain_tests.register_cpu_ci
CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def extract_class(path, name, names, bases, env):
    original = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    original.body = [
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    original.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            original,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
    return env[name]


def make_sender():
    # Keep the real sender clear/abort and manager drain methods. Only trace,
    # IPC notifications and failure reporting are external boundary doubles.
    env = dict(KVPoll=KVPoll)
    extract_class(
        ROOT / "python/sglang/srt/disaggregation/common/conn.py",
        "CommonKVSender",
        {"clear", "abort"},
        [],
        env,
    )
    cls = extract_class(
        CONN,
        "MooncakeKVSender",
        {
            "clear",
            "abort",
            "is_transfer_terminal",
            "is_transfer_drained",
            "forget_source_transfers",
        },
        ["CommonKVSender"],
        env,
    )
    manager_cls = load_methods(
        CONN,
        ["source_transfers_terminal", "source_transfers_drained"],
        dict(KVPoll=KVPoll),
    )
    mgr = manager_cls()
    mgr._source_drain = SourceDrainTracker()
    mgr.request_status = {7: KVPoll.Transferring}
    mgr.record_failure = Mock()
    mgr.deferred_bootstrap = None
    mgr.req_to_decode_prefix_len = {}
    mgr.transfer_infos = {}
    mgr._maybe_ack_drained_abort = Mock()
    mgr._wake_pd_hidden_ack_waiters = Mock()

    def update(room, status):
        mgr.request_status[room] = status
        if status == KVPoll.Failed:
            mgr._source_drain.poison(room)

    mgr.update_status = update
    sender = cls()
    sender.kv_mgr = mgr
    sender.bootstrap_room = 7
    sender._source_generation = mgr._source_drain.open(7)
    sender._source_transfers_forgotten = False
    sender.conclude_state = None
    sender.trace_ctx = Mock()
    sender.poll = lambda: sender.conclude_state
    return sender


class TestSourceDrainRetirement(CustomTestCase):
    def test_bootstrap_abort_clear_before_cleanup_remains_terminal(self):
        sender = make_sender()
        sender.abort()  # No KV chunk has ever been submitted.
        sender.clear()  # failure_exception clears before scheduler cleanup.
        for _ in range(3):
            self.assertTrue(sender.is_transfer_terminal())
            self.assertTrue(sender.is_transfer_drained())
            self.assertTrue(sender.forget_source_transfers())
            sender.abort()
            sender.clear()
        self.assertEqual(sender.kv_mgr.request_status, {})
        self.assertEqual(sender.kv_mgr._source_drain._generations, {})

    def test_forget_before_clear_removes_room_metadata(self):
        sender = make_sender()
        sender.abort()
        self.assertTrue(sender.forget_source_transfers())
        self.assertEqual(sender.kv_mgr.request_status, {})
        sender.clear()
        self.assertTrue(sender.is_transfer_terminal())

    def test_retired_sender_does_not_query_or_poison_reused_room(self):
        sender = make_sender()
        sender.abort()
        sender.clear()
        tracker = sender.kv_mgr._source_drain
        new_generation = tracker.open(7)
        sender.kv_mgr.request_status[7] = KVPoll.Transferring
        for _ in range(2):
            self.assertTrue(sender.is_transfer_terminal())
            self.assertTrue(sender.is_transfer_drained())
            self.assertTrue(sender.forget_source_transfers())
            sender.abort()
            sender.clear()
        self.assertTrue(tracker.owns_generation(7, new_generation))
        self.assertEqual(sender.kv_mgr.request_status[7], KVPoll.Transferring)
        self.assertNotIn((7, new_generation), tracker._closed)
        self.assertNotIn((7, new_generation), tracker._poisoned)

    def test_unknown_or_stale_generation_is_not_retired(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                sender = make_sender()
                tracker = sender.kv_mgr._source_drain
                generation = sender._source_generation
                tracker.poison(7, generation)
                tracker.forget(7, generation)  # Not acknowledged by this sender.
                new_generation = tracker.open(7)
                if unknown:
                    sender._source_generation = None
                self.assertFalse(sender.is_transfer_terminal())
                self.assertFalse(sender.is_transfer_drained())
                self.assertFalse(sender.forget_source_transfers())
                sender.abort()
                sender.clear()
                self.assertTrue(tracker.owns_generation(7, new_generation))
                self.assertNotIn((7, new_generation), tracker._closed)
                self.assertEqual(sender.kv_mgr.request_status[7], KVPoll.Transferring)

    def test_active_source_cannot_be_forgotten_until_completion(self):
        sender = make_sender()
        tracker = sender.kv_mgr._source_drain
        chunk = SimpleNamespace(room=7)
        tracker.publish(chunk, Mock(), lambda: True, sender._source_generation)
        tracker.begin(chunk)
        sender.abort()
        sender.clear()
        self.assertFalse(sender.is_transfer_terminal())
        self.assertFalse(sender.forget_source_transfers())
        tracker.complete(chunk)
        self.assertTrue(sender.is_transfer_terminal())
        sender.clear()
        self.assertTrue(sender.is_transfer_terminal())


if __name__ == "__main__":
    unittest.main()
