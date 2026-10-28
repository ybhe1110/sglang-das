"""Production ownership/collective bodies tested without Torch or native RDMA.

These model transport outcomes, not native DMA completion. Actual wheel fault
injection is still required: a returned failure must never be treated as a fence.
"""

import ast
import dataclasses
import json
import sys
import threading
import time
import unittest
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

from test_collective_timeout import load_collective_helpers
from test_external_kv_abort_drain import (
    CONN,
    ROOT,
    SourceDrainTracker,
    load_methods,
    register_cpu_ci,
)
from test_source_drain_retirement import extract_class

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
COMMON = ROOT / "python/sglang/srt/disaggregation/common/conn.py"
DECODE = ROOT / "python/sglang/srt/disaggregation/decode.py"

# Keep CustomTestCase's real setup/teardown safety without importing GPU test
# dependencies. CPU regressions run once, without the CI GPU-idle/retry policy.
CustomTestCase = extract_class(
    ROOT / "python/sglang/test/test_utils.py",
    "CustomTestCase",
    {"__init_subclass__", "setUp"},
    ["TestCase"],
    {"TestCase": unittest.TestCase, "_wait_for_gpu_idle_in_ci": lambda: None},
)


def manager_class():
    return load_methods(
        COMMON,
        [
            "register_deferred_abort_room",
            "note_abort_ack",
            "is_abort_release_safe",
            "clear_deferred_abort_state",
            "_collect_drained_abort_targets",
            "_maybe_ack_drained_abort",
            "_send_abort_acks",
            "register_deferred_ack_target",
            "poison_deferred_ack_room",
        ],
        {"DeferredAbortAckState": SimpleNamespace},
    )


def manager():
    m = manager_class()()
    m._deferred_abort_generation = 0
    m._deferred_abort_ack_tracker = {}
    m._abort_ack_lock = threading.RLock()
    m._deferred_ack_targets = {}
    m._deferred_ack_poisoned_rooms = set()
    m._staging_outstanding = defaultdict(int)
    m._send_abort_ack = Mock(return_value=True)
    return m


def mooncake_manager():
    override = load_methods(CONN, ["_collect_drained_abort_targets"], {})
    m = type("MooncakeUnderTest", (override, manager_class()), {})()
    m.__dict__.update(manager().__dict__)
    m._source_drain = SourceDrainTracker()
    return m


def wire_managers():
    """Keep production dispatch, inheritance and wire codecs, stub only IO.

    Include any Mooncake override of the shared methods, so a stale subclass
    signature cannot be hidden by binding the common methods directly.
    msgspec's data containers are replaced by stdlib dataclasses for this test.
    """
    env = dict(
        logger=Mock(),
        ABORT_TAG=b"ABORT",
        ABORT_ACK_TAG=b"ABORT_ACK",
        KVPoll=SimpleNamespace(Success=1, Failed=2, Bootstrapping=3, WaitingForInput=4),
        NamedTuple=NamedTuple,
        dataclasses=dataclasses,
        NetworkAddress=lambda ip, port: SimpleNamespace(
            to_tcp=lambda: f"tcp://{ip}:{port}", is_ipv6=False
        ),
        zmq=SimpleNamespace(DONTWAIT=1),
    )
    tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    names = {"AckTarget", "AbortNotification", "AbortAck", "DeferredAbortAckState"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    for n in nodes:
        if n.name != "AckTarget":
            n.bases, n.keywords = [], []
            n.decorator_list = [
                ast.Attribute(
                    ast.Name("dataclasses", ast.Load()), "dataclass", ast.Load()
                )
            ]
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0)] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(COMMON), "exec"), env)
    shared = {
        "register_deferred_abort_room",
        "note_abort_ack",
        "handle_abort_ack_message",
        "is_abort_release_safe",
        "clear_deferred_abort_state",
        "_collect_drained_abort_targets",
        "_prefill_unique_rank",
        "_send_abort_ack",
        "_send_abort_acks",
        "_maybe_ack_drained_abort",
        "register_deferred_ack_target",
        "poison_deferred_ack_room",
        "update_status",
    }
    base = extract_class(COMMON, "CommonKVManager", shared, [], env)
    cls = extract_class(
        CONN,
        "MooncakeKVManager",
        shared
        | {
            "_open_source_room",
            "_bind_abort_metadata",
            "_retire_abort_binding",
            "_handle_abort_notification",
            "_handle_deferred_abort_notification",
            "_handle_legacy_abort_notification",
        },
        ["CommonKVManager"],
        env,
    )

    def init(m):
        m.__dict__.update(manager().__dict__)
        del m._send_abort_ack  # Exercise actual dynamic dispatch, not a mock.
        m.enable_deferred_decode_kv_release = True
        m.request_status = {}
        m.attn_tp_rank = m.pp_rank = m.attn_cp_rank = 0
        m.pp_size = m.attn_cp_size = 1
        return m

    p, d = init(cls()), init(base())
    p._source_drain = SourceDrainTracker()
    p._abort_room_bindings = {}
    p._retired_abort_rooms = OrderedDict()
    sent = []

    def deliver(endpoint, frames, **kwargs):
        sent.append((endpoint, frames))
        d.handle_abort_ack_message(frames)

    p._send_multipart_locked = deliver
    return p, d, sent, env


class TestGenerationAndNativeDrain(CustomTestCase):
    def test_legacy_abort_is_upgraded_once_after_partial_metadata_publish(self):
        class Notification(SimpleNamespace):
            def to_zmq(self):
                return [self.generation]

        cls = load_methods(
            COMMON,
            ["ensure_abort_notified", "_send_abort_notification"],
            {
                "AbortNotification": Notification,
                "logger": Mock(),
                "time": time,
                "zmq": SimpleNamespace(DONTWAIT=1),
            },
            class_name="CommonKVReceiver",
        )
        r = cls()
        r.kv_mgr = m = manager()
        m.enable_deferred_decode_kv_release = True
        m.local_ip, m.rank_port = "peer", 1234
        r.bootstrap_room, r.init_time = 7, None
        r.bootstrap_infos = [{"rank_ip": "p", "rank_port": 1}]
        r.abort_notified, r._abort_generation = False, None
        sent = []

        def send(msg, **kwargs):
            sent.append(msg)
            if msg[0] is not None:
                # Synchronous ACK during send proves accounting was armed first.
                m.note_abort_ack(7, 0, msg[0])

        socket = SimpleNamespace(send_multipart=send)
        r._connect_to_bootstrap_server = lambda info, **kwargs: (
            socket,
            threading.Lock(),
        )
        r.ensure_abort_notified()
        self.assertEqual(sent, [[None]])
        r.ensure_abort_notified(force_arm=True)
        self.assertTrue(m.is_abort_release_safe(7, 1))
        r.ensure_abort_notified(force_arm=True)
        self.assertEqual(len(sent), 2)
        self.assertTrue(m.is_abort_release_safe(7, 1))

    def test_stale_and_duplicate_acks_cannot_release_reused_room(self):
        m = manager()
        old = m.register_deferred_abort_room(7)
        m.note_abort_ack(7, 0, old)
        m.note_abort_ack(7, 0, old)
        self.assertFalse(m.is_abort_release_safe(7, 2))
        m.clear_deferred_abort_state(7)
        new = m.register_deferred_abort_room(7)
        m.note_abort_ack(7, 1, old)
        self.assertFalse(m.is_abort_release_safe(7, 1))
        for rank in (0, 1):
            m.note_abort_ack(7, rank, new)
        self.assertTrue(m.is_abort_release_safe(7, 2))

    def test_ack_fanout_waits_for_source_tracker_not_just_chunk_counter(self):
        m = mooncake_manager()
        chunk = SimpleNamespace(room=7)
        m._source_drain.publish(chunk, Mock(), lambda: True)
        m._source_drain.begin(chunk)
        for port in (6000, 6001):
            m.register_deferred_ack_target(
                7, SimpleNamespace(ip="peer", port=port, generation=1)
            )
        m._maybe_ack_drained_abort(7)
        m._send_abort_ack.assert_not_called()
        m._source_drain.complete(chunk)
        m._source_drain.close(7)
        m._maybe_ack_drained_abort(7)
        self.assertEqual(m._send_abort_ack.call_count, 2)
        m._maybe_ack_drained_abort(7)
        self.assertEqual(m._send_abort_ack.call_count, 2)

    def test_failed_native_return_never_turns_into_drained_ack(self):
        m = mooncake_manager()
        chunk = SimpleNamespace(room=7)
        m._source_drain.publish(chunk, Mock(), lambda: True)
        m._source_drain.begin(chunk)
        with self.assertRaises(RuntimeError):
            m._source_drain.run(lambda: -1, success=lambda rc: rc == 0)
        m._source_drain.complete(chunk)
        m.register_deferred_ack_target(
            7, SimpleNamespace(ip="peer", port=1, generation=1)
        )
        m._maybe_ack_drained_abort(7)
        m._send_abort_ack.assert_not_called()
        self.assertFalse(m._source_drain.forget(7))

    def test_poison_suppresses_late_abort_and_zero_counter_ack(self):
        m = manager()
        m.poison_deferred_ack_room(7)
        m.register_deferred_ack_target(
            7, SimpleNamespace(ip="peer", port=1, generation=1)
        )
        m._maybe_ack_drained_abort(7)
        m._send_abort_ack.assert_not_called()

    def test_mooncake_abort_wire_dispatch_after_clear_and_room_reuse(self):
        """The deployed old override lost every ACK and killed decode after 30s."""
        for cleared in (False, True):
            with self.subTest(cleared=cleared):
                p, d, sent, env = wire_managers()
                generation = d.register_deferred_abort_room(7)
                source_generation = p._source_drain.open(7)
                p._abort_room_bindings[7] = (
                    source_generation,
                    {("decode", 6000): generation},
                )
                p.request_status[7] = env["KVPoll"].Bootstrapping
                if cleared:
                    p._source_drain.poison(7)
                    self.assertTrue(p._source_drain.forget(7, source_generation))
                    p.request_status.pop(7)
                notification = env["AbortNotification"](7, "decode", 6000, generation)
                for _ in range(2):
                    p._handle_abort_notification(notification.to_zmq())
                self.assertTrue(d.is_abort_release_safe(7, 1))
                self.assertFalse(d.is_abort_release_safe(7, 2))
                self.assertEqual(
                    sent[-1][1], [b"ABORT_ACK", b"7", b"0", str(generation).encode()]
                )
                self.assertNotIn(7, p._deferred_ack_targets)
                p._source_drain.poison(7)
                p._source_drain.forget(7, source_generation)
                d.clear_deferred_abort_state(7)
                d.register_deferred_abort_room(7)
                d.handle_abort_ack_message(sent[-1][1])
                self.assertFalse(d.is_abort_release_safe(7, 1))

    def test_wire_ack_waits_until_active_source_has_completed(self):
        p, d, sent, env = wire_managers()
        generation = d.register_deferred_abort_room(7)
        chunk = SimpleNamespace(room=7)
        p._source_drain.publish(chunk, Mock(), lambda: True)
        p._source_drain.begin(chunk)
        p._abort_room_bindings[7] = (
            p._source_drain._generations[7],
            {("decode", 6000): generation},
        )
        p.request_status[7] = env["KVPoll"].WaitingForInput
        p._handle_abort_notification(
            env["AbortNotification"](7, "decode", 6000, generation).to_zmq()
        )
        self.assertFalse(d.is_abort_release_safe(7, 1))
        self.assertEqual(sent, [])
        p._source_drain.complete(chunk)
        p._maybe_ack_drained_abort(7)
        self.assertTrue(d.is_abort_release_safe(7, 1))
        self.assertEqual(len(sent), 1)

    def test_failed_ack_retained_and_fanout_continues(self):
        p, d, sent, env = wire_managers()
        generation = d.register_deferred_abort_room(7)
        deliver = p._send_multipart_locked

        def fail_one(endpoint, frames, **kwargs):
            if endpoint.endswith(":6000"):
                raise OSError("control queue unavailable")
            deliver(endpoint, frames, **kwargs)

        p._send_multipart_locked = fail_one
        for port in (6000, 6001):
            p.register_deferred_ack_target(
                7, env["AckTarget"]("decode", port, generation)
            )
        p._maybe_ack_drained_abort(7)
        self.assertEqual(set(p._deferred_ack_targets[7]), {("decode", 6000)})
        self.assertEqual(len(sent), 1)
        p._send_multipart_locked = deliver
        p._maybe_ack_drained_abort(7)
        p._maybe_ack_drained_abort(7)
        self.assertNotIn(7, p._deferred_ack_targets)
        self.assertEqual(len(sent), 2)

    def test_ack_send_cannot_overwrite_concurrent_generation_or_poison(self):
        for action in ("replace", "poison"):
            with self.subTest(action=action):
                p, d, sent, env = wire_managers()
                p.register_deferred_ack_target(7, env["AckTarget"]("decode", 6000, 1))

                def send(endpoint, frames, **kwargs):
                    if action == "replace":
                        p.register_deferred_ack_target(
                            7, env["AckTarget"]("decode", 6000, 2)
                        )
                    else:
                        p.poison_deferred_ack_room(7)
                    raise OSError("send failed after concurrent state change")

                p._send_multipart_locked = send
                p._maybe_ack_drained_abort(7)
                if action == "replace":
                    self.assertEqual(
                        p._deferred_ack_targets[7][("decode", 6000)].generation, 2
                    )
                else:
                    self.assertNotIn(7, p._deferred_ack_targets)

    def test_receiver_retry_recovers_lost_ack_without_rearming(self):
        p, d, sent, env = wire_managers()
        clock = [100.0]
        env["time"] = SimpleNamespace(monotonic=lambda: clock[0])
        cls = load_methods(
            COMMON,
            [
                "ensure_abort_notified",
                "_send_abort_notification",
                "retry_abort_notification",
            ],
            env,
            class_name="CommonKVReceiver",
        )
        r = cls()
        r.kv_mgr = d
        d.local_ip, d.rank_port = "decode", 6000
        r._transfer_generation = 12345
        source_generation = p._open_source_room(7)
        p.request_status[7] = env["KVPoll"].Bootstrapping
        p._abort_room_bindings[7] = (
            source_generation,
            {("decode", 6000): r._transfer_generation},
        )
        r.bootstrap_room, r.init_time = 7, 1.0
        r.abort_notified, r._abort_generation = False, None
        r._last_abort_notification_time = None
        r.bootstrap_infos = [{"rank_ip": "prefill", "rank_port": 1}]
        notifications = []

        def send(frames, **kwargs):
            notifications.append(frames)
            self.assertEqual(kwargs["flags"], env["zmq"].DONTWAIT)
            if len(notifications) == 1:
                raise OSError("initial ABORT send failed")
            p._handle_abort_notification(frames)

        r._connect_to_bootstrap_server = lambda info, **kwargs: (
            SimpleNamespace(send_multipart=send),
            threading.Lock(),
        )
        deliver = p._send_multipart_locked
        p._send_multipart_locked = lambda *a, **kw: None  # Wire accepted, ACK lost.
        r.ensure_abort_notified()
        generation = r._abort_generation
        d.note_abort_ack(7, 1, generation)  # Preserve another rank's earlier ACK.
        self.assertFalse(d.is_abort_release_safe(7, 2))
        r.retry_abort_notification()
        self.assertEqual(len(notifications), 1)
        clock[0] += 1.0
        r.retry_abort_notification()
        self.assertEqual(notifications[0], notifications[1])
        self.assertFalse(d.is_abort_release_safe(7, 2))
        p._send_multipart_locked = deliver
        clock[0] += 1.0
        r.retry_abort_notification()
        self.assertEqual(notifications[0], notifications[2])
        self.assertEqual(r._abort_generation, generation)
        self.assertTrue(d.is_abort_release_safe(7, 2))

    def test_old_abort_cannot_cancel_reused_room_before_or_after_metadata(self):
        p, d, sent, env = wire_managers()
        old = 123456
        p._open_source_room(7)
        info = SimpleNamespace(endpoint="decode", dst_port=6000, abort_generation=old)
        self.assertTrue(p._bind_abort_metadata(7, info))
        p._source_drain.poison(7)
        self.assertTrue(p._source_drain.forget(7))
        p._retire_abort_binding(7)
        msg = env["AbortNotification"](7, "decode", 6000, old).to_zmq()
        p._handle_abort_notification(msg)  # Retired generation may still ACK.
        self.assertEqual(len(sent), 1)
        p._open_source_room(7)
        p.request_status[7] = env["KVPoll"].Bootstrapping
        for metadata_bound in (False, True):
            with self.subTest(metadata_bound=metadata_bound):
                if metadata_bound:
                    info.abort_generation = 789012
                    self.assertTrue(p._bind_abort_metadata(7, info))
                p._handle_abort_notification(msg)
                self.assertEqual(p.request_status[7], env["KVPoll"].Bootstrapping)
                self.assertEqual(len(sent), 1)
                self.assertNotIn(7, p._deferred_ack_targets)
        self.assertTrue(
            p._source_drain.publish(SimpleNamespace(room=7), Mock(), lambda: True)
        )

    def test_metadata_wire_carries_the_same_generation_as_abort(self):
        p, d, sent, env = wire_managers()
        env.update(
            time=time,
            json=json,
            dataclasses=dataclasses,
            np=SimpleNamespace(array=lambda *a, **kw: [], int32=int),
        )
        node = next(
            n
            for n in ast.parse(CONN.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "TransferInfo"
        )
        module = ast.Module(
            body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(CONN), "exec"), env)
        wire = env["TransferInfo"]
        cls = load_methods(
            CONN, ["send_metadata"], env, class_name="MooncakeKVReceiver"
        )
        r = cls()
        r._transfer_generation = 2**100 + 12345
        r.session_id, r.bootstrap_room, r.required_dst_info_num = "native", 7, 1
        r.bootstrap_infos = [{"is_dummy": True}]
        r.kv_mgr = SimpleNamespace(
            enable_staging=False, local_ip="decode", rank_port=6000
        )
        frames = []
        r._connect_to_bootstrap_server = lambda info: (
            SimpleNamespace(send_multipart=lambda msg: frames.append(msg)),
            threading.Lock(),
        )
        r.send_metadata(None)
        info = wire.from_zmq(frames[0])
        self.assertEqual(info.abort_generation, r._transfer_generation)
        p._open_source_room(7)
        p.transfer_infos = {}
        self.assertTrue(p._bind_abort_metadata(7, info, "native"))
        p.request_status[7] = env["KVPoll"].Bootstrapping
        gen = d.register_deferred_abort_room(7, generation=r._transfer_generation)
        p._handle_abort_notification(
            env["AbortNotification"](7, "decode", 6000, gen).to_zmq()
        )
        self.assertTrue(d.is_abort_release_safe(7, 1))
        self.assertEqual(p.request_status[7], env["KVPoll"].Failed)

    def test_new_metadata_before_sender_does_not_inherit_retired_binding(self):
        p, d, sent, env = wire_managers()
        old = SimpleNamespace(endpoint="decode", dst_port=6000, abort_generation=11)
        p._open_source_room(7)
        self.assertTrue(p._bind_abort_metadata(7, old))
        p._source_drain.poison(7)
        self.assertTrue(p._source_drain.forget(7))
        p._retire_abort_binding(7)
        self.assertFalse(p._bind_abort_metadata(7, old))
        new = SimpleNamespace(endpoint="decode", dst_port=6000, abort_generation=22)
        self.assertTrue(p._bind_abort_metadata(7, new))
        p._open_source_room(7)
        p.request_status[7] = env["KVPoll"].Bootstrapping
        p._handle_abort_notification(
            env["AbortNotification"](7, "decode", 6000, 11).to_zmq()
        )
        self.assertEqual(p.request_status[7], env["KVPoll"].Bootstrapping)
        self.assertEqual(sent, [])
        p._handle_abort_notification(
            env["AbortNotification"](7, "decode", 6000, 22).to_zmq()
        )
        self.assertEqual(len(sent), 1)

    def test_unknown_or_legacy_abort_never_authorizes_free(self):
        p, d, sent, env = wire_managers()
        p._open_source_room(7)
        p.request_status[7] = env["KVPoll"].Bootstrapping
        for generation in (None, 111):
            p._handle_abort_notification(
                env["AbortNotification"](7, "decode", 6000, generation).to_zmq()
            )
        self.assertEqual(sent, [])
        self.assertEqual(p.request_status[7], env["KVPoll"].Bootstrapping)
        self.assertFalse(
            p._bind_abort_metadata(
                7,
                SimpleNamespace(
                    endpoint="decode", dst_port=6000, abort_generation=None
                ),
            )
        )

    def test_metadata_before_sender_cannot_ack_before_source_owner_is_terminal(self):
        p, d, sent, env = wire_managers()
        info = SimpleNamespace(endpoint="decode", dst_port=6000, abort_generation=19)
        p.transfer_infos = {}
        self.assertTrue(p._bind_abort_metadata(7, info, "session"))
        msg = env["AbortNotification"](7, "decode", 6000, 19).to_zmq()
        p._handle_abort_notification(msg)
        self.assertEqual(sent, [])
        p.request_status[7] = env["KVPoll"].Bootstrapping
        p._open_source_room(7)
        p._handle_abort_notification(msg)
        self.assertEqual(p.request_status[7], env["KVPoll"].Failed)
        self.assertEqual(len(sent), 1)
        self.assertFalse(
            p._source_drain.publish(SimpleNamespace(room=7), Mock(), lambda: True)
        )

    def test_ack_send_releases_source_lock_before_network_io(self):
        p, d, sent, env = wire_managers()
        p.register_deferred_ack_target(7, env["AckTarget"]("decode", 6000, 1))
        sending, finish, acquired = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def send(*args, **kwargs):
            sending.set()
            self.assertTrue(finish.wait(3))

        def acquire_source():
            with p._source_drain.lock:
                acquired.set()

        p._send_multipart_locked = send
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(p._maybe_ack_drained_abort, 7)
            try:
                self.assertTrue(sending.wait(3))
                probe = pool.submit(acquire_source)
                self.assertTrue(
                    acquired.wait(1), "network send holds source drain lock"
                )
            finally:
                finish.set()
            pending.result(3)
            probe.result(3)

    def test_registration_at_final_pop_is_atomic(self):
        m = manager()
        m.register_deferred_ack_target(
            7, SimpleNamespace(ip="decode", port=1, generation=1)
        )
        replacement = SimpleNamespace(ip="decode", port=1, generation=2)
        attempted, finished = threading.Event(), threading.Event()
        threads = []
        lines = COMMON.read_text(encoding="utf-8").splitlines()

        def register():
            attempted.set()
            m.register_deferred_ack_target(7, replacement)
            finished.set()

        def trace(frame, event, arg):
            if (
                event == "line"
                and frame.f_code.co_name == "_maybe_ack_drained_abort"
                and "self._deferred_ack_targets.pop(room, None)"
                in lines[frame.f_lineno - 1]
                and not threads
            ):
                worker = threading.Thread(target=register)
                threads.append(worker)
                worker.start()
                self.assertTrue(attempted.wait(2))
                # Old code lets registration finish here, then deletes it.
                # With the state lock, registration waits until pop completes.
                finished.wait(0.2)
            return trace

        sys.settrace(trace)
        try:
            m._maybe_ack_drained_abort(7)
        finally:
            sys.settrace(None)
            for worker in threads:
                worker.join(3)
        self.assertTrue(threads)
        self.assertTrue(finished.is_set())
        self.assertIs(m._deferred_ack_targets[7][("decode", 1)], replacement)

    def test_control_send_does_not_wait_for_cache_or_send_lock(self):
        env = {"zmq": SimpleNamespace(DONTWAIT=1)}
        cls = load_methods(
            COMMON,
            ["_connect", "_send_multipart_locked"],
            env,
            class_name="CommonKVManager",
        )
        m = cls()
        m._socket_lock = threading.Lock()
        m._socket_send_locks = {"peer": threading.Lock()}
        sock = Mock()
        m._socket_cache, m._monitor_cache = {"peer": sock}, {}
        for lock in (m._socket_lock, m._socket_send_locks["peer"]):
            lock.acquire()
            with ThreadPoolExecutor(1) as pool:
                pending = pool.submit(
                    m._send_multipart_locked, "peer", [b"ACK"], flags=1
                )
                try:
                    with self.assertRaises(BlockingIOError):
                        pending.result(timeout=1)
                finally:
                    lock.release()
        sock.send_multipart.assert_not_called()

    def test_receiver_retry_does_not_wait_for_bootstrap_send_lock(self):
        p, d, sent, env = wire_managers()
        env["time"] = time
        cls = load_methods(
            COMMON,
            ["_send_abort_notification", "retry_abort_notification"],
            env,
            class_name="CommonKVReceiver",
        )
        r = cls()
        r._abort_generation, r._last_abort_notification_time = 123, None
        r.kv_mgr, r.bootstrap_room, r.bootstrap_infos = d, 7, [{}]
        d.local_ip, d.rank_port = "decode", 6000
        lock, sock = threading.Lock(), Mock()
        r._connect_to_bootstrap_server = lambda info, **kwargs: (sock, lock)
        lock.acquire()
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(r.retry_abort_notification)
            try:
                pending.result(timeout=1)
            finally:
                lock.release()
        sock.send_multipart.assert_not_called()
        self.assertEqual(r._abort_generation, 123)

    def test_receiver_cache_lock_is_nonblocking_for_control_messages(self):
        cls = load_methods(COMMON, ["_connect"], {}, class_name="CommonKVReceiver")
        r = cls()
        r._global_lock = threading.Lock()
        r._global_lock.acquire()
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(r._connect, "peer", nonblocking=True)
            try:
                with self.assertRaises(BlockingIOError):
                    pending.result(timeout=1)
            finally:
                r._global_lock.release()


def queue(env, mgr, rid="r", ready=False):
    cls = load_methods(
        DECODE,
        [
            "_reduce_deferred_flags",
            "resolve_deferred_releases",
            "_do_release",
            "has_pending_deferred_releases",
            "num_pending_deferred_releases",
            "release_memory_occupation",
        ],
        env,
        class_name="DecodeTransferQueue",
    )
    q = cls()
    q.gloo_group = object()
    q.queue = []
    q.enable_staging = False
    q.tree_cache = object()
    q._release_pd_hidden_rows = Mock()
    q.req_to_metadata_buffer_idx_allocator = Mock()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={3: 7})
    q._observe_deferred_kv_release = Mock()
    q._deferred_release_quarantine = {}
    generation = mgr.register_deferred_abort_room(7)
    if ready:
        mgr.note_abort_ack(7, 0, generation)
    receiver = SimpleNamespace(
        kv_mgr=mgr, clear=Mock(), retry_abort_notification=Mock()
    )
    req = SimpleNamespace(
        req=SimpleNamespace(rid=rid, bootstrap_room=7), kv_receiver=receiver
    )
    q._deferred_releases = [(req, 90.0, 110.0, 3, 1)]
    return q, req


class PairMin:
    """Bounded two-rank MIN simulator using real preflight tensor payloads."""

    def __init__(self):
        self.pending = []
        self.lengths = []
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2, action=self.reduce)

    def reduce(self):
        a, b = self.pending
        if len(a.values) != len(b.values):
            raise AssertionError("variable collective reached with mismatched shape")
        self.lengths.append(len(a.values))
        result = [min(x, y) for x, y in zip(a.values, b.values)]
        a.values[:] = b.values[:] = result
        self.pending.clear()

    def __call__(self, tensor, **kw):
        with self.lock:
            self.pending.append(tensor)
        self.barrier.wait(timeout=5)


class TestDecodeReleaseConsensus(CustomTestCase):
    def setUp(self):
        self.env = load_collective_helpers()
        self.env["torch"].uint8 = object()
        self.env.update(
            time=SimpleNamespace(monotonic=lambda: 100.0), release_kv_cache=Mock()
        )
        self.env["all_reduce_with_timeout"] = lambda *a, **kw: None

    def test_timeout_preserves_all_destination_owners(self):
        q, dr = queue(self.env, manager())
        q._deferred_releases[0] = (dr, 50.0, 99.0, 3, 1)
        with self.assertRaises(self.env["DisaggregationCollectiveError"]):
            q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 1)
        self.env["release_kv_cache"].assert_not_called()
        q._release_pd_hidden_rows.assert_not_called()
        q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        dr.kv_receiver.clear.assert_not_called()
        with self.assertRaises(RuntimeError):
            q.release_memory_occupation()

    def test_complete_acks_release_all_owners_once(self):
        q, dr = queue(self.env, manager(), ready=True)
        receiver = dr.kv_receiver
        q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        self.env["release_kv_cache"].assert_called_once_with(
            dr.req, q.tree_cache, is_insert=False
        )
        q._release_pd_hidden_rows.assert_called_once_with(dr)
        q.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        receiver.clear.assert_called_once_with()
        self.assertFalse(q.has_pending_deferred_releases())

    def test_partial_free_exception_is_quarantined_without_retry(self):
        q, dr = queue(self.env, manager(), ready=True)
        q._release_pd_hidden_rows.side_effect = RuntimeError("partially freed")
        with self.assertRaises(self.env["DisaggregationCollectiveError"]):
            q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 1)
        self.env["release_kv_cache"].assert_called_once()
        q._release_pd_hidden_rows.assert_called_once()
        q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()

    def test_waiting_queue_drives_retry_and_releases_only_after_ack(self):
        q, dr = queue(self.env, manager())
        receiver = dr.kv_receiver
        generation = receiver.kv_mgr._deferred_abort_ack_tracker[7].generation
        receiver.retry_abort_notification.side_effect = lambda: (
            receiver.kv_mgr.note_abort_ack(7, 0, generation)
        )
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 1)
        self.env["release_kv_cache"].assert_not_called()
        q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 0)
        self.env["release_kv_cache"].assert_called_once()
        receiver.retry_abort_notification.assert_called_once()

    def _run_pair(self, queues):
        pair = PairMin()
        self.env["all_reduce_with_timeout"] = pair
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(q.resolve_deferred_releases) for q in queues]
            errors = []
            for job in jobs:
                try:
                    job.result(timeout=10)
                except Exception as exc:
                    errors.append(exc)
        return pair, errors

    def test_rank_local_ack_readiness_cannot_release_one_rank_early(self):
        a, _ = queue(self.env, manager(), ready=True)
        b, _ = queue(self.env, manager(), ready=False)
        _, errors = self._run_pair([a, b])
        self.assertEqual(errors, [])
        self.assertEqual(
            [a.num_pending_deferred_releases(), b.num_pending_deferred_releases()],
            [1, 1],
        )
        self.env["release_kv_cache"].assert_not_called()

    def test_count_and_identity_drift_fail_before_variable_reduce(self):
        for mismatch in ("empty", "rid"):
            with self.subTest(mismatch=mismatch):
                a, _ = queue(self.env, manager(), ready=True)
                b, _ = queue(self.env, manager(), rid="other", ready=True)
                if mismatch == "empty":
                    b._deferred_releases = []
                pair, errors = self._run_pair([a, b])
                self.assertEqual(len(errors), 2)
                self.assertTrue(
                    all(
                        isinstance(e, self.env["DisaggregationCollectiveError"])
                        for e in errors
                    )
                )
                self.assertEqual(pair.lengths, [8])
                self.env["release_kv_cache"].assert_not_called()


class TestPDHiddenOwnership(CustomTestCase):
    def setUp(self):
        path = ROOT / "python/sglang/srt/disaggregation/hidden_events.py"
        cls = load_methods(
            path,
            [
                "_assert_chunk_source_quiesced",
                "free_chunk_rows",
                "release_or_mark_done",
            ],
            {},
        )
        self.h = cls()
        self.tracker = SourceDrainTracker()
        self.h.owner = SimpleNamespace(_source_drain=self.tracker)
        self.h.release_state_indices = lambda chunk: [[1, 2]]
        self.h.free_state_indices = Mock()
        self.h.mark_request_done = Mock()
        self.chunk = SimpleNamespace(room=7, pd_hidden_start=0)
        self.tracker.publish(self.chunk, Mock(), lambda: True)
        self.tracker.begin(self.chunk)

    def test_failed_native_access_keeps_streaming_and_full_hidden_rows(self):
        with self.assertRaises(RuntimeError):
            self.tracker.run(lambda: -1, success=lambda rc: rc == 0)
        for start in (0, None):
            self.chunk.pd_hidden_start = start
            with self.assertRaisesRegex(RuntimeError, "unresolved DMA"):
                self.h.release_or_mark_done(self.chunk)
        self.h.free_state_indices.assert_not_called()
        self.h.mark_request_done.assert_not_called()

    def test_hidden_rows_released_once_on_success(self):
        self.h.free_chunk_rows(self.chunk)
        self.h.free_chunk_rows(self.chunk)
        self.h.free_state_indices.assert_called_once()

    def test_partial_hidden_release_is_never_retried(self):
        self.h.free_state_indices.side_effect = RuntimeError("partial free")
        with self.assertRaisesRegex(RuntimeError, "partial free"):
            self.h.free_chunk_rows(self.chunk)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.h.free_chunk_rows(self.chunk)
        self.h.free_state_indices.assert_called_once()

    def test_decode_hidden_ack_timeout_does_not_drop_receive_owner(self):
        cls = load_methods(DECODE, ["_release_pd_hidden_rows"], {"logger": Mock()})
        q = cls()
        q.kv_manager = SimpleNamespace(
            wait_pd_hidden_ack_completions=lambda room: False
        )
        q.metadata_buffers = SimpleNamespace(pd_hidden_pool=Mock())
        req = SimpleNamespace(
            req=SimpleNamespace(rid="r", bootstrap_room=7), pd_hidden_dst_indices=[1, 2]
        )
        with self.assertRaisesRegex(RuntimeError, "ACK completion unknown"):
            q._release_pd_hidden_rows(req)
        self.assertEqual(req.pd_hidden_dst_indices, [1, 2])
        q.metadata_buffers.pd_hidden_pool.free.assert_not_called()


class TestRestoreAndOffload(CustomTestCase):
    def test_restore_failure_and_pending_are_part_of_reduced_poll(self):
        polls = SimpleNamespace(Failed=0, Transferring=1, Success=2)
        restore = SimpleNamespace(FAILED="failed", PENDING="pending", READY="ready")
        path = ROOT / "python/sglang/srt/disaggregation/decode_hicache_mixin.py"
        cls = load_methods(
            path,
            ["poll"],
            {"KVPoll": polls, "HiCacheRestoreResult": restore},
            class_name="HiCacheRestoreGatedKVReceiver",
        )
        for local, state, expected in (
            (2, "failed", 0),
            (2, "pending", 1),
            (2, "ready", 2),
            (1, "failed", 1),
            (0, "pending", 0),
        ):
            with self.subTest(local=local, state=state):
                gate = cls()
                gate.decode_req = SimpleNamespace(
                    kv_receiver=SimpleNamespace(poll=lambda: local),
                    hicache_restore_status=state,
                )
                self.assertEqual(gate.poll(), expected)

    def test_offload_retraction_refuses_local_or_peer_pending_copy(self):
        class Tensor:
            def __init__(self, values):
                self.value = values[0]

            def item(self):
                return self.value

        torch = SimpleNamespace(
            tensor=lambda values, **kw: Tensor(values),
            int=int,
            distributed=SimpleNamespace(ReduceOp=SimpleNamespace(MAX="max")),
        )
        env = {"torch": torch, "all_reduce_with_timeout": Mock()}
        path = (
            ROOT / "python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py"
        )
        cls = load_methods(path, ["drain_before_retraction"], env)
        for local, peer in ((False, False), (True, False), (False, True)):
            with self.subTest(local=local, peer=peer):
                m = cls()
                m.check_offload_progress = Mock()
                m.cache_controller = SimpleNamespace(
                    ack_write_queue=[1] if local else []
                )
                m.ongoing_offload, m.offload_inflight = {}, {}
                m.tp_world_size, m.tp_group = 2, object()

                def reduce(tensor, **kw):
                    m.check_offload_progress.assert_called_once()
                    tensor.value = tensor.value or peer

                env["all_reduce_with_timeout"].side_effect = reduce
                if local or peer:
                    with self.assertRaisesRegex(RuntimeError, "refusing to retract"):
                        m.drain_before_retraction()
                else:
                    m.drain_before_retraction()


if __name__ == "__main__":
    unittest.main()
