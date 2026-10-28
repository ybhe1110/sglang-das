"""Exercise production teardown bodies with real threads and controlled IO.

CPU tests prove ownership/order and bounded caller waits, not native DMA fences.
"""

import struct
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from test_external_kv_abort_drain import CONN, load_methods, register_cpu_ci
from test_pd_failure_protocol_cpu import COMMON

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def manager():
    cls = load_methods(
        CONN,
        ["shutdown", "_shutdown_transport", "_close_zmq_resources"],
        dict(time=time, threading=threading, logger=Mock()),
        class_name="MooncakeKVManager",
    )
    m = cls()
    m._shutdown_lock = threading.Lock()
    m._shutdown_complete = False
    m._shutdown_event = threading.Event()
    m._stop_heartbeat_checker = lambda **kw: m._shutdown_event.set()
    m._worker_threads, m._receiver_threads, m.executors = [], [], []
    m._socket_lock = threading.Lock()
    m._socket_cache, m._monitor_cache, m._socket_send_locks = {}, {}, {}
    m.deregister_buffer_to_engine = Mock()
    return m


class TestMooncakeShutdown(unittest.TestCase):
    def test_running_executor_does_not_block_caller_or_report_success(self):
        m = manager()
        release, entered = threading.Event(), threading.Event()

        def job():
            entered.set()
            release.wait()

        pool = ThreadPoolExecutor(1)
        m.executors = [pool]
        pool.submit(job)
        self.assertTrue(entered.wait(1))
        try:
            start = time.monotonic()
            self.assertFalse(m.shutdown(0.01))
            self.assertLess(time.monotonic() - start, 1)
            worker = m._shutdown_worker
            self.assertFalse(m.shutdown(0.01))
            self.assertIs(m._shutdown_worker, worker)
            m.deregister_buffer_to_engine.assert_not_called()
        finally:
            release.set()
            m._shutdown_worker.join(2)
            pool.shutdown()
        self.assertTrue(m.shutdown(1))
        self.assertTrue(m.shutdown(1))
        m.deregister_buffer_to_engine.assert_called_once()

    def test_receiver_exits_before_socket_close(self):
        m = manager()
        release, entered = threading.Event(), threading.Event()

        def receive():
            entered.set()
            release.wait()

        receiver = threading.Thread(target=receive)
        receiver.start()
        m._receiver_threads = [receiver]
        m.server_socket = Mock()
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(m.shutdown(0.01))
            m.server_socket.close.assert_not_called()
        finally:
            release.set()
            receiver.join(2)
            m._shutdown_worker.join(2)
        self.assertTrue(m.shutdown(1))
        m.server_socket.close.assert_called_once_with(linger=0)

    def test_busy_sender_keeps_socket_open_until_it_leaves(self):
        m = manager()
        sock, lock = Mock(), threading.Lock()
        m._socket_cache["peer"] = sock
        m._socket_send_locks["peer"] = lock
        lock.acquire()
        try:
            self.assertFalse(m.shutdown(0.01))
            sock.close.assert_not_called()
            m.deregister_buffer_to_engine.assert_not_called()
        finally:
            lock.release()
            m._shutdown_worker.join(2)
        self.assertTrue(m.shutdown(1))
        sock.close.assert_called_once()
        # An admitted sender must still be able to find its lock, check the
        # stop flag, and reject the stale socket instead of raising KeyError.
        self.assertIs(m._socket_send_locks["peer"], lock)

    def test_context_term_is_bounded_and_serialized(self):
        m = manager()
        entered, release = threading.Event(), threading.Event()

        def term():
            entered.set()
            release.wait()

        m._zmq_ctx = SimpleNamespace(term=Mock(side_effect=term))
        try:
            self.assertFalse(m.shutdown(0.01))
            self.assertTrue(entered.wait(1))
            self.assertFalse(m.shutdown(0.01))
            m._zmq_ctx.term.assert_called_once()
            m.deregister_buffer_to_engine.assert_not_called()
        finally:
            release.set()
            m._shutdown_worker.join(2)
        self.assertTrue(m.shutdown(1))

    def test_unresolved_source_is_retained_and_later_retry_finishes_once(self):
        m = manager()
        m._source_drain = SimpleNamespace(shutdown=Mock(return_value=[(7, 1)]))
        self.assertFalse(m.shutdown(1))
        m.deregister_buffer_to_engine.assert_not_called()
        m._source_drain.shutdown.return_value = []
        self.assertTrue(m.shutdown(1))
        self.assertTrue(m.shutdown(1))
        m.deregister_buffer_to_engine.assert_called_once()

    def test_close_failure_does_not_claim_success_or_release_buffers(self):
        m = manager()
        m.server_socket = Mock()
        m.server_socket.close.side_effect = RuntimeError("injected close failure")
        self.assertFalse(m.shutdown(1))
        m.deregister_buffer_to_engine.assert_not_called()
        m.server_socket.close.side_effect = None
        self.assertTrue(m.shutdown(1))
        m.deregister_buffer_to_engine.assert_called_once()

    def test_failed_monitor_close_keeps_reference_for_retry(self):
        m = manager()
        m._socket_cache["peer"] = Mock()
        m._socket_send_locks["peer"] = threading.Lock()
        monitor = Mock()
        m._monitor_cache["peer"] = monitor
        monitor.close.side_effect = RuntimeError("monitor close failed")
        self.assertFalse(m.shutdown(1))
        self.assertIs(m._monitor_cache["peer"], monitor)
        m.deregister_buffer_to_engine.assert_not_called()
        monitor.close.side_effect = None
        self.assertTrue(m.shutdown(1))
        self.assertEqual(m._monitor_cache, {})
        m.deregister_buffer_to_engine.assert_called_once()

    def test_new_sends_rejected_after_shutdown_begins(self):
        cls = load_methods(
            COMMON,
            ["_connect", "_send_multipart_locked"],
            {"zmq": SimpleNamespace(DONTWAIT=1)},
            class_name="CommonKVManager",
        )
        m = cls()
        m._socket_lock = threading.Lock()
        m._shutdown_event = threading.Event()
        m._shutdown_event.set()
        m._socket_cache = {"peer": Mock()}
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            m._send_multipart_locked("peer", [b"ACK"], flags=1)
        m._socket_cache["peer"].send_multipart.assert_not_called()

    def test_registration_send_failure_preserves_original_failure(self):
        class ZMQError(Exception):
            pass

        env = dict(
            struct=struct,
            pack_int_lists=lambda *a: b"",
            pack_string_list=lambda *a: b"",
            pack_state_types=lambda *a: b"",
            zmq=SimpleNamespace(ZMQError=ZMQError),
            KVPoll=SimpleNamespace(Failed=2),
        )
        cls = load_methods(CONN, ["_register_kv_args"], env)
        manager_cls = load_methods(COMMON, ["record_failure"], {})
        m = manager_cls()
        m.failure_records = {}
        m.failure_lock = threading.Lock()
        m.kv_args = SimpleNamespace(
            kv_data_ptrs=[],
            aux_data_ptrs=[],
            state_data_ptrs=[],
            state_item_lens=[],
            state_layer_ids=[],
            state_types=[],
            kv_layer_ids=[],
            engine_rank=0,
            kv_item_lens=[],
        )
        m.attn_tp_size = m.dcp_size = 1
        m.dcp_rank, m.enable_staging = 0, False
        m.local_ip, m.rank_port = "127.0.0.1", 1234
        m.update_status = Mock()
        r = cls()
        r.bootstrap_infos = [{"rank_ip": "127.0.0.2", "rank_port": 2345}]
        r.kv_mgr, r.bootstrap_room, r.session_id = m, 7, "test"
        sock = Mock()
        sock.send_multipart.side_effect = ZMQError("injected send failure")
        r._connect_to_bootstrap_server = lambda info: (sock, threading.Lock())
        self.assertFalse(r._register_kv_args())
        self.assertIn("_register_kv_args", m.failure_records[7])
        self.assertEqual(r.conclude_state, 2)
        m.update_status.assert_called_once_with(7, 2)


if __name__ == "__main__":
    unittest.main()
