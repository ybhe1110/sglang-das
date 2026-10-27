"""4.1 ReqKvInfo, role-switch and native capability backport regressions.

Uses real production method bodies at the hardware boundary. These tests do
not assert that a failed native operation has stopped DMA.
"""

import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_external_kv_abort_drain import (
    CONN,
    KVPoll,
    PREFILL,
    ROOT,
    SourceDrainTracker,
    load_methods,
    register_cpu_ci,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
ENGINE = (
    ROOT
    / "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
)


class TestV41PrefillOwnership(unittest.TestCase):
    def setUp(self):
        self.clock = 10.0
        self.cls = load_methods(
            PREFILL,
            ["_retire_aborted_prefill_result", "_prefill_cleanup_flags"],
            {
                "KVPoll": KVPoll,
                "time": NS(monotonic=lambda: self.clock),
                "envs": NS(SGLANG_DISAGGREGATION_COLLECTIVE_TIMEOUT=NS(get=lambda: 30)),
            },
        )

    def test_tp8_abort_parks_each_owner_once_and_waits_for_last_chunk(self):
        ranks = []
        for rank in range(8):
            s = self.cls()
            s.clear_pending_chunk_send = Mock()
            req = NS(
                rid="same",
                pending_bootstrap=True,
                inflight_middle_chunks=int(rank == 7),
                disagg_kv_sender=NS(abort=Mock(), is_transfer_terminal=lambda: True),
            )
            s.chunked_req = req
            s.disagg_prefill_inflight_queue = []
            self.assertFalse(s._retire_aborted_prefill_result(req))
            self.assertFalse(s._retire_aborted_prefill_result(req))
            self.assertEqual(s.disagg_prefill_inflight_queue, [req])
            self.assertIsNone(s.chunked_req)
            self.assertTrue(req.skip_radix_cache_insert)
            ranks.append((s, req))
        flags = [s._prefill_cleanup_flags([r], [KVPoll.Failed])[0][0] for s, r in ranks]
        self.assertEqual(min(flags), 0)
        ranks[-1][1].inflight_middle_chunks = 0
        self.assertEqual(
            min(s._prefill_cleanup_flags([r], [KVPoll.Failed])[0][0] for s, r in ranks),
            1,
        )
        for s, req in ranks:
            req._pd_cleanup_done = True
            s.disagg_prefill_inflight_queue.clear()
            self.assertFalse(s._retire_aborted_prefill_result(req))
            self.assertEqual(s.disagg_prefill_inflight_queue, [])

    def test_failed_source_deadline_retains_request(self):
        s = self.cls()
        req = NS(
            inflight_middle_chunks=0,
            disagg_kv_sender=NS(abort=Mock(), is_transfer_terminal=lambda: False),
        )
        self.assertEqual(s._prefill_cleanup_flags([req], [KVPoll.Failed]), ([0], [1]))
        self.clock += 31
        self.assertEqual(s._prefill_cleanup_flags([req], [KVPoll.Failed]), ([0], [0]))

    def test_pending_bootstrap_still_advances_without_retiring_source(self):
        s = self.cls()
        terminal = Mock(return_value=False)
        req = NS(
            inflight_middle_chunks=1,
            disagg_kv_sender=NS(abort=Mock(), is_transfer_terminal=terminal),
        )
        self.assertEqual(
            s._prefill_cleanup_flags([req], [KVPoll.Transferring]), ([1], [1])
        )
        terminal.assert_not_called()

    def test_decode_silent_drop_also_waits_for_remote_drain(self):
        path = ROOT / "python/sglang/srt/disaggregation/decode.py"
        cls = load_methods(path, ["_drop_uncommitted"], {"logger": Mock()})
        queue = cls()
        queue.enable_deferred_kv_release = True
        queue._clean_hicache_prefetch_resources = Mock()
        queue._defer_release = Mock()
        receiver = NS(
            abort=Mock(),
            ensure_abort_notified=Mock(),
            kv_mgr=NS(enable_deferred_decode_kv_release=True),
        )
        dr = NS(req=NS(rid="r"), kv_receiver=receiver)
        queue._drop_uncommitted(dr, stream_error=False)
        receiver.ensure_abort_notified.assert_called_once_with(force_arm=True)
        queue._defer_release.assert_called_once_with(dr)
        self.assertIs(dr.kv_receiver, receiver)

    def test_untracked_request_release_cannot_free_live_source(self):
        import ast

        path = ROOT / "python/sglang/srt/mem_cache/common.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "release_kv_cache"
        )
        # Substitute the exception import only; execute the production guard.
        fn.body = [n for n in fn.body if not isinstance(n, ast.ImportFrom)]
        for n in ast.walk(fn):
            if isinstance(n, ast.If):
                n.body = [x for x in n.body if not isinstance(x, ast.ImportFrom)]
        env = {"SourceTransferUnresolved": RuntimeError}
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                fn,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
        cache = Mock()
        req = NS(disagg_kv_sender=NS(is_transfer_terminal=lambda: False))
        with self.assertRaisesRegex(RuntimeError, "uncertain source"):
            env["release_kv_cache"](req, cache, checkpoint=False)
        cache.free_kv_row.assert_not_called()


class TestV41NativeShutdown(unittest.TestCase):
    def manager(self, outstanding):
        cls = load_methods(
            CONN,
            ["shutdown"],
            {"SourceTransferUnresolved": RuntimeError, "logger": Mock()},
            class_name="MooncakeKVManager",
        )
        m = cls()
        m._source_drain = NS(shutdown=Mock(return_value=outstanding))
        m._worker_threads = []
        m._socket_lock = threading.Lock()
        m._socket_cache, m._monitor_cache = {}, {}
        m.server_socket, m._zmq_ctx = Mock(), Mock()
        m.deregister_buffer_to_engine = Mock()
        m.disaggregation_mode = NS(value="prefill")
        return m

    def test_unresolved_source_prevents_socket_and_memory_teardown(self):
        m = self.manager([(7, 1)])
        with self.assertRaises(RuntimeError):
            m.shutdown()
        m.deregister_buffer_to_engine.assert_not_called()
        m._zmq_ctx.destroy.assert_not_called()

    def test_drained_manager_releases_registrations(self):
        m = self.manager([])
        m.shutdown()
        m.deregister_buffer_to_engine.assert_called_once()

    def test_alive_worker_prevents_deregistration(self):
        m = self.manager([])
        m._worker_threads = [NS(join=Mock(), is_alive=lambda: True)]
        with self.assertRaises(RuntimeError):
            m.shutdown()
        m.deregister_buffer_to_engine.assert_not_called()

    def engine(self, native):
        cls = load_methods(
            ENGINE,
            [
                "shutdown",
                "destroy_session",
                "_native_targets",
                "force_destroy_sessions",
            ],
            {"logger": Mock()},
            class_name="MooncakeTransferEngine",
        )
        e = cls()
        e._region_lock = e._session_lock = threading.RLock()
        e._shutdown = e._endpoint_destroyed = False
        e._registered_regions, e._peer_sessions = {(1, 16)}, {"peer"}
        e.engine = native
        return e

    def test_missing_endpoint_api_preserves_unresolved_owners(self):
        e = self.engine(NS())
        self.assertFalse(e.shutdown())
        self.assertEqual(e._peer_sessions, {"peer"})
        self.assertEqual(e._registered_regions, {(1, 16)})

    def test_failed_native_close_is_not_replayed_or_reported_success(self):
        native = NS(close=Mock(side_effect=RuntimeError("partial destroy")))
        e = self.engine(native)
        self.assertFalse(e.shutdown())
        self.assertFalse(e.shutdown())
        native.close.assert_called_once()
        self.assertEqual(e._peer_sessions, {"peer"})

    def test_native_false_is_failure_and_success_teardown_is_once(self):
        e = self.engine(NS(disconnect=Mock(return_value=False)))
        self.assertFalse(e.destroy_session("peer"))
        self.assertEqual(e._peer_sessions, {"peer"})
        native = NS(close=Mock(return_value=0))
        e = self.engine(native)
        self.assertTrue(e.shutdown())
        self.assertTrue(e.shutdown())
        native.close.assert_called_once()
        self.assertEqual(e._peer_sessions, set())


if __name__ == "__main__":
    unittest.main()
