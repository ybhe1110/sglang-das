"""Request identity and ownership on partial external-load failures."""

import unittest
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationCollectiveError
from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store import (
    mooncake_direct_linker as linker_module,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.native_read_guard import NativeReadGuard
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalLinkerCleanupError,
    UnifiedCacheLinkerWrapper,
    _PrivateLoad,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSessionOwnership(CustomTestCase):
    def setUp(self):
        self.linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.store = Mock()
        self.store.batch_get_session_end.return_value = 0
        self.linker.storage = NS(store=self.store, failed_get_cache=Mock())
        self.linker._storage_failed = Mock()
        self.linker.completed_loads = Queue()

    def test_partial_start_retries_only_failed_keys(self):
        self.store.batch_get_session_start.side_effect = [[0, -1, 0], [0]]
        started = []
        self.linker._start_load_sessions(["a", "b", "c"], started)
        self.assertEqual(started, [["a", "c"], ["b"]])
        self.assertEqual(
            self.store.batch_get_session_start.call_args_list,
            [call(["a", "b", "c"]), call(["b"])],
        )
        self.assertTrue(self.linker._end_load_sessions(started))
        self.assertEqual(
            self.store.batch_get_session_end.call_args_list,
            [call(["a", "c"]), call(["b"])],
        )
        self.assertEqual(started, [])

    def test_failed_retry_keeps_successful_subset_for_cleanup(self):
        self.store.batch_get_session_start.side_effect = [[0, -1], [-1]]
        started = []
        with self.assertRaisesRegex(RuntimeError, "session start failed"):
            self.linker._start_load_sessions(["a", "b"], started)
        self.assertEqual(started, [["a"]])
        self.assertTrue(self.linker._end_load_sessions(started))
        self.store.batch_get_session_end.assert_called_once_with(["a"])
        self.assertEqual(self.linker.num_completed_loads(), 0)

    def test_unknown_start_outcome_requires_process_teardown(self):
        for outcome in (RuntimeError("native error"), [0], None):
            with self.subTest(outcome=outcome):
                self.setUp()
                self.store.batch_get_session_start.side_effect = [outcome]
                started = []
                with self.assertRaises(Exception):
                    self.linker._start_load_sessions(["a", "b"], started)
                self.assertEqual(self.linker._uncertain_sessions, [("a", "b")])
                with self.assertRaisesRegex(RuntimeError, "ownership uncertain"):
                    self.linker.num_completed_loads()
                with self.assertRaisesRegex(RuntimeError, "ownership uncertain"):
                    self.linker.reset()
                self.store.batch_get_session_end.assert_not_called()

    def test_retry_exception_preserves_known_and_unknown_subsets(self):
        self.store.batch_get_session_start.side_effect = [[0, -1], RuntimeError("lost")]
        started = []
        with self.assertRaises(RuntimeError):
            self.linker._start_load_sessions(["a", "b"], started)
        self.assertEqual(self.linker._uncertain_sessions, [("b",)])
        self.linker._end_load_sessions(started)
        self.store.batch_get_session_end.assert_called_once_with(["a"])

    def test_end_failure_is_not_retried_and_other_sessions_are_closed(self):
        self.store.batch_get_session_end.side_effect = [
            RuntimeError("partial end"),
            0,
        ]
        started = [["a"], ["b"]]
        self.assertFalse(self.linker._end_load_sessions(started))
        self.assertTrue(self.linker._end_load_sessions(started))
        self.assertEqual(self.store.batch_get_session_end.call_count, 2)
        self.assertEqual(self.linker._uncertain_sessions, [("a",)])
        with self.assertRaisesRegex(RuntimeError, "ownership uncertain"):
            self.linker.num_completed_loads()

    def test_worker_does_not_publish_reclaimable_completion_when_uncertain(self):
        linker = self.linker
        linker.load_queue = Queue()
        linker.load_queue.put((7, {"rid": []}, NS(synchronize=Mock())))
        linker.load_queue.put(None)
        linker.layer_done_counter = NS(finish=Mock(return_value=False), fail=Mock())
        linker._finish_prefetch_metrics = Mock()

        def load(*args):
            linker._quarantine_sessions(["a"], "session end failed")
            return False

        linker.load_layer_wise = load
        linker.load_thread_func()
        self.assertTrue(linker.completed_loads.empty())
        self.assertEqual(linker.load_queue.unfinished_tasks, 0)

    def test_worker_publishes_only_after_session_cleanup(self):
        linker = self.linker
        events = []
        linker.load_queue = Queue()
        linker.load_queue.put((7, {"rid": []}, NS(synchronize=Mock())))
        linker.load_queue.put(None)
        linker.layer_done_counter = NS(finish=Mock(return_value=False), fail=Mock())
        linker._finish_prefetch_metrics = Mock()
        self.store.batch_get_session_start.side_effect = [[0, -1], [-1]]
        self.store.batch_get_session_end.side_effect = lambda keys: (
            events.append("end") or 0
        )

        def load(*args):
            started = []
            try:
                linker._start_load_sessions(["a", "b"], started)
            except RuntimeError:
                return False
            finally:
                linker._end_load_sessions(started)

        linker.load_layer_wise = load
        linker.completed_loads.put = Mock(
            side_effect=lambda item: events.append("completion")
        )
        linker.load_thread_func()
        self.assertEqual(events, ["end", "completion"])

    def test_end_nonzero_or_unknown_status_quarantines_without_retry(self):
        for status in (-1, 1, None, [0], False):
            with self.subTest(status=status):
                self.setUp()
                self.store.batch_get_session_end.return_value = status
                started = [["a"]]
                self.assertFalse(self.linker._end_load_sessions(started))
                self.assertEqual(started, [])
                self.assertEqual(self.linker._uncertain_sessions, [("a",)])
                self.assertTrue(self.linker._end_load_sessions(started))
                self.store.batch_get_session_end.assert_called_once_with(["a"])
                with self.assertRaisesRegex(RuntimeError, "ownership uncertain"):
                    self.linker.num_completed_loads()

    def test_real_layer_loader_cleans_partial_start_on_failure(self):
        linker = self.linker
        linker.tp_rank = 0
        linker._storage_ready = Mock(return_value=True)
        linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
            keys,
            1,
        )
        linker.storage._tag_keys = lambda keys: keys
        linker.pools = {PoolName.KV: NS(prepare_locations=lambda indices: [])}
        linker.layer_done_counter = NS(fail=Mock())
        self.store.batch_get_session_start.side_effect = [[0, -1], [-1]]
        with patch.object(
            linker_module, "arm_load_failure_injection", return_value=Mock()
        ):
            result = linker.load_layer_wise(
                7, [[NS(name=PoolName.KV, keys=["a", "b"], host_indices=[])]]
            )
        self.assertFalse(result)
        self.store.batch_get_session_end.assert_called_once_with(["a"])
        linker.layer_done_counter.fail.assert_called_once()

    def _range_failure_then_recovery(self, *, fail_at=0, raises=False, page_wise=False):
        """Run worker, loader, counter and session cleanup for two batches.

        Native reads and device events are simulated; this does not establish a
        native DMA fence. The second batch succeeds without restarting the worker.
        """
        # This fake never writes after returning. Real wheels need the explicit
        # binary-pinned contract, rather than inheriting that assumption.
        self.linker._native_read_guard = NativeReadGuard(
            self.linker._storage_failed,
            contract={"range_get_failure_drained": True},
        )
        self.addCleanup(self.linker._native_read_guard.close)
        linker = self.linker
        linker.tp_rank = 0
        linker.num_layers = 3
        linker.enable_page_wise_load = page_wise
        linker.page_wise_load_threshold = 1
        linker._storage_ready = Mock(return_value=True)
        linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
            keys,
            1,
        )
        linker.storage._tag_keys = lambda keys: keys
        linker.pools = {
            PoolName.KV: NS(
                prepare_locations=lambda indices: [1],
                get_prepared_layer_range_meta=lambda locations, layer: (
                    [[1024 + layer * 4]],
                    [[4]],
                    [[layer * 4]],
                ),
            )
        }
        linker.layer_done_counter = LayerWiseLoadCounter(3)
        linker._finish_prefetch_metrics = Mock()
        linker.load_queue = Queue()
        events = []
        self.store.batch_get_session_start.side_effect = lambda keys: (
            events.append(("start", tuple(keys))) or [0] * len(keys)
        )
        self.store.batch_get_session_end.side_effect = lambda keys: (
            events.append(("end", tuple(keys))) or 0
        )
        attempts = 0

        def read(keys, ptrs, sizes, offsets):
            nonlocal attempts
            current = attempts
            attempts += 1
            events.append(("read", tuple(keys)))
            if current == fail_at:
                if raises:
                    raise RuntimeError("injected synchronous read failure")
                return [0]  # Short read, not a successful zero-byte transfer.
            return [sum(parts) for parts in sizes]

        self.store.batch_get_into_multi_buffer_ranges.side_effect = read
        completions = linker.completed_loads
        publish = completions.put

        def record_completion(item):
            events.append(("completion", item))
            publish(item)

        completions.put = record_completion
        for rid in ("failed", "recovered"):
            index = linker.layer_done_counter.update_producer()
            transfers = [NS(name=PoolName.KV, keys=[rid], host_indices=[1])]
            linker.load_queue.put((index, {rid: transfers}, NS(synchronize=Mock())))
        linker.load_queue.put(None)
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(0):
            linker.load_thread_func()

        self.assertEqual(completions.get_nowait(), (["failed"], False))
        self.assertEqual(completions.get_nowait(), (["recovered"], True))
        self.assertTrue(completions.empty())
        self.assertEqual(linker.load_queue.unfinished_tasks, 0)
        self.assertEqual(linker.layer_done_counter.futures, {})
        self.assertIsNone(getattr(linker, "_session_cleanup_error", None))
        self.assertEqual(
            self.store.batch_get_session_end.call_args_list,
            [call(["failed"]), call(["recovered"])],
        )
        for rid, success in (("failed", False), ("recovered", True)):
            self.assertLess(
                events.index(("end", (rid,))),
                events.index(("completion", ([rid], success))),
            )
        self.assertEqual(attempts, (2 if page_wise else fail_at + 1 + 3))
        self.assertEqual(
            linker._finish_prefetch_metrics.call_args_list,
            [call(["failed"], False), call(["recovered"], True)],
        )

    def test_first_layer_short_read_aborts_batch_then_worker_recovers(self):
        self._range_failure_then_recovery()

    def test_first_layer_exception_aborts_batch_then_worker_recovers(self):
        self._range_failure_then_recovery(raises=True)

    def test_later_layer_failure_preserves_failed_terminal_then_recovers(self):
        self._range_failure_then_recovery(fail_at=1)

    def test_page_wise_short_read_cleans_sessions_then_recovers(self):
        self._range_failure_then_recovery(page_wise=True)

    def test_reset_does_not_clear_ownership_if_worker_fails_during_join(self):
        linker = self.linker
        linker.pending_loads = {}
        linker.load_queue = NS(join=lambda: linker._quarantine_sessions(["a"], "end"))
        linker.offload_queue = NS(join=Mock())
        with self.assertRaisesRegex(RuntimeError, "ownership uncertain"):
            linker.reset()


class TestReleaseQuarantine(CustomTestCase):
    def test_partial_component_free_is_retained_and_never_retried(self):
        wrapper = UnifiedCacheLinkerWrapper(
            NS(tree_core=NS()), NS(reset=Mock(), close=Mock())
        )
        req = NS(rid="partial-free")
        wrapper._remember_private_slot_range(req, 4, 8)
        components = [NS(update_external_linker_load=Mock()) for _ in range(3)]
        components[1].update_external_linker_load.side_effect = RuntimeError(
            "free failed"
        )
        transfers = [
            (component, NS(name=name, device_indices=torch.arange(4)))
            for component, name in zip(
                components, [PoolName.KV, PoolName.SWA, PoolName.MAMBA]
            )
        ]
        private = _PrivateLoad(req, transfers, 4, 8, transfers[0][1].device_indices)
        wrapper.private_loads[req.rid] = private
        wrapper.taken_loads = [([req.rid], False)]
        with self.assertRaises(ExternalLinkerCleanupError):
            wrapper.commit_completed_loads([False])
        record = wrapper.cleanup_quarantine[req.rid]
        self.assertIs(record["transfers"], transfers)
        self.assertEqual(record["states"], ("not_attempted", "uncertain", "released"))
        self.assertTrue(wrapper.has_pending_operations())
        self.assertTrue(req.external_kv_private_slots_released)
        for action in (
            lambda: wrapper.release_request(req.rid),
            lambda: wrapper.load_back(req),
            lambda: wrapper.match(None, req, None),
            wrapper.reset,
            wrapper.close,
            lambda: wrapper.commit_completed_loads([]),
        ):
            with self.assertRaises(ExternalLinkerCleanupError):
                action()
        self.assertEqual(
            [c.update_external_linker_load.call_count for c in components], [0, 1, 1]
        )

    def test_prequeue_abort_failure_is_also_quarantined(self):
        wrapper = UnifiedCacheLinkerWrapper(NS(tree_core=NS()), NS())
        req = NS(rid="prequeue")
        component = NS(
            update_external_linker_load=Mock(side_effect=RuntimeError("free"))
        )
        transfers = [(component, NS(name=PoolName.KV, device_indices=torch.arange(4)))]
        with self.assertRaises(ExternalLinkerCleanupError):
            wrapper._abort_component_transfers(req, transfers, 4, 8)
        self.assertEqual(wrapper.cleanup_quarantine[req.rid]["states"], ("uncertain",))


class TestMatchIdentity(CustomTestCase):
    @staticmethod
    def wrapper(reduce):
        return UnifiedCacheLinkerWrapper(
            NS(tree_core=NS(), _all_reduce_attn_groups=reduce), NS()
        )

    @staticmethod
    def preflight(wrapper, rid="a", **kwargs):
        return wrapper._match_preflight(
            rid, key_len=8, device_hit_len=8, tail_pages=0, local_valid=False, **kwargs
        )

    def test_same_geometry_different_request_fails_on_both_ranks(self):
        payloads = []
        for rid in ("a", "b"):
            self.preflight(
                self.wrapper(lambda tensor, op: payloads.append(tensor.clone())), rid
            )
        reduced = torch.minimum(*payloads)
        for rid in ("a", "b"):
            wrapper = self.wrapper(lambda tensor, op: tensor.copy_(reduced))
            with self.assertRaisesRegex(
                DisaggregationCollectiveError, "RID/sequence/phase"
            ):
                self.preflight(wrapper, rid)

    def test_sequence_phase_and_pp_mode_mismatch_are_fatal(self):
        for index in (12, 14, 16):
            with self.subTest(index=index):

                def mismatch(tensor, op):
                    tensor[index + 1] = -int(tensor[index]) - 1

                with self.assertRaises(DisaggregationCollectiveError):
                    self.preflight(self.wrapper(mismatch))

    def test_full_hit_still_negotiates_identity_and_advances_sequence(self):
        payloads = []
        wrapper = self.wrapper(lambda tensor, op: payloads.append(tensor.clone()))
        self.assertEqual(self.preflight(wrapper), (False, 0))
        self.assertEqual(self.preflight(wrapper), (False, 0))
        self.assertEqual([p.numel() for p in payloads], [18, 18])
        self.assertEqual([int(p[12]) for p in payloads], [0, 1])


if __name__ == "__main__":
    unittest.main()
