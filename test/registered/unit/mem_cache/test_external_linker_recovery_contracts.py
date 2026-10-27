"""v4.1 equivalents of release a6a01c1900 / 1ca38d9a3e session guards."""

import unittest
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, call

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalLinkerCleanupError,
    UnifiedCacheLinkerWrapper,
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
        self.linker._availability = NS(failed=Mock())
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


class TestReleaseIsolation(CustomTestCase):
    def test_partial_release_is_quarantined_and_never_retried(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cleanup_quarantine = {}
        req = NS(rid="partial-release")
        calls = []

        def partial_free(*args):
            calls.append("full-partially-freed")
            raise RuntimeError("free failed after a side effect")

        full = NS(name=PoolName.KV, device_indices=NS(numel=lambda: 8))
        swa = NS(name=PoolName.SWA, device_indices=NS(numel=lambda: 4))
        components = [
            NS(update_external_linker_load=Mock(side_effect=partial_free)),
            NS(
                update_external_linker_load=Mock(
                    side_effect=lambda *args: calls.append("swa")
                )
            ),
        ]
        transfers = list(zip(components, (full, swa)))
        with self.assertRaises(ExternalLinkerCleanupError):
            wrapper._abort_transfers(req, transfers, 12)
        self.assertEqual(calls, ["swa", "full-partially-freed"])
        self.assertEqual(
            wrapper.cleanup_quarantine[req.rid]["states"], ["released", "uncertain"]
        )
        self.assertIs(wrapper.cleanup_quarantine[req.rid]["req"], req)
        for action in (
            lambda: wrapper._abort_transfers(req, transfers, 12),
            lambda: wrapper.release_request(req.rid),
            wrapper.reset,
            wrapper.close,
        ):
            with self.assertRaises(ExternalLinkerCleanupError):
                action()
        for component in components:
            component.update_external_linker_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
