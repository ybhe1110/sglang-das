"""CPU regressions for #36988 adapted to the local source-drain owner.

Upstream can retire immediately; private FULL/SWA loads and outstanding
source reads require parking the result for synchronized cleanup instead.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from test_external_kv_abort_drain import PREFILL, load_methods, register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAbortedPrefillResult(unittest.TestCase):
    def setUp(self):
        cls = load_methods(PREFILL, ["_retire_aborted_prefill_result"], {})
        self.s = cls()
        self.s.clear_pending_chunk_send = Mock()
        self.s.abort_external_kv_request = Mock(return_value=False)
        self.s.disagg_prefill_inflight_queue = []
        self.s.chunked_req = None
        self.req = SimpleNamespace(
            rid="aborted", external_kv_cleanup_done=False, pending_bootstrap=True
        )

    def test_result_does_not_assert_sender_or_forward_drain(self):
        self.assertFalse(self.s._retire_aborted_prefill_result(self.req))
        self.assertEqual(self.s.disagg_prefill_inflight_queue, [self.req])
        self.assertFalse(self.req.pending_bootstrap)
        self.s.abort_external_kv_request.assert_called_once_with(
            self.req,
            forward_drained=False,
            sender_terminal=False,
            is_insert=False,
            emit_response=False,
        )

    def test_local_readiness_cannot_change_queue_admission(self):
        for ready in (False, True):
            with self.subTest(ready=ready):
                self.s.disagg_prefill_inflight_queue = []
                self.s.abort_external_kv_request.return_value = ready
                self.s._retire_aborted_prefill_result(self.req)
                self.assertEqual(self.s.disagg_prefill_inflight_queue, [self.req])

    def test_repeated_result_does_not_duplicate_owner(self):
        self.s._retire_aborted_prefill_result(self.req)
        self.s._retire_aborted_prefill_result(self.req)
        self.assertEqual(self.s.disagg_prefill_inflight_queue, [self.req])

    def test_live_chunk_keeps_deferred_cleanup_owner(self):
        self.s.chunked_req = self.req
        self.s._retire_aborted_prefill_result(self.req)
        self.assertIs(self.s.chunked_req, self.req)
        self.assertIs(self.s._pending_chunked_abort_req, self.req)
        self.assertIs(self.s._external_kv_chunked_abort_reqs[self.req.rid], self.req)
        self.assertTrue(self.req.external_kv_abort_response_via_chunked)

    def test_retired_result_cannot_reopen_cleanup(self):
        self.req.external_kv_cleanup_done = True
        self.s._retire_aborted_prefill_result(self.req)
        self.assertEqual(self.s.disagg_prefill_inflight_queue, [])
        self.s.abort_external_kv_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
