"""Pool imbalance diagnostics must preserve the verdict, not crash first."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler_components import invariant_checker as checker_module
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPoolInvariantDiagnostics(CustomTestCase):
    def setUp(self):
        self.parallel = patch.object(
            checker_module,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        )
        self.parallel.start()
        self.addCleanup(self.parallel.stop)
        self.checker = SchedulerInvariantChecker(
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            disaggregation_mode=None,
            page_size=256,
            full_tokens_per_layer=1024,
            swa_tokens_per_layer=1024,
            max_total_num_tokens=1024,
            tree_cache=SimpleNamespace(
                full_protected_size=lambda: 0,
                swa_protected_size=lambda: 0,
                linker=SimpleNamespace(
                    private_loads={"held": object()}, taken_loads=[]
                ),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
            req_to_token_pool=None,
            pool_stats_observer=SimpleNamespace(
                session_held_full_tokens=lambda: 0,
                session_held_swa_tokens=lambda: 0,
            ),
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

    @staticmethod
    def stats(full=1024, swa=1024):
        return PoolStats(
            full_num_used=1024 - full,
            full_token_usage=(1024 - full) / 1024,
            full_available_size=full,
            full_evictable_size=0,
            is_hybrid_swa=True,
            swa_available_size=swa,
            swa_evictable_size=0,
        )

    def test_idle_full_and_swa_imbalance_preserves_diagnostics(self):
        """The Oct 26 idle crash hid both missing and over-counted slots."""
        with self.assertNoLogs(checker_module.logger, level="WARNING"):
            leak, _ = self.checker._check_all_pools(self.stats())
        self.assertFalse(leak)
        for pool in ("full", "swa"):
            for delta in (-256, 256):
                with self.subTest(pool=pool, delta=delta):
                    with self.assertLogs(checker_module.logger, level="ERROR") as logs:
                        leak, messages = self.checker._check_all_pools(
                            self.stats(**{pool: 1024 + delta})
                        )
                    self.assertTrue(leak)
                    self.assertIn(f"available={1024 + delta}", " ".join(messages))
                    text = " ".join(logs.output)
                    self.assertIn(f"pool={pool} total=1024", text)
                    self.assertIn(f"accounted={1024 + delta} delta={delta}", text)
                    self.assertIn("inflight_reqs=0", text)
                    self.assertIn("page_size=256 private_loads=1", text)

    def test_diagnostics_follow_replaced_inflight_queue(self):
        owner = SimpleNamespace()
        self.checker.get_inflight_req_count = lambda: len(
            getattr(owner, "disagg_prefill_inflight_queue", ())
        )
        for queue in (None, [object()], [object(), object()]):
            if queue is not None:
                owner.disagg_prefill_inflight_queue = queue
            with self.assertLogs(checker_module.logger, level="ERROR") as logs:
                leak, _ = self.checker._check_all_pools(self.stats(full=768, swa=768))
            self.assertTrue(leak)
            self.assertEqual(len(logs.output), 2)
            for line in logs.output:
                self.assertIn(f"inflight_reqs={len(queue or [])}", line)

    def test_swa_verdict_and_log_use_same_ownership_sample(self):
        # A second observation would make the diagnostic contradict the verdict.
        self.checker.tree_cache.swa_protected_size = Mock(side_effect=[64, 128])
        self.checker.pool_stats_observer.session_held_swa_tokens = Mock(
            side_effect=[64, 128]
        )
        with self.assertLogs(checker_module.logger, level="ERROR") as logs:
            leak, message = self.checker._check_swa_pool(self.stats(swa=768))
        self.assertTrue(leak)
        self.assertIn("protected=64, session_held=64", message)
        self.assertIn("accounted=896 delta=-128", logs.output[0])
        self.assertIn("protected=64 session_held=64", logs.output[0])

    def test_strict_leak_reporting_is_not_suppressed(self):
        with patch.object(
            checker_module.envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE,
            "get",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "pool memory leak detected"):
                self.checker._report_leak("pool", "[swa] delta=-256")


if __name__ == "__main__":
    unittest.main()
