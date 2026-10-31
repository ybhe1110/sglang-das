"""Run production idle/accounting methods without importing Torch or a GPU runtime."""

import ast
import importlib.util
import logging
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_module("audit_ci", ROOT / "python/sglang/test/ci/ci_register.py").register_cpu_ci(
    est_time=2, suite="base-a-test-cpu"
)


def methods(path, class_name, names, **namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    cls.decorator_list, cls.bases = [], []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in cls.body} == set(names)
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
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


LOGGER = logging.getLogger("drained-audit-test")
Checker = methods(
    SRT / "managers/scheduler_components/invariant_checker.py",
    "SchedulerInvariantChecker",
    (
        "_check_pool_invariant",
        "_check_full_pool",
        "_check_swa_pool",
        "_check_all_pools",
        "_pool_owner_diagnostics",
        "audit_idle_pools",
    ),
    logger=LOGGER,
    logging=logging,
    os=os,
    time=time,
    get_parallel=lambda: SimpleNamespace(dcp_enabled=False),
)
Wrapper = methods(
    SRT / "mem_cache/unified_cache/unified_cache_linker.py",
    "UnifiedCacheLinkerWrapper",
    ("has_pending_operations",),
)
Backend = methods(
    SRT / "mem_cache/storage/mooncake_store/mooncake_direct_linker.py",
    "MooncakeDirectLinker",
    ("has_unresolved_transfers",),
)
Scheduler = methods(
    SRT / "managers/scheduler.py",
    "Scheduler",
    ("is_fully_idle",),
    DisaggregationMode=SimpleNamespace(PREFILL="p", DECODE="d"),
)
Verifier = load_module(
    "audit_verifier", ROOT / "test/manual/disaggregation/verify_drained_kv_audit.py"
)


class TestDrainedAudit(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.state = dict(active_reads=0, uncertain_reads=0)
        self.backend._native_read_guard = SimpleNamespace(snapshot=lambda: self.state)
        self.backend.load_queue = SimpleNamespace(unfinished_tasks=0)
        self.backend.offload_queue = SimpleNamespace(unfinished_tasks=0)
        self.wrapper = Wrapper()
        self.wrapper.cache_linker = self.backend
        for name in (
            "pending_loads",
            "failed_chains",
            "private_loads",
            "pending_offloads",
            "taken_loads",
            "cleanup_quarantine",
        ):
            setattr(self.wrapper, name, {})
        self.checker = Checker()
        self.checker.is_hybrid_swa = True
        self.checker.is_hybrid_ssm = False
        self.checker.full_tokens_per_layer = 1024
        self.checker.swa_tokens_per_layer = 512
        self.checker.tree_cache = SimpleNamespace(
            linker=self.wrapper,
            full_protected_size=lambda: 0,
            swa_protected_size=lambda: 0,
            retry_stranded_failed_linker_chains=Mock(),
        )
        self.checker.pool_stats_observer = SimpleNamespace(
            session_held_full_tokens=lambda: 0,
            session_held_swa_tokens=lambda: 0,
        )
        self.checker.token_to_kv_pool_allocator = SimpleNamespace(page_size=256)
        self.checker.page_size = 256
        self.checker.get_inflight_req_count = lambda: 0
        self.checker._last_idle_audit = float("-inf")
        self.ps = SimpleNamespace(
            full_available_size=768,
            full_evictable_size=256,
            swa_available_size=256,
            swa_evictable_size=256,
        )

    def test_drained_success_logs_both_actual_balances_and_rate_limits(self):
        with self.assertLogs(LOGGER, level="INFO") as captured:
            leak, messages = self.checker.audit_idle_pools(self.ps)
        self.assertFalse(leak)
        self.assertEqual(len(messages), 2)
        self.assertIn("status=pass drained=true", captured.output[0])
        self.assertIn("accounted=1024, delta=0", captured.output[0])
        self.assertIn("accounted=512, delta=0", captured.output[0])
        with self.assertNoLogs(LOGGER, level="INFO"):
            self.assertFalse(self.checker.audit_idle_pools(self.ps)[0])

    def test_missing_and_over_counted_full_swa_still_fail_after_success(self):
        self.checker.audit_idle_pools(self.ps)
        for pool in ("full", "swa"):
            for difference in (-256, 256):
                with self.subTest(pool=pool, delta=difference):
                    key = f"{pool}_available_size"
                    value = getattr(self.ps, key)
                    setattr(self.ps, key, value + difference)
                    with self.assertLogs(LOGGER, level="ERROR") as captured:
                        leak, messages = self.checker.audit_idle_pools(self.ps)
                    self.assertTrue(leak)
                    self.assertIn("status=fail drained=true", captured.output[-1])
                    self.assertIn(f"delta={difference}", " ".join(messages))
                    setattr(self.ps, key, value)

    def test_native_owner_is_not_hidden_by_empty_wrapper_queues(self):
        for key in ("active_reads", "uncertain_reads"):
            with self.subTest(state=key):
                self.state[key] = 1
                self.assertTrue(self.wrapper.has_pending_operations())
                with self.assertRaisesRegex(RuntimeError, "before linker drain"):
                    self.checker.audit_idle_pools(self.ps)
                self.state[key] = 0
        for queue in (self.backend.load_queue, self.backend.offload_queue):
            queue.unfinished_tasks = 1
            self.assertTrue(self.wrapper.has_pending_operations())
            queue.unfinished_tasks = 0
        self.assertFalse(self.wrapper.has_pending_operations())

    def scheduler(self):
        scheduler = Scheduler()
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler.chunked_req = scheduler.last_batch = None
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = lambda: True
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = "p"
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler._external_kv_cleanup_quarantine = {}
        scheduler._external_kv_cleanup_rids = set()
        scheduler.enable_unified_cache_external_linker = True
        scheduler.enable_hisparse = scheduler.enable_hierarchical_cache = False
        scheduler.tree_cache = self.checker.tree_cache
        return scheduler

    def test_prefill_idle_requires_native_and_cleanup_owners_to_drain(self):
        scheduler = self.scheduler()
        self.assertTrue(scheduler.is_fully_idle())
        scheduler._external_kv_cleanup_quarantine["old"] = object()
        self.assertFalse(scheduler.is_fully_idle())
        scheduler._external_kv_cleanup_quarantine.clear()
        scheduler._external_kv_cleanup_rids.add("old")
        self.assertFalse(scheduler.is_fully_idle())
        scheduler._external_kv_cleanup_rids.clear()
        self.state["active_reads"] = 1
        self.assertFalse(scheduler.is_fully_idle())
        self.state["active_reads"] = 0
        self.assertTrue(scheduler.is_fully_idle())


class TestAuditAcceptance(unittest.TestCase):
    def line(self, pid=1, when="2026-10-31T06:01:00Z", status="pass", delta=0):
        parts = []
        for name in ("full", "swa"):
            parts.append(
                f"[{name}] total=1024, available={768 + delta}, evictable=256, "
                f"protected=0, session_held=0, uncached=0, accounted={1024 + delta}, delta={delta}"
            )
        return (
            f"Idle KV pool audit: status={status} drained=true pid={pid} audit_time={when} "
            + " | ".join(parts)
        )

    def verify(self, lines, count=2):
        return Verifier.verify(
            lines,
            since=Verifier.timestamp("2026-10-31T06:00:00Z"),
            expected_processes=count,
        )

    def test_each_process_must_have_fresh_positive_proof(self):
        for lines in (
            [],
            ["no errors"],
            [self.line()],
            [self.line(1), self.line(1)],
            [self.line(1), self.line(2, when="2026-10-31T05:00:00Z")],
        ):
            with self.subTest(lines=lines):
                self.assertFalse(self.verify(lines)["passed"])
        self.assertTrue(self.verify([self.line(1), self.line(2)])["passed"])

    def test_imbalance_failure_missing_swa_or_mixed_generations_fail(self):
        for bad in (
            self.line(delta=-256),
            self.line(delta=256),
            self.line(status="fail"),
            self.line().split(" | ")[0],
            self.line().replace("accounted=1024", "accounted=999"),
        ):
            with self.subTest(bad=bad):
                self.assertFalse(self.verify([bad, self.line(2)])["passed"])
        self.assertFalse(
            self.verify([self.line(1), self.line(2), self.line(3)])["passed"]
        )

    def test_later_success_cannot_hide_failed_audit_in_acceptance_window(self):
        result = self.verify(
            [
                self.line(status="fail"),
                self.line(when="2026-10-31T06:02:00Z"),
                self.line(2),
            ]
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
