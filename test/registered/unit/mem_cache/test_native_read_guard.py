"""CPU tests: native return/error, not a Python deadline, owns the DMA fence."""

import hashlib
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    ROOT / "python/sglang/srt/mem_cache/storage/mooncake_store/native_read_guard.py"
)
spec = importlib.util.spec_from_file_location("native_read_guard_test", SOURCE)
guard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_module)
NativeReadGuard = guard_module.NativeReadGuard
NativeReadUncertain = guard_module.NativeReadUncertain

ci_spec = importlib.util.spec_from_file_location(
    "native_guard_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(ci_spec)
ci_spec.loader.exec_module(ci)
ci.register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNativeReadGuard(unittest.TestCase):
    def test_success_without_contract_is_allowed(self):
        guard = NativeReadGuard(Mock())
        self.assertEqual(
            guard.run(lambda: [4], batch=1, stage="layer0", expected=[4]), [4]
        )
        self.assertEqual(guard.snapshot()["uncertain_reads"], 0)
        guard.close()

    def test_failed_returns_never_become_safe_just_because_worker_returned(self):
        for result in (None, False, [-707], [0], [True], [4, 4]):
            with self.subTest(result=result):
                guard = NativeReadGuard(Mock())
                with self.assertRaises(NativeReadUncertain):
                    guard.run(lambda: result, batch=1, stage="swa", expected=[4])
                self.assertTrue(guard.admission_blocked())
                self.assertEqual(guard.snapshot()["active_reads"], 0)
                with self.assertRaises(NativeReadUncertain):
                    guard.check()
                with self.assertRaises(NativeReadUncertain):
                    guard.close()

    def test_verified_failed_call_can_abort_then_next_batch_succeeds(self):
        guard = NativeReadGuard(Mock(), contract={"range_get_failure_drained": True})
        with self.assertRaises(RuntimeError) as failure:
            guard.run(lambda: [-707], batch=1, stage="layer0", expected=[4])
        self.assertNotIsInstance(failure.exception, NativeReadUncertain)
        self.assertFalse(guard.admission_blocked())
        self.assertEqual(
            guard.run(lambda: [4], batch=2, stage="layer0", expected=[4]), [4]
        )
        guard.close()

    def test_exception_is_not_a_verified_cancel_or_session_end(self):
        guard = NativeReadGuard(Mock())
        operation = Mock(side_effect=TimeoutError("native stopped waiting"))
        with self.assertRaises(NativeReadUncertain):
            guard.run(operation, batch=1, stage="layer0", expected=[4])
        self.assertEqual(guard.snapshot()["uncertain_reads"], 1)

    def test_stall_opens_circuit_without_publishing_completion_or_freeing(self):
        entered, release, notified = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        guard = NativeReadGuard(lambda _: notified.set(), stall_s=0.02)
        terminal, errors = threading.Event(), []

        def native():
            entered.set()
            self.assertTrue(release.wait(2))
            return [4]

        def worker():
            try:
                guard.run(native, batch=1, stage="layer0", expected=[4])
            except Exception as error:
                errors.append(error)
            finally:
                terminal.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(notified.wait(1))
            self.assertFalse(terminal.is_set())
            self.assertTrue(guard.admission_blocked())
            self.assertEqual(guard.snapshot()["active_reads"], 1)
            with self.assertRaises(NativeReadUncertain):
                guard.close()
        finally:
            release.set()
            thread.join(2)
        self.assertTrue(terminal.is_set())
        self.assertEqual(len(errors), 1)
        self.assertNotIsInstance(errors[0], NativeReadUncertain)
        self.assertIn("slow read drained", str(errors[0]))
        self.assertFalse(guard.admission_blocked())
        guard.close()

    def test_long_context_byte_budget_is_not_a_fixed_whole_request_deadline(self):
        guard = NativeReadGuard(Mock(), stall_s=5, min_bytes_per_s=1024**2)

        def native():
            self.assertEqual(guard._active["budget"], 69)
            return [64 * 1024**2]

        guard.run(native, batch=1, stage="complete_page", expected=[64 * 1024**2])
        guard.close()

    def test_native_return_checks_deadline_if_watchdog_was_not_scheduled(self):
        notified = Mock()
        guard = NativeReadGuard(notified, stall_s=0.01)
        with patch.object(guard, "_watch", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "slow read drained"):
                guard.run(
                    lambda: (threading.Event().wait(0.02), [4])[1],
                    batch=1,
                    stage="layer0",
                    expected=[4],
                )
        notified.assert_called_once()
        self.assertEqual(guard.snapshot()["uncertain_reads"], 0)
        guard.close()

    def test_read_plan_needs_its_own_failure_contract(self):
        guard = NativeReadGuard(Mock(), contract={"range_get_failure_drained": True})
        with self.assertRaises(NativeReadUncertain):
            guard.run(
                Mock(side_effect=RuntimeError("plan failed")),
                batch=1,
                stage="plan",
                read_plan=True,
            )

    def test_read_plan_long_input_uses_explicit_byte_budget(self):
        guard = NativeReadGuard(Mock(), stall_s=5, min_bytes_per_s=1024**2)

        def native():
            self.assertEqual(guard._active["budget"], 133)

        guard.run(
            native, batch=1, stage="plan", read_plan=True, byte_count=128 * 1024**2
        )
        guard.close()

    def test_progress_only_updates_the_matching_batch(self):
        guard = NativeReadGuard(Mock())

        def native():
            before = guard._active["deadline"]
            guard.progress(99)
            self.assertEqual(guard._active["deadline"], before)
            guard.progress(1)
            self.assertGreaterEqual(guard._active["deadline"], before)
            return [4]

        guard.run(native, batch=1, stage="plan", expected=[4])
        guard.close()

    def test_invalid_settings_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                NativeReadGuard(Mock(), stall_s=value)
            with self.assertRaises(ValueError):
                NativeReadGuard(Mock(), min_bytes_per_s=value)

    def test_contract_requires_review_and_exact_binary_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, contract = root / "store.so", root / "contract.json"
            binary.write_bytes(b"audited wheel")
            data = {
                "schema_version": 1,
                "verification": "vendor review and canary run",
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "range_get_failure_drained": True,
            }
            contract.write_text(json.dumps(data))
            self.assertTrue(
                guard_module.read_failure_contract(str(contract), str(binary))
            )
            binary.write_bytes(b"new wheel")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                guard_module.read_failure_contract(str(contract), str(binary))
            data["range_get_failure_drained"] = 1
            contract.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "not verified"):
                guard_module.read_failure_contract(str(contract), str(binary))
        self.assertEqual(guard_module.read_failure_contract(None, None), {})


if __name__ == "__main__":
    unittest.main()
