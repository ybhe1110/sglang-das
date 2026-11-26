"""Critical-exit policy and shutdown races, without importing GPU modules."""

import ast
import importlib.util
import logging
import os
import pathlib
import signal
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[4]
path = ROOT / "python/sglang/srt/utils/watchdog.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if getattr(n, "name", None) == "SubprocessWatchdog")
env = dict(
    logging=logging,
    logger=logging.getLogger(__name__),
    os=os,
    signal=signal,
    threading=threading,
)
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    cls,
                ],
                type_ignores=[],
            )
        ),
        str(path),
        "exec",
    ),
    env,
)
SubprocessWatchdog = env["SubprocessWatchdog"]
spec = importlib.util.spec_from_file_location(
    "failstop_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {spec.name: ci}):
    spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCriticalSubprocessExit(unittest.TestCase):
    def test_every_unexpected_critical_exit_is_fatal_once(self):
        for exitcode in (0, 1, 42, -9):
            with self.subTest(exitcode=exitcode):
                callback = Mock()
                process = SimpleNamespace(
                    pid=123, exitcode=exitcode, is_alive=lambda: False
                )
                monitor = SubprocessWatchdog(
                    [process],
                    ["detokenizer"],
                    unexpected_exit_is_fatal=True,
                    on_failure=callback,
                )
                self.assertTrue(monitor._check_processes())
                self.assertFalse(monitor._check_processes())
                callback.assert_called_once_with("detokenizer", 123, exitcode)

    def test_legacy_clean_exit_policy_is_preserved(self):
        callback = Mock()
        proc = SimpleNamespace(pid=123, exitcode=0, is_alive=lambda: False)
        monitor = SubprocessWatchdog([proc], on_failure=callback)
        self.assertFalse(monitor._check_processes())
        callback.assert_not_called()

    def test_stop_precedes_expected_child_exit(self):
        callback = Mock()
        proc = SimpleNamespace(pid=123, exitcode=0, is_alive=lambda: False)
        monitor = SubprocessWatchdog(
            [proc],
            unexpected_exit_is_fatal=True,
            on_failure=callback,
        )
        monitor.stop()
        self.assertFalse(monitor._check_processes())
        callback.assert_not_called()

    def test_callback_can_stop_its_monitor(self):
        done = threading.Event()
        proc = SimpleNamespace(pid=123, exitcode=0, is_alive=lambda: False)
        monitor = SubprocessWatchdog(
            [proc], interval=0.01, unexpected_exit_is_fatal=True
        )

        def stop(*_):
            monitor.stop()
            done.set()

        monitor._on_failure = stop
        monitor.start()
        self.assertTrue(done.wait(2), "Monitor tried to join its own thread")

    def test_live_and_unstarted_processes_are_not_failures(self):
        callback = Mock()
        processes = [
            SimpleNamespace(pid=123, exitcode=None, is_alive=lambda: True),
            SimpleNamespace(pid=None, exitcode=None, is_alive=lambda: False),
        ]
        monitor = SubprocessWatchdog(
            processes,
            unexpected_exit_is_fatal=True,
            on_failure=callback,
        )
        self.assertFalse(monitor._check_processes())
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
