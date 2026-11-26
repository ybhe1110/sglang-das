"""Real two-worker Uvicorn retirement after losing a critical engine child."""

import importlib.util
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import psutil

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "http_failstop_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {spec.name: ci}):
    spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def wait_until(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TestHttpSupervisorFailstop(unittest.TestCase):
    def test_critical_exit_retires_parent_and_both_http_workers(self):
        for mode in ("clean", "exception", "killed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                state = pathlib.Path(directory)
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                fixture = pathlib.Path(__file__).with_name(
                    "subprocess_failstop_fixture.py"
                )
                with (state / "supervisor.log").open("w+") as logfile:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(fixture),
                            directory,
                            str(port),
                            "0" if mode == "clean" else "42",
                        ],
                        stdout=logfile,
                        stderr=subprocess.STDOUT,
                    )
                    child_pids = []
                    try:
                        self.assertTrue(
                            wait_until(
                                lambda: (
                                    len(list(state.glob("http-*.pid"))) == 2
                                    or process.poll() is not None
                                )
                            ),
                            "HTTP workers did not start",
                        )
                        logfile.flush()
                        logfile.seek(0)
                        self.assertIsNone(process.poll(), logfile.read())
                        child_pids = [int(p.read_text()) for p in state.glob("*.pid")]
                        self.assertEqual(len(child_pids), 3)
                        opener = urllib.request.build_opener(
                            urllib.request.ProxyHandler({})
                        )
                        with opener.open(
                            f"http://127.0.0.1:{port}/", timeout=3
                        ) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(response.read(), b"ok")
                        if mode == "killed":
                            psutil.Process(
                                int((state / "critical.pid").read_text())
                            ).kill()
                        else:
                            (state / "trigger").touch()
                        self.assertNotEqual(process.wait(timeout=10), 0)
                        self.assertTrue(
                            wait_until(
                                lambda: all(
                                    not psutil.pid_exists(pid) for pid in child_pids
                                ),
                                5,
                            )
                        )
                        with self.assertRaises((urllib.error.URLError, OSError)):
                            opener.open(f"http://127.0.0.1:{port}/", timeout=1)
                    finally:
                        if process.poll() is None:
                            owner = psutil.Process(process.pid)
                            for child in owner.children(recursive=True):
                                child.kill()
                            process.kill()
                            process.wait(timeout=5)
                        for pid in child_pids:
                            try:
                                psutil.Process(pid).kill()
                            except psutil.NoSuchProcess:
                                pass


if __name__ == "__main__":
    unittest.main()
