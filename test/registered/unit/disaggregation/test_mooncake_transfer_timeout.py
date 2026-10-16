"""CPU-only tests for the native Mooncake transfer deadline setup."""

import ast
import os
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = pathlib.Path(__file__).resolve().parents[4]
ENGINE = ROOT / "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
DOCKERFILE = ROOT / "docker/Dockerfile"


def load_configurator():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "configure_mooncake_transfer_timeout"
    )
    node.decorator_list = []
    module = ast.Module(
        body=[node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = {
        "os": os,
        "envs": SimpleNamespace(
            SGLANG_DISAGGREGATION_MOONCAKE_TRANSFER_TIMEOUT=SimpleNamespace(
                get=lambda: 30.0
            )
        ),
        "logger": Mock(),
    }
    exec(compile(module, str(ENGINE), "exec"), env)
    return env["configure_mooncake_transfer_timeout"], env


def load_runtime_helpers():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    nodes = [
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        for name in (
            "get_mooncake_package_version",
            "log_mooncake_runtime_configuration",
        )
    ]
    for node in nodes:
        node.decorator_list = []

    class Metadata:
        class PackageNotFoundError(Exception):
            pass

        @staticmethod
        def version(name):
            if name == "mooncake-transfer-engine-cuda13":
                return "0.3.13.post1"
            raise Metadata.PackageNotFoundError(name)

    logger = Mock()
    env = {
        "os": os,
        "envs": SimpleNamespace(
            SGLANG_DISAGGREGATION_MOONCAKE_TRANSFER_TIMEOUT=SimpleNamespace(
                get=lambda: 30.0
            )
        ),
        "importlib_metadata": Metadata,
        "logger": logger,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ENGINE), "exec"), env)
    return env["get_mooncake_package_version"], env["log_mooncake_runtime_configuration"], logger


class TestMooncakeTransferTimeout(unittest.TestCase):
    def test_docker_default_matches_deployed_mooncake_version(self):
        self.assertIn(
            "ARG MOONCAKE_VERSION=0.3.13.post1",
            DOCKERFILE.read_text(encoding="utf-8"),
        )

    def test_sets_native_timeout_when_operator_left_it_unset(self):
        configure, _ = load_configurator()
        previous = os.environ.pop("MC_TRANSFER_TIMEOUT", None)
        try:
            configure()
            self.assertEqual(os.environ["MC_TRANSFER_TIMEOUT"], "30.0")
        finally:
            os.environ.pop("MC_TRANSFER_TIMEOUT", None)
            if previous is not None:
                os.environ["MC_TRANSFER_TIMEOUT"] = previous

    def test_preserves_operator_timeout(self):
        configure, _ = load_configurator()
        previous = os.environ.get("MC_TRANSFER_TIMEOUT")
        os.environ["MC_TRANSFER_TIMEOUT"] = "7"
        try:
            configure()
            self.assertEqual(os.environ["MC_TRANSFER_TIMEOUT"], "7")
        finally:
            if previous is None:
                os.environ.pop("MC_TRANSFER_TIMEOUT", None)
            else:
                os.environ["MC_TRANSFER_TIMEOUT"] = previous

    def test_reports_installed_version_and_effective_native_configuration(self):
        get_version, log_configuration, logger = load_runtime_helpers()
        self.assertEqual(
            get_version(), "mooncake-transfer-engine-cuda13==0.3.13.post1"
        )
        previous_timeout = os.environ.get("MC_TRANSFER_TIMEOUT")
        previous_tcp = os.environ.get("MC_FORCE_TCP")
        os.environ["MC_TRANSFER_TIMEOUT"] = "30"
        os.environ["MC_FORCE_TCP"] = "0"
        try:
            log_configuration(SimpleNamespace(batch_transfer_sync_write=Mock()))
        finally:
            if previous_timeout is None:
                os.environ.pop("MC_TRANSFER_TIMEOUT", None)
            else:
                os.environ["MC_TRANSFER_TIMEOUT"] = previous_timeout
            if previous_tcp is None:
                os.environ.pop("MC_FORCE_TCP", None)
            else:
                os.environ["MC_FORCE_TCP"] = previous_tcp

        message, package, configured, native, api, force_tcp = logger.info.call_args.args
        self.assertIn("Mooncake runtime configuration", message)
        self.assertEqual(package, "mooncake-transfer-engine-cuda13==0.3.13.post1")
        self.assertEqual(configured, 30.0)
        self.assertEqual(native, "30")
        self.assertTrue(api)
        self.assertEqual(force_tcp, "0")


if __name__ == "__main__":
    unittest.main()
