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
CONN = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"


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


def load_source_watchdog():
    tree = ast.parse(CONN.read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_mooncake_source_transfer_watchdog_timeout_s"
    )
    node.decorator_list = []
    logger = Mock()
    env = {
        "os": os,
        "envs": SimpleNamespace(
            SGLANG_DISAGGREGATION_MOONCAKE_TRANSFER_TIMEOUT=SimpleNamespace(
                get=lambda: 30.0
            )
        ),
        "logger": logger,
    }
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(CONN), "exec"), env)
    return env["_mooncake_source_transfer_watchdog_timeout_s"]


class TestMooncakeTransferTimeout(unittest.TestCase):
    def test_engine_and_manager_expose_ordered_shutdown(self):
        engine_tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
        engine_shutdown = next(
            n
            for n in ast.walk(engine_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "shutdown"
        )
        native_methods = {
            n.func.attr
            for n in ast.walk(engine_shutdown)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        native_method_names = {
            n.value
            for n in ast.walk(engine_shutdown)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertTrue(
            native_methods.intersection({"shutdown", "close", "destroy"})
            or {"shutdown", "close", "destroy"}.issubset(native_method_names)
        )
        self.assertTrue(
            any(
                isinstance(n, ast.FunctionDef)
                and n.name == "shutdown_mooncake_transfer_engine"
                for n in ast.walk(engine_tree)
            )
        )

        conn_tree = ast.parse(CONN.read_text(encoding="utf-8"))
        manager_class = next(
            n
            for n in ast.walk(conn_tree)
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
        )
        manager_shutdown = next(
            n
            for n in manager_class.body
            if isinstance(n, ast.FunctionDef) and n.name == "shutdown"
        )
        shutdown_attrs = {
            n.func.attr
            for n in ast.walk(manager_shutdown)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertIn("shutdown", shutdown_attrs)
        self.assertIn("deregister_buffer_to_engine", shutdown_attrs)

        self.assertTrue(
            any(
                isinstance(n, ast.FunctionDef)
                and n.name == "force_destroy_sessions"
                for n in ast.walk(engine_tree)
            )
        )
        destroy_session = next(
            n
            for n in ast.walk(engine_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "destroy_session"
        )
        destroy_names = {
            n.value
            for n in ast.walk(destroy_session)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertTrue(
            destroy_names.intersection(
                {"destroy_session", "close_session", "remove_session"}
            )
        )

    def test_sync_batch_paths_use_source_watchdog(self):
        tree = ast.parse(CONN.read_text(encoding="utf-8"))
        helper = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_run_mooncake_batch_transfer"
        )
        attrs = {
            n.func.attr
            for n in ast.walk(helper)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertIn("_run_source_with_watchdog", attrs)
        source_helper = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_run_source_with_watchdog"
        )
        source_attrs = {
            n.func.attr
            for n in ast.walk(source_helper)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertIn("run_with_deadline", source_attrs)

        # Both direct synchronous batch-transfer call sites must go through
        # the helper; otherwise one layout path could still block forever.
        direct_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "run":
                continue
            if any(
                isinstance(child, ast.Attribute)
                and child.attr == "batch_transfer_sync"
                for child in ast.walk(node)
            ):
                direct_calls.append(node)
        self.assertEqual(direct_calls, [])

        # Staging gathers and CUDA event synchronization are source accesses
        # too; no direct unbounded source_drain.run call may remain in conn.py.
        unbounded_source_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_source_drain"
        ]
        self.assertEqual(unbounded_source_calls, [])

    def test_timeout_is_configured_before_native_engine_construction(self):
        tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
        # AST nodes do not retain parent pointers; identify the constructor by
        # the native TransferEngine call instead.
        constructors = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ]
        init = next(
            n
            for n in constructors
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "TransferEngine"
                for call in ast.walk(n)
            )
        )
        configure_lines = [
            n.lineno
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "configure_mooncake_transfer_timeout"
        ]
        native_lines = [
            n.lineno
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "TransferEngine"
        ]
        self.assertEqual(len(configure_lines), 1)
        self.assertEqual(len(native_lines), 1)
        self.assertLess(configure_lines[0], native_lines[0])

    def test_source_watchdog_has_grace_and_never_disables_liveness(self):
        watchdog = load_source_watchdog()
        previous = os.environ.get("MC_TRANSFER_TIMEOUT")
        try:
            os.environ["MC_TRANSFER_TIMEOUT"] = "7"
            self.assertEqual(watchdog(), 12.0)
            os.environ["MC_TRANSFER_TIMEOUT"] = "0"
            self.assertEqual(watchdog(), 5.0)
        finally:
            if previous is None:
                os.environ.pop("MC_TRANSFER_TIMEOUT", None)
            else:
                os.environ["MC_TRANSFER_TIMEOUT"] = previous

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
