"""CPU checks for backend-wide disaggregation shutdown invariants."""

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[4]
COMMON = ROOT / "python/sglang/srt/disaggregation/common/conn.py"
MOONCAKE = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"


def _class(tree, name):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(node, name):
    return next(
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name == name
    )


class TestBackendLifecycle(unittest.TestCase):
    def test_common_manager_owns_interruptible_heartbeat_shutdown(self):
        tree = ast.parse(COMMON.read_text(encoding="utf-8"))
        manager = _class(tree, "CommonKVManager")
        init = _method(manager, "__init__")
        self.assertTrue(
            any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "_shutdown_event"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "Event"
                for node in ast.walk(init)
            )
        )

        loop = _method(manager, "_run_heartbeat_checker_loop")
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "wait"
                for node in ast.walk(loop)
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                for node in ast.walk(loop)
            )
        )

        shutdown = _method(manager, "shutdown")
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_stop_heartbeat_checker"
                for node in ast.walk(shutdown)
            )
        )

    def test_mooncake_shutdown_stops_common_heartbeat_first(self):
        tree = ast.parse(MOONCAKE.read_text(encoding="utf-8"))
        manager = _class(tree, "MooncakeKVManager")
        shutdown = _method(manager, "shutdown")
        calls = [
            node
            for node in ast.walk(shutdown)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_stop_heartbeat_checker"
        ]
        self.assertTrue(calls)

    def test_ack_wakeup_sockets_close_before_context_term(self):
        tree = ast.parse(MOONCAKE.read_text(encoding="utf-8"))
        manager = _class(tree, "MooncakeKVManager")
        close = _method(manager, "_close_zmq_resources")
        close_line = min(
            node.lineno
            for node in ast.walk(close)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "close"
        )
        term_line = next(
            node.lineno
            for node in ast.walk(close)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "term"
        )
        source = ast.get_source_segment(
            MOONCAKE.read_text(encoding="utf-8"), close
        )
        self.assertIn("ack_wakeup_receiver", source)
        self.assertIn("ack_wakeup_sender", source)
        self.assertLess(close_line, term_line)


if __name__ == "__main__":
    unittest.main()
