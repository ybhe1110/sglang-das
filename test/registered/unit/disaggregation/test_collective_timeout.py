"""CPU tests for fail-fast P/D rank collectives."""

import ast
import pathlib
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock


ROOT = pathlib.Path(__file__).resolve().parents[4]
UTILS = ROOT / "python/sglang/srt/disaggregation/utils.py"
PREFILL = ROOT / "python/sglang/srt/disaggregation/prefill.py"
DISAGGREGATION = ROOT / "python/sglang/srt/disaggregation"


def load_collective_helpers():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    names = {
        "DisaggregationCollectiveError",
        "_collective_timeout_s",
        "_abort_collective_group",
        "all_reduce_with_timeout",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    ]

    class FakeTorch:
        Tensor = object

    class FakeDist:
        ProcessGroup = object

    logger = Mock()
    env = {
        "timedelta": __import__("datetime").timedelta,
        "Optional": Optional,
        "torch": FakeTorch,
        "dist": FakeDist,
        "envs": SimpleNamespace(
            SGLANG_DISAGGREGATION_COLLECTIVE_TIMEOUT=SimpleNamespace(
                get=lambda: 2.5
            )
        ),
        "logger": logger,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(UTILS), "exec"), env)
    return env


class TestCollectiveTimeout(unittest.TestCase):
    def test_collective_waits_asynchronously_with_configured_timeout(self):
        env = load_collective_helpers()
        work = Mock()
        group = Mock()
        dist = env["dist"]
        dist.all_reduce = Mock(return_value=work)
        env["all_reduce_with_timeout"](object(), op="min", group=group)
        dist.all_reduce.assert_called_once_with(
            unittest.mock.ANY,
            op="min",
            group=group,
            async_op=True,
        )
        work.wait.assert_called_once()
        timeout = work.wait.call_args.kwargs["timeout"]
        self.assertAlmostEqual(timeout.total_seconds(), 2.5)

    def test_collective_failure_aborts_group_and_raises(self):
        env = load_collective_helpers()
        work = Mock()
        work.wait.side_effect = TimeoutError("collective timeout")
        group = Mock()
        env["dist"].all_reduce = Mock(return_value=work)

        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_with_timeout"](object(), op="min", group=group)

        group.abort.assert_called_once_with()

    def test_collective_false_wait_result_is_a_timeout(self):
        env = load_collective_helpers()
        work = Mock()
        work.wait.return_value = False
        group = Mock()
        env["dist"].all_reduce = Mock(return_value=work)

        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_with_timeout"](object(), op="min", group=group)

        group.abort.assert_called_once_with()

    def test_collective_none_wait_result_is_a_timeout(self):
        # Some backends (e.g. older HCCL) return None from work.wait() when
        # the operation times out instead of returning False.  Both are treated
        # as a failure so the group is aborted and the caller gets an error.
        env = load_collective_helpers()
        work = Mock()
        work.wait.return_value = None
        group = Mock()
        env["dist"].all_reduce = Mock(return_value=work)

        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_with_timeout"](object(), op="min", group=group)

        group.abort.assert_called_once_with()

    def test_group_destroy_is_fallback_when_abort_is_unavailable(self):
        env = load_collective_helpers()
        work = Mock()
        work.wait.side_effect = RuntimeError("peer disconnected")
        group = object()
        env["dist"].all_reduce = Mock(return_value=work)
        env["dist"].destroy_process_group = Mock()

        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_with_timeout"](object(), op="min", group=group)

        env["dist"].destroy_process_group.assert_called_once_with(group)

    def test_prefill_raw_collectives_use_timeout_helper(self):
        for path in DISAGGREGATION.rglob("*.py"):
            if path == UTILS:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            direct = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "all_reduce"
            ]
            self.assertEqual(direct, [], path.as_posix())

        tree = ast.parse(PREFILL.read_text(encoding="utf-8"))
        imports_helper = any(
            isinstance(node, ast.Name) and node.id == "all_reduce_with_timeout"
            for node in ast.walk(tree)
        )
        self.assertTrue(imports_helper)


if __name__ == "__main__":
    unittest.main()
