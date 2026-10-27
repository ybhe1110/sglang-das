"""CPU tests for fail-fast P/D rank collectives."""

import ast
import pathlib
import unittest
import zlib
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import Mock

ROOT = pathlib.Path(__file__).resolve().parents[4]
UTILS = ROOT / "python/sglang/srt/disaggregation/utils.py"
PREFILL = ROOT / "python/sglang/srt/disaggregation/prefill.py"
DISAGGREGATION = ROOT / "python/sglang/srt/disaggregation"
UNIFIED_RADIX = ROOT / "python/sglang/srt/mem_cache/unified_radix_cache.py"


def load_collective_helpers():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    names = {
        "DisaggregationCollectiveError",
        "_collective_timeout_s",
        "_abort_collective_group",
        "all_reduce_with_timeout",
        "_all_reduce_flags_preflight",
        "_all_reduce_polls",
        "all_reduce_min_flags",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    ]

    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def __getitem__(self, index):
            return self.values[index]

        def tolist(self):
            return list(self.values)

    class FakeTorch:
        Tensor = object
        int64 = object()

        @staticmethod
        def tensor(values, **kwargs):
            return FakeTensor(values)

    class FakeDist:
        get_world_size = staticmethod(lambda group: 2)
        ProcessGroup = object

        class ReduceOp:
            MIN = "min"

    logger = Mock()
    env = {
        "timedelta": __import__("datetime").timedelta,
        "List": List,
        "Optional": Optional,
        "torch": FakeTorch,
        "dist": FakeDist,
        "envs": SimpleNamespace(
            SGLANG_DISAGGREGATION_COLLECTIVE_TIMEOUT=SimpleNamespace(get=lambda: 2.5)
        ),
        "logger": logger,
        "zlib": zlib,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(UTILS), "exec"), env)
    return env


class TestCollectiveTimeout(unittest.TestCase):
    def test_collective_waits_asynchronously_with_configured_timeout(self):
        env = load_collective_helpers()
        work = Mock()
        work.wait.return_value = True
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
        for path in (PREFILL, DISAGGREGATION / "decode_kvcache_offload_manager.py"):
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

    def test_unified_radix_collectives_use_timeout_helper(self):
        """The linker preflight/mask reduce must abort instead of hanging."""
        tree = ast.parse(UNIFIED_RADIX.read_text(encoding="utf-8"))
        direct = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "all_reduce"
        ]
        self.assertEqual(direct, [])
        self.assertTrue(
            any(
                isinstance(node, ast.Name) and node.id == "all_reduce_with_timeout"
                for node in ast.walk(tree)
            )
        )

    def test_cq_timeout_style_failure_aborts_collective_group(self):
        """A rank stalled in CQ/worker progress must fail the collective."""
        env = load_collective_helpers()
        work = Mock()
        work.wait.side_effect = TimeoutError("CQ completion did not progress")
        group = Mock()
        env["dist"].all_reduce = Mock(return_value=work)

        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_with_timeout"](object(), op="min", group=group)

        group.abort.assert_called_once_with()

    def test_cleanup_flags_shape_mismatch_is_fatal_before_variable_reduce(self):
        """A 4-vs-2 cleanup queue mismatch must never reach variable reduce."""
        env = load_collective_helpers()
        work = Mock()
        work.wait.return_value = True
        calls = []

        def all_reduce(tensor, **kwargs):
            calls.append(tensor)
            if len(calls) == 1:
                # Local queue has two entries; a peer has four.  The fixed
                # payload is still the same shape, so the mismatch is detected
                # in Python before a flags tensor can be submitted.
                tensor.values = [2, -4, 1, -1]
            return work

        env["dist"].all_reduce = Mock(side_effect=all_reduce)
        group = Mock()

        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_min_flags"]([0, 1], group, group)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].values, [2, -4, 1, -1])

    def test_empty_cleanup_queue_stops_after_fixed_preflight(self):
        """All ranks with no inflight requests avoid a zero-length reduce."""
        env = load_collective_helpers()
        work = Mock()
        work.wait.return_value = True
        calls = []

        def all_reduce(tensor, **kwargs):
            calls.append(tensor)
            return work

        env["dist"].all_reduce = Mock(side_effect=all_reduce)
        group = Mock()

        self.assertEqual(env["all_reduce_min_flags"]([], group, group), [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].values, [0, 0, 1, -1])
        self.assertEqual(calls[1].values, [0, 0, 2, -2])

    def test_cleanup_rid_order_mismatch_is_fatal_before_variable_reduce(self):
        """Ranks must poll and remove the same RID sequence."""
        env = load_collective_helpers()
        work = Mock()
        work.wait.return_value = True
        calls = []

        def all_reduce(tensor, **kwargs):
            calls.append(tensor)
            if len(calls) == 1:
                # Keep queue length and phase equal, but emulate a peer with a
                # different ordered RID digest.
                tensor.values[5] -= 1
            return work

        env["dist"].all_reduce = Mock(side_effect=all_reduce)
        group = Mock()
        with self.assertRaises(env["DisaggregationCollectiveError"]):
            env["all_reduce_min_flags"](
                [0, 1], group, group, request_ids=["rid-a", "rid-b"]
            )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
