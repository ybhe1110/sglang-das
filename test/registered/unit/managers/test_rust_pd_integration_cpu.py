"""Dependency-light execution of production PD ingress methods.

The collective transports are controlled doubles, not a GPU/Gloo integration
test. Linux SHM/ZMQ and Rust socket tests live alongside their implementations.
"""

import ast
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]


def load_nodes(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    if class_name:
        nodes = [
            ast.ClassDef(
                name="Subject", bases=[], keywords=[], body=nodes, decorator_list=[]
            )
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace["Subject"] if class_name else namespace


class Work:
    pass


class Abort:
    def __init__(self, rid, abort_all=False):
        self.rid, self.abort_all = rid, abort_all


class StateQuery:
    pass


class TestRustControlRouting(unittest.TestCase):
    def setUp(self):
        self.config = NS(
            enable_dp_attention=True, enable_dp_attention_local_control_broadcast=False
        )
        self.global_batches = [[] for _ in range(16)]
        self.calls = []

        def gather(output, local, group):
            self.calls.append(("global", group.rank, local))
            for rank, batch in enumerate(self.global_batches):
                output[rank] = batch

        self.namespace = dict(
            envs=NS(SGLANG_RUST_SERVER=NS(get=lambda: True)),
            get_parallel=lambda: self.config,
            is_ep_scale_joiner=lambda: False,
            all_gather_object=gather,
            broadcast_pyobj=Mock(side_effect=lambda value, *a, **kw: value),
            AbortReq=Abort,
            GetInternalStateReq=StateQuery,
            TokenizedGenerateReqInput=Work,
            TokenizedEmbeddingReqInput=Work,
            BatchTokenizedGenerateReqInput=Work,
            BatchTokenizedEmbeddingReqInput=Work,
        )
        self.cls = load_nodes(
            "python/sglang/srt/managers/scheduler_components/request_receiver.py",
            {
                "_broadcast_reqs_across_ranks",
                "_broadcast_rust_reqs",
                "_split_work_and_control_reqs",
            },
            self.namespace,
            "SchedulerRequestReceiver",
        )

    def receiver(self, rank, tp=16, cp=1):
        r = self.cls()
        r.ps = NS(
            pp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=rank if cp > 1 else 0,
            attn_tp_size=1,
            attn_cp_size=cp,
            tp_size=tp,
        )
        r.tp_group = r.tp_cpu_group = NS(rank=rank, ranks=list(range(tp)))
        r.attn_cp_group = r.attn_cp_cpu_group = NS(rank=rank, ranks=list(range(cp)))
        return r

    def test_every_dp_rank_keeps_own_work_cancel_and_query(self):
        for flag in (False, True):
            self.config.enable_dp_attention_local_control_broadcast = flag
            for rank in range(16):
                with self.subTest(rank=rank, local_control=flag):
                    items = [Work(), Abort(f"request-{rank}"), StateQuery()]
                    result = self.receiver(rank)._broadcast_reqs_across_ranks(items)
                    self.assertEqual(result, items)
                    self.assertEqual(self.calls[-1][2], [])

    def test_global_abort_from_nonzero_dp_reaches_all_once(self):
        abort = Abort("all", abort_all=True)
        self.global_batches[9] = [abort]
        for rank in range(16):
            result = self.receiver(rank)._broadcast_reqs_across_ranks(
                [abort] if rank == 9 else []
            )
            self.assertEqual(result, [abort])
            self.assertEqual(self.calls[-1][2], [abort] if rank == 9 else [])

    def test_cp8_local_cancel_is_broadcast_without_duplicate_global_delivery(self):
        self.global_batches = [[] for _ in range(8)]
        cancel = Abort("cp-request")
        self.namespace["broadcast_pyobj"] = Mock(return_value=[cancel])
        for rank in range(8):
            result = self.receiver(rank, tp=8, cp=8)._broadcast_reqs_across_ranks(
                [cancel] if rank == 0 else None
            )
            self.assertEqual(result, [cancel])
            self.assertEqual(self.calls[-1][2], [])
        self.assertEqual(self.namespace["broadcast_pyobj"].call_count, 8)

    def test_empty_steps_still_participate_in_global_control_collective(self):
        for rank in range(16):
            self.assertEqual(self.receiver(rank)._broadcast_reqs_across_ranks([]), [])
        self.assertEqual(len(self.calls), 16)

    def test_later_pp_stage_does_not_gather_forwarded_globals_again(self):
        abort = Abort("all", abort_all=True)
        for rank in range(16):
            receiver = self.receiver(rank)
            receiver.ps.pp_rank = 1
            self.assertEqual(receiver._broadcast_reqs_across_ranks([abort]), [abort])
        self.assertEqual(self.calls, [])

    def test_python_path_preserves_original_rank0_control_broadcast(self):
        self.namespace["envs"].SGLANG_RUST_SERVER.get = lambda: False
        root_cancel = Abort("root")
        other_cancel = Abort("other")
        self.namespace["broadcast_pyobj"] = Mock(return_value=[root_cancel])
        result = self.receiver(9)._broadcast_reqs_across_ranks([other_cancel])
        self.assertEqual(result, [root_cancel])
        self.assertEqual(self.calls, [])


class TestRustSnapshots(unittest.TestCase):
    def setUp(self):
        class Shm:
            def __init__(self, path, dp_size, dp_rank, publish_interval=1):
                self.path, self.dp_size, self.dp_rank = path, dp_size, dp_rank
                self.values, self.closed = [], False

            def write(self, value):
                self.values.append(value)

            def close(self):
                self.closed = True

        self.shm = Shm
        self.remote = Mock()
        self.config = NS(
            enable_dp_attention=True,
            nnodes=2,
            node_rank=1,
            dp_size=16,
            load_balance_method="round_robin",
        )
        self.ns = dict(
            time=time,
            logger=logging.getLogger(__name__),
            ShmLoadSnapshotWriter=Shm,
            ZmqLoadSnapshotWriter=Mock(return_value=self.remote),
            envs=NS(
                SGLANG_RUST_SERVER=NS(get=lambda: True),
                SGLANG_LOAD_SNAPSHOT_USE_ZMQ=NS(get=lambda: False),
            ),
            get_parallel=lambda: self.config,
            shm_path_for=lambda instance: f"local/{instance}",
            _zmq_addr_for=lambda args: "tcp://node0:12345",
        )
        load_nodes(
            "python/sglang/srt/managers/load_snapshot.py",
            {
                "MirroredLoadSnapshotWriter",
                "create_load_snapshot_writer",
                "should_use_zmq",
                "zmq_reader_owner",
            },
            self.ns,
        )

    def test_second_node_rank8_to15_publish_local_and_remote(self):
        paths = set()
        for rank in range(8, 16):
            writer = self.ns["create_load_snapshot_writer"](
                NS(instance_id="dp16"), 16, rank
            )
            value = NS(dp_rank=rank, num_running_reqs=rank)
            writer.write(value)
            self.assertEqual(writer.values, [value])
            self.remote.write.assert_called_with(value)
            self.assertEqual((writer.dp_size, writer.dp_rank), (16, rank))
            paths.add(writer.path)
            self.assertNotEqual(writer.path, "local/dp16")
            writer.close()
            self.assertTrue(writer.closed)
        self.assertEqual(len(paths), 8)

    def test_remote_transport_failure_preserves_local_and_reconnects(self):
        import zmq

        writer = self.ns["create_load_snapshot_writer"](NS(instance_id="dp16"), 16, 9)
        self.remote.write.side_effect = zmq.ZMQError(zmq.ENOTSOCK)
        writer.write("first")
        writer.write("during-backoff")
        self.assertEqual(writer.values, ["first", "during-backoff"])
        self.assertEqual(writer.remote_publish_errors, 1)
        self.remote.write.assert_called_once()
        self.remote.close.assert_called_once()
        self.remote.write.side_effect = None
        writer._remote_retry_at = 0
        writer.write("latest")
        self.remote.write.assert_called_with("latest")
        self.assertEqual(writer.values[-1], "latest")
        self.assertEqual(self.ns["ZmqLoadSnapshotWriter"].call_count, 2)
        writer.close()

    def test_remote_programming_error_is_not_silenced(self):
        writer = self.ns["create_load_snapshot_writer"](NS(instance_id="dp16"), 16, 9)
        self.remote.write.side_effect = ValueError("bad rank")
        with self.assertRaisesRegex(ValueError, "bad rank"):
            writer.write("local")
        self.assertEqual(writer.values, ["local"])
        writer.close()

    def test_remote_start_failure_closes_local_mapping(self):
        self.ns["ZmqLoadSnapshotWriter"].side_effect = RuntimeError("connect failed")
        with patch.object(self.shm, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "connect failed"):
                self.ns["create_load_snapshot_writer"](NS(instance_id="dp16"), 16, 9)
            close.assert_called_once()

    def test_remote_close_failure_does_not_leak_local_mapping(self):
        writer = self.ns["create_load_snapshot_writer"](NS(instance_id="dp16"), 16, 9)
        self.remote.close.side_effect = RuntimeError("close failed")
        with self.assertRaises(RuntimeError):
            writer.close()
        self.assertTrue(writer.closed)

    def test_rust_controller_owns_remote_collector_for_all_balance_methods(self):
        for method in ("round_robin", "follow_bootstrap_room", "total_tokens"):
            for node in (0, 1):
                self.config.node_rank, self.config.load_balance_method = node, method
                for caller in (
                    "DataParallelController",
                    "TokenizerManager",
                    "MultiTokenizerRouter",
                ):
                    self.assertEqual(
                        self.ns["zmq_reader_owner"](caller),
                        node == 0 and caller == "DataParallelController",
                    )

    def test_python_multinode_writer_remains_zmq_only(self):
        self.ns["envs"].SGLANG_RUST_SERVER.get = lambda: False
        self.assertIs(
            self.ns["create_load_snapshot_writer"](NS(instance_id="dp16"), 16, 9),
            self.remote,
        )


class TestBootstrapPort(unittest.TestCase):
    def test_prefill_default_custom_and_shared_api_port_are_not_rewritten(self):
        ns = dict(
            envs=NS(SGLANG_DISAGG_STAGING_BUFFER=NS(get=lambda: False)),
            logger=logging.getLogger(__name__),
            declare_resolution=Mock(),
        )
        load_nodes(
            "python/sglang/srt/arg_groups/pd_disaggregation_hook.py",
            {"handle_pd_disaggregation"},
            ns,
        )
        for port in (8998, 19000, 30000):
            args = NS(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="mooncake",
                dcp_size=1,
                port=30000,
                disaggregation_bootstrap_port=port,
            )
            ns["handle_pd_disaggregation"](args)
            self.assertEqual(args.disaggregation_bootstrap_port, port)
        ns["declare_resolution"].assert_not_called()

    def test_only_prefill_dp0_owns_configured_listener(self):
        module = NS(get_disagg=lambda: NS(disaggregation_bootstrap_port=8998))
        cls = load_nodes(
            "python/sglang/srt/managers/rust_server.py",
            {"_bootstrap_listener_args"},
            {},
            "RustServer",
        )
        with patch.dict("sys.modules", {"sglang.srt.runtime_context": module}):
            for mode, rank, owner in (
                ("prefill", 0, True),
                ("prefill", 1, False),
                ("decode", 0, False),
                ("decode", 9, False),
            ):
                scheduler = NS(
                    server_args=NS(disaggregation_mode=mode, host="0.0.0.0"),
                    ps=NS(attn_dp_rank=rank),
                )
                self.assertEqual(
                    cls._bootstrap_listener_args(scheduler),
                    dict(
                        serve_pd_bootstrap=owner,
                        bootstrap_addr="0.0.0.0:8998" if owner else None,
                    ),
                )
            scheduler.server_args.disaggregation_mode = "prefill"
            scheduler.server_args.host = "::"
            scheduler.ps.attn_dp_rank = 0
            self.assertEqual(
                cls._bootstrap_listener_args(scheduler)["bootstrap_addr"], "[::]:8998"
            )




class TestIngressIdentity(unittest.TestCase):
    def test_attention_and_regular_dp_identity(self):
        ns = {}
        cls = load_nodes("python/sglang/srt/managers/rust_server.py",
                         {"_ingress_identity"}, ns, "RustServer")
        for rank in range(16):
            scheduler = NS(server_args=NS(enable_dp_attention=True),
                           ps=NS(attn_dp_rank=rank, attn_dp_size=16,
                                 dp_rank=rank, dp_size=16))
            self.assertEqual(cls._ingress_identity(scheduler), (rank, 16))
        scheduler.server_args.enable_dp_attention = False
        scheduler.ps.attn_dp_rank = 0
        self.assertEqual(cls._ingress_identity(scheduler), (15, 16))
        scheduler.ps.dp_rank = 16
        with self.assertRaises(ValueError):
            cls._ingress_identity(scheduler)


if __name__ == "__main__":
    unittest.main()
