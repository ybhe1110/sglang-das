"""Regression for #37909 with real CPU/Gloo and the release bootstrap queue."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_collective_timeout import UTILS, load_collective_helpers
from test_external_kv_abort_drain import PREFILL, load_methods, register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def run_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        groups = {}
        for ranks in ([0, 1], [2, 3], [0, 2], [1, 3], [0], [1], [2], [3]):
            group = dist.new_group(ranks, timeout=timedelta(seconds=15))
            if rank in ranks:
                groups[tuple(ranks)] = group
        tp = groups[(0, 1) if rank < 2 else (2, 3)]
        cp = groups[(0, 2) if rank % 2 == 0 else (1, 3)]
        single = groups[(rank,)]
        # Run the real poll, timeout, resource vote and queue methods without
        # importing model kernels. Stub only request creation and allocation.
        env = load_collective_helpers()
        poll = SimpleNamespace(Failed=0, Bootstrapping=1, WaitingForInput=2)
        config = SimpleNamespace(
            optimistic_prefill_attempts=1,
            disaggregation_decode_allocation_policy="immediate",
        )
        env.update(
            torch=torch,
            dist=dist,
            KVPoll=poll,
            json=json,
            should_force_retry=lambda req: req.rid == "force",
            get_disagg=lambda: config,
            _uses_write_through_cache=lambda cache: cache.write_through,
        )
        env["envs"].SGLANG_TEST_DISAGG_FAILURE_PROB = SimpleNamespace(get=lambda: 0)
        names = [
            "poll_and_all_reduce_pp",
            "poll_and_all_reduce",
            "poll_and_all_reduce_attn_cp_tp_group",
            "_poll_with_failure_injection",
        ]
        functions = load_methods(UTILS, names, env)
        env.update({name: getattr(functions, name) for name in names})
        methods = [
            "pop_bootstrapped",
            "_aligned_bootstrap_requests",
            "_get_consensus_bootstrap_resource_ready",
            "_probe_bootstrap_ready",
        ]
        cls = load_methods(PREFILL, methods, env, class_name="PrefillBootstrapQueue")

        def queue(rids, tp_group=tp, cp_group=cp, credits=32):
            q = cls()
            q.pp_size = 1
            q.scheduler = SimpleNamespace(
                attn_tp_cpu_group=tp_group,
                attn_cp_cpu_group=cp_group,
                handle_bootstrap_failure=Mock(),
                tree_cache=SimpleNamespace(write_through=False),
                processed_tokens_counter=728,
            )
            q.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
                available_size=lambda: credits
            )
            q.kv_manager = SimpleNamespace(
                req_to_pd_hidden_meta={}, req_to_decode_prefix_len={}
            )
            q.ensure_metadata_buffer = Mock(return_value=True)
            q.finalize_bootstrap = Mock(return_value=True)
            q._requires_pd_hidden_transfer = lambda req: False
            q.queue = []
            for rid in rids:
                state = poll.Bootstrapping if rid == "opt" else poll.WaitingForInput
                if rid == "fail" and rank == 3:
                    state = poll.Failed
                q.queue.append(
                    SimpleNamespace(
                        rid=rid,
                        bootstrap_room=rid,
                        metadata_buffer_index=-1,
                        pending_bootstrap=True,
                        prefill_attempt_count=0,
                        swa_branching_seqlen=None,
                        kv=SimpleNamespace(cache_protected_len=0),
                        is_retracted=False,
                        time_stats=SimpleNamespace(set_wait_queue_entry_time=Mock()),
                        disagg_kv_sender=SimpleNamespace(poll=Mock(return_value=state)),
                    )
                )
            return q

        check = unittest.TestCase()

        def pop(q):
            good, failed = q.pop_bootstrapped(return_failed_reqs=True)
            return [r.rid for r in good], [r.rid for r in failed]

        for local in ([], [] if rank == 2 else ["only"], [str(rank)]):
            q = queue(local)
            check.assertEqual(pop(q), ([], []))
            check.assertEqual([r.rid for r in q.queue], local)
            for req in q.queue:
                req.disagg_kv_sender.poll.assert_not_called()

        # 8 vs 1; exercise each axis independently as well as the 2x2 mesh.
        for tp_group, cp_group, wide in (
            (tp, single, rank % 2 == 0),
            (single, cp, rank < 2),
            (tp, cp, rank == 0),
        ):
            extra = [f"local-{i}" for i in range(7)] if wide else []
            q = queue(extra + ["shared"], tp_group, cp_group)
            check.assertEqual(pop(q), (["shared"], []))
            check.assertEqual([r.rid for r in q.queue], extra)
            # The lagging ranks subsequently receive the remaining RIDs.
            if not wide:
                q.queue.extend(queue([f"local-{i}" for i in range(7)]).queue)
            check.assertEqual(pop(q), ([f"local-{i}" for i in range(7)], []))

        # Same length, different order and membership. The first rank's FIFO
        # order drives the next model batch as well as the resource collectives.
        local = (
            ["z", str(rank), "fail", "a"]
            if rank == 0
            else ["a", "fail", str(rank), "z"]
        )
        q = queue(local)
        if rank == 3:
            q.kv_manager.req_to_decode_prefix_len = {"z": 256, "a": 512}
        check.assertEqual(pop(q), (["z", "a"], ["fail"]))
        check.assertEqual([r.rid for r in q.queue], [str(rank)])
        check.assertEqual(q.kv_manager.req_to_decode_prefix_len, {"z": 256, "a": 512})

        for rid in ("normal", "opt", "force"):
            q = queue([rid], credits=0 if rank == 1 else 1)
            check.assertEqual(pop(q), ([], []))
            q.ensure_metadata_buffer.assert_not_called()
            q.finalize_bootstrap.assert_not_called()
            check.assertEqual(len(q.queue), 1)

        # v4.1 permits optimistic admission with zero ordinary attempts under
        # prefill_complete, but a write-through SWA branch on even one rank
        # must hold the whole request until that rank can proceed.
        config.optimistic_prefill_attempts = 0
        config.disaggregation_decode_allocation_policy = "prefill_complete"
        q = queue(["opt"])
        req = q.queue[0]
        q.scheduler.tree_cache.write_through = True
        req.swa_branching_seqlen = 1 if rank == 2 else None
        check.assertEqual(pop(q), ([], []))
        q.ensure_metadata_buffer.assert_not_called()
        req.swa_branching_seqlen = None
        check.assertEqual(pop(q), (["opt"], []))
        check.assertEqual(req.arrival_processed_tokens, 728)

        q = queue(["dup", "dup"] if rank == 1 else ["dup"])
        with check.assertRaisesRegex(
            env["DisaggregationCollectiveError"], "Duplicate RID"
        ):
            pop(q)
        q = queue(["b", "a"], single, single)
        check.assertIs(q._aligned_bootstrap_requests(), q.queue)
        check.assertEqual(pop(q), (["b", "a"], []))
        q = queue(["b", "a"])
        q.pp_size = 2
        q._aligned_bootstrap_requests = Mock(
            side_effect=AssertionError("PP uses its own consensus")
        )
        good, failed = q.pop_bootstrapped(True, pp_good_rids=["b"], pp_bad_rids=["a"])
        check.assertEqual([req.rid for req in good], ["b"])
        check.assertEqual([req.rid for req in failed], ["a"])
    finally:
        dist.destroy_process_group()


class TestPrefillBootstrapRIDAlign(unittest.TestCase):
    def test_attention_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = (Path(tmp) / "store").as_uri()
            mp.spawn(run_rank, args=(rendezvous,), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
