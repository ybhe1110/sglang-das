"""Run production Rust-ingress routing with real Gloo collectives, no model.

Example: torchrun --standalone --nproc-per-node=16 scripts/verify_rust_dp_collectives.py
Two hosts: use torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0/1
           --master-addr=<node0> --master-port=29501 ...
Also run --cp-size=4 (DP4/CP4) and --cp-size=16 (DP1/CP16).
This checks collective routing, not Rust HTTP/rings or KV/DMA safety.
"""
import argparse
import ast
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import torch.distributed as dist


@dataclass
class Work:
    rid: str


@dataclass
class Abort:
    rid: str
    abort_all: bool = False


class StateQuery:
    pass


def broadcast(value, rank, group, src):
    objects = [value]
    dist.broadcast_object_list(objects, src=src, group=group)
    return objects[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", type=int, default=1)
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        cp = args.cp_size
        assert cp > 0 and size % cp == 0
        source = Path(__file__).resolve().parents[1] / "python/sglang/srt/managers/scheduler_components/request_receiver.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SchedulerRequestReceiver")
        names = {"_broadcast_rust_reqs", "_split_work_and_control_reqs"}
        methods = [n for n in cls.body if getattr(n, "name", None) in names]
        assert len(methods) == len(names)
        unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(name="Receiver", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
        ns = dict(broadcast_pyobj=broadcast, all_gather_object=dist.all_gather_object,
                  AbortReq=Abort, GetInternalStateReq=StateQuery,
                  TokenizedGenerateReqInput=Work, TokenizedEmbeddingReqInput=Work,
                  BatchTokenizedGenerateReqInput=Work, BatchTokenizedEmbeddingReqInput=Work)
        exec(compile(ast.fix_missing_locations(unit), str(source), "exec"), ns)
        receiver = ns["Receiver"]()
        receiver.ps = NS(pp_rank=0, attn_tp_rank=0, attn_cp_rank=rank % cp,
                         attn_tp_size=1, attn_cp_size=cp, tp_size=size)
        receiver.tp_cpu_group = dist.group.WORLD
        for start in range(0, size, cp):
            ranks = list(range(start, start + cp))
            group = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                receiver.attn_cp_cpu_group = group
                receiver.attn_cp_group = NS(rank=rank % cp, ranks=ranks)
        leader = rank // cp * cp
        global_source = size - cp
        for step in range(20):
            local = [] if rank == leader else None
            if rank == leader and step % 3:
                local = [Work(f"r{leader}-{step}"), Abort(f"r{leader}-{step}")]
            if rank == global_source and step % 2:
                local.append(Abort(f"all-{step}", True))
            actual = receiver._broadcast_rust_reqs(local)
            expected = []
            if step % 3:
                expected += [Work(f"r{leader}-{step}"), Abort(f"r{leader}-{step}")]
            if step % 2:
                expected += [Abort(f"all-{step}", True)]
            assert actual == expected, (rank, step, actual, expected)
        dist.barrier()
        if rank == 0:
            print(f"PASS: real Gloo world={size}, DP={size // cp}, CP={cp}; local cancels, empty steps and global fanout")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
