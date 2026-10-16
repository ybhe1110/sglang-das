"""Check the backported snapshot contract without importing GPU dependencies."""
import ast
import json
from pathlib import Path
from typing import Optional

import msgspec

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "python/sglang/srt/managers/load_snapshot.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
names = {
    "MemoryMetrics", "SpeculativeMetrics", "LoRAMetrics",
    "DisaggregationMetrics", "QueueMetrics", "LoadSnapshot",
}
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
scope = {"msgspec": msgspec, "Optional": Optional, "DPBudgetSnapshot": object}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
scope["_CORE_KEYS"] = tuple(
    key for key in scope["LoadSnapshot"].__struct_fields__
    if key not in {"memory", "speculative", "lora", "disaggregation", "queues", "dp_budget"}
)
fixture = json.loads((ROOT / "rust/sglang-server/testdata/load_snapshot.json").read_text())
for index, payload in enumerate(fixture["payloads"]):
    snapshot = msgspec.msgpack.decode(bytes.fromhex(payload), type=scope["LoadSnapshot"])
    for name, include in [("all", None), ("core", {"core"}), ("core,queues", {"core", "queues"})]:
        assert snapshot.to_dict(include) == fixture["to_dict"][name][index], name
print("PASS: Python snapshot wire payloads and JSON fixture agree")
