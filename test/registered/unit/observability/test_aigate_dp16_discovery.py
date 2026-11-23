"""CPU regressions for deployed DP16 discovery failures; no inference or cluster."""

import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    discovery = importlib.import_module("reconcile_aigate_dp16")


def engine(role, index, ip, discovered=True, name=None):
    name = name or f"{role}-{index}"
    labels = {
        discovery.RBG + "group-name": "engine",
        discovery.RBG + "role-name": role,
        discovery.RBG + "component-index": str(index if role == "decode" else 0),
        discovery.RBG + "component-size": "2" if role == "decode" else "1",
    }
    if discovered:
        labels.update(
            {
                discovery.AIGATE + k: v
                for k, v in {
                    "router-group": "router",
                    "model-id": "model",
                    "role": role,
                    "transfer-group": "transfer",
                    "serving-mode": "disaggregated",
                }.items()
            }
        )
    port = 30000
    return {
        "metadata": {
            "name": name,
            "namespace": "aiservice",
            "uid": name + "-uid",
            "labels": labels,
        },
        "spec": {
            "hostNetwork": True,
            "nodeName": "node-" + ip,
            "containers": [
                {
                    "name": "engine",
                    "ports": (
                        [{"name": "sglang-http", "containerPort": port}]
                        if discovered
                        else []
                    ),
                }
            ],
        },
        "status": {
            "phase": "Running",
            "hostIP": ip,
            "podIP": ip,
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def plan(pods):
    return discovery.make_plan(
        pods,
        namespace="aiservice",
        engine_group="engine",
        name_prefix="test",
        router_group="router",
        transfer_group="transfer",
        model="model",
        pause_image="pause:test",
    )


class TestDP16Discovery(CustomTestCase):
    def setUp(self):
        self.pods = [
            engine("decode", 0, "10.0.0.15"),
            engine("decode", 1, "10.0.0.5", False),
        ]
        self.pods += [engine("prefill", i, f"10.0.0.{i + 1}") for i in range(3)]

    def test_missing_worker_discovery_registers_rank8_and_no_duplicate_metrics(self):
        result = plan(self.pods)
        self.assertEqual(len(result["registrations"]["items"]), 15)
        ports = [
            p["spec"]["containers"][0]["ports"][0]["containerPort"]
            for p in result["registrations"]["items"]
        ]
        self.assertEqual(
            sorted(ports), sorted([*range(30001, 30008), *range(30000, 30008)])
        )
        self.assertEqual(len(result["targets"]), 5)
        targets = {t["targets"][0]: t["labels"]["role"] for t in result["targets"]}
        self.assertEqual(targets["10.0.0.15:30000"], "decode")
        self.assertEqual(targets["10.0.0.5:30000"], "decode")
        self.assertNotIn("10.0.0.5:30008", targets)

    def test_already_registered_worker_needs_only_fourteen(self):
        self.pods[1] = engine("decode", 1, "10.0.0.5")
        self.assertEqual(len(plan(self.pods)["registrations"]["items"]), 14)

    def test_apply_requires_live_rust_rank_identity(self):
        desired = plan(self.pods)
        ranks = {i["address"]: i["rank"] for i in desired["ingresses"]}
        discovery.validate_ingresses(
            desired, lambda addr: {"global_dp_rank": ranks[addr], "global_dp_size": 16}
        )
        with self.assertRaisesRegex(ValueError, "expected Rust DP16"):
            discovery.validate_ingresses(
                desired, lambda addr: {"loads": [{"dp_rank": 0}]}
            )
        with self.assertRaisesRegex(ValueError, "expected Rust DP16"):
            discovery.validate_ingresses(
                desired, lambda addr: {"global_dp_rank": 0, "global_dp_size": 16}
            )

    def test_reschedule_moves_address_owner_and_affinity(self):
        old = plan(self.pods)
        self.pods[1] = engine("decode", 1, "10.0.0.22", False)
        self.pods[1]["metadata"]["uid"] = "replacement"
        new = plan(self.pods)
        self.assertEqual(new["ingresses"][9]["address"], "10.0.0.22:30001")
        reg = new["registrations"]["items"][-1]
        self.assertEqual(reg["metadata"]["ownerReferences"][0]["uid"], "replacement")
        affinity = reg["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        self.assertEqual(
            affinity["nodeSelectorTerms"][0]["matchFields"][0]["values"],
            ["node-10.0.0.22"],
        )
        self.assertNotEqual(
            old["registrations"]["items"][-1]["metadata"]["name"],
            reg["metadata"]["name"],
        )

    def test_notready_and_terminating_engines_withdraw_targets(self):
        for mutate in (
            lambda p: p["metadata"].update(deletionTimestamp="now"),
            lambda p: p["status"]["conditions"][0].update(status="False"),
        ):
            with self.subTest(mutate=mutate):
                pods = copy.deepcopy(self.pods)
                mutate(pods[1])
                result = plan(pods)
                self.assertEqual(len(result["ingresses"]), 8)
                self.assertEqual(len(result["targets"]), 4)

    def test_wrong_worker_base_port_rejected(self):
        self.pods[1] = engine("decode", 1, "10.0.0.5")
        self.pods[1]["spec"]["containers"][0]["ports"][0]["containerPort"] = 30008
        with self.assertRaisesRegex(ValueError, "30000"):
            plan(self.pods)

    def test_duplicate_nodes_or_wrong_role_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique node indices"):
            plan(self.pods + [copy.deepcopy(self.pods[0])])
        self.pods[0]["metadata"]["labels"][discovery.AIGATE + "role"] = "prefill"
        with self.assertRaisesRegex(ValueError, "fix Aigate labels"):
            plan(self.pods)

    def test_namespace_and_engine_scope(self):
        extra = engine("decode", 0, "10.0.0.99")
        extra["metadata"]["namespace"] = "other"
        self.assertEqual(plan(self.pods + [extra]), plan(self.pods))
        extra["metadata"]["namespace"] = "aiservice"
        extra["metadata"]["labels"][discovery.RBG + "group-name"] = "other"
        self.assertEqual(plan(self.pods + [extra]), plan(self.pods))

    def test_reconcile_recreates_missing_and_deletes_only_owned_uid(self):
        desired = plan(self.pods)
        old = copy.deepcopy(desired["registrations"]["items"][0])
        old["metadata"].update(name="test-old", uid="old-registration-uid")
        unrelated = engine("prefill", 9, "10.0.0.9")
        run = Mock()
        discovery.reconcile(desired, [old, unrelated], "aiservice", "test", run)
        self.assertEqual(run.call_count, 16)
        delete = run.call_args_list[0]
        self.assertEqual(
            delete.kwargs["body"]["preconditions"], {"uid": "old-registration-uid"}
        )
        self.assertTrue(delete.args[3].endswith("/pods/test-old"))
        run.reset_mock()
        discovery.reconcile(
            desired,
            desired["registrations"]["items"] + [unrelated],
            "aiservice",
            "test",
            run,
        )
        run.assert_not_called()

    def test_name_collision_refused(self):
        desired = plan(self.pods)
        foreign = copy.deepcopy(desired["registrations"]["items"][0])
        foreign["metadata"]["labels"].pop(discovery.MANAGED)
        run = Mock()
        with self.assertRaisesRegex(ValueError, "collision"):
            discovery.reconcile(desired, [foreign], "aiservice", "test", run)
        run.assert_not_called()

    def test_managed_label_never_allows_deleting_an_engine(self):
        pod = copy.deepcopy(self.pods[0])
        pod["metadata"]["labels"][discovery.MANAGED] = "test"
        run = Mock()
        with self.assertRaisesRegex(ValueError, "non-registration"):
            discovery.reconcile(plan(self.pods), [pod], "aiservice", "test", run)
        run.assert_not_called()

    def test_atomic_targets_replace_stale_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.json"
            discovery.atomic_json(
                path, [{"targets": ["10.0.0.15:30000"], "labels": {"role": "prefill"}}]
            )
            discovery.atomic_json(path, plan(self.pods)["targets"])
            self.assertEqual(json.loads(path.read_text()), plan(self.pods)["targets"])
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["targets.json"])


if __name__ == "__main__":
    unittest.main()
