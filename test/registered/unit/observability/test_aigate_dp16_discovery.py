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
    verify = importlib.import_module("verify_aigate_dp16")


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
    port = 30000 + (8 * index if role == "decode" else 0)
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
                    "ports": [{"name": "sglang-http", "containerPort": port}]
                    if discovered
                    else [],
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
        self.assertEqual(sorted(ports), list(range(30001, 30016)))
        self.assertEqual(len(result["targets"]), 5)
        targets = {t["targets"][0]: t["labels"]["role"] for t in result["targets"]}
        self.assertEqual(targets["10.0.0.15:30000"], "decode")
        self.assertEqual(targets["10.0.0.5:30008"], "decode")
        self.assertNotIn("10.0.0.5:30000", targets)

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
        self.assertEqual(new["ingresses"][9]["address"], "10.0.0.22:30009")
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
        self.pods[1]["spec"]["containers"][0]["ports"][0]["containerPort"] = 30000
        with self.assertRaisesRegex(ValueError, "30008"):
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

    def test_container_restart_invalidates_baseline(self):
        before = plan(self.pods)
        self.pods[0]["status"]["containerStatuses"] = [
            {"name": "engine", "containerID": "new", "restartCount": 1}
        ]
        after = plan(self.pods)
        self.assertNotEqual(before["engine_epochs"], after["engine_epochs"])
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify.compare(
                {"model": "model", "engines": before["engine_epochs"]},
                {"model": "model", "engines": after["engine_epochs"]},
            )

    def test_atomic_targets_replace_stale_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.json"
            discovery.atomic_json(
                path, [{"targets": ["10.0.0.15:30000"], "labels": {"role": "prefill"}}]
            )
            discovery.atomic_json(path, plan(self.pods)["targets"])
            self.assertEqual(json.loads(path.read_text()), plan(self.pods)["targets"])
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["targets.json"])

    def test_registry_one_rank_is_not_coverage(self):
        desired = plan(self.pods)
        replicas = [
            {"addr": t["targets"][0], "role": t["labels"]["role"]}
            for t in desired["targets"]
            if t["labels"]["role"] == "prefill"
        ]
        replicas.append({"addr": "10.0.0.15:30000", "role": "decode"})
        models = {"models": [{"model": "model", "replicas": replicas}]}
        with self.assertRaisesRegex(ValueError, "registry mismatch"):
            verify.check_registry(desired, models)
        replicas.pop()
        replicas += [
            {"addr": i["address"], "role": "decode"} for i in desired["ingresses"]
        ]
        verify.check_registry(desired, models)

    def test_prometheus_wrong_role_or_missing_node_rejected(self):
        desired = plan(self.pods)
        active = [
            {
                "labels": {"job": "sglang", "role": t["labels"]["role"]},
                "scrapeUrl": "http://" + t["targets"][0] + "/metrics",
                "health": "up",
                "lastError": "",
            }
            for t in desired["targets"]
        ]
        verify.check_prometheus_targets(desired, active, "sglang")
        with self.assertRaisesRegex(ValueError, "targets differ"):
            verify.check_prometheus_targets(desired, active[:-1], "sglang")
        active[0]["labels"]["role"] = "wrong"
        with self.assertRaisesRegex(ValueError, "role mismatch"):
            verify.check_prometheus_targets(desired, active, "sglang")

    def test_before_after_requires_every_rank_and_both_stream_modes(self):
        before = {
            "engines": ["a"],
            "model": "model",
            "ranks": {},
            "finished": {},
            "ttft": {},
            "e2e": {},
        }
        after = copy.deepcopy(before)
        after["ranks"] = {str(i): 2 for i in range(16)}
        for metric in verify.GENERATION_METRICS:
            before[metric] = {}
            after[metric] = {"true": 16, "false": 16}
        self.assertEqual(len(verify.compare(before, after)), 16)
        after["ranks"]["9"] = 0
        with self.assertRaisesRegex(ValueError, "rank 9"):
            verify.compare(before, after)
        after["ranks"]["9"] = 2
        after["ttft"]["false"] = 0
        with self.assertRaisesRegex(ValueError, "is_streaming=false"):
            verify.compare(before, after)

    def test_histogram_and_label_validation(self):
        labels = 'engine_type="decode",model_name="model",is_streaming="true"'
        text = f'sglang:e2e_request_latency_seconds_count{{{labels}}} 3\nsglang:e2e_request_latency_seconds_bucket{{{labels},le="+Inf"}} 3\n'
        self.assertEqual(
            verify.read_metrics(text, "model", "decode")["e2e"], {"true": 3}
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify.read_metrics(
                text.replace('le="+Inf"} 3', 'le="+Inf"} 2'), "model", "decode"
            )
        with self.assertRaisesRegex(ValueError, "engine_type mismatch"):
            verify.read_metrics(text, "model", "prefill")

    def test_capture_checks_gateway_loads_and_prometheus(self):
        desired = plan(self.pods)
        models = {
            "models": [
                {
                    "model": "model",
                    "replicas": [
                        {"addr": i["address"], "role": "decode"}
                        for i in desired["ingresses"]
                    ]
                    + [
                        {"addr": t["targets"][0], "role": "prefill"}
                        for t in desired["targets"]
                        if t["labels"]["role"] == "prefill"
                    ],
                }
            ]
        }
        urls = {"http://gateway/v1/models": json.dumps(models)}
        active = []
        for target in desired["targets"]:
            address = target["targets"][0]
            role = target["labels"]["role"]
            active.append(
                {
                    "labels": {"job": "sglang", "role": role},
                    "scrapeUrl": f"http://{address}/metrics",
                    "health": "up",
                    "lastError": "",
                }
            )
            labels = f'engine_type="{role}",model_name="model",is_streaming="true"'
            metrics = f"sglang:num_requests_total{{{labels}}} 16\n"
            for name in ("e2e_request_latency_seconds", "time_to_first_token_seconds"):
                metrics += f'sglang:{name}_count{{{labels}}} 16\nsglang:{name}_bucket{{{labels},le="+Inf"}} 16\nsglang:{name}_sum{{{labels}}} 2.5\n'
            if role == "decode":
                start = 0 if address.endswith(":30000") else 8
                for rank in range(start, start + 8):
                    metrics += f'sglang:per_stage_req_latency_seconds_count{{engine_type="decode",model_name="model",dp_rank="{rank}",stage="decode_transferred"}} 2\n'
            urls[f"http://{address}/metrics"] = metrics
        urls["http://prometheus/api/v1/targets"] = json.dumps(
            {"data": {"activeTargets": active}}
        )
        for ingress in desired["ingresses"]:
            urls[f"http://{ingress['address']}/v1/loads"] = json.dumps(
                {"global_dp_rank": ingress["rank"], "global_dp_size": 16}
            )

        def get(url):
            if url.startswith("http://prometheus/api/v1/query?"):
                return json.dumps(
                    {
                        "data": {
                            "result": [
                                {
                                    "metric": {
                                        "role": "decode",
                                        "is_streaming": "true",
                                    },
                                    "value": [0, "32"],
                                }
                            ]
                        }
                    }
                )
            return urls[url]

        with patch.object(verify, "get", side_effect=get):
            snapshot = verify.capture(
                desired, "http://gateway", "http://prometheus", "sglang"
            )
            self.assertEqual(snapshot["finished"], {"true": 32})
            self.assertEqual(snapshot["ttft_sum"], {"true": 5})
            self.assertEqual(len(snapshot["ranks"]), 16)
            urls[f"http://{desired['ingresses'][9]['address']}/v1/loads"] = json.dumps(
                {"global_dp_rank": 0, "global_dp_size": 16}
            )
            with self.assertRaisesRegex(ValueError, "wrong listener identity"):
                verify.capture(desired, "http://gateway", "http://prometheus", "sglang")


if __name__ == "__main__":
    unittest.main()
