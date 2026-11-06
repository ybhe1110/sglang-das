# Aigate DP16 discovery and Rust metrics

This integration targets one RBG deployment with single-node P DP1 replicas and
one two-node Decode TP16/DP16 engine (eight HCUs per node). All engines use
host networking, the same base API port, `SGLANG_RUST_SERVER=1`, and
`--enable-metrics`. Python and the Rust extension must both include `f94b51c5ee`
or later. No additional Cargo dependencies or vendor changes are needed here.

The November 6 inspection found working Rust metrics but stale Prometheus
static targets: four down targets and a Decode endpoint labeled `role=prefill`.
Aigate had only one Decode endpoint despite all 16 listeners being reachable.
This directory fixes discovery and supplies a read-only acceptance check.
It does not change private Aigate code or apply anything just by installing SGLang.

## Generate from engine Pods

Use `scripts/reconcile_aigate_dp16.py` instead of the old static 14-Pod renderer.
The old renderer assumes rank0 AND rank8 are already registered, and its fixed
node names do not follow an engine reschedule.

The reconciler reads RBG `group-name`, `role-name`, `component-index` and
`component-size` labels. It supports exactly one D pair per configured group.
Do not select by Aigate labels: the D worker may have none. It generates:

- All missing rank registrations, including rank8 when the worker has no
  Aigate labels/port. Already advertised rank0/rank8 are not duplicated.
- Owner references to the engine Pod UID and scheduling affinity to its actual
  node. Names change with the UID. Stale managed registrations are deleted with
  UID preconditions; engine Pods and unrelated registrations are never patched.
- One Prometheus target per ready engine Pod, deriving `role`, `node_name`, IP
  and port from that Pod. D node1 uses `base_port + 8`, never `base_port`.

All discovered P engines must already have the correct Aigate labels and named
`sglang-http` port. A discovered D engine with an incorrect named port is rejected,
not silently double-registered. Fix those fields in the RBG workload template.
The configured base port/topology must match the engine launch arguments.
The acceptance check validates real listener identities before traffic testing.

First render a reviewable plan (no Kubernetes writes):

```sh
set -a
. ./examples/monitoring/aigate-dp16/discovery.env.example
set +a
# Set PAUSE_IMAGE to the locally approved image before applying.
python3 scripts/reconcile_aigate_dp16.py \
  --namespace "$NAMESPACE" --engine-group "$ENGINE_GROUP" \
  --name-prefix "$NAME_PREFIX" --router-group "$ROUTER_GROUP" \
  --transfer-group "$TRANSFER_GROUP" --model "$MODEL" \
  --pause-image "$PAUSE_IMAGE" --base-port "$BASE_PORT" \
  --output-dir /tmp/sglang-discovery-review
```

Outputs are `plan.json` and `sglang-targets.json`. For offline review, add
`--pods-json pods.json` from `kubectl get pods -n aiservice -o json`.
Offline snapshots cannot be applied or watched. Verify 16 ingresses and, for
3P+2D, five targets. A worker without Aigate discovery needs **15** registration
Pods; one already advertising rank8 needs **14**. Readiness probes target the
existing Rust `/health` listener; pause containers do not proxy inference.
Delete any previous manually created registration Pods before enabling this
reconciler, to avoid duplicate endpoints/host-port reservations. The reconciler
will not delete foreign registrations for you.

After review, append `--apply --watch 15` and use the persistent output directory
`/var/lib/sglang-discovery`. Apply mode first verifies each live Decode listener's
`/v1/loads` global rank and size; a Python frontend or incorrect Rust topology is
rejected before publishing the plan or changing registrations.
The watcher restores manually deleted registrations
and follows engine Pod replacements. Not-ready/terminating engines are withdrawn.
An API/validation error retains the last valid file and logs an error; monitor
the watcher log and Prometheus target health. Run only one writer per name prefix
and output directory. Engine image/topology changes require revalidation.

For persistence on the Prometheus host, install Python 3.10+, kubectl and the
repository at `/opt/sglang`, create the `sglang-discovery` OS user, and install the
provided `discovery.service` plus `/etc/sglang-discovery/discovery.env`.
Supply a working kubeconfig with the namespace permissions in `rbac.yaml` and
credential renewal appropriate to your cluster. The Role permits Pod creation
and deletion throughout the namespace; script ownership checks narrow its use,
but Kubernetes RBAC itself does not restrict these verbs by label.
Then enable the unit with `systemctl enable --now discovery.service`.
Run in dry-run mode first on clusters with Pod admission/security restrictions.

## Prometheus and labels

Replace the old static SGLang scrape job with `prometheus.yaml`, preserving other
jobs in your real configuration. Validate with `promtool check config` and reload
Prometheus once. Subsequent file_sd address changes are automatic. If Prometheus
runs in a container, mount the **directory**, not only the JSON file, because the
writer replaces the file atomically. Restrict the file_sd path to
`sglang-targets.json`; `plan.json` is not a target file.

Do not scrape all 16 Rust ports for these metrics. Listeners within each engine
Pod share `PROMETHEUS_MULTIPROC_DIR` and expose that registry's aggregate. This
layout assumes each engine Pod has its own registry directory, as in the normal
launcher. Sharing one registry across Pods needs a different scrape layout.
Aigate should still poll **all 16** `/v1/loads` endpoints for rank-local load.

Frontend metrics retain native Python names and labels: `model_name`,
`engine_type`, `is_streaming`, and configured custom labels. `role` is a target
label and must agree with `engine_type`; frontend metrics do not acquire a
`dp_rank` label just because the scrape URL is rank9. Scheduler stage metrics
do have rank labels. P intentionally has no frontend TTFT series. Unused labeled
counters/histograms may be absent until the first request; health probes alone
do not create generation observations. An idle latency `rate` can be zero/NaN.

Example queries (scope model and job when sharing a Prometheus):

```promql
up{job="sglang"}
sum by (engine_type, is_streaming) (rate(sglang:num_requests_total{job="sglang"}[5m]))
histogram_quantile(0.95, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket{job="sglang",engine_type="decode"}[5m])))
sum by (dp_rank) (increase(sglang:per_stage_req_latency_seconds_count{job="sglang",engine_type="decode",stage="decode_transferred"}[5m]))
```

## Gateway routing and acceptance

Keep the existing Aigate Decode `round_robin` policy for coverage acceptance.
Discovery must show 16 distinct Decode addresses in `/v1/models`. A request for
rank9 goes to node1:`base_port+9`. `routed_dp_rank=9` validates the destination;
it does not forward a request from the rank0 listener to rank9. It may be omitted
when the selected address is already exact. Each P/D pair needs a new shared
bootstrap room/pair key. P uses its own rank, not the selected D rank.

Do not infer global identity from `loads[0].dp_rank` (compatibility value 0).
Use top-level `global_dp_rank/global_dp_size`. If later implementing load-aware
selection in Aigate, include pending admissions and freshness/failure handling
for each listener's load snapshot. Round robin balances request counts, not
different request lengths. This repository does not implement private EPP policy.

Install `prometheus-client` for the acceptance script. With the final plan and
Prometheus configuration active, take a baseline:

```sh
python3 scripts/verify_aigate_dp16.py \
  --plan /var/lib/sglang-discovery/plan.json \
  --gateway http://AIGATE_HOST:3000 --prometheus http://PROMETHEUS_HOST:9090 \
  --snapshot /tmp/dp16-before.json
```

Send ordinary and streaming Chat traffic through Aigate (at least several full
16-request rounds; keep the model and candidate set stable). Allow all requests
to finish and wait at least two Prometheus scrape intervals, then run the same
command with `--snapshot /tmp/dp16-after.json --baseline /tmp/dp16-before.json`.

The check requires exact P/D discovery, 16 correct live rank identities, one
healthy scrape per engine, correct role labels, positive per-rank KV-transferred
request deltas, and positive ordinary/streaming completion, TTFT and E2E deltas
in both raw metrics and Prometheus, including histogram count and sum growth.
It checks histogram +Inf/count consistency.
It sends no inference requests and does not claim a performance result. Also
check your workload's HTTP errors/token counts; transferred requests alone are
not proof that every generation completed. Refresh the plan for each snapshot;
Pod replacement or a container restart requires a new baseline.

References: [Prometheus file discovery](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#file_sd_config),
[Kubernetes owner references](https://kubernetes.io/docs/concepts/architecture/garbage-collection/).
