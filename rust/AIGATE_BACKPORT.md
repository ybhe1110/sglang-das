# Rust frontend on the internal 0.5.18 baseline

Base: release/20260825_v0.5.18 at 585bc2d54e.
Branch: feat/rust-frontend-aigate.

The baseline already contains the community Rust tokenizer/detokenizer,
OpenAI handlers (#33103), PD bootstrap (#33125), and TCP TTFT fix (#33269).
This branch backports /v1/loads from upstream #44532
(26a2f18c0258b93edd315e03cb6f6d9a81881d63), adapting its module paths,
Python configuration bridge, error helper and tests to this baseline.
The newer num_prealloc_ready_tokens field is omitted to match the baseline's
Python snapshot schema rather than reporting a fabricated zero.

## Deployment contract

- Enable the existing frontend with SGLANG_RUST_SERVER=1.
- tokenizer-worker-num and detokenizer-worker-num configure separate Rust
  pools; they need not match and are already wired on this baseline.
- Use the Rust version pinned in rust/rust-toolchain.toml.
- GET /v1/loads?include=core,disagg returns the listener's own DP slot,
  exposed as dp_rank=0, following upstream. Register each actual IP:port
  separately in Aigate. This endpoint does not aggregate DP16.
- Multi-node DP attention publishes through MirroredLoadSnapshotWriter: a
  per-listener local SHM file plus the existing ZMQ stream to node 0's DPC.
  The Rust endpoint reads its local file even if the remote collector is
  unavailable. A delayed remote snapshot cannot overwrite the local mirror.
- Never interpret 503, empty loads, or stale snapshot timestamps as idle.
- Validate Aigate's private schema and polling policy against real responses.
- PD bootstrap preserves --disaggregation-bootstrap-port (default 8998).
  Only the prefill DP0 frontend owns that registry. It starts before rank
  registration and closes with the runtime. An explicitly shared API/bootstrap
  port uses one listener; port conflicts fail startup. D/Aigate should continue
  to supply the configured bootstrap port.
- Chat forwards bootstrap_host, bootstrap_port, bootstrap_room,
  bootstrap_pair_key, decode_tp_size, routed_dp_rank and disagg_prefill_dp_rank
  to the same scheduler wire fields as /generate. Mooncake still requires a
  fresh shared pair key from Aigate for each P/D request lifecycle. PD chat
  requires n=1; route separate choices with separate room/key pairs.
- Individual cancellations and internal-state queries stay within the serving
  attention TP/CP group. Global controls are gathered from all Rust ingress
  leaders at PP0, then fanned out once. Later PP stages do not gather them again.
  Empty PP0 steps participate too; benchmark the full-group gather overhead.
- Cancellation survives a full scheduler ring. The Rust intake waits for ring
  capacity or runtime shutdown; this does not certify native DMA cancellation.
- Rebuild the Rust extension together with Python: the Server constructor now
  receives bootstrap_addr and serve_pd_bootstrap.

This preserves the branch's per-DP-listener architecture. DPC still manages
schedulers and may collect ZMQ load reports, but does not dispatch requests
received on these Rust listeners. For dual-node DP16, Aigate must discover
all 16 actual ingress addresses (or use an explicitly deployed aggregation
layer); registering only the two base ports cannot reach the other ranks.

## Validation

Run the lightweight schema check with Python and msgspec:

    python scripts/verify_rust_loads_backport.py

On the target Linux build environment:

    cd rust
    cargo test --locked -p sglang-server api_server::loads
    cargo test --locked -p sglang-server api_server::openai::chat
    cargo test --locked -p sglang-server api_server::disaggregation::bootstrap
    cargo test --locked -p sglang-server abort_survives_full_ring

Dependency-light Python routing/configuration regressions:

    python -m pytest test/registered/unit/managers/test_rust_pd_integration_cpu.py

Actual SHM/ZMQ tests (Linux and the SGLang Python test dependencies required):

    python -m pytest test/registered/unit/managers/test_load_snapshot_backends.py

Then run test/registered/disaggregation/test_disaggregation_rust_server.py and poll
each DP endpoint during generation. Check running, prealloc and transfer
queues and snapshot freshness. Exercise cancellation and streaming output.
For the intended topology, use P TP8/CP8 and D dual-node TP16/DP16. Send a
distinct request to each D ingress, cancel on nonzero ranks, and verify that
other ranks' requests continue. Test both /generate and /v1/chat/completions
through Aigate with identical P/D room/key fields. Observe source-drain,
quarantine and pool accounting after drain, and compare TTFT/throughput against
the Python frontend. Do not infer end-to-end safety from these CPU tests.

Production qualification still requires the GPU/HCU acceptance checks below.
The isolated Linux build and CPU checks for the chat fix are recorded separately.
It is a scoped loads backport and PD integration fix, not a wholesale merge of
latest upstream.


## DP16 ingress identity and failure contract

The ingress identity comes from attention-DP rank/size (ordinary DP uses its
scheduler DP identity). Python and Rust reject invalid identity or a snapshot
writer whose slot/size disagrees with that identity. Rebuild the extension:
`ServerArgs` now requires `global_dp_rank` and `global_dp_size`.

Both `/generate` and `/v1/chat/completions` reject a supplied `routed_dp_rank`
that does not match the ingress, including invalid single-rank destinations.
Native batches are fully checked before submission. An omitted rank means
"this ingress"; it does not trigger DPC forwarding.

JSON `/v1/loads` adds top-level `global_dp_rank` and `global_dp_size`.
`loads[].dp_rank=0`, `?dp_rank=0` and Prometheus load labels retain their existing
per-listener semantics. The reader still uses the writer-derived physical slot.
These metadata fields do not change the load snapshot wire schema.

TP16/DP16/CP1/PP1 spans both nodes in ONE TP group; its CPU group already spans
all 16 ingress leaders. Keep the full TP gather. For TP/CP subgroups, only
leaders contribute global controls but all ranks receive them. Do not reuse the
separate StepInfo/MLPSync scheduler group for request controls.

Snapshot writer initialization failure now fails Rust scheduler startup rather
than silently starting an ingress with no load writer. A disconnected collector
normally uses ZMQ's nonblocking queue. A transport exception after local write
preserves the local snapshot, increments `remote_publish_errors`, closes the
bad socket and retries construction after 5 seconds on a later publication.
Warnings are limited to once per 30 seconds. Configuration/programming errors
are not silently swallowed. The counter is internal, not a `/metrics` export.

## Aigate integration requirements

For persistent Kubernetes rank registration and dynamic Prometheus targets, use
the [DP16 discovery reconciler and acceptance checks](../examples/monitoring/aigate-dp16/README.md).
They cover the missing rank8 registration, Pod rescheduling, stale scrape-role
labels, and before/after request-counter and histogram validation.

The 129.1 cluster's Aigate EPP discovers one named `sglang-http` port per
selected Pod. In the original deployment only the D leader's base port was
registered. Requests to that listener went to DP0. See the
[tested DP16 registration layout and benchmark results](benchmarks/aigate-dp16/README.md)
for the fix and the per-rank checks. For other deployments, verify the
following integration requirements:

- Discover all 16 real D ingress addresses. With the same base port B on both
  hosts: node0 serves B+0 through B+7, node1 serves B+8 through B+15.
  Use actual startup addresses if host/base-port configuration differs.
- Associate each address with the new top-level global identity (or equivalent
  registration metadata). Do not use `loads[0].dp_rank=0` as a cluster identity.
  Continue polling each address without a rank query, or with `dp_rank=0`.
- Send to the selected address; set `routed_dp_rank` to that address's rank,
  or omit it. P and D destination ranks may differ. Never copy D rank 9 into
  a P-DP0 request. `disagg_prefill_dp_rank` on D identifies the selected P rank.
- Generate one fresh room/pair key per request lifecycle; the matching P and D
  requests share them. Re-pairing retries get new identities. PD Chat uses n=1.
- Preserve P DP0's configured bootstrap address/port (default 8998).
- Chat clients can force a fixed output length with `max_tokens`, `min_tokens`
  and `ignore_eos`. Preserve these fields on both P and D requests. The Chat
  adapter maps `min_tokens` to `min_new_tokens` and applies `ignore_eos`
  before normalizing scheduler sampling parameters.
- Keep cancellation bound to the owning ingress and propagate client disconnects
  to downstream requests. A P failure must cancel its paired D request.
- Treat missing/503/stale load snapshots as unavailable, not zero load. Judge
  freshness by the snapshot timestamp, not the envelope HTTP timestamp.
- If JSON parsing rejects unknown fields, allow the two new identity fields.
- With `--enable-metrics`, scrape `/metrics` on the API port for the native
  Prometheus multiprocess registry, including HTTP counts, generation counts,
  TTFT, inter-token and end-to-end histograms. Listeners sharing one
  `PROMETHEUS_MULTIPROC_DIR` expose the same aggregate: scrape one target per
  registry to avoid counting it multiple times. Continue polling each rank's
  `/v1/loads` separately for routing. With metrics disabled, `/metrics` is 404.

No Aigate code change is required if it already implements those contracts.
If it registers only two base ports, routes through DPC, copies the D rank to P,
or treats the local load rank as global, its discovery/routing must change.
The bundled sgl-model-gateway is a separate implementation, not proof about
Aigate; its PD router uses each worker's base_url and per-worker preparation.

## Additional verification

Run the dependency-light suite (requires pyzmq for fault injection):

    python -m unittest discover -s test/registered/unit/managers -p test_rust_pd_integration_cpu.py

Run real CPU collectives on Linux with PyTorch installed, no model required:

    torchrun --standalone --nproc-per-node=16 scripts/verify_rust_dp_collectives.py
    torchrun --standalone --nproc-per-node=16 scripts/verify_rust_dp_collectives.py --cp-size=4
    torchrun --standalone --nproc-per-node=16 scripts/verify_rust_dp_collectives.py --cp-size=16

On BOTH hosts, substitute their respective node rank 0/1 and node0 address:

    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 --master-addr=NODE0 --master-port=29501 scripts/verify_rust_dp_collectives.py

From rust/, run the full server unit suite after rebuilding, including ingress
HTTP rejection, configuration identity, bootstrap and load rendering tests:

    cargo test --locked -p sglang-server

These checks do not certify KV DMA safety. Complete the target GPU/HCU tests
above, including all 16 ingresses, nonzero-rank cancellation, full ring,
collector outage, P failure, repeated pairing and source-drain accounting.
Record TTFT/ITL/throughput versus Python before production rollout.


Legacy gateway compatibility: both native GenerateBody and ChatCompletionBody
accept `data_parallel_rank` as a serde alias for `routed_dp_rank`; it receives
the same ingress check. Send one spelling, not both (duplicate fields are
rejected). This covers the bundled gateway's older request field, but does
not change its base_url addressing: each registered worker still needs the
actual selected ingress URL, not a shared node base port.

Initial DP ingress validation: 16 dependency-light CPU regressions and the
snapshot fixture/schema check passed; Python compilation and diff whitespace
checks passed. Real Gloo/GPU/HCU acceptance remains separate from CPU checks.

## DeepSeek V4 Chat without a tokenizer template

This backports the missing-template fallback from upstream PR #38939
(`07e1918924`) and adapts the DSV4 request normalization from PR #42664
(`45abea0269`). `dynamo-renderer` is pinned to 5.1.2 with the corresponding lock
updates. The existing loads implementation also now declares its missing
`rustix` dependency explicitly.

Python resolves the DSV4 encoder and checkpoint reasoning-effort profile using
the same helpers as `OpenAIServingChat`. It passes those values, HF model type,
BOS id, default template kwargs and the existing thinking/effort environment
defaults to Rust through `ModelConfig`. Update Python and rebuild the Rust
extension together. There is no new environment variable: keep
`SGLANG_RUST_SERVER=1`. DSV4 needs no fabricated `chat_template` in the model
directory and no `--chat-template`; its native encoder takes precedence as it
does in Python. Startup reports `loaded native DeepSeek V4 chat encoder`.

The Chat endpoint keeps the complete body for native rendering, including
`chat_template_kwargs`, `reasoning`, `task`, continuation and message-level
tools. SGLang's thinking/effort precedence, tool serialization and final
assistant handling are preserved. A continuation is tokenized separately and
loses at most one leading BOS. The reasoning parser receives the prompt's
initial state for both streaming and unary responses. PD room/pair/rank fields
continue through the existing validated routing path.

Aigate needs no template code if it forwards Chat JSON. It must preserve these
fields and send matching messages/tools/thinking/effort to P and D. Keep both
sides on the same Python/Rust build, checkpoint, profile and default kwargs.
The existing per-rank discovery and PD pairing requirements still apply.
The private Aigate code has not been audited here.

Validation commands (Linux, from the repository root):

```bash
cd rust
cargo test --locked -p sglang-server --lib
cargo build --locked -p sglang-server
cd sglang-server
PYTHONPATH=../../python python tests/scripts/generate_dsv4_parity.py \
  tests/fixtures/parity/deepseek-v4-flash-0731.json \
  --model-path /work/models --reasoning-effort-profile official \
  --output tests/fixtures/parity/deployment.json
cargo test --locked -p sglang-server dsv4_deployment_token_parity -- --ignored --nocapture
```

The fixture generator calls the actual Python serving method and tokenizer; it
does not use Rust output as its oracle. Use the deployment's actual profile
override, or omit the flag to detect it from the checkpoint. The deployment
fixture is ignored by Git and is required by the explicitly ignored token test.

Validation on 2026-11-05 used an isolated temporary build directory in the
existing Linux image, Rust 1.99.1 and the deployment's `/work/models` tokenizer
(resolved profile `official`). The server suite passed 266 tests; the separate
token test checked 41 cases: 36 successful requests matched prompt text and
token IDs, 3 matched Python rejection, and 2 retained upstream `known_gap`
markers. Those two cases serialize floating-point JSON in tool schemas/history
differently (`1e-6` versus Python's `1e-06`); they are not claimed as exact
parity. The test fails if an unmarked case diverges or a marked gap disappears.
HTTP tests cover PD rank 9, preserved bootstrap identities, and both streaming
and non-streaming reasoning. No live service was restarted or replaced by
these checks; an image rollout and real Aigate-to-PD generation test remain
deployment acceptance steps.

## Bootstrap heartbeat, cancellation and CP8 failure recovery

The dedicated bootstrap port serves `GET /health` with HTTP 200 and body `OK`
as soon as it can accept heartbeats. Use `/route` to check rank registration
and the main API port's health check to check scheduler readiness.

Aigate can cancel a single request on its owning ingress with
`POST /abort_request` and JSON `{"rid":"<client RID>"}`. HTTP 200 acknowledges
the cancellation request. The endpoint matches active client IDs exactly,
forwards their unique internal IDs, and ignores a repeat after completion.
Missing/blank IDs and `abort_all:true` return 400. Requests still tokenizing
cannot subsequently be submitted after their cancellation is processed.

On P, a rank that finishes failed-transfer cleanup keeps the request in the
inflight queue until all CP/TP peers vote that cleanup has completed. The
source-drain checks and the queue identity/shape checks remain active.

The reproducible fault fixture is `test/manual/disaggregation/cp8_transfer_replay.py`.
Use it only on an isolated P, with its test sender installed as described in
the module. On DSV4 Flash with eight HCUs, the original P exited with
`count=[0,1] phase=[2,2]` when one rank drained ahead of its peers. With the
patch, an injected transfer failure returned 500 and explicit cancellation
returned 499. All eight ranks acknowledged source release; the next generation
after each fault returned 200 on the same P process. These requests use the
real HTTP, model and scheduler paths, with a controlled fake KV sender in place
of physical RDMA.
