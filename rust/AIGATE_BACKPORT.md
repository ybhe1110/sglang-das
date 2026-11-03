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

The backport has not been compiled on Linux or tested on GPU hardware.
Do not treat this branch as production-qualified until those checks pass.
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

The private Aigate implementation is not present in this repository; the
following are acceptance criteria, not a claim that its implementation passes.

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
- Keep cancellation bound to the owning ingress and propagate client disconnects
  to downstream requests. A P failure must cancel its paired D request.
- Treat missing/503/stale load snapshots as unavailable, not zero load. Judge
  freshness by the snapshot timestamp, not the envelope HTTP timestamp.
- If JSON parsing rejects unknown fields, allow the two new identity fields.
- `/metrics` is still absent. `/v1/loads?format=prometheus` is only load metrics;
  do not advertise it as full Python frontend metric compatibility.

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

Local validation for this change: 16 dependency-light CPU regressions and the
snapshot fixture/schema check passed; Python compilation and diff whitespace
checks passed. Rust tests and real Gloo/GPU/HCU runs remain unexecuted on the
Windows editing host (no Rust toolchain, PyTorch or configured WSL).
