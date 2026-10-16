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
- Snapshot publication must use ShmLoadSnapshotWriter. ZMQ-only publication
  (including some multi-node layouts) returns 503 and needs additional work.
- Never interpret 503, empty loads, or stale snapshot timestamps as idle.
- Validate Aigate's private schema and polling policy against real responses.
- PD bootstrap uses the prefill API listener.

## Validation

Run the lightweight schema check with Python and msgspec:

    python scripts/verify_rust_loads_backport.py

On the target Linux build environment:

    cd rust
    cargo test --locked -p sglang-server api_server::loads

Then run the Python load-snapshot tests and PD end-to-end tests, and poll
each DP endpoint during generation. Check running, prealloc and transfer
queues and snapshot freshness. Exercise cancellation and streaming output.

The backport has not been compiled on Linux or tested on GPU hardware.
Do not treat this branch as production-qualified until those checks pass.
It is a scoped loads backport, not a wholesale merge of latest upstream.
