# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Serving OpenAI traffic through AIgate

Build the Rust extension and install the Python package from the same checkout.
Set `SGLANG_RUST_SERVER=1` on the engine and use its usual
`python -m sglang.launch_server` command. An image with the matching
extension already installed can also set `SGLANG_RUST_BUILD_MODE=never`.
To roll back an instance, drain its active streams and restart it with the Rust
server switch unset.

The versioned and unversioned tokenize/detokenize routes have two dedicated
worker pools. A declared body of at most 32 KiB uses the short pool. Larger
bodies and requests without Content-Length use the long pool (8 MiB maximum).
Each pool defaults to one OS worker and eight outstanding requests, including
queued jobs, uploads and buffered replies. Configure these per listener with
`SGLANG_RUST_UTILITY_WORKERS` and `SGLANG_RUST_UTILITY_CAPACITY`. Each utility
call accepts at most 1024 batch items and emits at most 16 MiB of JSON.
Capacity exhaustion returns 503, size limits return 413 and the 60-second
preparation deadline returns 504. Chat and native generation have a separate
preparation pool sized by the tokenizer worker count, eight slots per worker,
and a 64 MiB input limit. Streaming generation keeps its existing detokenizer
and scheduling pipeline. With metrics enabled, the metrics listener exposes
`sglang:rust_http_in_flight` and `sglang:rust_http_rejected_total` by lane and DP rank.

AIgate can post OpenAI messages to `/v1/tokenize` and read `count`. For
DeepSeek V4/V4.1, this route and `/v1/chat/completions` use the same renderer
and encode path. For a multimodal V4.1 checkpoint serving text PD traffic, use
`--language-model-only` and the matching `deepseekv41` parser when enabling tool
calls. Send the same messages, tool definitions and template options
on both requests. In PD mode the P/D requests use one choice, a shared request
ID and the same bootstrap room and pair key. Advertise the configured prefill
bootstrap port separately from each engine's HTTP port.

Each attention-DP listener serves its own `/v1/loads` snapshot.
`loads[0].dp_rank` is listener-local zero. Use `global_dp_rank` and
`global_dp_size` for topology. In the two-node, eight-listeners-per-node
decode layout, both hosts use ports `base_port` through `base_port + 7`.
The discovery helpers in `scripts/render_aigate_dp16_registrations.py` and
`scripts/reconcile_aigate_dp16.py` describe this layout. The reconciler
requires an explicit `--apply` to change Kubernetes resources.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.
