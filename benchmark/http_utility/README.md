# HTTP tokenizer utility isolation

This branch implements P0/P1 of the 2026-11-18 Python HTTP-blocking design on
`release/20261106_v0.5.22-dsv4.1` (`b55eb27730`). It does not change generation,
the scheduler, KV ownership, PD/EPD protocols, or GPU kernels.

## Execution and ownership

`/tokenize`, `/v1/tokenize`, `/detokenize`, and `/v1/detokenize` use a dedicated
per-HTTP-worker service. The identity conversion bypasses generation's tokenizer
executor. Chat conversion, template rendering, encode/decode, result-model
construction, and JSON serialization run on a utility thread. Existing helper
branches and error mappings are preserved. FastAPI still parses/validates the
bounded JSON request on its event loop. On validation failure the four utility
routes return HTTP 400 with a flat error body of at most 256 bytes. The summary
uses the first diagnostic and the error count, omitting input and context objects.

Detailed utility request logging (levels 2/3) runs in the admitted work
item after batch and item limits pass and before tokenization. Only whitelisted
request headers and a shallow logger configuration snapshot are captured on the
loop. Payload dumping, formatting
and redaction run on the utility thread. A dedicated writer in each HTTP process
writes request logs for both generation and utilities to their configured sinks.
Generation checks log capacity before submitting payload formatting to a separate
single-thread executor. The request awaits formatting so its payload remains
stable, then continues while the writer handles the sink. For token-ID input,
the generation tokenizer executor decodes the text before generation resumes,
even if the log record was dropped. When tokenizer offload is disabled, the HTTP
worker uses a private decoder for this step. Changes to the file target open it
off the event loop before publishing the new logging configuration.
The utility worker waits for an accepted record to finish writing and retains
its request reservations.
The FIFO holds up to 128 waiting records and 8 MiB of Python message strings,
including the active write in the byte limit. On overflow the writer drops the
new record and counts the loss. Shutdown drains accepted records with a one-second
join budget. A sink still blocked at process exit can leave records unwritten.
Work cancelled before execution skips logging as well as tokenization. If a
request is cancelled during logging, the worker skips tokenization when the
log sink returns.

Each utility thread owns its tokenizer, template manager and Chat handler/cache.
During startup, each thread reloads the built-in Tiktoken or patched Mistral
tokenizer from its resolved local file so its native backend and bound methods
belong to that thread. Other tokenizers and the template manager use deep copies.
A custom tokenizer that cannot be independently copied fails startup explicitly.
The utility backend captures text-serving metadata, not engine sockets or GPU
state. Runtime tokenizer/template replacement requires restarting HTTP workers.

ASGI middleware admits requests **before** reading their bodies. Admission
reserves one work slot, a maximum input byte allowance, one response slot, and a
maximum output byte allowance. It rejects immediately instead of creating an
unbounded admission-waiter queue. Input bytes are a conservative cost proxy that
also covers Chat schemas and extra fields; input values/IDs and batch counts have
additional limits. For a validated detokenize request, the item check counts a
flat ID list by length. For a batch, it counts each row and adds its length.
The outer list counts once even when empty. Only a list of rows is subject to
the detokenize batch limit. Other request fields follow the usual item counting rules.
Total input-item count is bounded by
`utility_max_admitted * utility_max_input_items`. These are bounds, not a
model-calibrated prediction of CPU time or peak Python heap usage.

All admitted requests wait in one FIFO. When a worker becomes free it takes
the oldest waiting request and executes it with its own backend. A cancelled
queued entry stays charged until physically dequeued. A cancelled
running helper stays charged until it returns. The pool's Future is deliberately
not cancelled, since cancellation alone does not remove its physical queue item.
Work slots return on physical completion. Input reservations also cover FastAPI's
request retention through response send. Output reservations shrink to serialized
size (with a 256-byte error-response floor) at completion and remain charged until ASGI send completes/fails or the
response is discarded. Output size is checked after serialization, so temporary
encoder/output-model allocations still depend on the bounded input and tokenizer.

The middleware is the sole transport receive owner after buffering the body.
Disconnect cancels the application task; it does not pretend to interrupt native
encode/decode. Shutdown follows the bounded drain policy below.

Child-task cleanup is retained by the ticket and shielded from repeated HTTP
owner cancellation. If the owner exits while cleanup is still pending, a completion
callback closes the ticket only after both application and disconnect watcher
finish. This prevents orphaned leases without releasing live response buffers.

### Shutdown policy

1. Stop service/executor admission and cancel pending replies. Keep physical
   queue entries and their reservations until they actually complete.
2. Allow `utility_shutdown_timeout` seconds (default 10) for HTTP owners and
   physical work to finish and executors to join off the event loop.
3. If the grace period expires, request cancellation of remaining HTTP owners,
   log the outstanding work/input/response/output budgets, and raise `TimeoutError`
   from `aclose()` into lifespan shutdown. Stop loop-lag sampling on either path.
   Do not clear tickets, force counters to zero, or report successful shutdown.
4. While the event loop remains alive, late HTTP cleanup and worker completion
   still release reservations exactly once. A later `aclose()` can finish draining.

This bounds the service's cooperative drain wait, not the lifetime of a stuck
native thread or a cancellation-resistant ASGI task. `shutdown(wait=False)` does
not terminate running threads, and the ASGI server's own pre-lifespan drain has
its own timing. Deployment supervision must enforce a final process termination
deadline, allowing time for both the ASGI drain and this grace period. The library
does not call `os._exit()` or kill the shared serving process. Full server/supervisor
shutdown behavior remains a deployment acceptance test.

## Configuration and new HTTP policies

All limits are **per HTTP worker**, registered in the `serving` configuration
namespace and checked during server-argument resolution.

| CLI parameter | Default |
| --- | ---: |
| `--utility-workers` | 1 |
| `--utility-max-admitted` | 8 |
| `--utility-max-input-bytes` | 2097152 (2 MiB per request) |
| `--utility-max-total-input-bytes` | 16777216 (16 MiB) |
| `--utility-max-input-items` | 262144 (includes JSON values and token IDs) |
| `--utility-max-batch-size` | 1024 (texts, token lists, or messages) |
| `--utility-max-output-bytes` | 8388608 (8 MiB per response) |
| `--utility-max-total-output-bytes` | 67108864 (64 MiB) |
| `--utility-timeout` | 60 seconds, including body, queue, work, and send |
| `--utility-body-timeout` | 15 seconds |
| `--utility-shutdown-timeout` | 10 seconds |

Worst-case reservations can reject a small request even when actual bytes in
flight are lower than the configured ceiling. This is intentional conservative
admission. Input limits do not depend on `Content-Length`; chunked bodies are
counted as received. Overload returns a flat 429 error with `Retry-After: 1`;
size/item limits return 413; unavailable/shutting-down service returns 503;
deadline expiry before headers returns 504. An expired send after headers is
cancelled without appending a second response. Disconnect sends no synthetic 499.
Error sends have a separate bounded wait of at most `utility_body_timeout`.
The output limit must be at least 256 bytes to accommodate bounded error envelopes.
Utility error messages that would exceed that envelope use a short fallback while
preserving the status code. Error sends check the remaining reservation explicitly;
they do not reuse the interrupted success response's buffered headers/byte count.
Duplicate `http.response.start` messages raise a protocol error, both before and
after the first headers have been forwarded.
Admission rejection sends have a one-second timeout and never retain a request body.
Capacity fields require positive integers (not booleans or floats). Duration
fields accept positive finite integers/floats, but reject booleans and strings.

**New compatibility policy:** compressed bodies (`x-body-compressed` or a
non-identity `Content-Encoding`) are rejected with 415 on utility endpoints.
The existing optional global decompressor can allocate an unbounded expanded
body before routing, so it must not run ahead of utility admission. Other
endpoints retain their existing compression behavior. Clients of utilities
should send uncompressed JSON and adjust large-request limits explicitly.

## Multi-worker rollout and metrics

Start with one HTTP worker, then use the existing `--tokenizer-worker-num 2` (and
later higher values within measured CPU/RSS budgets). Each HTTP process creates
its service inside lifespan; no executor is inherited from the parent. Existing
API-key, native-gRPC, HTTP/2, and topology restrictions still apply. Extra HTTP
workers do not create extra model ranks. Allow CPU for generation tokenization,
tokenizer internal parallelism, and scheduler processes in addition to utilities.

With `--enable-metrics`, `sglang:utility_*` metrics carry `worker_pid`: work/input
reservations, pending responses/output bytes, cancelled work, rejection/deadline
counters, queue/execution time histograms, and HTTP loop lag. Startup logs report
the tokenizer class, `is_fast`, PID, and actual transformers/tokenizers versions.

Threads remove the direct synchronous HTTP-loop call, but do not guarantee a
latency ceiling for Python-heavy or GIL-holding native code. A separate Utility
process backend, short/long scheduling, aging, and tenant quotas are conditional
or future phases, not implemented here. Measure real model encode/decode/template
behavior before deciding whether process isolation is necessary.

## Validation

In an installed SGLang test environment, run each file in its own process:

```bash
python test/registered/unit/entrypoints/test_utility_executor.py
python test/registered/unit/entrypoints/test_serving_tokenize_utility.py
```

Tests exercise physical-queue cancellation, running failures, completion/cancel
races, response ownership under slow/failed sends, disconnects, input limits,
deadlines, shutdown, aliases, unchanged error/shape semantics, Chat argument
forwarding, private tokenizer state, and a real locally constructed HF fast
tokenizer without downloading weights. The template spy checks utility argument
forwarding; it does not replace a deployment test of the target model's templates.

`http_mixed_probe.py` is the supplied design's probe, copied without behavioral
changes. Use independent baseline and modified deployments, identical model,
CPU budget, text/IDs, and generation settings. Prepare legal model-specific
payloads before the measurement window:

```bash
python benchmark/http_utility/http_mixed_probe.py \
  --base-url http://127.0.0.1:30000 \
  --tool-path /v1/tokenize --tool-body long.json --short-body short.json \
  --stream-path /v1/chat/completions --stream-body stream.json \
  --inject-after-events 10 --tool-count 3 --tool-concurrency 3 \
  --duration 20 --interval-ms 20 --output utility_fixed.jsonl
```

Repeat for detokenize and Chat tools, one/multiple HTTP workers, and sustained
overload; use at least five warmed rounds. Compare per-PID loop lag, p99 probe
latency, SSE gaps, TTFT, generation throughput, CPU/RSS, refusals, and budget return
to baseline. The probe's client dispatch lag/skips must be included in analysis.
No GPU/DCU mixed-load latency or full production model compatibility is claimed
by the CPU unit tests. Roll back by deploying the parent release version; never
replay a generation request that has already streamed output.

### Locate GET latency within a long detokenize request

Start the instrumented HTTP workers with `SGLANG_ENABLE_UTILITY_PHASE_PROBE=1`
in their service environment. Timing headers are emitted only for explicitly
marked successful requests. The client needs Python 3.10+ and `httpx` (tested
with 0.28.1). Run from the repository root on 129.13 or a host with a direct
connection to that Pod. Use the same uncompressed request body as the three-way
comparison, `current-evidence/detokenize.json` from its replay ZIP (250,000 IDs,
SHA-256 `69cc82578c6ae01b488fd8901815ccb76251be34f84e9c6925d2c024eabb2940`).
On 129.1 the corresponding file is under `/tmp/http-retest-20261123-1916/`.

```bash
python3 benchmark/http_utility/detokenize_phase_probe.py \
  --base-url http://10.113.129.13:30000 \
  --body /tmp/http-retest-20261123-1916/detokenize.json \
  --offsets-ms 0 25 50 75 100 125 150 200 250 --rounds 5 \
  --output detok-129-13-phase.jsonl
```

The script uses one warm-up and one long POST per offset per round (45 measured
POSTs here). It shuffles offsets within each round and sends both GETs over
separate, prewarmed HTTP/1.1 connections to the POST worker. A second worker
receives the same GET pair as a control. Each response verifies instance ID,
PID and client socket port. A missing header, connection change, error response
or inconsistent clock bounds stops the run. Use `--single-worker` only for a
one-worker deployment. Set `SGLANG_API_KEY` on the client if required. Each run
needs a fresh output filename.

Read the last JSONL record for the verdict per route. `trial.stages_ms` breaks
down body receipt, JSON parsing, dispatch/validation before submission, work
queue, item accounting, normalization, the tokenizer's flat-list `decode`,
serialization and return to the HTTP loop. Raw timestamps are in nanoseconds.
Each `sample` reports the actual client dispatch offset, GET RTT, baseline on
that connection (median of the idle GETs immediately before and after),
send-to-ASGI waiting bounds, ASGI-to-handler and handler times, and overlap
with decode before the handler and across the whole RTT. `/v1/loads` handler
time includes the awaited load query. The server timestamps headers immediately
before handing them to ASGI; the client records completion of the full response.

Client and server monotonic clock origins may differ. Each trial bounds their
offset using dispatch/ASGI entry and header preparation/client completion on
the pinned worker. The intersection gives the clock uncertainty reported in
the samples. For GETs at least `--slow-ms` (default 10 ms) above baseline, the
summary sums the excess RTT and bounds the fraction coincident with decode.
The lower bound allows the entire baseline to have overlapped decode. A lower
bound above 50% means `majority_overlaps_decode`; an upper bound below 50%
means `majority_outside_decode`. At least three distinct rounds with slow
samples are required, otherwise the verdict is `inconclusive`. These bounds
express timing uncertainty, not statistical confidence. Compare the per-offset
medians with the other worker and the GET handler duration when interpreting
the result. Temporal overlap alone cannot distinguish GIL contention from CPU
scheduling or native runtime activity.

### Local validation (2026-11-18)

The initial 22 tests passed in an isolated Windows CPU harness: 17 executor/ASGI lifecycle
tests and 5 serving/HTTP/tokenizer tests. The real HF tokenizer checks used the
repository-pinned `transformers==5.19.0` and `tokenizers==0.23.2`. No weights were
downloaded. CLI field registration and parsing, Prometheus metric creation and
budget return, and Python syntax/static checks were also verified. An AST
comparison confirmed that both original synchronous utility method bodies were
preserved; a regression probe confirmed execution moved from the event-loop
thread to a worker thread.

The local environment lacks PyTorch and the complete SGLang runtime. Temporary
harnesses bypassed heavyweight package initialization and substituted the test
base and unused engine imports; they loaded the real protocol, serving base,
utility handlers, executor, middleware, and tokenizer implementations. Chat
argument forwarding used a template spy. These results do **not** constitute a
full application startup, target-model Chat-template compatibility test, or
HCU mixed-load acceptance run. Run the commands and deployment probes above in
the complete serving environment before rollout.

Static review also found the pre-existing duplicate
`validation_exception_handler` function name in `http_server.py` (Ruff F811).
This unrelated baseline issue was left unchanged.

The review follow-up passed all 27 CPU tests (22 executor/ASGI plus 5 serving
tests). Added coverage exercises shutdown grace expiry with blocked native work
and a sender delaying cancellation, late cleanup after shutdown failure, a 504
replacing a tiny success response, invalid configuration types, and duplicate
response starts before/after headers. The response-budget, type-validation and
duplicate-start regressions were also run against the pre-fix implementations
and failed there. The same isolated-harness limitations above apply.

The 2026-11-22 follow-up passed 29 CPU tests (23 executor/ASGI plus 6 serving
tests). New regressions cover shutdown cancellation during disconnect cleanup,
eventual budget return after the HTTP owner exits, and level-2 text / level-3 JSON
logging with a blocked sink, worker-thread payload dumps, whitelisted headers,
and exclusion of unapproved headers. Both regressions fail against the previous
commit. The local harness loads the actual request logger and formatting code;
it substitutes the unavailable distributed runtime and unused watermark config
types while loading the redaction function from repository source. Full HCU
serving, model-specific Chat initialization and mixed-load latency remain outside
these CPU checks.
