# L3 storage fault / no-L1-fallback comparison

## Baseline and scope

- Baseline: `release/20260825_v0.5.18` at `cb4cc11864940a862b558ebe38eec5a470edb4c1`
  (the local branch referred to as release/0.5.18).
- Experiment: `codex/release-0.5.18-no-l1-fallback`, branched directly from that SHA.
- Disable **storage fault -> cache miss -> L1 recomputation**. Keep successful
  cache misses, full L1 hits, request-private slots, post-load radix publication,
  FULL/SWA ownership exclusion, session quarantine, and P/D source-drain.
- Do not disable strict pool checks or add estimated `uncached` counts.
- This is a diagnostic branch, not a proposed production availability policy.

| Stage | Baseline | Experiment |
| --- | --- | --- |
| Full L1 hit | Use local KV | Same, including identity preflight |
| Healthy L3 reply says absent | Recompute missing tokens | Same |
| Lookup RPC/worker timeout, exception, malformed reply, open circuit | Return miss and recompute | Fixed-size TP/CP verdict, then fatal exit on every participating rank; no mask collective |
| PP1 admitted hit becomes missing, or revalidation fails | Release temporary slots, recompute | MIN verdict, release not-yet-submitted slots, fatal exit |
| PP>1 has already agreed on prefix | Preserve shape, report load failure | Same; no independent per-stage recomputation |
| Active load fails | Agree failure, abort request, drain and clean up | Same |
| Session cleanup/DMA ownership unknown | Retain/quarantine; do not recycle | Same |
| Write-back fails | FIFO failed completion releases tree pin | Same; later lookup sees circuit failure |

The metadata worker, cooldown, bounded waits and device capture remain enabled.
Simply setting the old `SGLANG_MOONCAKE_L1_FALLBACK=0` would also bypass metadata
timeouts and would not disable revalidation fallback. This branch ignores that
environment variable. Startup logs explicitly report `L1_fallback=False`.

Lookup adds a one-element verdict collective after the existing identity/geometry
preflight. **Do not mix baseline and experimental ranks in a TP/CP/PP deployment.**
Upgrade and rollback the full communicating service group.

## What the existing evidence establishes

The 2026-10-29 19:39 P pool report has a FULL deficit of 1,310,464 tokens
(5,119 pages at page size 256) and a SWA deficit of 3,072 tokens (12 pages).
The earlier observed failed load was 406 pages; it does not alone explain the
FULL deficit. The idle report showed private_loads/taken_load_batches/inflight=0.
The exact first missing owner has not been identified.

The 22:01–22:06 read-only collection separately established host-global OOM on
129.10, with two Mooncake clients allocating fixed 256 GiB and 150 GiB segments,
plus D workers and other processes on a 503 GiB host. A native client restart is
not evidence that the post-load radix insertion caused a client memory leak.
Fixed segment occupancy and Python KV pool accounting are different resources.

Before A/B inference tests, fix the host RAM budget and confirm D can complete
startup. Otherwise host OOM remains a confounder, independent of fallback.

## Local ownership risks to distinguish from client failure

The post-load/private-slot path was introduced locally in `550a5d924e`;
`352079b9e7` adds FULL/SWA release exclusion, `a6a01c1900` adds partial-session
and release quarantine, and `1ca38d9a3e` preserves private prefix ownership
outside PP. The fault-to-L1 policy is older (`7b6a7a9014`), not inherent to
post-load publication.

- Recompute increases P allocations and write-back work. A stale release interval
  could make cleanup skip newly allocated slots, but the current pre-queue
  fallback clears that interval. A specific missed path still needs a page ledger.
- Successful load -> request -> tree and failed load -> linker ABORT -> request
  cleanup are distinct ownership transitions; FULL and SWA may have different
  physical extents. Previous double-free guards do not prove every later
  chunked/donation/retraction transition correct.
- A source-drain or unknown session may intentionally retain slots. It must
  remain a tracked owner, not become an untracked deficit. This branch preserves
  that safety rule and does not manufacture a native terminal fence.
- A source-only probe reproduced an existing allocation-exception gap in
  `load_back`: FULL component allocation succeeds, a subsequent component's
  `build_external_linker_transfer(LOAD)` raises, and the earlier allocation is
  neither aborted nor registered in `private_loads`. In the probe, one fake
  allocator slot remained allocated while private/pending maps were empty.
  That branch is unchanged for this comparison; the exception normally exits
  the scheduler, so it is not evidence for the observed later idle imbalance.
- `create_storage` registers buffers before returning its new store. A
  registration exception has no explicit close of the partially created store
  on this release path. Whether native destructors reclaim it needs verification;
  initialization retries could otherwise retain native resources. This is not
  proof that a remote Mooncake client's fixed segment or heap leaks.

These are separate hypotheses/coverage gaps, not evidence that `#40896` caused
the measured host OOM. Keep unrelated ownership fixes separate from this A/B diff.

## Local checks

```bash
python test/registered/unit/mem_cache/test_mooncake_l1_fallback.py
python test/registered/unit/disaggregation/test_collective_timeout.py
```

The first suite executes production method bodies with CPU boundary doubles:
metadata worker/circuit, one-rank failure consensus, cold miss vs fault, pre-queue
release ownership and quarantine. It does not emulate native DMA or GPU allocators.
On a dependency-complete DCU test image also run:

```bash
python -m pytest -q test/registered/unit/mem_cache/test_unified_cache_linker.py \
  test/registered/unit/mem_cache/test_unified_cache_linker_load_failure.py \
  test/registered/unit/mem_cache/test_external_linker_recovery_contracts.py
```

## Test-cluster A/B procedure (not executed by creating this branch)

1. Keep the same model, wheel/image versions, PP/TP/CP topology, page size, cache
   capacities, input IDs, concurrency and client retry policy. Record source SHA
   **and the experimental diff**, image digest, Pod UID/container start and node.
   Only use the dedicated ybhe test service; do not fault shared storage clients
   or a shared master that serves unrelated workloads.
2. For A (release) and B (experiment), independently warm the same L3 prefixes,
   reset L1 through a clean group deployment, then run identical normal cold/hot
   and multi-chunk workloads. Include normal end and explicit cancellation.
   Wait for linker work, request cleanup and source-drain to finish and retain
   strict FULL/SWA pool reports.
3. Inject each fault at a controlled boundary: before lookup; between lookup and
   revalidation; after native load starts; during write-back. Prefer a dedicated
   test endpoint/native harness. A metadata exception is different from a valid
   key-absent reply. If a client failure is hidden by replica selection or returns
   a valid absent reply, B can still recompute and that round does not isolate
   the removed fault-fallback path. Record the actual branch taken.
4. For asynchronous-load-only regression, the existing test-image settings
   `SGLANG_TEST_LINKER_LOAD_FAILURE_PROB=1` and
   `SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS=0` inject before a layer range-get call.
   They **do not** exercise lookup/revalidation fallback or prove cancellation
   of an outstanding DMA. The ReadPlan route also needs its own native fault
   harness; do not count this injection as ReadPlan coverage.
5. After a fault, continue new requests for A to exercise L1 recomputation while
   old owners drain. B should fail the affected scheduler at a metadata verdict,
   or retain the existing request-abort path for already admitted I/O. Save logs
   before automatic restart. Compare equivalent completed work and fault counts,
   not just wall time; B may process fewer requests.
6. Repeat outage/recovery rounds. With actual hardware, test 1M-context inputs
   only after shorter cases pass; reserve output tokens within context length.

Read-only snapshots before fault / after fault / after cleanup / after old PID
exit: Pod status, current and previous logs, events, node/cgroup RAM and OOM
counters, `rdma resource show`, `ibv_devinfo`, `ss -tanp`, native CQ/outstanding
logs. Access 129.1 through interactive SSH PTY, then issue kubectl commands.
Do not infer native resource recovery from Python shutdown warnings.

## Ownership evidence and interpretation

To identify the first loss, instrument a test-only per-RID ledger at allocator
and ownership boundaries. Record `(Pod UID, rank, RID, request generation,
allocation epoch, FULL/SWA physical page, request row interval, operation,
previous owner, next owner, outcome)` for allocate, prepare, queue, worker terminal,
handoff, free, quarantine and source-drain. A physical page can be reused across
epochs; row positions alone do not identify a physical allocation. At each
checkpoint require one owner, or exactly one successful release for that epoch.
This ledger is required for a conclusive leak diagnosis and is not implemented
by the source-only tests above.

- `accounted = available + evictable + protected + session_held + uncached`
- `delta = accounted - total`; negative is missing accounting, positive can be
  duplicated accounting/free or a non-atomic snapshot. Do not sum rank reports.
- If A leaks but B exits before doing comparable work, the result only localizes
  a fault/recompute window. Process exit can erase evidence; this is **not proof**
  that disabling fallback fixed ownership.
- If B still loses pages on successful loads or admitted-load abort, investigate
  private-slot handoff, FULL/SWA cleanup, chunked checkpoints and source-drain.
- If both variants fail from host-global OOM, the experiment says nothing about
  P's GPU KV slot ownership.
- Unknown native DMA/session outcomes must keep their owner until a trustworthy
  terminal fence or process exit. Do not force-free to make the pool report pass.
