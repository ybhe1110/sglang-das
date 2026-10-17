# DPC load snapshot drain benchmark

`bench_dpc_drain.py` measures the control-plane cost and visibility latency of
the DPC ZMQ-owner fix. It does not start a model server.

The `control`, `dispatch`, and `transport` modes import the SGLang runtime and
should be run in the same Linux/container environment used by the service.
The repository checkout alone is not sufficient on Windows because the server
runtime depends on Linux process and accelerator modules.

Run the control-plane measurements in the SGLang environment:

```bash
python benchmark/dp_load_snapshot/bench_dpc_drain.py \
  --mode control --duration 30 --output-json control.json

python benchmark/dp_load_snapshot/bench_dpc_drain.py \
  --mode dispatch --samples 10000 --dp-size 16 --output-json dispatch.json
```

The transport benchmark compares the old dispatch-triggered behavior with the
new periodic drain. Use a short dispatch interval for a smoke test; use the
observed idle interval when reproducing the production problem.

```bash
# New behavior: owner polls every 20 ms.
python benchmark/dp_load_snapshot/bench_dpc_drain.py \
  --mode transport --drain-mode periodic --drain-interval 0.02 \
  --samples 100 --output-json transport-periodic.json

# Old behavior model: owner only polls when a dispatch arrives.
python benchmark/dp_load_snapshot/bench_dpc_drain.py \
  --mode transport --drain-mode dispatch --dispatch-interval 1.0 \
  --samples 100 --output-json transport-dispatch.json
```

The transport result reports source-timestamp-to-SHM visibility latency. The
periodic result should be bounded by roughly one drain interval plus transport
overhead; the dispatch-triggered result follows the dispatch interval.

For real Decode output speed, run the same command against the same model,
hardware, DP size, request mix, and concurrency before and after the fix:

```bash
python benchmark/dp_load_snapshot/bench_dpc_drain.py \
  --mode decode --host 127.0.0.1 --port 30000 \
  --concurrency 16 --requests 64 --output-tokens 128 \
  --output-json decode-after.json
```

The Decode result contains aggregate output tok/s, per-request Decode tok/s,
TTFT, and inter-token latency. The script uses `usage.completion_tokens` when
the streaming endpoint returns it; otherwise it reports an explicit chunk-count
estimate. Compare the JSON files from the pre-fix and post-fix servers rather
than mixing runs from different hardware or model processes.
