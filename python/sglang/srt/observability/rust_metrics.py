"""Embedded Rust frontend bridge to the scheduler's Prometheus multiprocess files."""

import msgspec


class RustMetricsExporter:
    def __init__(self, server_args):
        # Import after the engine sets PROMETHEUS_MULTIPROC_DIR, as in FastAPI.
        from prometheus_client import CollectorRegistry, generate_latest, multiprocess

        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.observability.metrics_collector import (
            STAT_LOGGER_ROLE_TOKENIZER,
            TokenizerMetricsCollector,
            resolve_collector_class,
        )
        from sglang.srt.utils.common import create_http_metric_collectors

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        self.render = lambda: generate_latest(registry)
        self.http_requests, self.http_responses, self.active, self.keys = (
            create_http_metric_collectors()
        )
        labels = {
            "model_name": server_args.served_model_name,
            "engine_type": DisaggregationMode.to_engine_type(
                server_args.disaggregation_mode
            ),
        }
        if server_args.enable_priority_scheduling:
            labels["priority"] = ""
        labels.update(
            dict.fromkeys(server_args.tokenizer_metrics_allowed_custom_labels or (), "")
        )
        self.request_labels = set(
            server_args.tokenizer_metrics_allowed_custom_labels or ()
        )
        self.custom_header = (
            server_args.tokenizer_metrics_custom_labels_header
            if self.request_labels
            else None
        )
        if server_args.enable_priority_scheduling:
            self.request_labels.add("priority")
        labels.update(server_args.extra_metric_labels or {})
        self.labels = labels
        collector = resolve_collector_class(
            server_args, STAT_LOGGER_ROLE_TOKENIZER, TokenizerMetricsCollector
        )
        self.collector = collector(
            server_args=server_args,
            labels=labels,
            bucket_time_to_first_token=server_args.bucket_time_to_first_token,
            bucket_inter_token_latency=server_args.bucket_inter_token_latency,
            bucket_e2e_request_latency=server_args.bucket_e2e_request_latency,
        )

    def __call__(self, batch, scrape):
        for event in msgspec.msgpack.decode(batch):
            self._record(event)
        return self.render() if scrape else b""

    def _record(self, event):
        kind = event["type"]
        if kind in ("http_start", "http_end"):
            labels = {"endpoint": event["path"], "method": event["method"]}
            if kind == "http_start":
                self.http_requests.labels(**labels).inc()
                self.active.labels(**labels).inc()
                if event["key"]:
                    self.keys.inc(event["key"])
            else:
                if event["status"] is not None:
                    self.http_responses.labels(
                        **labels, status_code=str(event["status"])
                    ).inc()
                self.active.labels(**labels).dec()
                if event["key"]:
                    self.keys.dec(event["key"])
            return

        labels = self.labels
        if event.get("labels"):
            labels = labels | {
                k: v for k, v in event["labels"].items() if k in self.request_labels
            }
        if kind == "first":
            self.collector.observe_time_to_first_token(
                labels, event["seconds"], stream=event["stream"]
            )
        elif kind == "interval":
            self.collector.observe_inter_token_latency(
                labels, event["seconds"], event["tokens"]
            )
        elif kind == "finished":
            usage = event["usage"] or {}
            self.collector.observe_one_finished_request(
                labels,
                event["prompt"],
                event["tokens"],
                usage.get("cached"),
                event["seconds"],
                event["grammar"],
                cached_tokens_details=usage.get("details"),
                spec_verify_ct=usage.get("spec_verify", 0),
                is_streaming=event["stream"],
            )
        elif kind == "aborted":
            self.collector.observe_one_aborted_request(self.labels)
        else:
            raise ValueError(f"Unknown Rust metric event: {kind}")
