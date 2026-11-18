"""Per-PID utility budgets and latency; initialized after metrics setup."""

import os


class UtilityMetrics:
    def __init__(self, registry=None):
        from prometheus_client import Counter, Gauge, Histogram

        kwargs = {} if registry is None else {"registry": registry}
        labels = {"worker_pid": str(os.getpid())}

        def metric(cls, name, description, **extra):
            return cls(
                "sglang:utility_" + name, description, labels.keys(), **kwargs, **extra
            ).labels(**labels)

        self.budgets = {
            name: metric(
                Gauge, name, "Tokenizer utility " + name, multiprocess_mode="livesum"
            )
            for name in (
                "work",
                "input_bytes",
                "responses",
                "output_bytes",
                "cancelled_work",
            )
        }
        self.rejected = metric(
            Counter, "rejected_total", "Utility admission rejections"
        )
        self.timed_out = metric(Counter, "timeouts_total", "Utility HTTP deadlines")
        self.queue_seconds = metric(
            Histogram, "queue_seconds", "Utility admission to execution"
        )
        self.execution_seconds = metric(
            Histogram, "execution_seconds", "Utility synchronous execution"
        )
        self.loop_lag_seconds = metric(
            Histogram, "loop_lag_seconds", "HTTP worker event-loop lag"
        )

    def observe(self, snapshot):
        for key, value in snapshot.items():
            self.budgets[key].set(value)
