"""Ensure the same Prometheus queries work when switching inference frontends."""

import os
import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=160, stage="base-b", runner_config="1-gpu-small")


def scrape(url):
    response = requests.get(url + "/metrics", timeout=10)
    response.raise_for_status()
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    }


@unittest.skipUnless(is_rust_server_built(), "Rust server extension required")
class TestRustFrontendMetrics(CustomTestCase):
    def test_native_prometheus_queries(self):
        schemas = []
        for mode in ("0", "1"):
            with self.subTest(rust_server=mode):
                process = popen_launch_server(
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                    DEFAULT_URL_FOR_TEST,
                    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    env={**os.environ, "SGLANG_RUST_SERVER": mode},
                    other_args=[
                        "--enable-metrics",
                        "--served-model-name",
                        "metrics-model",
                        "--tokenizer-metrics-allowed-custom-labels",
                        "tenant",
                        "--extra-metric-labels",
                        '{"deployment":"test"}',
                        "--bucket-e2e-request-latency",
                        "0.05",
                        "1",
                        "10",
                    ],
                )
                try:
                    schemas.append(self._check_requests(DEFAULT_URL_FOR_TEST))
                finally:
                    kill_process_tree(process.pid)
        self.assertEqual(schemas[0], schemas[1])

    def _check_requests(self, url, *, model_name="metrics-model", deployment="test"):
        schema = set()
        for stream in (False, True):
            before = scrape(url)
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": [42, 43],
                    "stream": stream,
                    "custom_labels": {"tenant": "probe"},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4,
                        "ignore_eos": True,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            if stream:
                self.assertIn("data: [DONE]", response.text)
            else:
                self.assertIn("meta_info", response.json())
            after = scrape(url)
            labels = dict(
                model_name=model_name,
                engine_type="unified",
                deployment=deployment,
                is_streaming=str(stream).lower(),
                tenant="probe",
            )
            for name in (
                "sglang:num_requests_total",
                "sglang:time_to_first_token_seconds_count",
                "sglang:e2e_request_latency_seconds_count",
            ):
                key = (name, tuple(sorted(labels.items())))
                self.assertEqual(after[key] - before.get(key, 0), 1)
            key = (
                "sglang:http_requests_total",
                (("endpoint", "/generate"), ("method", "POST")),
            )
            self.assertEqual(after[key] - before.get(key, 0), 1)
            for metric in (
                "sglang:time_to_first_token_seconds",
                "sglang:e2e_request_latency_seconds",
            ):
                buckets = {
                    k
                    for k in after
                    if k[0] == metric + "_bucket"
                    and labels.items() <= dict(k[1]).items()
                }
                self.assertTrue(buckets)
                inf_key = (
                    metric + "_bucket",
                    tuple(sorted({**labels, "le": "+Inf"}.items())),
                )
                self.assertEqual(after[inf_key] - before.get(inf_key, 0), 1)
                schema.update(buckets)
        return schema


if __name__ == "__main__":
    unittest.main()
