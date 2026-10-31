"""The failure-case client must not turn transport failures into passing aborts."""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPairedL3FailureValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[3]
            / "manual/disaggregation/paired_l3_validation.py"
        )
        spec = importlib.util.spec_from_file_location("paired_l3_validation", path)
        cls.client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.client)

    def stream(self, records, *, http_error=None, read_error=None, expect_abort=True):
        response = Mock()
        response.raise_for_status.side_effect = http_error
        response.iter_lines.side_effect = read_error
        response.iter_lines.return_value = [
            b"data: " + json.dumps(record).encode() for record in records
        ] + [b"data: [DONE]"]
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.dict(
            "sys.modules",
            {"requests": SimpleNamespace(post=Mock(return_value=context))},
        ):
            return self.client.request(
                "http://test-only",
                {},
                2,
                expect_abort,
                "external KV" if expect_abort else None,
            )

    @staticmethod
    def abort():
        return {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "status_code": 500,
                    "message": "Aborted: external KV cache load failed.",
                }
            }
        }

    def test_load_abort_is_a_terminal_even_with_http_200(self):
        result = self.stream([self.abort()])
        self.assertEqual(result["terminal_count"], 1)
        self.assertEqual(result["meta"]["finish_reason"]["type"], "abort")

    def test_generic_500_is_not_an_expected_load_abort(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            self.stream([], http_error=RuntimeError("HTTP 500"))

    def test_timeout_is_not_an_expected_load_abort(self):
        with self.assertRaises(TimeoutError):
            self.stream([], read_error=TimeoutError())

    def test_eof_without_terminal_fails(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one terminal"):
            self.stream([])

    def test_duplicate_abort_fails(self):
        with self.assertRaisesRegex(RuntimeError, "got 2"):
            self.stream([self.abort(), self.abort()])

    def test_success_without_injection_is_not_a_passing_failure_case(self):
        record = {"meta_info": {"finish_reason": {"type": "length"}}}
        with self.assertRaisesRegex(RuntimeError, "Unexpected terminal"):
            self.stream([record])

    def test_unrelated_abort_cannot_validate_external_load_failure(self):
        record = self.abort()
        record["meta_info"]["finish_reason"]["message"] = "Cancelled by client"
        with self.assertRaisesRegex(RuntimeError, "expected cause"):
            self.stream([record])

    def test_recovery_cannot_pass_with_another_abort(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected terminal"):
            self.stream([self.abort()], expect_abort=False)

    def test_recovery_accepts_normal_terminal(self):
        record = {"meta_info": {"finish_reason": {"type": "length"}}}
        result = self.stream([record], expect_abort=False)
        self.assertEqual(result["meta"]["finish_reason"]["type"], "length")


if __name__ == "__main__":
    unittest.main()
