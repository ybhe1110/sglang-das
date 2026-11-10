"""Sampler synchronization must not allocate an unused singleton communicator."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSamplerSync(CustomTestCase):
    def test_singleton_sync_survives_unavailable_transport_memory(self):
        """A one-rank reduction is identity even when NCCL cannot allocate.

        CP/DP grammar requests used to initialize a redundant communicator and
        fail allocating 32 MiB. Inject the failure at the communication boundary;
        sampling must preserve the tokens without entering that backend.
        """
        for force_sync, grammars in ((False, [object()]), (True, None)):
            with self.subTest(force_sync=force_sync):
                tokens = torch.tensor([7, 2, 19], dtype=torch.int64)
                before = tokens.clone()
                with (
                    patch.object(sampler, "SYNC_TOKEN_IDS_ACROSS_TP", force_sync),
                    patch.object(sampler.dist, "get_world_size", return_value=1),
                    patch.object(
                        sampler.dist,
                        "all_reduce",
                        side_effect=RuntimeError("NCCL transport allocation failed"),
                    ),
                ):
                    sampler.Sampler._sync_token_ids_across_tp(
                        SimpleNamespace(tp_sync_group=object()),
                        tokens,
                        SimpleNamespace(grammars=grammars),
                    )
                torch.testing.assert_close(tokens, before, rtol=0, atol=0)

    def test_multi_rank_sync_does_not_hide_transport_failure(self):
        """Real TP groups still require synchronization and must fail loudly."""
        with (
            patch.object(sampler, "SYNC_TOKEN_IDS_ACROSS_TP", False),
            patch.object(sampler.dist, "get_world_size", return_value=2),
            patch.object(
                sampler.dist,
                "all_reduce",
                side_effect=RuntimeError("peer transport failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "peer transport failed"),
        ):
            sampler.Sampler._sync_token_ids_across_tp(
                SimpleNamespace(tp_sync_group=object()),
                torch.tensor([7]),
                SimpleNamespace(grammars=[object()]),
            )


if __name__ == "__main__":
    unittest.main()
