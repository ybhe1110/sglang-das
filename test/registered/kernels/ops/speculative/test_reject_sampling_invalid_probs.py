"""Malformed draft probabilities must reject and leave a finite residual.

These cases reproduce #37134's unconditional acceptance of zero/negative q
and corruption of the residual by infinities. For local CPU diagnostics the
same kernel can run with TRITON_INTERPRET=1 (set before starting Python).
"""

import os
import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_hcu_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: HIP/HCU use this Triton chain sampler for non-greedy verify.
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")
# backend-specific: DSPARK on DCU shares the invalid-q acceptance/residual path.
register_hcu_ci(est_time=5, suite="stage-b-test-1-hcu-small")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


class TestRejectSamplingInvalidProbabilities(CustomTestCase):
    def _verify(self, draft_probs, coin=0.05, final_coin=0.75):
        target_probs = torch.tensor(
            [[[0.6, 0.3, 0.1, 0.0], [0.6, 0.3, 0.1, 0.0]]], device=DEVICE
        )
        predicts = torch.full((2,), -1, dtype=torch.int32, device=DEVICE)
        accept_index = torch.full((1, 2), -1, dtype=torch.int32, device=DEVICE)
        num_correct_drafts = torch.empty((1,), dtype=torch.int32, device=DEVICE)
        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=num_correct_drafts,
            candidates=torch.tensor([[0, 1]], device=DEVICE),
            retrive_index=torch.tensor([[0, 1]], device=DEVICE),
            retrive_next_token=None,
            retrive_next_sibling=None,
            uniform_samples=torch.tensor([[coin]], device=DEVICE),
            uniform_samples_for_final_sampling=torch.tensor(
                [final_coin], device=DEVICE
            ),
            target_probs=target_probs,
            draft_probs=torch.tensor([[draft_probs]], device=DEVICE),
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        count = int(num_correct_drafts.item())
        bonus = int(predicts[accept_index[0, count]].item())
        return count, bonus

    def test_invalid_proposed_probability_rejects(self):
        for q in (0.0, -1.0, float("-inf"), float("inf"), float("nan"), 2.0):
            with self.subTest(q=q):
                # Every valid draft entry is zero; after rejecting malformed q
                # the residual is target p, whose 0.75 quantile is token 1.
                self.assertEqual(self._verify([0.0, q, 0.0, 0.0]), (0, 1))

    def test_invalid_residual_entries_are_ignored_in_both_passes(self):
        for q in (-1.0, float("-inf"), float("inf"), float("nan"), 2.0):
            with self.subTest(q=q):
                # Valid q at the proposed token forces rejection. After
                # sanitizing q[0], residual = [0.6, 0, 0.1, 0]. Its 0.5
                # quantile is token 0, never the zero-probability last token.
                self.assertEqual(
                    self._verify([q, 0.9, 0.0, 0.0], coin=0.9, final_coin=0.5),
                    (0, 0),
                )

    def test_valid_acceptance_and_residual_are_preserved(self):
        self.assertEqual(self._verify([0.5, 0.25, 0.25, 0.0]), (1, 1))
        # Rejection residual is [0.1, 0, 0.1, 0], so its 0.75 quantile is 2.
        self.assertEqual(self._verify([0.5, 0.5, 0.0, 0.0], coin=0.9), (0, 2))


if __name__ == "__main__":
    unittest.main()
