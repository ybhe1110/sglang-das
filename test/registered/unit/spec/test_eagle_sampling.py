"""Regressions for #37134 while retaining HCU's target-only linear MTP.

CPU tensors exercise proposal selection, HCU sampling and graph output
ownership. Only device kernels and model/graph execution are replaced.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import (
    _should_auto_enable_hip_rejection_sampling,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative import eagle_utils, eagle_worker_v2, spec_utils
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHipRejectionDefaults(CustomTestCase):
    def test_only_supported_plain_hip_configs_are_auto_enabled(self):
        defaults = dict(
            is_hip=True,
            is_hcu=False,
            use_rejection_sampling=False,
            algorithm="EAGLE",
            token_map=None,
            eagle_topk=1,
            accept_threshold_single=1.0,
            accept_threshold_acc=1.0,
            enable_deterministic_inference=False,
        )
        self.assertTrue(_should_auto_enable_hip_rejection_sampling(**defaults))
        for override in (
            dict(is_hcu=True),
            dict(is_hip=False),
            dict(use_rejection_sampling=True),
            dict(algorithm="EAGLE3"),
            dict(algorithm="DSPARK"),
            dict(token_map="draft_to_target.pt"),
            dict(eagle_topk=2),
            dict(accept_threshold_single=0.5),
            dict(accept_threshold_acc=0.5),
            dict(enable_deterministic_inference=True),
        ):
            with self.subTest(override=override):
                self.assertFalse(
                    _should_auto_enable_hip_rejection_sampling(**(defaults | override))
                )


class TestDraftProposal(CustomTestCase):
    def test_mixed_greedy_rows_keep_argmax_and_sampling_rows_keep_q(self):
        logits = torch.tensor([[0.2, 0.1, 0.0]]).repeat(128, 1)
        temperatures = torch.ones(128, 1)
        top_ks = torch.tensor([1, TOP_K_ALL] * 64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37134)
            reference_q, _, reference_tokens = spec_utils.sample_draft_proposal(
                logits, temperatures
            )
            torch.manual_seed(37134)
            q, probabilities, tokens = spec_utils.sample_draft_proposal(
                logits, temperatures, top_ks
            )
        self.assertTrue(torch.equal(q, reference_q))
        self.assertTrue(torch.equal(tokens[::2], logits[::2].argmax(-1, keepdim=True)))
        self.assertTrue(torch.equal(tokens[1::2], reference_tokens[1::2]))
        self.assertTrue(bool((tokens[1::2] != 0).any()))
        torch.testing.assert_close(probabilities, q.gather(1, tokens))

    def test_multi_layer_in_graph_proposal_honors_greedy_rows(self):
        logits = torch.tensor([[0.2, 0.1, 0.0]]).repeat(128, 1)
        temperatures = torch.ones(128, 1)
        top_ks = torch.tensor([1, TOP_K_ALL] * 64)
        runner = MultiLayerEagleDraftExtendCudaGraphRunner.__new__(
            MultiLayerEagleDraftExtendCudaGraphRunner
        )
        runner.step = 0
        runner.prune_draft_extend_logits = True
        runner.buffers = SimpleNamespace(
            temperatures=temperatures,
            top_ks=top_ks,
            draft_probs=torch.empty(128, 1, 3),
        )
        output = SimpleNamespace(next_token_logits=logits)

        def softmax_into(logits, temperatures, out):
            return out.copy_(torch.softmax(logits / temperatures, dim=-1))

        with (
            torch.random.fork_rng(devices=[]),
            patch(
                "sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner."
                "wide_row_softmax_triton",
                side_effect=softmax_into,
            ),
        ):
            torch.manual_seed(37134)
            runner._sample_draft_proposal(output, 128)
            torch.manual_seed(37134)
            q = torch.softmax(logits / temperatures, dim=-1)
            _, sampled_tokens = spec_utils.fast_sample(q)
        torch.testing.assert_close(runner.buffers.draft_probs[:, 0], q)
        self.assertTrue(bool((output.topk_index[::2] == 0).all()))
        self.assertTrue(torch.equal(output.topk_index[1::2], sampled_tokens[1::2]))
        torch.testing.assert_close(output.topk_p, q.gather(1, output.topk_index))


class TestEagleVerifySampling(CustomTestCase):
    def _verify(self, *, hcu, rejection, temperature=1.0, top_p=1.0, greedy=False):
        bs, width = 128, 2
        sampling_info = SimpleNamespace(
            temperatures=torch.full((bs, 1), temperature),
            top_ks=torch.full((bs,), 1 if greedy else TOP_K_ALL),
            top_ps=torch.full((bs,), top_p),
            min_ps=torch.zeros(bs),
            sampling_seed=None,
            need_min_p_sampling=False,
            need_top_k_sampling=greedy,
            need_top_p_sampling=top_p < 1.0,
            is_all_greedy=greedy,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
        )
        batch = SimpleNamespace(
            device="cpu",
            seq_lens=torch.ones(bs, dtype=torch.int64),
            sampling_info=sampling_info,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        verify_input = SimpleNamespace(
            tree_topk=1,
            draft_token_num=width,
            max_tree_depth=width,
            draft_token=torch.zeros(bs * width, dtype=torch.int64),
            positions=torch.arange(bs * width),
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            draft_probs=torch.full((bs, 1, 3), 1 / 3),
        )
        logits = torch.tensor([[2.0, 1.0, 0.0]]).repeat(bs * width, 1)
        observed = {}

        def tree_walk(**kwargs):
            observed["tree_tokens"] = kwargs["target_predict"].clone()
            kwargs["predicts"].copy_(kwargs["target_predict"].flatten())
            kwargs["accept_token_num"].zero_()
            return (
                kwargs["predicts"],
                kwargs["accept_index"],
                kwargs["accept_token_num"],
            )

        def chain_sample(**kwargs):
            observed["target_probs"] = kwargs["target_probs"].clone()
            kwargs["predicts"].zero_()
            kwargs["accept_token_num"].zero_()

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(
                    speculative_use_rejection_sampling=rejection,
                    speculative_accept_threshold_single=1.0,
                    speculative_accept_threshold_acc=1.0,
                )
            )
            for name, value in dict(
                _is_hcu=hcu,
                _is_hip=True,
                _is_cuda=False,
                _is_cpu=False,
                _is_npu=False,
                _is_xpu=False,
            ).items():
                stack.enter_context(patch.object(eagle_utils, name, value))
            stack.enter_context(
                patch.object(
                    eagle_utils, "lightop_top_k_top_p_sampling_from_probs", None
                )
            )
            stack.enter_context(
                patch.object(
                    eagle_utils, "verify_tree_greedy_func", side_effect=tree_walk
                )
            )
            stack.enter_context(
                patch.object(
                    eagle_utils,
                    "borrow_graph_pool",
                    lambda **_: contextlib.nullcontext(),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.distributed.get_tp_group",
                    return_value=SimpleNamespace(world_size=1),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton",
                    side_effect=chain_sample,
                )
            )
            stack.enter_context(patch.object(spec_utils, "SIMULATE_ACC_LEN", -1))
            stack.enter_context(torch.random.fork_rng(devices=[]))
            torch.manual_seed(37134)
            eagle_utils.eagle_sample(
                verify_input, batch, SimpleNamespace(next_token_logits=logits)
            )
        return observed

    def test_hcu_keeps_target_sampling_with_or_without_rejection_flag(self):
        for rejection in (False, True):
            with self.subTest(rejection=rejection):
                hot = self._verify(hcu=True, rejection=rejection, temperature=2.0)
                cold = self._verify(hcu=True, rejection=rejection, temperature=0.05)
                nucleus = self._verify(hcu=True, rejection=rejection, top_p=0.5)
                self.assertNotIn("target_probs", hot)
                self.assertTrue(bool((hot["tree_tokens"] != 0).any()))
                self.assertTrue(bool((cold["tree_tokens"] == 0).all()))
                self.assertTrue(bool((nucleus["tree_tokens"] == 0).all()))

    def test_plain_hip_rejection_reaches_temperature_scaled_distribution(self):
        result = self._verify(hcu=False, rejection=True, temperature=2.0)
        self.assertNotIn("tree_tokens", result)
        expected = torch.softmax(torch.tensor([2.0, 1.0, 0.0]) / 2.0, dim=-1)
        torch.testing.assert_close(result["target_probs"][0, 0], expected)

    def test_greedy_and_plain_hip_without_rejection_stay_greedy(self):
        for hcu, rejection, greedy in (
            (True, True, True),
            (False, True, True),
            (False, False, False),
        ):
            with self.subTest(hcu=hcu, rejection=rejection, greedy=greedy):
                result = self._verify(hcu=hcu, rejection=rejection, greedy=greedy)
                self.assertNotIn("target_probs", result)
                self.assertTrue(bool((result["tree_tokens"] == 0).all()))


class TestDraftGraphOutputOwnership(CustomTestCase):
    def test_target_forward_cannot_overwrite_retained_draft_probs(self):
        graph_probs = torch.tensor([[[0.2, 0.8]]])
        original = graph_probs.clone()
        worker = eagle_worker_v2.EagleDraftWorker.__new__(
            eagle_worker_v2.EagleDraftWorker
        )
        worker.req_to_token_pool = None
        worker.draft_runner = SimpleNamespace(canary_manager=None)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.seed_dsa_topk_from_draft_extend = False
        worker.target_worker = None
        worker.tree_mask_mode = None
        worker.device = "cpu"
        worker.cuda_graph_runner = SimpleNamespace(
            execute=lambda _: (None, None, None, graph_probs)
        )
        with (
            patch.object(
                eagle_worker_v2,
                "prepare_for_draft",
                return_value=(
                    SimpleNamespace(
                        forward_mode=SimpleNamespace(is_idle=lambda: False)
                    ),
                    True,
                ),
            ),
            patch.object(
                eagle_worker_v2,
                "build_eagle_verify_input",
                side_effect=lambda *args, **kwargs: args[5],
            ),
        ):
            retained = worker.draft(SimpleNamespace(spec_info=SimpleNamespace()))
        graph_probs.fill_(float("-inf"))  # target forward reuses the graph pool
        self.assertTrue(torch.equal(retained, original))


if __name__ == "__main__":
    unittest.main()
