import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import compute_spec_context_reserve
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.runtime_context import get_context, get_parallel, get_spec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSchedulerInitReqMaxNewTokens(CustomTestCase):
    """Property tests for Scheduler.init_req_max_new_tokens.

    Rules enforced when clipping a request's max_new_tokens:
      1. context: input_len + max_new_tokens + spec_context_reserve < max_req_len
      2. admission budget (PrefillAdder):
         ceil_page(input_len) + max_new_tokens + page_size * shard_widening
         < max_total_num_tokens * shard_widening
      3. env limit: <= SGLANG_MAX_NEW_TOKENS_LIMIT when set and positive
      4. never above the requested value
      5. min_new_tokens <= max_new_tokens afterwards

    Each case asserts all rules hold and the result is tight: one more token
    would violate a rule or exceed the request. Over-long inputs degenerate to
    max_new_tokens = 0 and are rejected by later admission checks.
    """

    @classmethod
    def setUpClass(cls):
        # Silence the per-request capping warning; the sweep triggers it a lot.
        cls._scheduler_logger = logging.getLogger("sglang.srt.managers.scheduler")
        cls._old_level = cls._scheduler_logger.level
        cls._scheduler_logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        cls._scheduler_logger.setLevel(cls._old_level)

    def setUp(self):
        config = get_context().override_server_args(speculative_algorithm=None)
        config.install()
        self.addCleanup(config.restore)
        # Tests use an unsharded topology unless they explicitly override it.
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _new_scheduler(
        self,
        max_req_len: int = 128,
        max_total_num_tokens: int = 1024,
        page_size: int = 1,
        kv_shard_widening: int = 1,
        enable_overlap: bool = True,
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = max_req_len
        scheduler.max_total_num_tokens = max_total_num_tokens
        scheduler.page_size = page_size
        scheduler.kv_shard_widening = kv_shard_widening
        scheduler.spec_context_reserve = compute_spec_context_reserve(enable_overlap)
        scheduler.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        scheduler.sliding_window_size = None
        scheduler.chunked_prefill_size = None
        scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=max_total_num_tokens,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        scheduler.token_to_kv_pool_allocator.page_size = page_size
        return scheduler

    def _new_req(self, max_new_tokens, input_len: int = 8, min_new_tokens: int = 0):
        return SimpleNamespace(
            rid="test-req",
            set_finish_with_abort=Mock(),
            origin_input_ids=[0] * input_len,
            sampling_params=SimpleNamespace(
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens
            ),
        )

    def test_speculative_context_exhaustion_reports_error(self):
        """Positive/default output requests must not silently become prefill-only.

        Explicit zero remains valid, and KV-only clipping is not a speculative
        context error. Check the real abort reason returned to the tokenizer.
        """
        for context_len in (128, 1048576):
            for requested, reserve, remaining, expected, aborted in (
                (8, 30, 30, 0, True),
                (None, 30, 30, 0, True),
                (8, 30, 29, 0, True),
                (0, 30, 30, 0, False),
                (0, 30, 29, 0, False),
                (8, 30, 31, 1, False),
                (8, 0, 30, 8, False),
                (8, 0, 0, 0, False),
                (0, 0, 0, 0, False),
            ):
                with (
                    self.subTest(
                        context_len=context_len,
                        requested=requested,
                        reserve=reserve,
                        remaining=remaining,
                    ),
                    envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None),
                ):
                    scheduler = self._new_scheduler(
                        max_req_len=context_len,
                        max_total_num_tokens=2 * context_len,
                    )
                    scheduler.spec_context_reserve = reserve
                    minimum = 0 if requested == 0 else 4
                    req = self._new_req(requested, input_len=0, min_new_tokens=minimum)
                    input_len = context_len - remaining - 1
                    req.origin_input_ids = range(input_len)
                    req.to_finish = None
                    req.release_mm_inputs_on_abort = Mock()
                    req.set_finish_with_abort = MethodType(
                        Req.set_finish_with_abort, req
                    )
                    scheduler.init_req_max_new_tokens(req)
                    self.assertEqual(req.sampling_params.max_new_tokens, expected)
                    self.assertEqual(
                        req.sampling_params.min_new_tokens, min(minimum, expected)
                    )
                    if aborted:
                        self.assertIsNotNone(req.to_finish)
                        reason = req.to_finish.to_json()
                        self.assertEqual(reason["type"], "abort")
                        self.assertEqual(reason["status_code"], 400)
                        self.assertEqual(reason["err_type"], "BadRequestError")
                        self.assertIn(f"context ({input_len})", reason["message"])
                        self.assertIn("reserve=30", reason["message"])
                        self.assertIn("disable speculative decoding", reason["message"])
                    else:
                        self.assertIsNone(req.to_finish)
                        self.assertEqual(len(req.origin_input_ids), input_len)

    def test_kv_capacity_exhaustion_is_not_speculative_context_error(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None):
            scheduler = self._new_scheduler(max_total_num_tokens=10)
            scheduler.spec_context_reserve = 30
            req = self._new_req(8, input_len=8)
            req.to_finish = None
            req.release_mm_inputs_on_abort = Mock()
            req.set_finish_with_abort = MethodType(Req.set_finish_with_abort, req)
            scheduler.init_req_max_new_tokens(req)
            self.assertEqual(req.sampling_params.max_new_tokens, 0)
            self.assertIsNone(req.to_finish)

    def _init_and_check(self, scheduler, req) -> int:
        """Run init_req_max_new_tokens, then assert all admission rules hold
        and the result is tight. Returns the resulting max_new_tokens."""
        requested = req.sampling_params.max_new_tokens
        scheduler.init_req_max_new_tokens(req)
        max_new_tokens = req.sampling_params.max_new_tokens

        input_len = len(req.origin_input_ids)
        page_size = scheduler.page_size
        shard_widening = scheduler.kv_shard_widening
        paged_input_len = -(-input_len // page_size) * page_size
        limit = scheduler.max_new_tokens_limit
        limit_active = limit is not None and limit > 0

        def satisfies_rules(candidate: int) -> bool:
            context_ok = (
                input_len + candidate + scheduler.spec_context_reserve
                < scheduler.max_req_len
            )
            budget_ok = (
                paged_input_len + candidate + page_size * shard_widening
                < scheduler.max_total_num_tokens * shard_widening
            )
            limit_ok = not limit_active or candidate <= limit
            requested_ok = requested is None or candidate <= requested
            return context_ok and budget_ok and limit_ok and requested_ok

        self.assertGreaterEqual(max_new_tokens, 0)
        if max_new_tokens > 0:
            self.assertTrue(satisfies_rules(max_new_tokens))
        self.assertFalse(satisfies_rules(max_new_tokens + 1))
        self.assertLessEqual(req.sampling_params.min_new_tokens, max_new_tokens)
        return max_new_tokens

    def test_limit_disabled_by_default(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 64)

    def test_limit_clips_explicit_request(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 16)

    def test_limit_applies_when_request_unset(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=None)
            self.assertEqual(self._init_and_check(scheduler, req), 16)

    def test_non_positive_limit_is_ignored(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(limit):
                    scheduler = self._new_scheduler()
                    req = self._new_req(max_new_tokens=64)
                    self.assertEqual(self._init_and_check(scheduler, req), 64)

    def test_context_rule_binds_tighter_than_limit(self):
        max_req_len, input_len = 32, 20
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler(max_req_len=max_req_len)
            req = self._new_req(max_new_tokens=64, input_len=input_len)
            self.assertEqual(
                self._init_and_check(scheduler, req), max_req_len - input_len - 1
            )

    def test_budget_rule_binds_tighter_than_limit(self):
        max_total_num_tokens, page_size, input_len = 24, 4, 8
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(32):
            scheduler = self._new_scheduler(
                max_total_num_tokens=max_total_num_tokens, page_size=page_size
            )
            req = self._new_req(max_new_tokens=64, input_len=input_len)
            paged_input_len = -(-input_len // page_size) * page_size
            self.assertEqual(
                self._init_and_check(scheduler, req),
                max_total_num_tokens - paged_input_len - page_size - 1,
            )

    def test_dcp_budget_is_already_in_logical_tokens(self):
        with get_parallel().override(attn_dcp_size=4):
            scheduler = self._new_scheduler(max_total_num_tokens=24, page_size=4)
            req = self._new_req(max_new_tokens=64, input_len=8)

            self.assertEqual(self._init_and_check(scheduler, req), 11)

    def test_sharded_budget_reserves_one_page_per_shard(self):
        max_total_num_tokens, page_size, input_len, shard_widening = 8, 4, 8, 3
        scheduler = self._new_scheduler(
            max_req_len=128,
            max_total_num_tokens=max_total_num_tokens,
            page_size=page_size,
            kv_shard_widening=shard_widening,
        )
        req = self._new_req(max_new_tokens=64, input_len=input_len)

        max_new_tokens = self._init_and_check(scheduler, req)

        widened_capacity = max_total_num_tokens * shard_widening
        per_req_overhead = page_size * shard_widening
        self.assertEqual(
            max_new_tokens,
            widened_capacity - input_len - per_req_overhead - 1,
        )
        self.assertLess(
            input_len + max_new_tokens + per_req_overhead,
            widened_capacity,
        )

    def test_min_new_tokens_clamped_to_limit(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64, min_new_tokens=32)
            self.assertEqual(self._init_and_check(scheduler, req), 16)
            self.assertEqual(req.sampling_params.min_new_tokens, 16)

    def test_admission_rules_sweep(self):
        for page_size in (1, 4, 16):
            for input_len in (1, 8, 100):
                for requested in (None, 0, 5, 64, 1 << 20):
                    for limit in (None, 0, 16, 1 << 20):
                        for max_req_len, max_total_num_tokens in (
                            (128, 1024),
                            (32, 24),
                            (128, 24),
                        ):
                            with self.subTest(
                                page_size=page_size,
                                input_len=input_len,
                                requested=requested,
                                limit=limit,
                                max_req_len=max_req_len,
                                max_total_num_tokens=max_total_num_tokens,
                            ):
                                with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(limit):
                                    scheduler = self._new_scheduler(
                                        max_req_len=max_req_len,
                                        max_total_num_tokens=max_total_num_tokens,
                                        page_size=page_size,
                                    )
                                    req = self._new_req(
                                        max_new_tokens=requested, input_len=input_len
                                    )
                                    self._init_and_check(scheduler, req)

    def test_unified_budget_rounds_prompt_and_decode_together(self):
        bundle = init_unified_swa_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=1,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=4,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            total_bytes=768,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
        )
        for algorithm in (None, "DSPARK"):
            with get_spec().override(
                speculative_algorithm=algorithm,
                speculative_num_draft_tokens=17 if algorithm else None,
            ):
                scheduler = self._new_scheduler(page_size=4)
                scheduler.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
                scheduler.sliding_window_size = 4
                scheduler.chunked_prefill_size = 4
                scheduler.max_new_tokens_limit = None
                for prompt_len in (4, 5, 6, 7):
                    with self.subTest(algorithm=algorithm, prompt_len=prompt_len):
                        req = self._new_req(max_new_tokens=1, input_len=prompt_len)
                        scheduler.init_req_max_new_tokens(req)
                        self.assertEqual(req.sampling_params.max_new_tokens, 1)

    def test_speculative_reserve_preserves_allocator_capacity_clipping(self):
        """Context headroom must not bypass the tighter sharded KV budget."""
        with (
            get_spec().override(
                speculative_algorithm="DSPARK", speculative_num_draft_tokens=17
            ),
            envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None),
        ):
            for shard_widening in (1, 3):
                with self.subTest(shard_widening=shard_widening):
                    scheduler = self._new_scheduler(
                        max_total_num_tokens=24,
                        page_size=4,
                        kv_shard_widening=shard_widening,
                    )
                    req = self._new_req(100, input_len=8)
                    self.assertEqual(
                        self._init_and_check(scheduler, req),
                        (24 - 4) * shard_widening - 8 - 1,
                    )

    def test_speculative_boundary_inflight_windows(self):
        """Completion clipping happens after verify and, in overlap, after the
        next forward is launched. Every final-window phase must stay inside
        model context even when the pending verify accepts its entire window.
        A RoPE table enlarged by only one window cannot guarantee this.
        """
        context_len = 1048576
        input_len = context_len - 2000
        for algorithm in ("DSPARK", "DFLASH", "EAGLE3", "UNO"):
            for window in (6, 17):
                for overlap in (False, True):
                    for requested in (None, 4000):
                        with (
                            self.subTest(
                                algorithm=algorithm,
                                window=window,
                                overlap=overlap,
                                requested=requested,
                            ),
                            get_spec().override(
                                speculative_algorithm=algorithm,
                                speculative_num_draft_tokens=window,
                                speculative_num_steps=window - 1,
                                speculative_eagle_topk=1,
                                speculative_adaptive=False,
                            ),
                            envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None),
                        ):
                            scheduler = self._new_scheduler(
                                max_req_len=context_len - 1,
                                max_total_num_tokens=2 * context_len,
                                enable_overlap=overlap,
                            )
                            req = self._new_req(requested, input_len=0)
                            req.origin_input_ids = range(input_len)
                            scheduler.init_req_max_new_tokens(req)
                            output_limit = req.sampling_params.max_new_tokens
                            completion_limit = input_len + output_limit
                            forward_width = window + (algorithm == "UNO")
                            for remaining in range(1, forward_width + 1):
                                # The last output token has not yet been fed
                                # back into the model (hence the extra -1).
                                prefix_len = completion_limit - remaining - 1
                                last_position = prefix_len + forward_width - 1
                                if overlap:
                                    last_position += forward_width
                                self.assertLess(last_position, context_len)

    def test_speculative_short_request_and_minimum_clipping(self):
        with (
            get_spec().override(
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=17,
                speculative_num_steps=1,
                speculative_eagle_topk=1,
            ),
            envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None),
        ):
            scheduler = self._new_scheduler()
            self.assertEqual(self._init_and_check(scheduler, self._new_req(16)), 16)
            req = self._new_req(64, input_len=80, min_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 13)
            self.assertEqual(req.sampling_params.min_new_tokens, 13)
            self.assertEqual(
                self._init_and_check(scheduler, self._new_req(None, input_len=100)), 0
            )

    def test_eagle_tree_reserve_larger_than_verify_window(self):
        with (
            get_spec().override(
                speculative_algorithm="EAGLE3",
                speculative_num_draft_tokens=6,
                speculative_num_steps=5,
                speculative_eagle_topk=4,
            ),
            envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None),
        ):
            for overlap, expected in ((False, 99), (True, 79)):
                scheduler = self._new_scheduler(enable_overlap=overlap)
                self.assertEqual(
                    self._init_and_check(scheduler, self._new_req(None)), expected
                )

    def test_adaptive_reserves_largest_configured_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.json"
            path.write_text(json.dumps({"1": {"candidate_steps": [1, 5, 16]}}))
            with (
                get_context().override_server_args(
                    speculative_algorithm="EAGLE3",
                    speculative_num_draft_tokens=2,
                    speculative_num_steps=1,
                    speculative_eagle_topk=1,
                    speculative_adaptive=True,
                    speculative_adaptive_config=str(path),
                ),
                envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None),
            ):
                scheduler = self._new_scheduler()
                self.assertEqual(
                    self._init_and_check(scheduler, self._new_req(None)), 85
                )


if __name__ == "__main__":
    unittest.main()
