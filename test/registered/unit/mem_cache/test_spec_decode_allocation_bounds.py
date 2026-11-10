"""Speculative allocation must fit its row, including page_size=1 double buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_page_size,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import (
    get_context,
    get_parallel,
    get_schedule,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _write_rows_on_cpu(indices, rows, starts, ends, locations, batch_size):
    """CPU stand-in for the device writer, including its flat pointer addressing.

    Unlike a bounded Python row slice, this exposes a write into the next row
    if the production host guard is removed.
    """
    flat = rows.view(-1)
    offset = 0
    for i in range(batch_size):
        start = int(indices[i]) * rows.shape[1] + int(starts[i])
        count = int(ends[i] - starts[i])
        flat[start : start + count] = locations[offset : offset + count]
        offset += count


class TestSpecDecodeAllocationBounds(CustomTestCase):
    def setUp(self):
        config = get_context().override_server_args(
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=17,
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            page_size=1,
        )
        config.install()
        self.addCleanup(config.restore)
        parallel = get_parallel().override(attn_dcp_size=1)
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)

    def test_row_headroom_covers_double_buffer_and_page_rounding(self):
        # The old page_size=1 allowance was only window + 4; a larger window
        # could overwrite the following request's mapping before verification.
        context_len = 128
        for algorithm, window, steps, topk in (
            ("DSPARK", 6, 1, 1),
            ("DSPARK", 17, 1, 1),
            ("DFLASH", 17, 1, 1),
            ("EAGLE3", 6, 5, 4),
            ("UNO", 8, 1, 1),
        ):
            for page_size, dcp_size in ((1, 1), (4, 1), (1, 4), (16, 2)):
                with (
                    self.subTest(
                        algorithm=algorithm,
                        window=window,
                        page=page_size,
                        dcp=dcp_size,
                    ),
                    get_context().override_server_args(
                        speculative_algorithm=algorithm,
                        speculative_num_draft_tokens=window,
                        speculative_num_steps=steps,
                        speculative_eagle_topk=topk,
                    ),
                    get_schedule().override(page_size=page_size),
                    get_parallel().override(attn_dcp_size=dcp_size),
                ):
                    alloc_page = get_alloc_page_size()
                    reserve = get_alloc_reserve_per_decode()
                    row_width = context_len + get_req_to_token_extra_context_len()
                    for committed in range(
                        context_len - alloc_page + 1, context_len + 1
                    ):
                        cur = committed // alloc_page * alloc_page
                        req = SimpleNamespace(
                            kv=SimpleNamespace(
                                kv_committed_len=committed, kv_allocated_len=cur
                            ),
                        )
                        _, nxt, needed = page_aligned_decode_alloc_lens(
                            [req], reserve=reserve, page_size=alloc_page
                        )
                        self.assertLessEqual(nxt[0], row_width)
                        self.assertEqual(nxt[0] % alloc_page, 0)
                        self.assertGreaterEqual(nxt[0] - committed, reserve)
                        self.assertEqual(needed, nxt[0] - cur)

    def test_non_speculative_headroom_unchanged(self):
        with get_context().override_server_args(
            speculative_algorithm=None, speculative_num_draft_tokens=None
        ):
            self.assertEqual(get_req_to_token_extra_context_len(), 4)

    def _new_allocation(self):
        pool = ReqToTokenPool(
            size=3, max_context_len=12, device="cpu", enable_memory_saver=False
        )
        pool.req_to_token.fill_(-7)
        allocator = TokenToKVPoolAllocator(
            size=64, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        initial = allocator.alloc(4)
        pool.req_to_token[1:3, :2] = initial.view(2, 2).to(torch.int32)
        reqs = [
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=2)) for _ in range(2)
        ]
        cache = SimpleNamespace(
            token_to_kv_pool_allocator=allocator, supports_prefix_sharing=lambda: False
        )
        return pool, allocator, reqs, cache

    def _allocate(self, pool, reqs, cache, ends):
        starts = torch.tensor([r.kv.kv_allocated_len for r in reqs], dtype=torch.int64)
        ends = torch.tensor(ends, dtype=torch.int64)
        with patch(
            "sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func",
            side_effect=_write_rows_on_cpu,
        ):
            alloc_for_spec_decode(
                cache,
                pool,
                reqs=reqs,
                req_pool_indices=torch.tensor([1, 2]),
                cur_kv_lens=starts,
                cur_kv_lens_cpu=starts,
                nxt_kv_lens=ends,
                nxt_kv_lens_cpu=ends,
                num_needed_tokens=int((ends - starts).sum()),
            )

    def test_out_of_bounds_rejected_before_any_batch_mutation(self):
        pool, allocator, reqs, cache = self._new_allocation()
        before = pool.req_to_token.clone()
        free_before = allocator.free_pages.clone()
        # Only the second request overflows: the first must not be allocated
        # or written either, and the neighboring row must retain its sentinel.
        with self.assertRaisesRegex(ValueError, r"row width: 13 > 12"):
            self._allocate(pool, reqs, cache, [12, 13])
        self.assertTrue(torch.equal(pool.req_to_token, before))
        self.assertTrue(torch.equal(allocator.free_pages, free_before))
        self.assertEqual([r.kv.kv_allocated_len for r in reqs], [2, 2])

    def test_exact_width_write_and_reuse_preserve_neighbor_rows(self):
        pool, allocator, reqs, cache = self._new_allocation()
        expected = pool.req_to_token.clone()
        expected[1:3, 2:] = torch.arange(5, 25, dtype=torch.int32).view(2, 10)
        self._allocate(pool, reqs, cache, [12, 12])
        self.assertTrue(torch.equal(pool.req_to_token, expected))
        self.assertEqual(allocator.available_size(), 40)
        self.assertEqual([r.kv.kv_allocated_len for r in reqs], [12, 12])
        self._allocate(pool, reqs, cache, [12, 12])
        self.assertTrue(torch.equal(pool.req_to_token, expected))
        self.assertEqual(allocator.available_size(), 40)

    def test_empty_allocation_is_a_noop(self):
        pool, allocator, _, cache = self._new_allocation()
        before = pool.req_to_token.clone()
        empty = torch.empty(0, dtype=torch.int64)
        alloc_for_spec_decode(
            cache,
            pool,
            reqs=[],
            req_pool_indices=empty,
            cur_kv_lens=empty,
            cur_kv_lens_cpu=empty,
            nxt_kv_lens=empty,
            nxt_kv_lens_cpu=empty,
            num_needed_tokens=0,
        )
        self.assertTrue(torch.equal(pool.req_to_token, before))
        self.assertEqual(allocator.available_size(), 60)


if __name__ == "__main__":
    unittest.main()
