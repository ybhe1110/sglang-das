"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            # Slot 0 is the padding sink and must never contribute to the
            # available count.  More importantly, a cleanup retry can present
            # the same slot twice (for example after an external KV load is
            # aborted while its request is being retired).  Token allocators
            # used to append both copies, making available_size() exceed the
            # static capacity and turning pool usage negative.  Paged
            # allocators already de-duplicate at page granularity; keep the
            # token allocator's accounting equally idempotent.
            free_index = torch.unique(free_index[free_index > 0])
            if free_index.numel() == 0:
                return
            occupied = self.free_pages
            if self.need_sort and self.release_pages.numel() > 0:
                occupied = torch.cat((occupied, self.release_pages))
            free_index = free_index[~torch.isin(free_index, occupied)]
            if free_index.numel() == 0:
                return
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
