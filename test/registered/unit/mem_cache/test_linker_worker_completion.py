"""Native/session return must precede publication of private-load completion."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker as mc
from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker as umbp
from sglang.srt.mem_cache.unified_cache.linker_lifecycle import LayerWiseLoadCounter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLinkerWorkerCompletion(CustomTestCase):
    def test_mooncake_terminal_follows_read_and_session_cleanup(self):
        for failure in (None, "start", "read", "cleanup"):
            with self.subTest(failure=failure):
                backend = mc.MooncakeDirectLinker.__new__(mc.MooncakeDirectLinker)
                counter = LayerWiseLoadCounter(2)
                index = counter.update_producer()
                events = []

                def read(*args):
                    self.assertFalse(counter.terminals[index].done())
                    events.append("read")
                    return [0, 0] if failure == "read" else [32, 32]

                def end(keys):
                    self.assertFalse(counter.terminals[index].done())
                    events.append(("end", list(keys)))
                    if failure == "cleanup":
                        raise RuntimeError("session cleanup failed")

                store = SimpleNamespace(
                    batch_get_session_start=lambda keys: (
                        [0, -1] if failure == "start" else [0, 0]
                    ),
                    batch_get_into_multi_buffer_ranges=read,
                    batch_get_session_end=end,
                )
                backend.storage = SimpleNamespace(
                    store=store,
                    _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
                    _tag_keys=lambda keys: keys,
                )
                backend.pools = {
                    PoolName.KV: SimpleNamespace(
                        prepare_locations=lambda indices: [0, 1],
                        get_prepared_layer_range_meta=lambda locations, layer: (
                            [[100], [200]],
                            [[32], [32]],
                            [[0], [0]],
                        ),
                    )
                }
                backend.num_layers = 2
                backend.device = "cpu"
                backend.layer_done_counter = counter
                backend._availability = SimpleNamespace(failed=lambda error: None)
                backend.load_queue = Queue()
                transfer = PoolTransfer(name=PoolName.KV, keys=["a", "b"])
                backend.load_queue.put((index, {"rid": [transfer]}, Mock()))
                backend.load_queue.put(None)
                with patch.object(mc, "device_module", Mock()):
                    backend.load_thread_func()
                self.assertEqual(
                    events[-1], ("end", ["a"] if failure == "start" else ["a", "b"])
                )
                self.assertEqual(
                    events.count("read"),
                    0 if failure == "start" else 1 if failure == "read" else 2,
                )
                self.assertTrue(counter.terminals[index].done())
                self.assertEqual(counter.finish(index), failure is None)
                self.assertEqual(backend.load_queue.unfinished_tasks, 0)
                self.assertEqual(counter.futures, {})
                self.assertEqual(counter.errors, {})

    def test_umbp_worker_exception_unblocks_layers_but_waits_for_return(self):
        backend = umbp.UMBPDirectLinker.__new__(umbp.UMBPDirectLinker)
        counter = LayerWiseLoadCounter(2)
        index = counter.update_producer()

        def native_call(*args):
            counter.complete(index, 0)
            self.assertFalse(counter.terminals[index].done())
            raise RuntimeError("native load failed")

        backend.layer_done_counter = counter
        backend._run_layer_wise_batch = native_call
        backend._load_queue = Queue()
        backend._load_queue.put((index, ["rid"], [], Mock()))
        backend._load_queue.put(None)
        backend._load_thread_func()
        self.assertTrue(counter.terminals[index].done())
        self.assertFalse(counter.finish(index))
        self.assertEqual(backend._load_queue.unfinished_tasks, 0)
        self.assertEqual(counter.errors, {})


if __name__ == "__main__":
    unittest.main()
