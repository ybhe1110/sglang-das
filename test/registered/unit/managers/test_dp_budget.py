"""CPU regressions for causal DP reservations and their actual load transports."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

from sglang.srt.managers.dp_budget import (
    DPBudget,
    LoadBalanceMethod,
    SchedulerDPBudget,
)
from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    ZmqLoadSnapshotWriter,
    ZmqShmLoadSnapshotReader,
    snapshot_decoder,
    snapshot_encoder,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(tokens=10, terminal=False):
    """Return a CPU Req stand-in with tokens and optional terminal state."""
    return SimpleNamespace(seqlen=tokens, finished=Mock(return_value=terminal))


def _scheduler(**overrides):
    """Return a Scheduler queue stand-in, applying the supplied overrides."""
    state = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        grammar_manager=SimpleNamespace(grammar_queue=[]),
        disagg_prefill_bootstrap_queue=None,
        disagg_prefill_inflight_queue=[],
        disagg_decode_prealloc_queue=None,
        disagg_decode_transfer_queue=None,
        last_batch=None,
        cur_batch_for_debug=None,
        result_queue=[],
        chunked_req=None,
    )
    state.__dict__.update(overrides)
    return state


class TestDPBudget(unittest.TestCase):
    def test_new_empty_snapshots_do_not_erase_reservations(self):
        """Assert zero snapshots between dispatches preserve rank/token balance."""
        for method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        ):
            with self.subTest(method=method):
                budget = DPBudget(4)
                sources = [SchedulerDPBudget(rank) for rank in range(4)]
                ranks = []
                for publication in range(1, 49):
                    loads = []
                    for rank, source in enumerate(sources):
                        load = LoadSnapshot(dp_rank=rank, timestamp=publication)
                        source.attach(_scheduler(), load)
                        loads.append(load)
                    budget.update_budget(loads)
                    ranks.append(budget.dispatch(method, estimated_tokens=10))
                self.assertEqual([ranks.count(rank) for rank in range(4)], [12] * 4)
                self.assertEqual(budget.total_requests, [12] * 4)
                self.assertEqual(budget.total_tokens, [120] * 4)

    def test_partial_and_complete_confirmation_do_not_double_count(self):
        """Assert a prefix releases only its reservation and retains later work."""
        budget, source = DPBudget(1), SchedulerDPBudget(0)
        budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 10)
        first = SimpleNamespace(dp_budget_ticket=budget.ticket(0))
        budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 20)
        second = SimpleNamespace(dp_budget_ticket=budget.ticket(0))
        source.begin(first)
        source.complete()
        load = LoadSnapshot(timestamp=1, num_waiting_reqs=1, num_total_tokens=10)
        source.attach(_scheduler(waiting_queue=[_req(10)]), load)
        budget.update_budget([load])
        self.assertEqual((budget.total_requests, budget.total_tokens), ([2], [30]))
        source.begin(second)
        source.complete()
        load = LoadSnapshot(timestamp=2, num_waiting_reqs=2, num_total_tokens=30)
        source.attach(_scheduler(waiting_queue=[_req(10), _req(20)]), load)
        budget.update_budget([load])
        self.assertEqual((budget.total_requests, budget.total_tokens), ([2], [30]))
        empty = LoadSnapshot(timestamp=3)
        source.attach(_scheduler(), empty)
        budget.update_budget([empty])
        self.assertEqual((budget.total_requests, budget.total_tokens), ([0], [0]))

    def test_rejected_or_short_request_can_be_confirmed_by_empty_snapshot(self):
        """Assert terminal work releases its reservation even if never published live."""
        budget, source = DPBudget(1), SchedulerDPBudget(0)
        budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 10)
        source.begin(SimpleNamespace(dp_budget_ticket=budget.ticket(0)))
        source.complete()
        load = LoadSnapshot(timestamp=1)
        source.attach(_scheduler(), load)
        budget.update_budget([load])
        self.assertEqual((budget.total_requests, budget.total_tokens), ([0], [0]))

    def test_duplicate_and_out_of_order_publications_are_ignored(self):
        """Assert delayed source loads cannot undo newer confirmations or reservations."""
        budget, source = DPBudget(1), SchedulerDPBudget(0)
        old = LoadSnapshot(timestamp=10)
        source.attach(_scheduler(), old)
        budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        source.begin(SimpleNamespace(dp_budget_ticket=budget.ticket(0)))
        source.complete()
        new = LoadSnapshot(timestamp=5, num_running_reqs=1)
        source.attach(_scheduler(), new)
        budget.update_budget([new])
        budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        budget.update_budget([new, old])
        self.assertEqual(budget.total_requests, [2])

    def test_foreign_controller_epoch_cannot_clear_reservations(self):
        """Assert old-controller SHM contents do not confirm new-controller work."""
        old_budget, source = DPBudget(1), SchedulerDPBudget(0)
        old_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        source.begin(SimpleNamespace(dp_budget_ticket=old_budget.ticket(0)))
        source.complete()
        load = LoadSnapshot(timestamp=1)
        source.attach(_scheduler(), load)
        budget = DPBudget(1)
        budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 20)
        budget.update_budget([load])
        self.assertEqual((budget.total_requests, budget.total_tokens), ([1], [20]))

    def test_source_restart_is_rejected(self):
        """Assert a new scheduler generation cannot silently reset dispatch accounting."""
        budget, source = DPBudget(1), SchedulerDPBudget(0)
        load = LoadSnapshot(timestamp=1)
        source.attach(_scheduler(), load)
        budget.update_budget([load])
        restarted = LoadSnapshot(timestamp=2)
        SchedulerDPBudget(0).attach(_scheduler(), restarted)
        with self.assertRaisesRegex(ValueError, "restarted"):
            budget.update_budget([restarted])

    def test_future_or_regressing_confirmation_is_rejected(self):
        """Assert impossible ACKs fail without changing the last valid budget."""
        budget, source = DPBudget(1), SchedulerDPBudget(0)
        budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 10)
        source.begin(SimpleNamespace(dp_budget_ticket=budget.ticket(0)))
        source.complete()
        load = LoadSnapshot(timestamp=1, num_running_reqs=1, num_total_tokens=10)
        source.attach(_scheduler(), load)
        budget.update_budget([load])
        for ack, tokens in ((2, 10), (0, 0), (1, 0), (1, 11)):
            with self.subTest(ack=ack, tokens=tokens):
                bad = msgspec.structs.replace(
                    load,
                    dp_budget=msgspec.structs.replace(
                        load.dp_budget,
                        publication=2,
                        confirmed_requests=ack,
                        confirmed_tokens=tokens,
                    ),
                )
                with self.assertRaisesRegex(ValueError, "confirmation"):
                    budget.update_budget([bad])
                self.assertEqual(
                    (budget.total_requests, budget.total_tokens), ([1], [10])
                )

    def test_ticket_fifo_and_rank_validation(self):
        """Assert source handling rejects sequence gaps, wrong ranks and foreign epochs."""
        source = SchedulerDPBudget(0)
        for ticket in (("epoch", 0, 2, 10), ("epoch", 1, 1, 10)):
            with self.assertRaisesRegex(ValueError, "ticket"):
                source.begin(SimpleNamespace(dp_budget_ticket=ticket))
        source.begin(SimpleNamespace(dp_budget_ticket=("epoch", 0, 1, 10)))
        source.complete()
        with self.assertRaisesRegex(ValueError, "ticket"):
            source.begin(SimpleNamespace(dp_budget_ticket=("other", 0, 2, 20)))

    def test_snapshot_cannot_ack_incomplete_request_handler(self):
        """Assert publishing during a request handler leaves its ticket unconfirmed."""
        source = SchedulerDPBudget(0)
        source.begin(SimpleNamespace(dp_budget_ticket=("epoch", 0, 1, 10)))
        with self.assertRaisesRegex(ValueError, "handling"):
            source.attach(_scheduler(), LoadSnapshot())
        self.assertEqual(source.confirmed_requests, 0)

    def test_hidden_queues_are_deduplicated_and_terminal_work_is_removed(self):
        """Assert grammar, overlap, chunked and held P/D work stays in the routing load."""
        source = SchedulerDPBudget(0)
        source.begin(SimpleNamespace(dp_budget_ticket=("epoch", 0, 1, 10)))
        source.complete()
        normal, grammar, held, inflight, overlap, terminal = (
            _req(10),
            _req(20),
            _req(30),
            _req(40),
            _req(50),
            _req(60, True),
        )
        state = _scheduler(
            waiting_queue=[normal],
            grammar_manager=SimpleNamespace(grammar_queue=[grammar, terminal]),
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[SimpleNamespace(req=normal)],
                retracted_queue=[],
                held_rebootstrap_reqs=[held],
            ),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            disagg_prefill_inflight_queue=[inflight],
            last_batch=SimpleNamespace(reqs=[normal, grammar, overlap, terminal]),
            result_queue=[(SimpleNamespace(reqs=[overlap]), None)],
            chunked_req=overlap,
        )
        load = LoadSnapshot(num_waiting_reqs=1, num_total_tokens=10)
        source.attach(state, load)
        self.assertEqual(
            (load.dp_budget.total_requests, load.dp_budget.total_tokens), (5, 150)
        )
        self.assertEqual((load.num_waiting_reqs, load.num_total_tokens), (1, 10))
        for req in (grammar, held, inflight, overlap):
            req.finished.return_value = True
        source.attach(state, load)
        self.assertEqual(
            (load.dp_budget.total_requests, load.dp_budget.total_tokens), (1, 10)
        )

    def test_confirmation_round_trips_in_msgpack_and_stays_internal(self):
        """Assert load codecs carry accounting metadata without exposing it in the API."""
        source = SchedulerDPBudget(0)
        source.begin(SimpleNamespace(dp_budget_ticket=("epoch", 0, 1, 10)))
        source.complete()
        load = LoadSnapshot(timestamp=1)
        source.attach(_scheduler(), load)
        decoded = snapshot_decoder.decode(snapshot_encoder.encode(load))
        self.assertEqual(decoded, load)
        self.assertNotIn("dp_budget", decoded.to_dict({"core"}))
        self.assertNotIn("dp_budget", decoded.to_dict({"all"}))

    def test_confirmation_round_trips_in_shm_and_zmq(self):
        """Assert actual SHM and ZMQ readers preserve ACKs and unconfirmed reservations."""
        for transport in ("shm", "zmq"):
            with self.subTest(
                transport=transport
            ), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "load.shm")
                if transport == "shm":
                    writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
                    reader = ShmLoadSnapshotReader(path, dp_size=1)
                else:
                    endpoint = "ipc://" + os.path.join(tmp, "load.sock")
                    reader = ZmqShmLoadSnapshotReader(endpoint, path, dp_size=1)
                    writer = ZmqLoadSnapshotWriter(endpoint, dp_size=1, dp_rank=0)
                try:
                    budget, source = DPBudget(1), SchedulerDPBudget(0)
                    budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 10)
                    first = SimpleNamespace(dp_budget_ticket=budget.ticket(0))
                    budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, 20)
                    source.begin(first)
                    source.complete()
                    load = LoadSnapshot(
                        timestamp=1, num_waiting_reqs=1, num_total_tokens=10
                    )
                    source.attach(_scheduler(waiting_queue=[_req(10)]), load)
                    writer.write(load)
                    if transport == "zmq":
                        self.assertTrue(reader._socket.poll(1000))
                    received = reader.read_all()
                    self.assertEqual(received, [load])
                    budget.update_budget(received)
                    self.assertEqual(
                        (budget.total_requests, budget.total_tokens), ([2], [30])
                    )
                finally:
                    reader.close()
                    writer.close()


if __name__ == "__main__":
    unittest.main()
