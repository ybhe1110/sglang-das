"""Causal accounting for load-aware DP dispatch, independent of GPU execution."""

from enum import Enum, auto
from typing import Optional
from uuid import uuid4

import msgspec


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        """Parse a load-balancing method.

        Args:
            method: Case-insensitive enum name.
        Returns:
            Matching LoadBalanceMethod.
        Raises:
            ValueError: The name is unsupported.
        """
        try:
            return cls[method.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


# Controller epoch, target rank, dispatch sequence, cumulative estimated tokens.
DPBudgetTicket = tuple[str, int, int, int]


class DPBudgetSnapshot(msgspec.Struct, array_like=True):
    """Source load and the dispatch prefix included in the same snapshot."""

    epoch: Optional[str]
    generation: str
    publication: int
    confirmed_requests: int
    confirmed_tokens: int
    total_requests: int
    total_tokens: int


class DPBudget:
    """Source load plus dispatches not yet confirmed by that source."""

    def __init__(self, dp_size: int):
        """Initialize empty per-rank budgets.

        Args:
            dp_size: Number of ranks controlled by this process.
        Returns:
            None.
        """
        self.dp_size = dp_size
        self.epoch = uuid4().hex
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.last_timestamp = [0.0] * dp_size
        self.sent_requests = [0] * dp_size
        self.sent_tokens = [0] * dp_size
        self.confirmed_requests = [0] * dp_size
        self.confirmed_tokens = [0] * dp_size
        self.generations = [None] * dp_size
        self.publications = [0] * dp_size

    def update_budget(self, loads):
        """Apply loads without erasing unconfirmed reservations; return None.

        Args:
            loads: Per-rank LoadSnapshot objects from SHM or ZMQ.
        Returns:
            None.
        Raises:
            ValueError: A source restarts or confirms an invalid dispatch prefix.
        """
        for load in loads:
            rank = load.dp_rank
            cut = load.dp_budget
            if cut is None:
                if self.generations[rank] is not None:
                    raise ValueError("DP budget confirmation missing from source")
                # A snapshot without an ACK cannot confirm any local dispatch.
                if load.timestamp <= self.last_timestamp[rank]:
                    continue
                requests = load.num_running_reqs + load.num_waiting_reqs
                tokens = load.num_total_tokens
            else:
                if cut.epoch not in (None, self.epoch):
                    continue
                generation = self.generations[rank]
                if generation is not None and generation != cut.generation:
                    raise ValueError(
                        "DP budget source restarted; restart controller too"
                    )
                if cut.publication <= self.publications[rank]:
                    continue
                ack, ack_tokens = cut.confirmed_requests, cut.confirmed_tokens
                if (
                    not self.confirmed_requests[rank] <= ack <= self.sent_requests[rank]
                    or not self.confirmed_tokens[rank]
                    <= ack_tokens
                    <= self.sent_tokens[rank]
                    or (
                        ack == self.confirmed_requests[rank]
                        and ack_tokens != self.confirmed_tokens[rank]
                    )
                    or (
                        ack == self.sent_requests[rank]
                        and ack_tokens != self.sent_tokens[rank]
                    )
                    or (cut.epoch is None and (ack or ack_tokens))
                    or cut.total_requests < 0
                    or cut.total_tokens < 0
                ):
                    raise ValueError("Invalid DP budget confirmation")
                self.generations[rank] = cut.generation
                self.publications[rank] = cut.publication
                self.confirmed_requests[rank] = ack
                self.confirmed_tokens[rank] = ack_tokens
                requests, tokens = cut.total_requests, cut.total_tokens
            self.last_timestamp[rank] = load.timestamp
            self.total_requests[rank] = (
                requests + self.sent_requests[rank] - self.confirmed_requests[rank]
            )
            self.total_tokens[rank] = (
                tokens + self.sent_tokens[rank] - self.confirmed_tokens[rank]
            )

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        """Choose a rank and reserve one request and estimated_tokens.

        Args:
            method: Load-aware rank selection policy.
            estimated_tokens: Nonnegative token reservation for this dispatch.
        Returns:
            Selected rank, or None for methods without load-aware accounting.
        """
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be nonnegative")
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None
        self.sent_requests[rank] += 1
        self.sent_tokens[rank] += estimated_tokens
        self.total_requests[rank] += 1
        self.total_tokens[rank] += estimated_tokens
        return rank

    def ticket(self, rank: int) -> DPBudgetTicket:
        """Build an IPC ticket for the latest reservation.

        Args:
            rank: Rank returned by the immediately preceding dispatch.
        Returns:
            Controller epoch, rank, sequence and cumulative estimated tokens.
        """
        return (self.epoch, rank, self.sent_requests[rank], self.sent_tokens[rank])


class SchedulerDPBudget:
    """Confirm FIFO dispatches only after handling and account for hidden queues."""

    def __init__(self, rank: int):
        """Initialize a fresh scheduler generation.

        Args:
            rank: Rank whose load this scheduler publishes.
        Returns:
            None.
        """
        self.rank = rank
        self.epoch = None
        self.generation = uuid4().hex
        self.publication = 0
        self.confirmed_requests = 0
        self.confirmed_tokens = 0
        self.pending = None

    def begin(self, req) -> None:
        """Validate req's optional ticket before handling; return None.

        Args:
            req: Tokenized request or control message with an optional ticket.
        Returns:
            None.
        Raises:
            ValueError: The ticket has a foreign epoch/rank or a FIFO gap.
        """
        ticket = getattr(req, "dp_budget_ticket", None)
        if ticket is None:
            return
        epoch, rank, sequence, tokens = ticket
        if (
            self.pending is not None
            or rank != self.rank
            or self.epoch not in (None, epoch)
            or sequence != self.confirmed_requests + 1
            or tokens < self.confirmed_tokens
        ):
            raise ValueError("Invalid DP budget dispatch ticket")
        self.epoch = epoch
        self.pending = ticket

    def complete(self) -> None:
        """Confirm the handled or immediately rejected ticket.

        Returns:
            None. With no pending ticket, no counters change.
        """
        if self.pending is not None:
            _, _, self.confirmed_requests, self.confirmed_tokens = self.pending
            self.pending = None

    def attach(self, scheduler, snapshot) -> None:
        """Attach source load and its causal confirmation to snapshot in place.

        Args:
            scheduler: Scheduler owning the live queues and batches.
            snapshot: LoadSnapshot built from the same scheduler state.
        Returns:
            None. Public load counters retain their existing meanings.
        Raises:
            ValueError: A request handler has not yet completed.
        """
        if self.pending is not None:
            raise ValueError("Cannot publish a DP budget during request handling")
        extras = _unreported_requests(scheduler) if self.epoch is not None else []
        self.publication += 1
        snapshot.dp_budget = DPBudgetSnapshot(
            self.epoch,
            self.generation,
            self.publication,
            self.confirmed_requests,
            self.confirmed_tokens,
            snapshot.num_running_reqs + snapshot.num_waiting_reqs + len(extras),
            # Hidden queues may already own KV. This is a conservative routing
            # estimate, not an allocation or a change to public KV statistics.
            snapshot.num_total_tokens + sum(req.seqlen for req in extras),
        )


def _unreported_requests(scheduler) -> list:
    """Return live Req objects omitted by public running/waiting counters.

    Args:
        scheduler: Scheduler containing ordinary and P/D queues and overlap batches.
    Returns:
        Requests deduplicated by object identity, excluding terminal requests.
    """
    normal = list(scheduler.running_batch.reqs) + list(scheduler.waiting_queue)
    hidden = list(scheduler.grammar_manager.grammar_queue)
    bootstrap = scheduler.disagg_prefill_bootstrap_queue
    if bootstrap is not None:
        normal.extend(bootstrap.queue)
        hidden.extend(scheduler.disagg_prefill_inflight_queue)
    prealloc = scheduler.disagg_decode_prealloc_queue
    if prealloc is not None:
        normal.extend(prealloc.queue)
        normal.extend(prealloc.retracted_queue)
        normal.extend(scheduler.disagg_decode_transfer_queue.queue)
        hidden.extend(prealloc.held_rebootstrap_reqs)
    for batch in (
        scheduler.last_batch,
        scheduler.cur_batch_for_debug,
        *(item[0] for item in getattr(scheduler, "result_queue", ())),
    ):
        if batch is not None:
            hidden.extend(batch.reqs)
    if scheduler.chunked_req is not None:
        hidden.append(scheduler.chunked_req)
    normal_ids = {id(getattr(req, "req", req)) for req in normal}
    extra = {}
    for value in hidden:
        req = getattr(value, "req", value)
        if id(req) not in normal_ids and not req.finished():
            extra[id(req)] = req
    return list(extra.values())
