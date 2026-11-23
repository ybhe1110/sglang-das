"""Bounded, per-HTTP-worker execution for stateless tokenizer utilities.

No HTTP objects or engine IPC handles cross the executor boundary. Cancellation
abandons the reply, not the physical queue item: that item retains its work and
response reservations until it has actually returned (or been dequeued).
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from functools import partial
from typing import Callable

logger = logging.getLogger(__name__)
ERROR_RESPONSE_BYTES = 256


@dataclass(frozen=True)
class UtilityLimits:
    workers: int = 1
    max_admitted: int = 8
    max_input_bytes: int = 2 * 1024 * 1024
    max_total_input_bytes: int = 16 * 1024 * 1024
    max_input_items: int = 262144
    max_batch_size: int = 1024
    max_output_bytes: int = 8 * 1024 * 1024
    max_total_output_bytes: int = 64 * 1024 * 1024
    timeout: float = 60.0
    body_timeout: float = 15.0
    shutdown_timeout: float = 10.0

    @classmethod
    def from_config(cls, config):
        return cls(
            **{f.name: getattr(config, f"utility_{f.name}") for f in fields(cls)}
        )

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            is_duration = f.name in {"timeout", "body_timeout", "shutdown_timeout"}
            valid_type = (
                type(value) in (int, float) if is_duration else type(value) is int
            )
            if (
                not valid_type
                or value <= 0
                or (is_duration and not math.isfinite(value))
            ):
                raise ValueError(
                    f"--utility-{f.name.replace('_', '-')} must be positive"
                )
        if self.workers > self.max_admitted:
            raise ValueError("--utility-workers must not exceed --utility-max-admitted")
        if self.max_output_bytes < ERROR_RESPONSE_BYTES:
            raise ValueError(
                "--utility-max-output-bytes must be at least 256 for errors"
            )
        for kind in ("input", "output"):
            if getattr(self, f"max_total_{kind}_bytes") < getattr(
                self, f"max_{kind}_bytes"
            ):
                raise ValueError(f"Total utility {kind} budget must fit one request")


class UtilityError(Exception):
    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.status = status


class UtilityTicket:
    """Loop-owned admission; the worker only reads the cancellation Event."""

    def __init__(self, service):
        self.service = service
        self.cancelled = threading.Event()
        self.submitted = False
        self.work_done = False
        self.http_done = False
        self.response_done = False
        self.reply = None
        self.input_bytes = service.limits.max_input_bytes
        self.output_bytes = service.limits.max_output_bytes
        self.admitted_at = time.monotonic()
        self.started_at = None
        self.http_owner = None
        self.http_cleanup = None

    def close(self):
        """Called by the ASGI owner on send completion, disconnect or error."""
        if self.http_done:
            return
        self.http_done = True
        self.http_owner = None
        self.cancelled.set()
        if self.reply is not None:
            if self.reply.done() and not self.reply.cancelled():
                self.reply.exception()
            self.reply.cancel()
            self.reply = None
        if not self.submitted:
            self.service._release_work(self)
        if self.work_done:
            self.service._release_response(self)
        self.service._observe()


class UtilityService:
    def __init__(self, limits: UtilityLimits, metrics=None, *, validate_request=None):
        self.limits = limits
        self.validate_request = validate_request
        self.loop = asyncio.get_running_loop()
        self.metrics = metrics
        self.accepting = False
        self.tickets = set()
        self.work_used = 0
        self.input_bytes = 0
        self.response_used = 0
        self.output_bytes = 0
        self._workers = []
        self._idle = []
        self._waiting = deque()
        self._lag_task = None
        self._empty = asyncio.Event()
        self._empty.set()

    async def start(self, backend_factory: Callable):
        """Build private tokenizer/handler instances before HTTP starts serving."""
        try:
            for _ in range(self.limits.workers):
                pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="sgl-utility"
                )
                self._workers.append(pool)
                backend = await self.loop.run_in_executor(pool, backend_factory)
                self._idle.append((pool, backend))
            self.accepting = True
            if self.metrics:
                self._lag_task = asyncio.create_task(self._observe_loop_lag())
        except BaseException:
            await self.aclose()
            raise

    def admit(self) -> UtilityTicket:
        if not self.accepting:
            raise UtilityError("Tokenizer utility service is unavailable", 503)
        lim = self.limits
        if (
            self.work_used >= lim.max_admitted
            or self.response_used >= lim.max_admitted
            or self.input_bytes + lim.max_input_bytes > lim.max_total_input_bytes
            or self.output_bytes + lim.max_output_bytes > lim.max_total_output_bytes
        ):
            if self.metrics:
                self.metrics.rejected.inc()
            raise UtilityError("Tokenizer utility capacity exhausted; retry later")
        ticket = UtilityTicket(self)
        self.tickets.add(ticket)
        self._empty.clear()
        self.work_used += 1
        self.input_bytes += ticket.input_bytes
        self.response_used += 1
        self.output_bytes += ticket.output_bytes
        self._observe()
        return ticket

    async def run(self, ticket: UtilityTicket, kind: str, request, *, before_run=None):
        if ticket.service is not self or ticket.submitted or ticket.http_done:
            raise RuntimeError("Utility ticket is not available for submission")
        if not self.accepting:
            raise UtilityError("Tokenizer utility service is shutting down", 503)
        # The Pydantic request is exclusively owned until the worker returns.
        # No deepcopy/model_dump of a large payload on the HTTP loop.
        ticket.submitted = True
        reply = ticket.reply = self.loop.create_future()

        def invoke(backend):
            if ticket.cancelled.is_set():
                return None
            ticket.started_at = time.monotonic()
            if self.metrics:
                self.metrics.queue_seconds.observe(
                    ticket.started_at - ticket.admitted_at
                )
            try:
                if self.validate_request is not None:
                    self.validate_request(request, self.limits)
                    if ticket.cancelled.is_set():
                        return None
                if before_run is not None:
                    before_run()
                    if ticket.cancelled.is_set():
                        return None
                result = backend(kind, request)
                if len(result.body) > self.limits.max_output_bytes:
                    raise UtilityError(
                        "Tokenizer utility response exceeds byte limit", 413
                    )
                return result
            finally:
                if self.metrics:
                    self.metrics.execution_seconds.observe(
                        time.monotonic() - ticket.started_at
                    )

        # Every queued item owns an admission slot, including after cancellation.
        self._waiting.append((ticket, invoke))
        self._dispatch()
        try:
            return await asyncio.shield(reply)
        except asyncio.CancelledError:
            ticket.cancelled.set()
            if reply.done() and not reply.cancelled():
                reply.exception()
            reply.cancel()
            self._observe()
            raise

    def _dispatch(self):
        while self._idle and self._waiting:
            ticket, invoke = self._waiting.popleft()
            if ticket.cancelled.is_set():
                self._finish(ticket)
                continue
            worker = self._idle[-1]
            pool, backend = worker
            try:
                future = pool.submit(invoke, backend)
            except Exception as exc:
                self._finish(ticket, error=exc)
                continue
            self._idle.pop()
            future.add_done_callback(
                partial(self.loop.call_soon_threadsafe, self._settle, ticket, worker)
            )

    def _settle(self, ticket, worker, future):
        try:
            result = future.result()  # Always consume the backend exception.
        except BaseException as exc:
            self._finish(ticket, error=exc)
        else:
            self._finish(ticket, result=result)
        self._idle.append(worker)
        self._dispatch()

    def _finish(self, ticket, *, result=None, error=None):
        reply = ticket.reply
        if reply is not None and not reply.done():
            if error is not None:
                reply.set_exception(error)
            else:
                if result is not None:
                    # A pre-header deadline can replace this result with an error.
                    size = max(ERROR_RESPONSE_BYTES, len(result.body))
                    self.output_bytes -= ticket.output_bytes - size
                    ticket.output_bytes = size
                reply.set_result(result)
        self._release_work(ticket)
        if ticket.http_done:
            self._release_response(ticket)
        self._observe()

    def _release_work(self, ticket):
        if not ticket.work_done:
            ticket.work_done = True
            self.work_used -= 1

    def _release_response(self, ticket):
        if not ticket.response_done:
            ticket.response_done = True
            self.response_used -= 1
            # FastAPI may retain the parsed request until send has completed.
            self.input_bytes -= ticket.input_bytes
            self.output_bytes -= ticket.output_bytes
            self.tickets.remove(ticket)
            if not self.tickets:
                self._empty.set()

    def snapshot(self):
        return {
            "work": self.work_used,
            "input_bytes": self.input_bytes,
            "responses": self.response_used,
            "output_bytes": self.output_bytes,
            "cancelled_work": sum(
                t.cancelled.is_set() and not t.work_done for t in self.tickets
            ),
        }

    def _observe(self):
        if self.metrics:
            self.metrics.observe(self.snapshot())

    async def _observe_loop_lag(self):
        while True:
            target = self.loop.time() + 0.1
            await asyncio.sleep(0.1)
            self.metrics.loop_lag_seconds.observe(max(0, self.loop.time() - target))

    async def aclose(self):
        self.accepting = False
        # The HTTP owner, including a slow sender, must finish before its
        # response reservation is released. Wake callers waiting for work, but
        # never pretend a still-running ASGI send has finished.
        for ticket in list(self.tickets):
            ticket.cancelled.set()
            if ticket.reply is not None and not ticket.reply.done():
                ticket.reply.cancel()
        # Stop executor admission without cancelling physical queue entries.
        # Their completion callbacks must still account for actual work.
        for pool in self._workers:
            pool.shutdown(wait=False)

        async def drain():
            await self._empty.wait()
            # All submitted work has physically returned; joining is now safe.
            for pool in self._workers:
                await asyncio.to_thread(pool.shutdown, wait=True)

        try:
            await asyncio.wait_for(drain(), self.limits.shutdown_timeout)
            self._workers.clear()
            self._idle.clear()
        except TimeoutError:
            for ticket in list(self.tickets):
                owner = ticket.http_owner
                if owner is not None and owner is not asyncio.current_task():
                    owner.cancel()
            logger.error(
                "Utility shutdown exceeded %.3fs: %s; reservations retained. "
                "Process supervisor must enforce the final termination deadline "
                "if native work or HTTP cancellation does not complete.",
                self.limits.shutdown_timeout,
                self.snapshot(),
            )
            # Do not clear tickets: late HTTP cleanup and _settle remain valid.
            raise TimeoutError(
                "Tokenizer utility shutdown grace period expired"
            ) from None
        finally:
            if self._lag_task is not None:
                self._lag_task.cancel()
                await asyncio.gather(self._lag_task, return_exceptions=True)
                self._lag_task = None
            self._observe()
