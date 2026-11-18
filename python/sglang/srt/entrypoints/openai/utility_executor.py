"""Bounded, per-HTTP-worker execution for stateless tokenizer utilities.

No HTTP objects or engine IPC handles cross the executor boundary. Cancellation
abandons the reply, not the physical queue item: that item retains its work and
response reservations until it has actually returned (or been dequeued).
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from typing import Callable


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

    @classmethod
    def from_config(cls, config):
        return cls(
            **{f.name: getattr(config, f"utility_{f.name}") for f in fields(cls)}
        )

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"--utility-{f.name.replace('_', '-')} must be positive"
                )
        if self.workers > self.max_admitted:
            raise ValueError("--utility-workers must not exceed --utility-max-admitted")
        if self.max_output_bytes < 256:
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

    def close(self):
        """Called by the ASGI owner on send completion, disconnect or error."""
        if self.http_done:
            return
        self.http_done = True
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
    def __init__(self, limits: UtilityLimits, metrics=None):
        self.limits = limits
        self.loop = asyncio.get_running_loop()
        self.metrics = metrics
        self.accepting = False
        self.tickets = set()
        self.work_used = 0
        self.input_bytes = 0
        self.response_used = 0
        self.output_bytes = 0
        self._workers = []
        self._pending = []
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
                self._workers.append((pool, None))
                backend = await self.loop.run_in_executor(pool, backend_factory)
                self._workers[-1] = (pool, backend)
                self._pending.append(0)
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

    async def run(self, ticket: UtilityTicket, kind: str, request):
        if ticket.service is not self or ticket.submitted or ticket.http_done:
            raise RuntimeError("Utility ticket is not available for submission")
        if not self.accepting:
            raise UtilityError("Tokenizer utility service is shutting down", 503)
        # The Pydantic request is exclusively owned until the worker returns.
        # No deepcopy/model_dump of a large payload on the HTTP loop.
        ticket.submitted = True
        reply = ticket.reply = self.loop.create_future()
        index = min(range(len(self._pending)), key=self._pending.__getitem__)
        self._pending[index] += 1
        pool, backend = self._workers[index]

        def invoke():
            if ticket.cancelled.is_set():
                return None
            ticket.started_at = time.monotonic()
            if self.metrics:
                self.metrics.queue_seconds.observe(
                    ticket.started_at - ticket.admitted_at
                )
            try:
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

        try:
            future = pool.submit(invoke)
        except BaseException:
            self._pending[index] -= 1
            self._release_work(ticket)
            raise
        future.add_done_callback(
            lambda done: self.loop.call_soon_threadsafe(
                self._settle, ticket, index, done
            )
        )
        try:
            return await asyncio.shield(reply)
        except asyncio.CancelledError:
            ticket.cancelled.set()
            if reply.done() and not reply.cancelled():
                reply.exception()
            reply.cancel()
            self._observe()
            raise

    def _settle(self, ticket, index, future):
        self._pending[index] -= 1
        reply = ticket.reply
        try:
            result = future.result()  # Always consume the backend exception.
        except BaseException as exc:
            if reply is not None and not reply.done():
                reply.set_exception(exc)
        else:
            if reply is not None and not reply.done():
                if result is not None:
                    size = len(result.body)
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
        await self._empty.wait()
        for pool, _ in self._workers:
            # wait=True must never block the serving event loop.
            await asyncio.to_thread(pool.shutdown, wait=True)
        self._workers.clear()
        if self._lag_task is not None:
            self._lag_task.cancel()
            await asyncio.gather(self._lag_task, return_exceptions=True)
            self._lag_task = None
        self._observe()
