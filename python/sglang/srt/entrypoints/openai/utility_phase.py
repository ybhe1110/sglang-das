"""Opt-in per-request monotonic timestamps for the pinned HTTP utility probe."""

import json
import os
import time
import uuid
from contextvars import ContextVar
from functools import wraps

_trace = ContextVar("sglang_utility_phase_trace", default=None)
_HEADER = b"x-sglang-phase-probe"


def mark_phase(name):
    trace = _trace.get()
    if trace is not None:
        trace[name] = time.perf_counter_ns()


def bind_worker_trace(fn):
    """Transfer only this request's trace to its executor thread."""
    trace = _trace.get()
    if trace is None:
        return fn

    def traced(*args):
        token = _trace.set(trace)
        try:
            return fn(*args)
        finally:
            _trace.reset(token)

    return traced


def time_probe_handler(fn):
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        trace = _trace.get()
        if trace is None:
            return await fn(*args, **kwargs)
        trace["handler_begin"] = time.perf_counter_ns()
        try:
            return await fn(*args, **kwargs)
        finally:
            trace["handler_end"] = time.perf_counter_ns()

    return wrapped


class UtilityPhaseMiddleware:
    def __init__(self, app):
        self.app = app
        self.instance = uuid.uuid4().hex

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or (scope["method"], scope["path"])
            not in {
                ("GET", "/v1/models"),
                ("GET", "/v1/loads"),
                ("POST", "/v1/detokenize"),
                ("POST", "/detokenize"),
            }
            or (_HEADER, b"1") not in scope["headers"]
        ):
            return await self.app(scope, receive, send)

        trace = {"asgi_enter": time.perf_counter_ns()}
        token = _trace.set(trace)

        async def send_timed(message):
            # Errors/cancellations can outlive their HTTP owner on the worker.
            # Publish only completed successful work and never retain a registry.
            if message["type"] == "http.response.start" and message["status"] == 200:
                trace["headers_ready"] = time.perf_counter_ns()
                payload = json.dumps(
                    {"pid": os.getpid(), "instance": self.instance, "ns": trace},
                    separators=(",", ":"),
                ).encode("ascii")
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != b"cache-control"
                ]
                message = dict(
                    message,
                    headers=headers
                    + [(_HEADER, payload), (b"cache-control", b"no-store")],
                )
            await send(message)

        try:
            return await self.app(scope, receive, send_timed)
        finally:
            _trace.reset(token)
