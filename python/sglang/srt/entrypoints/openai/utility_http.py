"""One ASGI owner for utility body limits, disconnects and response leases."""

from __future__ import annotations

import asyncio

from starlette.responses import JSONResponse

from sglang.srt.entrypoints.openai.utility_executor import (
    ERROR_RESPONSE_BYTES,
    UtilityError,
)

UTILITY_PATHS = frozenset(
    {"/tokenize", "/v1/tokenize", "/detokenize", "/v1/detokenize"}
)


def utility_validation_error_response(errors):
    # FastAPI already materialized this sequence. Inspect a bounded part of
    # the first diagnostic; input, ctx and the remaining items can be large.
    summary = "Invalid tokenizer request"
    if len(errors) > 1:
        summary += f" ({len(errors)} validation errors)"
    first = errors[0] if errors else {}
    location = ".".join(
        part[:24] for part in first.get("loc", ())[:3] if isinstance(part, str)
    )
    reason = first.get("msg", "")
    detail = " ".join(filter(None, (location, reason[:80])))
    payload = {
        "object": "error",
        "message": f"{summary}: {detail}" if detail else summary,
        "type": "Bad Request",
        "param": None,
        "code": 400,
    }
    response = JSONResponse(payload, status_code=400)
    if len(response.body) > ERROR_RESPONSE_BYTES:
        # JSON escaping and UTF-8 can expand even a short diagnostic.
        payload["message"] = summary
        response = JSONResponse(payload, status_code=400)
    return response


def utility_error_response(error: UtilityError):
    response = JSONResponse(
        {
            "object": "error",
            "message": str(error),
            "type": "UtilityError",
            "param": None,
            "code": error.status,
        },
        status_code=error.status,
        headers={"Retry-After": "1"} if error.status == 429 else None,
    )
    if len(response.body) > ERROR_RESPONSE_BYTES:
        return utility_error_response(
            UtilityError("Tokenizer utility request failed", error.status)
        )
    return response


class UtilityHTTPMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        root = scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or path.rstrip("/") not in UTILITY_PATHS
        ):
            return await self.app(scope, receive, send)

        service = getattr(scope["app"].state, "utility_service", None)
        try:
            if service is None:
                raise UtilityError("Tokenizer utility service is unavailable", 503)
            ticket = service.admit()
            ticket.http_owner = asyncio.current_task()
        except UtilityError as error:
            try:
                await asyncio.wait_for(
                    utility_error_response(error)(scope, receive, send), 1.0
                )
            except TimeoutError:
                pass
            return

        tasks = []
        cleanup = None
        response_started = False

        def cancel_children():
            nonlocal cleanup
            if cleanup is None:
                for task in tasks:
                    task.cancel()
                cleanup = asyncio.gather(*tasks, return_exceptions=True)
            return cleanup

        async def send_error(error):
            # Even rejection/timeout responses need a finite slow-client wait.
            response = utility_error_response(error)
            if len(response.body) > min(
                ticket.output_bytes, service.limits.max_output_bytes
            ):
                raise RuntimeError("Utility error response exceeds reserved output")
            try:
                await asyncio.wait_for(
                    response(scope, receive, send),
                    service.limits.body_timeout,
                )
            except TimeoutError:
                pass

        try:
            headers = dict(scope.get("headers", []))
            # The optional global decompressor buffers unbounded expanded input
            # before routing. Do not let it bypass this endpoint's byte budget.
            if (
                b"x-body-compressed" in headers
                or headers.get(b"content-encoding", b"identity") != b"identity"
            ):
                raise UtilityError("Send uncompressed JSON to tokenizer utilities", 415)
            if b"content-length" in headers:
                try:
                    length = int(headers[b"content-length"])
                except ValueError:
                    raise UtilityError("Invalid Content-Length", 400) from None
                if length < 0:
                    raise UtilityError("Invalid Content-Length", 400)
                if length > service.limits.max_input_bytes:
                    raise UtilityError(
                        "Tokenizer utility input exceeds byte limit", 413
                    )

            async def read_body():
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return None
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > service.limits.max_input_bytes:
                        raise UtilityError(
                            "Tokenizer utility input exceeds byte limit", 413
                        )
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        return bytes(body)

            loop = asyncio.get_running_loop()
            deadline = loop.time() + service.limits.timeout
            body = await asyncio.wait_for(
                read_body(), min(service.limits.body_timeout, service.limits.timeout)
            )
            if body is None:
                return
            scope = dict(scope, utility_ticket=ticket)
            disconnected = asyncio.Event()
            pending_start = None
            sent_bytes = 0

            async def replay_receive():
                nonlocal body
                if body is not None:
                    message = {"type": "http.request", "body": body, "more_body": False}
                    body = None
                    return message
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def watch_disconnect():
                # The body is fully read. This is now the sole receive owner;
                # downstream receives a replay and the shared disconnect event.
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        disconnected.set()
                        return

            async def send_response(message):
                nonlocal response_started, pending_start, sent_bytes
                if message["type"] == "http.response.start":
                    if pending_start is not None or response_started:
                        raise RuntimeError("Duplicate utility http.response.start")
                    pending_start = message
                    return
                if message["type"] == "http.response.body":
                    sent_bytes += len(message.get("body", b""))
                    if sent_bytes > service.limits.max_output_bytes:
                        raise UtilityError(
                            "Tokenizer utility response exceeds byte limit", 413
                        )
                if pending_start is not None:
                    response_started = True
                    await send(pending_start)
                    pending_start = None
                await send(message)

            application = asyncio.create_task(
                self.app(scope, replay_receive, send_response)
            )
            watcher = asyncio.create_task(watch_disconnect())
            tasks = [application, watcher]
            done, _ = await asyncio.wait(
                tasks,
                timeout=max(0, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if watcher in done:
                await watcher  # Surface a broken receive transport too.
                return
            if application in done:
                await application
            else:
                raise TimeoutError
        except TimeoutError:
            if service.metrics:
                service.metrics.timed_out.inc()
            if not response_started:
                # Stop the handler before starting a timeout response.
                await asyncio.shield(cancel_children())
                await send_error(
                    UtilityError("Tokenizer utility deadline exceeded", 504)
                )
        except UtilityError as error:
            if not response_started:
                await send_error(error)
        finally:
            # A second owner cancellation must neither cancel the drain nor
            # lose the lease. The ticket retains the drain until both children
            # really finish, even if this middleware task has already exited.
            pending = ticket.http_cleanup = cancel_children()

            def finish_cleanup(_):
                ticket.http_cleanup = None
                ticket.close()

            if pending.done():
                finish_cleanup(pending)
            else:
                pending.add_done_callback(finish_cleanup)
            await asyncio.shield(pending)
