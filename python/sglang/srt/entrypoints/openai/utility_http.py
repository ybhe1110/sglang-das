"""One ASGI owner for utility body limits, disconnects and response leases."""

from __future__ import annotations

import asyncio

from sglang.srt.entrypoints.openai.utility_executor import UtilityError
from starlette.responses import JSONResponse

UTILITY_PATHS = frozenset(
    {"/tokenize", "/v1/tokenize", "/detokenize", "/v1/detokenize"}
)


def utility_error_response(error: UtilityError):
    return JSONResponse(
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
        except UtilityError as error:
            try:
                await asyncio.wait_for(
                    utility_error_response(error)(scope, receive, send), 1.0
                )
            except TimeoutError:
                pass
            return

        tasks = []
        response_started = False

        async def send_error(error):
            # Even rejection/timeout responses need a finite slow-client wait.
            try:
                await asyncio.wait_for(
                    utility_error_response(error)(scope, receive, send),
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
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await send_error(
                    UtilityError("Tokenizer utility deadline exceeded", 504)
                )
        except UtilityError as error:
            if not response_started:
                await send_error(error)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            ticket.close()
