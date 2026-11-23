import asyncio
from collections.abc import Coroutine
from typing import Any


async def wait_for_cleanup(cleanup: Coroutine[Any, Any, None]) -> None:
    """Finish cleanup and surface its errors before propagating caller cancellation."""
    task = asyncio.create_task(cleanup)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    task.result()
    if cancelled is not None:
        raise cancelled
