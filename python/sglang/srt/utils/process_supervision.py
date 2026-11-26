"""Fail-stop cleanup independent of an HTTP supervisor's signal handlers."""

import logging
import os
import threading

import psutil

logger = logging.getLogger(__name__)


def fatal_subprocess_exit(name: str, pid: int, exitcode: int, timeout: float = 5.0):
    """Retire the whole engine after losing a critical, stateful subprocess.

    This runs in the owner's watchdog thread. SIGQUIT may have been replaced
    by Uvicorn/Granian; neither an event-loop callback nor a signal is needed.
    The timer also bounds cleanup if process enumeration or logging blocks.
    """
    try:
        deadline = threading.Timer(timeout, os._exit, args=(1,))
        deadline.daemon = True
        deadline.start()
        logger.error(
            "Critical subprocess %s (pid=%s) exited with code %s; retiring engine",
            name,
            pid,
            exitcode,
        )
        children = psutil.Process().children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=timeout)
    finally:
        os._exit(1)
