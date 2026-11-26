"""Real HTTP supervisor fixture; model work is a disposable child process."""

import ast
import importlib.util
import logging
import multiprocessing as mp
import os
import pathlib
import signal
import sys
import threading
import time

import uvicorn

ROOT = pathlib.Path(__file__).resolve().parents[3]


def load_watchdog():
    path = ROOT / "python/sglang/srt/utils/watchdog.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if getattr(n, "name", None) == "SubprocessWatchdog"
    )
    env = dict(
        logger=logging.getLogger(__name__), os=os, signal=signal, threading=threading
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env["SubprocessWatchdog"]


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        state_dir = pathlib.Path(os.environ["SGLANG_FAILSTOP_FIXTURE_DIR"])
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                (state_dir / f"http-{os.getpid()}.pid").write_text(str(os.getpid()))
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    else:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def critical_worker(directory, exitcode):
    path = pathlib.Path(directory)
    (path / "critical.pid").write_text(str(os.getpid()))
    while not (path / "trigger").exists():
        time.sleep(0.02)
    os._exit(exitcode)


def main():
    directory, port, exitcode = sys.argv[1:]
    os.environ["SGLANG_FAILSTOP_FIXTURE_DIR"] = directory
    child = mp.get_context("spawn").Process(
        target=critical_worker, args=(directory, int(exitcode))
    )
    child.start()
    spec = importlib.util.spec_from_file_location(
        "failstop_callback", ROOT / "python/sglang/srt/utils/process_supervision.py"
    )
    callback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(callback)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, signal.SIG_IGN)
    monitor = load_watchdog()(
        [child],
        ["detokenizer"],
        interval=0.05,
        unexpected_exit_is_fatal=True,
        on_failure=callback.fatal_subprocess_exit,
    )
    monitor.start()
    try:
        uvicorn.run(
            "subprocess_failstop_fixture:app",
            host="127.0.0.1",
            port=int(port),
            workers=2,
            loop="asyncio",
            log_level="warning",
        )
    finally:
        monitor.stop()
        if child.is_alive():
            child.terminate()
        child.join(timeout=2)


if __name__ == "__main__":
    main()
