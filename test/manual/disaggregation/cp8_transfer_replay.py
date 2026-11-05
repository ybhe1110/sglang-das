"""Fault fixture for an isolated CP8 prefill server using SGLang's fake sender.

In the disposable server's fake/conn.py append:
    from cp8_transfer_replay import install
    install(FakeKVSender)
Create /tmp/cp8-transfer-replay, record the server PID in server.pid there,
and put this module on PYTHONPATH. Run it in the same pod after /health is ready.
Real model forwards and scheduler collectives run; the two reserved rooms
control the transport outcome and local source drain without touching RDMA.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONTROL = Path("/tmp/cp8-transfer-replay")
FAIL, CANCEL = 720081, 720082


def install(cls):
    from sglang.srt.disaggregation.base.conn import KVPoll
    from torch import distributed as dist

    original_init, original_poll = cls.__init__, cls.poll

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.replay_room = kwargs.get("bootstrap_room")

    def poll(self):
        if self.replay_room not in (FAIL, CANCEL) or not self.has_sent:
            return original_poll(self)
        rank = dist.get_rank()
        prefix = CONTROL / str(self.replay_room)
        if not getattr(self, "replay_started", False):
            prefix.with_suffix(f".rank{rank}").touch()
            self.replay_started = True
        failed = (
            self.conclude_state == KVPoll.Failed or prefix.with_suffix(".fail").exists()
        )
        if failed and (rank == 0 or prefix.with_suffix(".drained").exists()):
            return KVPoll.Failed
        return KVPoll.Transferring

    def forget(self):
        if self.replay_room in (FAIL, CANCEL):
            rank = dist.get_rank()
            prefix = CONTROL / str(self.replay_room)
            assert rank == 0 or prefix.with_suffix(".drained").exists(), (
                "early source release"
            )
            prefix.with_suffix(f".released{rank}").touch()
        return True

    cls.__init__, cls.poll, cls.forget_source_transfers = init, poll, forget


def run(url, baseline):
    import requests

    CONTROL.mkdir(exist_ok=True)

    def generate(room):
        return requests.post(
            f"{url}/generate",
            json={
                "rid": f"replay-{room}",
                "input_ids": [42] * 256,
                "bootstrap_host": "2.2.2.2",
                "bootstrap_room": room,
                "bootstrap_pair_key": str(room),
                "sampling_params": {"max_new_tokens": 1},
            },
            timeout=180,
        )

    def healthy():
        return requests.get(f"{url}/health", timeout=8).status_code == 200

    pid = int((CONTROL / "server.pid").read_text())
    assert healthy()
    for room in (FAIL, CANCEL):
        prefix = CONTROL / str(room)
        for f in CONTROL.glob(f"{room}.*"):
            f.unlink()
        with ThreadPoolExecutor(1) as pool:
            response = pool.submit(generate, room)
            for _ in range(1200):
                if len(list(CONTROL.glob(f"{room}.rank*"))) == 8:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError(
                    "request did not reach transfer on all eight ranks"
                )
            if room == FAIL:
                prefix.with_suffix(".fail").touch()
            else:
                abort = requests.post(
                    f"{url}/abort_request", json={"rid": f"replay-{room}"}, timeout=5
                )
                assert abort.status_code == 200, abort.text
            # Let several scheduler loops run while only rank zero has drained.
            time.sleep(1)
            prefix.with_suffix(".drained").touch()
            if baseline:
                for _ in range(150):
                    stat = Path(f"/proc/{pid}/stat")
                    if not stat.exists() or stat.read_text().split(") ")[1][0] == "Z":
                        print("baseline: P exited on split drain", flush=True)
                        return
                    time.sleep(0.1)
                raise AssertionError("baseline P did not exit")
            result = response.result(timeout=30)
            if room == CANCEL:
                assert result.status_code == 499, result.text
            assert result.status_code >= 400 or "abort" in result.text.lower(), (
                result.text
            )
            for _ in range(100):
                if len(list(CONTROL.glob(f"{room}.released*"))) == 8:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("some ranks did not release the failed transfer")
            good = generate(room + 100)
            assert good.status_code == 200, good.text
            meta = good.json()["meta_info"]
            assert meta["id"] == f"replay-{room + 100}", meta
            assert meta["finish_reason"]["type"] in {"stop", "length"}, meta
            os.kill(pid, 0)
            assert healthy()
            print(
                f"room={room} failed={result.status_code}, next request={good.status_code}, P pid={pid}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    run(args.url, args.baseline)
