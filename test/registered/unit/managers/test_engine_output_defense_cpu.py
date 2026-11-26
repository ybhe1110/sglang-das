"""Engine defenses for gateway-owned, unique-per-attempt request IDs.

Run production method bodies without importing GPU/platform-only modules.
The gateway RID is preserved; duplicate inputs are rejected before mutation.
"""

import ast
import asyncio
import collections
import dataclasses
import importlib.util
import logging
import pathlib
import sys
import time
import unittest
import uuid
from http import HTTPStatus
from types import SimpleNamespace
from typing import Optional  # noqa: F401 - msgspec resolves source annotations
from unittest.mock import patch

import msgspec
from fastapi import BackgroundTasks
from starlette.responses import Response

ROOT = pathlib.Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def source_namespace(relative, names=None, env=None, skip_imports=False):
    path = SRT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if names is not None:
        nodes = [node for node in nodes if getattr(node, "name", None) in names]
        assert len(nodes) == len(names), names
    if skip_imports:
        nodes = [
            n
            for n in nodes
            if not isinstance(n, ast.ImportFrom)
            or not (n.module or "").startswith("sglang.")
        ]
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    namespace = {"__name__": __name__, **(env or {})}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def methods(relative, class_name, names, env):
    path = SRT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", None) == class_name)
    body = [n for n in cls.body if getattr(n, "name", None) in names]
    assert len(body) == len(names), names
    cls.bases, cls.keywords, cls.decorator_list, cls.body = [], [], [], body
    namespace = {"__name__": __name__, **env}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


spec = importlib.util.spec_from_file_location(
    "output_defense_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {spec.name: ci}):
    spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEALTH_CHECK_RID_PREFIX = source_namespace("constants.py")["HEALTH_CHECK_RID_PREFIX"]
wire = source_namespace(
    "managers/io_struct.py",
    {"BaseReq", "AbortReq", "HealthCheckOutput", "ShutdownReq"},
    {"msgspec": msgspec},
)
ReqState = source_namespace(
    "managers/tokenizer_manager.py", {"ReqState"}, {"dataclasses": dataclasses}
)["ReqState"]


class TimeStats:
    trace_ctx = SimpleNamespace(tracing_enable=False)
    first_token_time = 0

    def get_e2e_latency(self):
        return 0.1

    def __getattr__(self, name):
        return lambda *a, **kw: None


class GenerateInput(SimpleNamespace):
    def __getitem__(self, i):
        return self.items[i]


class Packet(SimpleNamespace):
    def __getattr__(self, name):
        return None


serving = SimpleNamespace(tokenizer_worker_num=1, batch_notify_size=1)
Manager = methods(
    "managers/tokenizer_manager.py",
    "TokenizerManager",
    {
        "_init_req_state",
        "_discard_pending_req_states",
        "_handle_abort_req",
        "_handle_batch_output",
        "_handle_health_check_output",
        "_send_one_request",
        "create_abort_task",
        "abort_request",
        "_wait_one_response",
        "_handle_abort_finish_reason",
    },
    dict(
        wire,
        asyncio=asyncio,
        logger=logging.getLogger(__name__),
        ReqState=ReqState,
        APIServerReqTimeStats=lambda **kw: TimeStats(),
        BackgroundTasks=BackgroundTasks,
        GenerateReqInput=GenerateInput,
        BatchStrOutput=Packet,
        BatchTokenIDOutput=type("TokenPacket", (), {}),
        BatchEmbeddingOutput=type("EmbeddingPacket", (), {}),
        HEALTH_CHECK_RID_PREFIX=HEALTH_CHECK_RID_PREFIX,
        is_health_check_generate_req=lambda obj: obj.rid.startswith(
            HEALTH_CHECK_RID_PREFIX
        ),
        unwrap_from_pickle=lambda obj: obj,
        wrap_shm_features=lambda obj: obj,
        get_serving=lambda: serving,
        get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
        _REQUEST_STATE_WAIT_TIMEOUT=0.002,
        HTTPStatus=HTTPStatus,
    ),
)


def manager():
    m = Manager()
    m.rid_to_state, m.sent = {}, []
    m.enable_trace = m.enable_metrics = m.enable_lora = False
    m.disaggregation_mode = "none"
    m.incremental_streaming_output = False
    m.dump_requests_folder = m.crash_dump_folder = ""
    m._dispatch_to_scheduler = m.sent.append
    m.config_value = lambda name: "1"
    m.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch=lambda objs: [], cancel_for_dispatch=lambda objs: None
    )
    return m


def request(rid="gateway-attempt-1"):
    return GenerateInput(
        rid=rid,
        is_single=True,
        received_time=0,
        stream=False,
        return_logprob=False,
        return_sampling_mask=False,
        lora_path=None,
        log_metrics=False,
        background=False,
    )


def packet(rid, text="ok"):
    return Packet(
        rids=[rid],
        finished_reasons=[{"type": "length"}],
        output_strs=[text],
        output_ids=[[1]],
        prompt_tokens=[1],
        completion_tokens=[1],
        reasoning_tokens=[0],
        cached_tokens=[0],
        retraction_counts=[0],
    )


class TestRequestDefenses(unittest.TestCase):
    def test_gateway_rid_is_preserved_on_dispatch_and_response(self):
        m = manager()
        obj = request()
        m._init_req_state(obj)
        state = m.rid_to_state[obj.rid]
        tokenized = SimpleNamespace(
            rid=obj.rid,
            time_stats=state.time_stats,
            mm_inputs=None,
            wrap_pickle_fields=lambda: None,
        )
        m._send_one_request(tokenized)
        self.assertEqual(m.sent[0].rid, obj.rid)
        asyncio.run(m._handle_batch_output(packet(obj.rid)))
        self.assertEqual(state.out_list[0]["meta_info"]["id"], obj.rid)
        self.assertEqual(state.get_text(), "ok")
        self.assertFalse(m.rid_to_state)

    def test_duplicate_rejection_is_atomic_for_whole_batch(self):
        m = manager()
        obj = request()
        m._init_req_state(obj)
        old = m.rid_to_state[obj.rid]
        for rids in (["new", obj.rid], ["new", "new"]):
            with self.subTest(rids=rids):
                batch = GenerateInput(
                    rid=rids,
                    items=[request(rid) for rid in rids],
                    is_single=False,
                    received_time=0,
                    bootstrap_room=None,
                )
                with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
                    m._init_req_state(batch)
                self.assertEqual(m.rid_to_state, {obj.rid: old})

    def test_late_cleanup_cannot_cancel_or_remove_new_owner(self):
        m = manager()
        old_obj = request()
        owned = m._init_req_state(old_obj)
        m.rid_to_state.pop(old_obj.rid)
        newer = request(old_obj.rid)
        m._init_req_state(newer)
        new_state = m.rid_to_state[newer.rid]
        m._discard_pending_req_states(old_obj, owned_states=owned)
        self.assertIs(m.rid_to_state[newer.rid], new_state)
        self.assertFalse(m.sent)

    def test_failed_batch_cleanup_cancels_only_captured_states(self):
        m = manager()
        other = request("another-attempt")
        m._init_req_state(other)
        batch = GenerateInput(
            rid=["attempt-a", "attempt-b"],
            items=[request("attempt-a"), request("attempt-b")],
            is_single=False,
            received_time=0,
            bootstrap_room=None,
        )
        owned = m._init_req_state(batch)
        m._discard_pending_req_states(batch, owned_states=owned)
        self.assertEqual([obj.rid for obj in m.sent], batch.rid)
        self.assertEqual(list(m.rid_to_state), [other.rid])

    def test_delayed_abort_cannot_cancel_later_owner(self):
        async def run():
            m = manager()
            obj = request()
            background = m.create_abort_task(obj)
            m._init_req_state(obj)
            real_sleep = asyncio.sleep
            newer = request(obj.rid)

            async def replace_owner(_):
                m.rid_to_state.pop(obj.rid)
                m._init_req_state(newer)
                await real_sleep(0)

            with patch.object(asyncio, "sleep", replace_owner):
                await background()
            self.assertIs(m.rid_to_state[obj.rid].obj, newer)
            self.assertFalse(m.sent)

        asyncio.run(run())

    def test_disconnect_waiter_does_not_cancel_replacement(self):
        async def run():
            m = manager()
            obj = request()
            m._init_req_state(obj)

            async def disconnected():
                return True

            waiter = asyncio.create_task(
                m._wait_one_response(
                    obj, SimpleNamespace(is_disconnected=disconnected)
                ).__anext__()
            )
            await asyncio.sleep(0)
            m.rid_to_state.pop(obj.rid)
            newer = request(obj.rid)
            m._init_req_state(newer)
            with self.assertRaisesRegex(ValueError, "disconnected"):
                await waiter
            self.assertIs(m.rid_to_state[obj.rid].obj, newer)
            self.assertFalse(m.sent)

        asyncio.run(run())

    def test_error_finish_cleanup_does_not_delete_replacement(self):
        m = manager()
        obj = request()
        old = m._init_req_state(obj)[0][1]
        m.rid_to_state.pop(obj.rid)
        newer = request(obj.rid)
        m._init_req_state(newer)
        out = {"meta_info": {"finish_reason": {"type": "abort", "status_code": 500}}}
        self.assertIs(asyncio.run(m._handle_abort_finish_reason(out, old, True)), out)
        self.assertIs(m.rid_to_state[obj.rid].obj, newer)

    def test_busy_health_reply_is_correlated_and_removed(self):
        m = manager()
        obj = request(f"{HEALTH_CHECK_RID_PREFIX}_unique")
        state = m._init_req_state(obj)[0][1]
        m._handle_health_check_output(wire["HealthCheckOutput"](rid="unrelated"))
        self.assertFalse(state.event.is_set())
        reply = wire["HealthCheckOutput"](rid=obj.rid)
        reply = msgspec.msgpack.decode(msgspec.msgpack.encode(reply), type=type(reply))
        m._handle_health_check_output(reply)
        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertEqual(state.out_list[0]["meta_info"]["id"], obj.rid)
        self.assertFalse(m.rid_to_state)


class TestRouterShutdown(unittest.TestCase):
    def test_owner_watchdog_stops_before_shutdown_forward(self):
        async def run():
            events = []
            messages = iter([wire["ShutdownReq"]()])

            async def receive(_):
                try:
                    return next(messages)
                except StopIteration:
                    raise EOFError from None

            async def send(_, obj):
                events.append(("forward", obj))

            dummy = type("OtherControl", (), {})
            cls = methods(
                "managers/multi_tokenizer_mixin.py",
                "MultiTokenizerRouter",
                {"router_worker_obj"},
                dict(
                    wire,
                    async_sock_recv=receive,
                    async_sock_send=send,
                    TokenizerWorkerRegistrationReq=dummy,
                    PauseGenerationReqInput=dummy,
                    ContinueGenerationReqInput=dummy,
                ),
            )
            router = cls()
            router.receive_from_worker = router.send_to_scheduler = None
            router._subprocess_watchdog = SimpleNamespace(
                stop=lambda: events.append(("stop", None))
            )
            with self.assertRaises(EOFError):
                await router.router_worker_obj()
            self.assertEqual([kind for kind, _ in events], ["stop", "forward"])

        asyncio.run(run())


class TestHealthProbeCorrelation(unittest.TestCase):
    def probe(self, behavior):
        m = SimpleNamespace(
            gracefully_exit=False,
            server_status="up",
            is_generation=True,
            last_receive_tstamp=time.time(),
            cleaned=[],
        )
        m._discard_pending_req_states = lambda obj: m.cleaned.append(obj.rid)

        async def generate(obj, raw):
            if behavior == "unrelated":
                while True:
                    m.last_receive_tstamp = time.time()
                    await asyncio.sleep(0.001)
            elif behavior == "reply":
                yield {"meta_info": {"id": obj.rid}}

        m.generate_request = generate
        values = source_namespace(
            "entrypoints/http_server.py",
            {"health_generate"},
            dict(
                asyncio=asyncio,
                uuid=uuid,
                logger=logging.getLogger(__name__),
                _global_state=SimpleNamespace(tokenizer_manager=m),
                HEALTH_CHECK_RID_PREFIX=HEALTH_CHECK_RID_PREFIX,
                HEALTH_CHECK_TIMEOUT=0.02,
                Response=Response,
                GenerateReqInput=GenerateInput,
                ServerStatus=SimpleNamespace(
                    Starting="starting", Up="up", UnHealthy="unhealthy"
                ),
                envs=SimpleNamespace(
                    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=SimpleNamespace(
                        get=lambda: True
                    )
                ),
                get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
                DisaggregationMode=SimpleNamespace(NULL=SimpleNamespace(value="null")),
            ),
        )
        return values["health_generate"](
            SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        ), m

    def test_only_own_reply_is_healthy(self):
        for behavior, status in (("unrelated", 503), ("empty", 503), ("reply", 200)):
            with self.subTest(behavior=behavior):
                coro, m = self.probe(behavior)
                self.assertEqual(asyncio.run(coro).status_code, status)
                self.assertEqual(len(m.cleaned), 1)


decode_env = source_namespace(
    "managers/detokenizer_manager.py",
    {"DecodeStatus", "LimitedCapacityDict"},
    dict(dataclasses=dataclasses, OrderedDict=collections.OrderedDict),
)
Decoder = methods(
    "managers/detokenizer_manager.py",
    "DetokenizerManager",
    {"_decode_batch_token_id_output", "_clamp_decode_ids", "trim_matched_stop"},
    dict(
        decode_env,
        Counter=collections.Counter,
        DETOKENIZER_MAX_STATES=65536,
        find_printable_text=lambda s: s,
    ),
)


def decoder():
    m = Decoder()
    m.vocab_size, m.disable_tokenizer_batch_decode = 100, True
    m.decode_status = decode_env["LimitedCapacityDict"](65536)
    m.tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: "".join(chr(65 + i) for i in ids)
    )
    return m


def tokens(rids, reasons):
    n = len(rids)
    return SimpleNamespace(
        rids=rids,
        decoded_texts=[""] * n,
        decode_ids=[[i + 1] for i in range(n)],
        read_offsets=[0] * n,
        finished_reasons=reasons,
        no_stop_trim=[True] * n,
        skip_special_tokens=[False] * n,
        spaces_between_special_tokens=[False] * n,
    )


class TestDetokenizerDefenses(unittest.TestCase):
    def test_duplicate_rid_rejected_before_state_mutation(self):
        for reasons in (
            [None, None],
            [{"type": "length"}, None],
            [None, {"type": "length"}],
        ):
            with self.subTest(reasons=reasons):
                m = decoder()
                existing = decode_env["DecodeStatus"]("", [3], 0, 0)
                m.decode_status["gateway-attempt"] = existing
                with self.assertRaisesRegex(
                    RuntimeError, "Duplicate request IDs.*gateway-attempt"
                ):
                    m._decode_batch_token_id_output(
                        tokens(["gateway-attempt"] * 2, reasons)
                    )
                self.assertIs(m.decode_status["gateway-attempt"], existing)
                self.assertEqual(existing.decode_ids, [3])
                self.assertEqual(list(m.decode_status), ["gateway-attempt"])

    def test_unique_gateway_rids_and_next_request_complete(self):
        m = decoder()
        result = m._decode_batch_token_id_output(
            tokens(["attempt-a", "attempt-b"], [{"type": "length"}, None])
        )
        self.assertEqual(result, ["B", "C"])
        self.assertEqual(list(m.decode_status), ["attempt-b"])
        self.assertEqual(
            m._decode_batch_token_id_output(
                tokens(["attempt-c"], [{"type": "length"}])
            ),
            ["B"],
        )

    def test_exception_survives_cleanup_failure(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):

                class BrokenManager:
                    def __init__(self, *args):
                        pass

                    def event_loop(self):
                        raise KeyError("missing-state")

                    multi_http_worker_event_loop = event_loop

                    def maybe_clear_socket_mapping(self):
                        raise RuntimeError("cleanup failed")

                values = source_namespace(
                    "managers/detokenizer_manager.py",
                    {"run_detokenizer_process"},
                    dict(
                        DetokenizerManager=BrokenManager,
                        kill_itself_when_parent_died=lambda: None,
                        setproctitle=SimpleNamespace(setproctitle=lambda _: None),
                        configure_logger=lambda _: None,
                        publish=lambda *a, **kw: None,
                        psutil=SimpleNamespace(
                            Process=lambda: SimpleNamespace(parent=lambda: None)
                        ),
                        logger=logging.getLogger(__name__),
                        get_exception_traceback=lambda: "failure",
                    ),
                )
                with self.assertRaisesRegex(KeyError, "missing-state"):
                    values["run_detokenizer_process"](
                        SimpleNamespace(tokenizer_worker_num=workers), None
                    )


if __name__ == "__main__":
    unittest.main()
