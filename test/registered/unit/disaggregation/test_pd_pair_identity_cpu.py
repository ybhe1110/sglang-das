"""CPU tests for P/D identity admission, parallel sampling and rebootstrap."""

import ast
import asyncio
import copy
import hashlib
import importlib.util
import threading
import unittest
import uuid
from array import array
from collections import defaultdict
from contextlib import nullcontext
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import test_pd_failure_protocol_cpu as protocol
from test_external_kv_abort_drain import ROOT, load_methods, register_cpu_ci
from test_source_drain_retirement import extract_class

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "pd_parallel_sampling",
    ROOT / "python/sglang/srt/disaggregation/parallel_sampling.py",
)
parallel_sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel_sampling)


class Request(SimpleNamespace):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid


class TestPairIdentity(protocol.CustomTestCase):
    def test_invalid_pair_rejected_at_prefill_admission_without_scheduler_exception(
        self,
    ):
        class InvalidPair(ValueError):
            pass

        create = Mock(side_effect=InvalidPair("fresh pair required"))
        abort = Mock()
        cls = load_methods(
            ROOT / "python/sglang/srt/disaggregation/prefill.py",
            ["create_sender"],
            dict(
                TransferBackend=SimpleNamespace(FAKE="fake", MOONCAKE="mooncake"),
                FAKE_BOOTSTRAP_HOST="2.2.2.2",
                KVClassType=SimpleNamespace(SENDER="sender"),
                get_kv_class=lambda *a: create,
                InvalidBootstrapPairKey=InvalidPair,
                prepare_abort=abort,
                HTTPStatus=HTTPStatus,
            ),
        )
        q = cls()
        q._check_if_req_exceed_kv_capacity = lambda req: False
        q.transfer_backend, q.tp_rank, q.pp_rank = "mooncake", 0, 0
        q.kv_manager, q.scheduler, q.bootstrap_port = Mock(), Mock(), 8998
        req = Request(bootstrap_host="peer", bootstrap_room=7, bootstrap_pair_key="old")
        self.assertFalse(q.create_sender(req, 8))
        abort.assert_called_once_with(
            req, "fresh pair required", status_code=HTTPStatus.BAD_REQUEST
        )
        q.scheduler.output_streamer.stream_output.assert_called_once()

    def test_invalid_pair_rejected_at_decode_admission_without_enqueue(self):
        class InvalidPair(ValueError):
            pass

        abort = Mock()
        cls = load_methods(
            ROOT / "python/sglang/srt/disaggregation/decode.py",
            ["add"],
            dict(
                is_unadmitted_reject=lambda req: False,
                InvalidBootstrapPairKey=InvalidPair,
                prepare_abort=abort,
                HTTPStatus=HTTPStatus,
            ),
            class_name="DecodePreallocQueue",
        )
        q = cls()
        q._check_if_req_exceed_kv_capacity = lambda req: False
        q._create_receiver_and_enqueue = Mock(side_effect=InvalidPair("missing pair"))
        q.scheduler = Mock()
        req = Request()
        q.add(req)
        abort.assert_called_once_with(
            req, "missing pair", status_code=HTTPStatus.BAD_REQUEST
        )
        q.scheduler.output_streamer.stream_output.assert_called_once()

    def tokenizer(self, mode):
        cls = load_methods(
            ROOT / "python/sglang/srt/managers/tokenizer_manager.py",
            ["_create_tokenized_object"],
            dict(
                array=array,
                GenerateReqInput=Request,
                TokenizedGenerateReqInput=SimpleNamespace,
                DisaggregationMode=SimpleNamespace(PREFILL="prefill", DECODE="decode"),
                FAKE_BOOTSTRAP_HOST="2.2.2.2",
                get_disagg=lambda: SimpleNamespace(
                    disaggregation_transfer_backend="mooncake"
                ),
                envs=SimpleNamespace(
                    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=SimpleNamespace(
                        get=lambda: True
                    )
                ),
            ),
        )
        m = cls()
        m.preferred_sampling_params = None
        m.sampling_params_class = lambda **kwargs: Mock()
        m.tokenizer = None
        m.model_config = SimpleNamespace(vocab_size=100)
        m.disaggregation_mode = mode
        m.rid_to_state = {"p-rid": Mock(), "d-rid": Mock()}
        return m

    def test_shared_pair_key_survives_tokenization_with_different_rids(self):
        values = []
        for mode, rid in (("prefill", "p-rid"), ("decode", "d-rid")):
            obj = Request(
                rid=rid,
                sampling_params={},
                bootstrap_host="peer",
                bootstrap_room=7,
                bootstrap_pair_key="shared-capability",
            )
            tokenized = self.tokenizer(mode)._create_tokenized_object(obj, "hello", [1])
            values.append(tokenized.bootstrap_pair_key)
            self.assertEqual(tokenized.rid, rid)
        self.assertEqual(values, ["shared-capability", "shared-capability"])

    def test_missing_key_rejected_before_scheduler_but_fake_health_check_allowed(self):
        for mode, rid in (("prefill", "p-rid"), ("decode", "d-rid")):
            with self.subTest(mode=mode):
                obj = Request(
                    rid=rid, sampling_params={}, bootstrap_host="peer", bootstrap_room=7
                )
                m = self.tokenizer(mode)
                with self.assertRaisesRegex(ValueError, "bootstrap_pair_key"):
                    m._create_tokenized_object(obj, "hello", [1])
                obj.bootstrap_host = "2.2.2.2"
                self.assertIsNone(
                    m._create_tokenized_object(obj, "health", [1]).bootstrap_pair_key
                )

    def test_internal_rebootstrap_rotates_key_identically_on_all_ranks_and_forwards_it(
        self,
    ):
        receivers = []
        cls = load_methods(
            ROOT / "python/sglang/srt/disaggregation/decode.py",
            ["_create_receiver_and_enqueue"],
            dict(
                hashlib=hashlib,
                _is_fake_transfer=lambda req: False,
                TransferBackend=SimpleNamespace(FAKE="fake", MOONCAKE="mooncake"),
                KVClassType=SimpleNamespace(RECEIVER="receiver"),
                get_kv_class=lambda *args: (
                    lambda **kw: receivers.append(kw) or SimpleNamespace()
                ),
                _bootstrap_addr=lambda req: "prefill:8998",
                DecodeRequest=SimpleNamespace,
            ),
        )
        payload_cls = load_methods(
            ROOT / "python/sglang/srt/managers/schedule_batch.py",
            ["build_rebootstrap_payload"],
            {},
            class_name="Req",
        )
        keys = []
        for rank in range(8):
            q = cls()
            q.transfer_backend, q.kv_manager, q.queue = "mooncake", Mock(), []
            req = Request(
                bootstrap_room=7,
                bootstrap_pair_key="initial-pair",
                origin_input_ids=[1],
                output_ids=[2],
                sampling_params=Request(),
            )
            q._create_receiver_and_enqueue(req, is_rebootstrap=True)
            key = req.bootstrap_pair_key
            self.assertNotEqual(key, "initial-pair")
            self.assertEqual(receivers[-1]["bootstrap_pair_key"], key)
            self.assertEqual(
                payload_cls.build_rebootstrap_payload(req)["bootstrap_pair_key"], key
            )
            keys.append(key)
            q._create_receiver_and_enqueue(req, is_rebootstrap=True)
            self.assertNotEqual(req.bootstrap_pair_key, key)
        self.assertEqual(len(set(keys)), 1)


class ParallelBatch:
    def __init__(self, mode, identities, n, stream):
        self.items = [
            Request(
                rid=f"{mode}-{i}",
                bootstrap_host="prefill",
                bootstrap_room=room,
                bootstrap_pair_key=key,
                sampling_params={"n": n, "max_new_tokens": 16},
            )
            for i, (room, key) in enumerate(identities)
        ]
        self.rid = [item.rid for item in self.items]
        self.batch_size, self.parallel_sample_num, self.stream = (
            len(self.items),
            n,
            stream,
        )

    def __getitem__(self, index):
        return self.items[index]


class TestParallelPairIdentity(protocol.CustomTestCase):
    def expand(self, mode, identities, n=10, stream=False, emitted=None):
        """Run production P/D admission and fan-out without GPU/tokenizer imports."""
        path = ROOT / "python/sglang/srt/managers/tokenizer_manager.py"
        cls = next(
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
        )
        cls.bases = []
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name
            in {
                "_handle_batch_request",
                "_collect_batch_responses",
                "_stream_batch_responses",
            }
        ]
        env = dict(
            asyncio=asyncio,
            copy=copy,
            logger=Mock(),
            nullcontext=nullcontext,
            get_bool_env_var=lambda name: False,
            assign_parallel_sample_bootstrap=parallel_sampling.assign_parallel_sample_bootstrap,
        )
        tree = ast.Module(
            body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), cls],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
        manager = TestPairIdentity().tokenizer(mode)
        for node in cls.body:
            setattr(
                type(manager), node.name, getattr(env["TokenizerManager"], node.name)
            )
        batch = ParallelBatch(mode, identities, n, stream)
        manager.rid_to_state = {
            item.rid: SimpleNamespace(time_stats=Mock(), obj=item)
            for item in batch.items
        }
        emitted = [] if emitted is None else emitted

        async def tokenize(req):
            return manager._create_tokenized_object(req, "prompt", [1, 2])

        def init(req):
            manager.rid_to_state[req.rid] = SimpleNamespace(time_stats=Mock(), obj=req)

        def send(req):
            raw = manager.rid_to_state[req.rid].obj
            self.assertEqual(self.identity(req), self.identity(raw))
            emitted.append(req)

        async def response(req, request):
            yield {"meta_info": {"id": req.rid}}

        manager._tokenize_one_request = tokenize
        manager._init_req_state = init
        manager._send_one_request = send
        manager._wait_one_response = response
        manager._should_use_batch_tokenization = lambda *args: False

        async def run():
            return [out async for out in manager._handle_batch_request(batch, None)]

        outputs = asyncio.run(run())
        self.assertEqual([self.identity(item) for item in batch.items], identities)
        if stream:
            self.assertEqual(
                sorted(out["index"] for out in outputs),
                list(range(len(identities) * n)),
            )
        else:
            self.assertEqual(len(outputs[0]), len(identities) * n)
        return emitted

    @staticmethod
    def identity(req):
        return req.bootstrap_room, req.bootstrap_pair_key

    def test_parallel_children_match_across_pd_and_bootstrap_attempts(self):
        """Warmups/samples must not share rooms, even with reused batch parents."""
        parents = [((1 << 64) - 1, "k" * 128)] * 2
        for stream in (False, True):
            with self.subTest(stream=stream):
                p = self.expand("prefill", parents, stream=stream)
                d = self.expand("decode", parents, stream=stream)
                identities = [self.identity(req) for req in p]
                self.assertEqual(identities, [self.identity(req) for req in d])
                self.assertNotEqual([req.rid for req in p], [req.rid for req in d])
                self.assertEqual(len(identities), 22)
                self.assertEqual(len({room for room, _ in identities}), 22)
                self.assertEqual(len({key for _, key in identities}), 22)
                self.assertTrue(set(identities).isdisjoint(parents))
                self.assertTrue(all(0 < room < 1 << 63 for room, _ in identities))
                self.assertTrue(
                    all(key.isascii() and 0 < len(key) <= 128 for _, key in identities)
                )
                retried = self.expand(
                    "prefill", [(room, "retry") for room, _ in parents]
                )
                self.assertTrue(
                    {r.bootstrap_room for r in p}.isdisjoint(
                        r.bootstrap_room for r in retried
                    )
                )
                self.assertTrue(
                    {r.bootstrap_pair_key for r in p}.isdisjoint(
                        r.bootstrap_pair_key for r in retried
                    )
                )

    def test_n10_after_warmup_cancels_and_releases_each_destination_once(self):
        """The probe's warmup must not reject samples or strand their drain ACKs."""
        p_reqs = self.expand("prefill", [(7, "shared-pair")])
        d_reqs = self.expand("decode", [(7, "shared-pair")])
        p, d, sent, env = protocol.wire_managers()
        warmup = p_reqs[0]
        p._open_source_room(*self.identity(warmup))
        p._source_drain.close(warmup.bootstrap_room)
        self.assertTrue(p._source_drain.forget(warmup.bootstrap_room))
        p._retire_abort_binding(warmup.bootstrap_room)

        protocol.TestDecodeReleaseConsensus.setUp(self)
        clock = [100.0]
        env["time"] = self.env["time"] = SimpleNamespace(monotonic=lambda: clock[0])
        self.env["_matches_abort_rid"] = load_methods(
            protocol.DECODE, ["_matches_abort_rid"], {}
        )._matches_abort_rid
        receiver_cls = extract_class(
            protocol.COMMON,
            "CommonKVReceiver",
            {
                "abort",
                "ensure_abort_notified",
                "_send_abort_notification",
                "retry_abort_notification",
                "clear",
                "_clear",
            },
            [],
            env,
        )
        d.local_ip, d.rank_port = "decode", 6000
        d.record_failure = Mock()
        d.defer_decode_allocation = False
        d.required_prefill_response_num_table = {}
        d.prefill_response_tracker = defaultdict(set)
        d.addr_to_rooms_tracker = defaultdict(set)
        q, _ = protocol.queue(self.env, d)
        d.clear_deferred_abort_state(7)
        q.queue, q._deferred_releases = [], []
        q.metadata_buffers.bootstrap_room = {}
        q.tp_rank, q.locally_aborted_rids = 0, set()
        q.deferred_kv_release_timeout = 30.0
        active = []
        for index, (p_req, d_req) in enumerate(zip(p_reqs[1:], d_reqs[1:])):
            room, key = self.identity(p_req)
            p._open_source_room(room, key)
            receiver = receiver_cls()
            receiver.kv_mgr = d
            receiver.bootstrap_room, receiver.bootstrap_addr, receiver.init_time = (
                d_req.bootstrap_room,
                "prefill",
                1.0,
            )
            receiver._owns_bootstrap_room = True
            receiver._lifecycle_lock = threading.RLock()
            receiver._transfer_generation = (1 << 96) + index
            receiver._bootstrap_pair_key = d_req.bootstrap_pair_key
            receiver.abort_notified, receiver._abort_generation = False, None
            receiver._last_abort_notification_time = None
            receiver.bootstrap_infos = [
                {"rank_ip": "prefill", "rank_port": 7000, "prefill_ack_rank": 0}
            ]
            receiver._connect_to_bootstrap_server = lambda info, **kw: (
                SimpleNamespace(
                    send_multipart=lambda frames, **kw: p._handle_abort_notification(
                        frames
                    )
                ),
                threading.Lock(),
            )
            self.assertTrue(
                p._bind_abort_metadata(
                    room,
                    SimpleNamespace(
                        bootstrap_pair_key=receiver._bootstrap_pair_key,
                        endpoint="decode",
                        dst_port=6000,
                        abort_generation=receiver._transfer_generation,
                    ),
                )
            )
            if index % 2:
                chunk = SimpleNamespace(room=room)
                p._source_drain.publish(chunk, Mock(), lambda: True)
                p._source_drain.begin(chunk)
                active.append(chunk)
            q.queue.append(
                SimpleNamespace(
                    req=d_req, kv_receiver=receiver, metadata_buffer_index=index
                )
            )
            q.metadata_buffers.bootstrap_room[index] = room

        requests = list(q.queue)
        cancel = SimpleNamespace(rid="", abort_all=True)
        q.abort_matching(cancel)
        q.abort_matching(cancel)
        for req in requests:
            q._defer_release(req)
        q.queue = []
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 5)
        self.assertEqual(self.env["release_kv_cache"].call_count, 5)
        self.assertEqual(
            {call.args[0].rid for call in self.env["release_kv_cache"].call_args_list},
            {req.req.rid for req in requests[::2]},
        )
        clock[0] += 31.0
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 5)
        self.assertEqual(self.env["release_kv_cache"].call_count, 5)
        for index in range(1, 10, 2):
            self.assertEqual(
                q.metadata_buffers.bootstrap_room[index],
                requests[index].req.bootstrap_room,
            )
        for chunk in active:
            p._source_drain.complete(chunk)
            p._maybe_ack_drained_abort(chunk.room)
        q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 0)
        self.assertEqual(self.env["release_kv_cache"].call_count, 10)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.free.call_count, 10)
        self.assertEqual(
            {
                call.args[0]
                for call in q.req_to_metadata_buffer_idx_allocator.free.call_args_list
            },
            set(range(10)),
        )
        self.assertTrue(all(req.kv_receiver is None for req in requests))
        self.assertEqual(set(q.metadata_buffers.bootstrap_room.values()), {0})
        self.assertEqual(d._deferred_abort_ack_tracker, {})
        self.assertEqual(
            {env["AbortAck"].from_zmq(frames).room for _, frames in sent},
            {req.bootstrap_room for req in d_reqs[1:]},
        )

    def test_n1_and_unpaired_backends_keep_their_contract(self):
        parents = [(7, "first"), (8, "second")]
        self.assertEqual(
            [self.identity(req) for req in self.expand("decode", parents, n=1)], parents
        )
        ordinary = self.expand("none", [(None, None)], n=3)
        self.assertEqual([self.identity(req) for req in ordinary], [(None, None)] * 4)
        legacy = self.expand("none", [(7, None)], n=3)
        self.assertEqual(len({req.bootstrap_room for req in legacy}), 4)
        self.assertTrue(all(req.bootstrap_pair_key is None for req in legacy))

    def test_invalid_parent_pair_is_rejected_before_any_parallel_dispatch(self):
        for mode in ("prefill", "decode"):
            for key in (None, "", "x" * 129, "\u00e9"):
                with self.subTest(mode=mode, key=key):
                    emitted = []
                    with self.assertRaisesRegex(ValueError, "bootstrap_pair_key"):
                        self.expand(mode, [(7, key)], emitted=emitted)
                    self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
