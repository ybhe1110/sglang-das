"""CPU tests for shared P/D identity admission and internal rebootstrap."""

import hashlib
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

from test_external_kv_abort_drain import ROOT, load_methods, register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Request(SimpleNamespace):
    def __getattr__(self, name):
        return None


class TestPairIdentity(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
