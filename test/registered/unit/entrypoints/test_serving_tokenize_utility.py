"""Full tool work leaves the loop and preserves tokenize/detokenize semantics."""

import asyncio
import base64
import json
import logging
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DetokenizeRequest,
    TokenizeRequest,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,
    OpenAIServingTokenize,
    check_utility_items,
    make_utility_backend_factory,
)
from sglang.srt.entrypoints.openai.utility_executor import UtilityLimits, UtilityService
from sglang.srt.entrypoints.openai.utility_http import (
    UTILITY_PATHS,
    UtilityHTTPMiddleware,
)
from sglang.srt.utils.request_logger import RequestLogger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tokenizer:
    model_max_length = 10**30

    def __init__(self):
        self.thread_ids = []

    def encode(self, text, add_special_tokens=True):
        self.thread_ids.append(threading.get_ident())
        if text == "bad-value":
            raise ValueError("invalid text")
        if text == "bad-runtime":
            raise RuntimeError("backend error")
        return ([1] if add_special_tokens else []) + [ord(c) + 2 for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        self.thread_ids.append(threading.get_ident())
        if -1 in tokens:
            raise ValueError("decode failed")
        if -2 in tokens:
            raise RuntimeError("backend error")
        return "".join(
            ("" if skip_special_tokens else "<bos>") if t == 1 else chr(t - 2)
            for t in tokens
        )


def _manager(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(context_len=8192, is_multimodal=False),
        config_value=lambda name: None,
        model_path="test",
        served_model_name="test",
        request_logger=SimpleNamespace(log_requests=False),
    )


class TestServingUtility(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def asyncSetUp(self):
        self.manager = _manager(_Tokenizer())
        self.service = UtilityService(
            UtilityLimits(), validate_request=check_utility_items
        )
        await self.service.start(make_utility_backend_factory(self.manager, None))
        self.tokenize = OpenAIServingTokenize(
            self.manager, utility_service=self.service
        )
        self.detokenize = OpenAIServingDetokenize(
            self.manager, utility_service=self.service
        )

    async def asyncTearDown(self):
        for ticket in list(self.service.tickets):
            ticket.close()
        await self.service.aclose()

    async def call(self, handler, request):
        ticket = self.service.admit()
        raw = Request({"type": "http", "headers": [], "utility_ticket": ticket})
        try:
            response = await handler.handle_request(request, raw)
            return response.status_code, json.loads(response.body)
        finally:
            ticket.close()

    async def test_detailed_logging_uses_admitted_worker_and_preserves_headers(self):
        """Slow sinks and payload dumps must not execute on the HTTP loop."""
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        for fmt, level, handler, request in (
            ("text", 2, self.tokenize, TokenizeRequest(prompt="hello")),
            ("json", 3, self.detokenize, DetokenizeRequest(tokens=[67, 68])),
        ):
            with self.subTest(format=fmt, level=level):
                entered, gate = asyncio.Event(), threading.Event()
                messages, sink_threads, dump_threads = [], [], []

                class Sink(logging.Handler):
                    def emit(self, record):
                        sink_threads.append(threading.get_ident())
                        messages.append(record.getMessage())
                        loop.call_soon_threadsafe(entered.set)
                        # On broken code, fail the thread assertion rather than
                        # deadlocking the test's own event loop.
                        if threading.get_ident() != loop_thread:
                            if not gate.wait(5):
                                raise TimeoutError("Test did not release log sink")

                logger = RequestLogger(True, level, fmt, None)
                await self.enterAsyncContext(logger.background_writes())
                target = logging.Logger("utility-test", level=logging.INFO)
                target.addHandler(Sink())
                logger.targets = [target]
                self.manager.request_logger = logger
                model = type(request)
                original_dump = model.model_dump

                def dump(instance, *args, **kwargs):
                    dump_threads.append(threading.get_ident())
                    return original_dump(instance, *args, **kwargs)

                ticket = self.service.admit()
                raw = Request(
                    {
                        "type": "http",
                        "headers": [
                            (b"x-smg-routing-key", b"route-a"),
                            (b"authorization", b"secret"),
                        ],
                        "utility_ticket": ticket,
                    }
                )
                with patch.object(model, "model_dump", dump):
                    task = asyncio.create_task(handler.handle_request(request, raw))
                    try:
                        await asyncio.wait_for(entered.wait(), 2)
                        self.assertTrue(dump_threads)
                        self.assertTrue(
                            all(t != loop_thread for t in dump_threads + sink_threads)
                        )
                        self.assertEqual(self.service.work_used, 1)
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                    finally:
                        gate.set()
                        response = await asyncio.wait_for(task, 2)
                        ticket.close()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(messages), 1)
                self.assertIn("route-a", messages[0])
                self.assertNotIn("secret", messages[0])
                self.assertEqual(self.service.response_used, 0)

    async def test_generation_logging_progresses_while_utility_holds_sink(self):
        """A slow utility log must not block the generation hook or grow its queue."""
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        sink_timed_out = threading.Event()
        records = []

        class Sink(logging.Handler):
            def emit(self, record):
                if not records:
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(2):
                        sink_timed_out.set()
                records.append(record.getMessage())

        logger = self.manager.request_logger = RequestLogger(True, 2, "text", None)
        target = logging.Logger("mixed-requests")
        target.addHandler(Sink())
        logger.targets = [target]
        async with logger.background_writes() as writer:
            with patch.object(writer, "MAX_RECORDS", 1):
                tool = asyncio.create_task(
                    self.call(self.tokenize, TokenizeRequest(prompt="utility"))
                )
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    dumps = []
                    original_dump = ChatCompletionRequest.model_dump

                    def dump(obj, **kwargs):
                        dumps.append(threading.get_ident())
                        return original_dump(obj, **kwargs)

                    with patch.object(ChatCompletionRequest, "model_dump", dump):
                        for text in ("generation-accepted", "generation-dropped"):
                            request = ChatCompletionRequest(
                                messages=[{"role": "user", "content": text}]
                            )
                            await OpenAIServingBase._log_received_request(
                                self.tokenize, request, None
                            )
                    self.assertEqual(len(dumps), 1)
                    self.assertNotIn(threading.get_ident(), dumps)
                    self.assertFalse(sink_timed_out.is_set())
                    self.assertEqual(writer.dropped, 1)
                    self.assertFalse(tool.done())
                finally:
                    release.set()
                    await asyncio.wait_for(tool, 2)
        self.assertEqual(len(records), 2)
        self.assertIn("utility", records[0])
        self.assertIn("generation-accepted", records[1])

    async def test_generation_receipt_and_finish_format_off_loop_and_preserve_decoded_text(
        self,
    ):
        """Generation must retain its decoded text even when a log is dropped."""
        from sglang.srt.utils import request_logger as log_module

        @dataclass
        class Input:
            rid: str
            input_ids: list
            text: object = None

        original_format = log_module.format_json_log
        formats = []

        def format_record(event, data):
            formats.append((event, threading.get_ident()))
            return original_format(event, data)

        for private in (False, True):
            with self.subTest(private=private), ThreadPoolExecutor(
                max_workers=1
            ) as owner:
                source, decoder = _Tokenizer(), _Tokenizer()
                logger = RequestLogger(True, 2, "json", None)
                target = logging.Logger("generation")
                logger.targets = [target]
                obj = Input("first", [[1, 67], [68]])
                with self.assertLogs(target, level="INFO") as logs, patch.object(
                    log_module, "format_json_log", format_record
                ):
                    async with logger.background_writes(
                        tokenizer_executor=None if private else owner,
                        private_decoder_factory=(lambda: decoder) if private else None,
                    ) as writer:
                        await logger.log_received_request(obj, source)
                        self.assertEqual(obj.text, ["<bos>A", "B"])
                        await logger.log_finished_request(
                            obj, {"meta_info": {}, "text": "ok"}
                        )
                        with patch.object(writer, "MAX_RECORDS", 0):
                            dropped = Input("dropped", [67])
                            await logger.log_received_request(dropped, source)
                            self.assertEqual(dropped.text, "A")
                received, finished = [json.loads(r.getMessage()) for r in logs.records]
                self.assertIsNone(received["obj"]["text"])
                self.assertEqual(finished["obj"]["text"], obj.text)
                self.assertEqual(finished["out"]["text"], "ok")
                self.assertNotIn(threading.get_ident(), [tid for _, tid in formats])
                actual_decoder, unused = (
                    (decoder, source) if private else (source, decoder)
                )
                self.assertEqual(unused.thread_ids, [])
                self.assertEqual(len(actual_decoder.thread_ids), 3)
                self.assertNotIn(threading.get_ident(), actual_decoder.thread_ids)
                if not private:
                    owner_tid = await asyncio.get_running_loop().run_in_executor(
                        owner, threading.get_ident
                    )
                    self.assertEqual(set(source.thread_ids), {owner_tid})

    async def test_cancelled_log_preparation_keeps_its_slot_until_dequeued(self):
        """Cancelled queued payloads cannot create space behind a busy formatter."""
        loop = asyncio.get_running_loop()
        started, release = asyncio.Event(), threading.Event()
        dumps = []
        logger = RequestLogger(True, 2, "text", None)
        logger.targets = []

        class Payload:
            def __init__(self, text):
                self.text = text

            def model_dump(self, **kwargs):
                dumps.append(self.text)
                if self.text == "first":
                    loop.call_soon_threadsafe(started.set)
                    if not release.wait(3):
                        raise TimeoutError("Test did not release formatter")
                return {"text": self.text}

        async def log(text):
            await logger.log_openai_received_request(Payload(text), None)

        async with logger.background_writes() as writer:
            with patch.object(writer, "MAX_RECORDS", 2):
                first = asyncio.create_task(log("first"))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    queued = asyncio.create_task(log("cancelled"))
                    await asyncio.sleep(0)
                    queued.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await queued
                    await asyncio.wait_for(log("full"), 1)
                    self.assertEqual(writer.dropped, 1)
                finally:
                    release.set()
                    await first
        self.assertEqual(dumps, ["first"])

    async def test_file_target_switch_is_atomic_and_level_changes_do_not_reopen_it(
        self,
    ):
        """Slow or failed opens must leave the current log target usable."""
        from sglang.srt.utils import log_utils

        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        entered, release = asyncio.Event(), threading.Event()
        opens = []
        original_handler = log_utils.TimedRotatingFileHandler

        def open_file(*args, **kwargs):
            opens.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            if threading.get_ident() != loop_thread and not release.wait(3):
                raise TimeoutError("Test did not release file open")
            return original_handler(*args, **kwargs)

        logger = RequestLogger(True, 2, "text", None)
        old_target = logging.Logger("previous-target")
        logger.targets = [old_target]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            log_utils, "TimedRotatingFileHandler", open_file
        ):
            change = asyncio.create_task(
                logger.configure(log_requests_target=[directory])
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                with self.assertLogs(old_target, level="INFO") as before:
                    logger._log("during open")
                self.assertEqual(before.records[0].getMessage(), "during open")
            finally:
                release.set()
                await change
            target = logger.targets[0]
            try:
                await logger.configure(log_requests_level=3)
                await logger.configure(log_requests_target=[directory])
                self.assertEqual(len(opens), 1)
                self.assertNotIn(loop_thread, opens)
                file = next(Path(directory).glob("*.log"))
                with self.assertRaises(OSError):
                    await logger.configure(
                        log_requests=False, log_requests_target=[str(file)]
                    )
                self.assertTrue(logger.log_requests)
                self.assertEqual(logger.log_requests_target, [directory])
                logger._log("after switch")
                self.assertIn("after switch", file.read_text(encoding="utf-8"))
            finally:
                for handler in target.handlers[:]:
                    target.removeHandler(handler)
                    handler.close()

    async def test_blocked_log_write_retains_bytes_and_allows_shutdown(self):
        """A blocked sink stays charged and cannot indefinitely delay HTTP shutdown."""
        loop = asyncio.get_running_loop()
        entered, finished = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        records = []

        class Sink(logging.Handler):
            def emit(self, record):
                loop.call_soon_threadsafe(entered.set)
                release.wait(5)
                records.append(record.getMessage())
                loop.call_soon_threadsafe(finished.set)

        logger = RequestLogger(True, 2, "text", None)
        target = logging.Logger("blocked-sink")
        target.addHandler(Sink())
        logger.targets = [target]
        try:
            async with logger.background_writes() as writer:
                with patch.object(writer, "MAX_BYTES", 800):
                    logger._log("x" * 512)
                    await asyncio.wait_for(entered.wait(), 2)
                    logger._log("y" * 512)
                    self.assertEqual(writer.dropped, 1)
            self.assertEqual(records, [])
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 2)
        self.assertEqual(records, ["x" * 512])

    async def test_prompt_branches_and_special_tokens_match_original(self):
        for prompt in ("", "中文🙂", ["hello", "中文", ""], []):
            for special in (False, True):
                request = TokenizeRequest(prompt=prompt, add_special_tokens=special)
                status, result = await self.call(self.tokenize, request)
                self.assertEqual(status, 200)
                reference = _Tokenizer()
                expected = (
                    reference.encode(prompt, special)
                    if isinstance(prompt, str)
                    else [reference.encode(t, special) for t in prompt]
                )
                self.assertEqual(result["tokens"], expected)
                self.assertEqual(
                    result["count"],
                    len(expected)
                    if isinstance(prompt, str)
                    else [len(t) for t in expected],
                )
                self.assertEqual(result["max_model_len"], 8192)
        # Generation's tokenizer is never used by any utility thread.
        self.assertEqual(self.manager.tokenizer.thread_ids, [])

    async def test_decode_shapes_empty_input_and_error_mapping(self):
        for tokens in ([], [1, 67, 68], [[1, 67], [], [68]]):
            for special in (False, True):
                status, result = await self.call(
                    self.detokenize,
                    DetokenizeRequest(tokens=tokens, skip_special_tokens=special),
                )
                reference = _Tokenizer()
                expected = (
                    [reference.decode(t, special) for t in tokens]
                    if tokens and isinstance(tokens[0], list)
                    else reference.decode(tokens, special)
                )
                self.assertEqual((status, result), (200, {"text": expected}))
        for tokens, status, kind in (
            ([-1], 400, "DecodeError"),
            ([-2], 500, "InternalServerError"),
        ):
            actual, error = await self.call(
                self.detokenize, DetokenizeRequest(tokens=tokens)
            )
            self.assertEqual((actual, error["type"]), (status, kind))
            self.assertNotIn("error", error)  # Existing flat error envelope.
        for text, status in (("bad-value", 400), ("bad-runtime", 500)):
            actual, error = await self.call(self.tokenize, TokenizeRequest(prompt=text))
            self.assertEqual((actual, error["code"]), (status, status))

    async def test_chat_conversion_and_actual_encoding_run_on_worker(self):
        # The spy stands in for model-specific template rendering, not the
        # utility's protocol conversion or its fallback encode operation.
        seen = []
        worker_tokenizer = _Tokenizer()
        worker_handler = OpenAIServingTokenize(_manager(worker_tokenizer))

        def render(request, multimodal):
            seen.append((request, threading.get_ident()))
            return SimpleNamespace(prompt_ids="中文", prompt=None)

        worker_handler.chat_serving = SimpleNamespace(
            _validate_request=lambda request: None, _process_messages=render
        )
        await self.service.aclose()
        self.service = UtilityService(UtilityLimits())
        from fastapi.responses import ORJSONResponse

        await self.service.start(
            lambda: (
                lambda kind, req: ORJSONResponse(
                    worker_handler._run_non_streaming_sync(req).model_dump()
                )
            )
        )
        self.tokenize.utility_service = self.service
        request = TokenizeRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            continue_final_message=True,
            chat_template_kwargs={"custom": "retained", "add_generation_prompt": False},
            input_ids=[7, 8],
        )
        status, result = await self.call(self.tokenize, request)
        self.assertEqual(status, 200)
        self.assertEqual(result["tokens"], [ord(c) + 2 for c in "中文"])
        converted, thread_id = seen[0]
        self.assertNotEqual(thread_id, threading.get_ident())
        self.assertEqual(worker_tokenizer.thread_ids, [thread_id])
        self.assertEqual(converted.tools[0].function.name, "f")
        self.assertTrue(converted.continue_final_message)
        self.assertEqual(
            converted.chat_template_kwargs,
            {"custom": "retained", "add_generation_prompt": False},
        )
        self.assertEqual(converted.input_ids, [7, 8])

    async def test_real_fast_tokenizer_round_trip_without_model_download(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        native = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "hello": 1, "world": 2, "中文": 3}, unk_token="[UNK]"
            )
        )
        native.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=native, unk_token="[UNK]")
        manager = _manager(tokenizer)
        service = UtilityService(UtilityLimits(workers=2))
        await service.start(make_utility_backend_factory(manager, None))
        try:

            async def check(text):
                ticket = service.admit()
                try:
                    response = await service.run(
                        ticket, "tokenize", TokenizeRequest(prompt=text)
                    )
                    self.assertEqual(
                        json.loads(response.body)["tokens"], tokenizer.encode(text)
                    )
                finally:
                    ticket.close()

            await asyncio.gather(
                *(check(text) for text in ("hello world", "中文", "", "🙂"))
            )
            for ids in ([], [1, 2], [[1], [], [3, 0]]):
                ticket = service.admit()
                try:
                    response = await service.run(
                        ticket, "detokenize", DetokenizeRequest(tokens=ids)
                    )
                    expected = (
                        [tokenizer.decode(row, skip_special_tokens=True) for row in ids]
                        if ids and isinstance(ids[0], list)
                        else tokenizer.decode(ids, skip_special_tokens=True)
                    )
                    self.assertEqual(json.loads(response.body)["text"], expected)
                finally:
                    ticket.close()
        finally:
            await service.aclose()

    async def start_private_tokenizers(self, source):
        service = UtilityService(UtilityLimits(workers=2))
        self.addAsyncCleanup(service.aclose)
        # Capture the tokenizer supplied to each worker's chat handler at startup.
        with patch(
            "sglang.srt.entrypoints.openai.serving_tokenize.OpenAIServingChat"
        ) as chat:
            await service.start(
                make_utility_backend_factory(_manager(source), object())
            )
        return [call.args[0].tokenizer for call in chat.call_args_list]

    async def test_tiktoken_workers_have_independent_encodings_and_templates(self):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(
                json.dumps(
                    {
                        "word_split": "V1",
                        "regular_tokens": [
                            {"bytes": [i], "token": i} for i in range(256)
                        ],
                        "special_tokens": [{"bytes": list(b"<|eos|>"), "token": 256}],
                    }
                )
            )
            source = TiktokenTokenizer(str(path))
            workers = await self.start_private_tokenizers(source)
        messages = [{"role": "user", "content": "hello"}]
        reference = source.apply_chat_template(messages, True, True)
        self.assertEqual(len({id(t.tokenizer) for t in [source, *workers]}), 3)
        self.assertEqual(
            len({id(t.chat_template_jinja.environment) for t in [source, *workers]}), 3
        )
        workers[0].tokenizer._control_tokens["pad"] = "worker only"
        self.assertEqual(workers[1].tokenizer._control_tokens["pad"], "<|pad|>")
        self.assertEqual(source.tokenizer._control_tokens["pad"], "<|pad|>")
        with patch.object(source.tokenizer, "_core_bpe", None):
            for worker in workers:
                self.assertEqual(worker.encode("hi"), [104, 105])
                self.assertEqual(worker.decode([104, 105]), "hi")
                self.assertEqual(
                    worker.apply_chat_template(messages, True, True), reference
                )

    async def test_mistral_patched_methods_use_each_workers_own_backend(self):
        from transformers.tokenization_mistral_common import MistralCommonBackend

        from sglang.srt.utils.hf_transformers.mistral_utils import (
            patch_mistral_common_tokenizer,
        )

        with tempfile.TemporaryDirectory() as directory:
            # A byte-only Tekken vocabulary exercises the real backend offline.
            path = Path(directory) / "tekken.json"
            path.write_text(
                json.dumps(
                    {
                        "config": {
                            "version": "v3",
                            "pattern": r".+",
                            "num_vocab_tokens": 256,
                            "default_num_special_tokens": 1000,
                            "default_vocab_size": 1256,
                        },
                        "vocab": [
                            {
                                "rank": i,
                                "token_bytes": base64.b64encode(bytes([i])).decode(
                                    "ascii"
                                ),
                                "token_str": bytes([i]).decode(
                                    "utf-8", errors="replace"
                                ),
                            }
                            for i in range(256)
                        ],
                    }
                )
            )
            source = patch_mistral_common_tokenizer(
                MistralCommonBackend(path, model_max_length=128)
            )
            source.chat_template = "custom template"
            source.additional_stop_token_ids = {42}
            workers = await self.start_private_tokenizers(source)
        messages = [{"role": "user", "content": "hello"}]
        ids = source.encode("hello")
        text = source.decode(ids, skip_special_tokens=True)
        chat_ids = source.apply_chat_template(messages, return_dict=False)
        self.assertEqual(len({id(t.tokenizer) for t in [source, *workers]}), 3)
        # A closure still bound to the generation tokenizer would use this
        # disabled backend or its original chat method.
        with (
            patch.object(source, "tokenizer", None),
            patch.object(
                source, "_orig_apply_chat_template", side_effect=AssertionError
            ),
        ):
            for worker in workers:
                self.assertEqual(worker.chat_template, source.chat_template)
                self.assertEqual(worker.additional_stop_token_ids, {42})
                self.assertIsNot(
                    worker.additional_stop_token_ids, source.additional_stop_token_ids
                )
                self.assertEqual(worker.model_max_length, 128)
                self.assertEqual(worker.encode("hello"), ids)
                self.assertEqual(worker.decode(ids, skip_special_tokens=True), text)
                self.assertEqual(
                    worker.apply_chat_template(messages, return_dict=False), chat_ids
                )

    async def test_batch_limit_uses_only_the_endpoint_input(self):
        await self.service.aclose()
        self.service = UtilityService(
            UtilityLimits(max_batch_size=2, max_input_items=32),
            validate_request=check_utility_items,
        )
        await self.service.start(make_utility_backend_factory(self.manager, None))
        self.tokenize.utility_service = self.detokenize.utility_service = self.service
        limit = self.service.limits.max_batch_size
        for request, status in (
            (TokenizeRequest(prompt="hi", tokens=1), 200),
            (TokenizeRequest(prompt="hi", tokens=list(range(32))), 413),
            (TokenizeRequest(prompt=[""] * limit, tokens=[[1]]), 200),
            (TokenizeRequest(prompt=[""] * (limit + 1), tokens=[[1]]), 413),
            (
                TokenizeRequest(
                    messages=[{"role": "user", "content": "hi"}] * (limit + 1),
                    tokens=[[1]],
                ),
                413,
            ),
            (DetokenizeRequest(tokens=[[67]] * (limit + 1)), 413),
            (DetokenizeRequest(tokens=[67] * (limit + 1)), 200),
        ):
            with self.subTest(model=type(request).__name__, status=status):
                handler = (
                    self.tokenize
                    if isinstance(request, TokenizeRequest)
                    else self.detokenize
                )
                actual, _ = await self.call(handler, request)
                self.assertEqual(actual, status)

        # Hold the sink with an unrelated record. Both over-limit paths must
        # still return 413 without submitting their detailed logs.
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        records = []

        class Sink(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())
                loop.call_soon_threadsafe(entered.set)
                release.wait(5)

        logger = self.manager.request_logger = RequestLogger(True, 2, "json", None)
        target = logging.Logger("over-limit")
        target.addHandler(Sink())
        logger.targets = [target]
        async with logger.background_writes():
            logger._log("busy")
            try:
                await asyncio.wait_for(entered.wait(), 2)
                for handler, request in (
                    (self.tokenize, TokenizeRequest(prompt=["a", "b", "c"])),
                    (self.detokenize, DetokenizeRequest(tokens=[67] * 33)),
                ):
                    status, _ = await asyncio.wait_for(self.call(handler, request), 2)
                    self.assertEqual(status, 413)
                    self.assertEqual(self.service.work_used, 0)
            finally:
                release.set()
        self.assertEqual(records, ["busy"])

    async def test_fastapi_validation_aliases_and_budget_policy(self):
        """Production validation errors must fit the utility budget without echoing inputs."""
        import httpx

        from sglang.srt.entrypoints.http_server import app as production_app

        app = FastAPI()
        app.router.routes.extend(
            route
            for route in production_app.routes
            if getattr(route, "path", None) in UTILITY_PATHS
        )
        app.add_exception_handler(
            RequestValidationError,
            production_app.exception_handlers[RequestValidationError],
        )
        app.state.utility_service = self.service
        app.state.openai_serving_tokenize = self.tokenize
        app.state.openai_serving_detokenize = self.detokenize
        app.add_middleware(UtilityHTTPMiddleware)
        app.add_middleware(CORSMiddleware, allow_origins=["*"])

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for path in ("/v1/tokenize", "/tokenize"):
                response = await client.post(path, json={"prompt": "中文"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 3)
                response = await client.post(
                    path, json={"prompt": "private prompt", "messages": []}
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["type"], "Bad Request")
                self.assertNotIn("private prompt", response.text)
                self.assertLessEqual(len(response.content), 256)
            for path in ("/v1/detokenize", "/detokenize"):
                response = await client.post(path, json={"tokens": []})
                self.assertEqual(response.json(), {"text": ""})
                self.assertEqual(response.status_code, 200)
                with patch.object(
                    RequestValidationError, "__str__", side_effect=AssertionError
                ):
                    response = await client.post(path, json={"tokens": ["invalid"] * 8})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["code"], 400)
                self.assertIn("tokens", response.json()["message"])
                self.assertLessEqual(len(response.content), 256)
            response = await client.post(
                "/tokenize",
                content=b"{}",
                headers={"x-body-compressed": "zstd", "Origin": "http://client"},
            )
            self.assertEqual(response.status_code, 415)
            self.assertEqual(response.headers["access-control-allow-origin"], "*")
            response = await client.post(
                "/tokenize",
                content=b"{}",
                headers={
                    "Content-Length": str(self.service.limits.max_input_bytes + 1)
                },
            )
            self.assertEqual(response.status_code, 413)
            response = await client.post(
                "/tokenize",
                json={"prompt": [""] * (self.service.limits.max_batch_size + 1)},
            )
            self.assertEqual(response.status_code, 413)
            self.assertFalse(self.service.tickets)


if __name__ == "__main__":
    unittest.main()
