import copy
import logging
import os
from functools import partial
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, Response
from pydantic import BaseModel

from sglang.srt.entrypoints.openai.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.utility_executor import UtilityError
from sglang.srt.entrypoints.openai.utility_http import utility_error_response
from sglang.srt.entrypoints.openai.utility_phase import mark_phase
from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer
from sglang.srt.utils.request_logger import _extract_whitelisted_headers

logger = logging.getLogger(__name__)


class _OpenAIServingUtility(OpenAIServingBase):
    async def _log_received_request(self, request, raw_request):
        # Log inside the admitted utility work item, before conversion. The base
        # hook must not dump/format the payload or write to a sink on the loop.
        pass

    async def _convert_to_internal_request_async(self, request, raw_request=None):
        # Utility conversion is an identity; never queue it behind generation
        # tokenization. The complete synchronous operation runs in our pool.
        return request, request

    async def _run_utility(self, kind, request, raw_request):
        ticket = raw_request.scope.get("utility_ticket") if raw_request else None
        if ticket is None or self.utility_service is None:
            return utility_error_response(
                UtilityError("Utility admission is unavailable", 503)
            )
        try:
            before_run = None
            request_logger = self.tokenizer_manager.request_logger
            if request_logger.log_requests and request_logger.log_requests_level >= 2:
                # Snapshot configuration and whitelisted headers, not the body
                # or ASGI Request. Only this admitted worker waits for its log.
                before_run = partial(
                    copy.copy(request_logger).log_utility_request,
                    request,
                    headers=_extract_whitelisted_headers(raw_request),
                )
            return await self.utility_service.run(
                ticket, kind, request, before_run=before_run
            )
        except UtilityError as error:
            return utility_error_response(error)


class OpenAIServingTokenize(_OpenAIServingUtility):
    """Handler for /v1/tokenize requests"""

    def __init__(
        self, tokenizer_manager, template_manager=None, *, utility_service=None
    ):
        super().__init__(tokenizer_manager)
        self.utility_service = utility_service
        self.chat_serving: Optional[OpenAIServingChat] = (
            OpenAIServingChat(tokenizer_manager, template_manager)
            if template_manager is not None and utility_service is None
            else None
        )

    def _request_id_prefix(self) -> str:
        return "tok-"

    def _convert_to_internal_request(
        self, request: TokenizeRequest, raw_request: Request
    ) -> tuple[TokenizeRequest, TokenizeRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: TokenizeRequest,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Response:
        return await self._run_utility("tokenize", request, raw_request)

    def _run_non_streaming_sync(self, request: TokenizeRequest):
        try:
            tokenizer = self.tokenizer_manager.tokenizer
            max_model_len = getattr(tokenizer, "model_max_length", -1)
            # ORJSON accepts signed negative and unsigned positive 64-bit integers.
            if isinstance(max_model_len, int) and not (
                -(2**63) <= max_model_len < 2**64
            ):
                max_model_len = self.tokenizer_manager.model_config.context_len

            if request.messages is not None:
                token_ids = self._tokenize_chat_request(request)
                tokens = token_ids
                count = len(token_ids)
            elif isinstance(request.prompt, str):
                token_ids = tokenizer.encode(
                    request.prompt,
                    add_special_tokens=request.add_special_tokens,
                )
                tokens = token_ids
                count = len(token_ids)
            elif isinstance(request.prompt, list):
                token_ids_list = [
                    tokenizer.encode(
                        text, add_special_tokens=request.add_special_tokens
                    )
                    for text in request.prompt
                ]
                tokens = token_ids_list
                count = [len(ids) for ids in token_ids_list]
            else:
                return self.create_error_response(
                    f"Invalid prompt type: {type(request.prompt)}. Expected str or List[str]."
                )

            return TokenizeResponse(
                tokens=tokens, count=count, max_model_len=max_model_len
            )
        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error("Error during tokenization", exc_info=True)
            return self.create_error_response(
                f"Internal server error during tokenization: {e}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _tokenize_chat_request(self, request: TokenizeRequest) -> List[int]:
        if self.chat_serving is None:
            raise ValueError("Chat template tokenization requires a template manager.")

        chat_request = request.to_chat_completion_request()
        validation_error = self.chat_serving._validate_request(chat_request)
        if validation_error:
            raise ValueError(validation_error)

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self.chat_serving._process_messages(
            chat_request, is_multimodal
        )

        prompt_ids = processed_messages.prompt_ids
        if isinstance(prompt_ids, list) and (
            prompt_ids or not processed_messages.prompt
        ):
            return prompt_ids
        if isinstance(prompt_ids, str):
            return self.tokenizer_manager.tokenizer.encode(
                prompt_ids, add_special_tokens=False
            )
        if processed_messages.prompt:
            return self.tokenizer_manager.tokenizer.encode(
                processed_messages.prompt, add_special_tokens=False
            )

        raise ValueError("Failed to render chat messages into token ids.")


class OpenAIServingDetokenize(_OpenAIServingUtility):
    """Handler for /v1/detokenize requests"""

    def __init__(self, tokenizer_manager, *, utility_service=None):
        super().__init__(tokenizer_manager)
        self.utility_service = utility_service

    def _request_id_prefix(self) -> str:
        return "detok-"

    def _convert_to_internal_request(
        self, request: DetokenizeRequest, raw_request: Request
    ) -> tuple[DetokenizeRequest, DetokenizeRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: DetokenizeRequest,
        request: DetokenizeRequest,
        raw_request: Request,
    ) -> Response:
        return await self._run_utility("detokenize", request, raw_request)

    def _run_non_streaming_sync(self, request: DetokenizeRequest):
        try:
            tokenizer = self.tokenizer_manager.tokenizer

            if (
                isinstance(request.tokens, list)
                and request.tokens
                and isinstance(request.tokens[0], int)
            ):
                mark_phase("normalize_begin")
                if not all(isinstance(t, int) for t in request.tokens):
                    return self.create_error_response(
                        "Invalid input: 'tokens' must be a list of integers."
                    )
                tokens_to_decode = [int(t) for t in request.tokens]
                mark_phase("decode_begin")
                text = tokenizer.decode(
                    tokens_to_decode, skip_special_tokens=request.skip_special_tokens
                )
                mark_phase("decode_end")
                text_out: Union[str, List[str]] = text
            elif (
                isinstance(request.tokens, list)
                and request.tokens
                and isinstance(request.tokens[0], list)
            ):
                texts: List[str] = []
                for token_list in request.tokens:
                    if not all(isinstance(t, int) for t in token_list):
                        return self.create_error_response(
                            f"Invalid input: Sublist in 'tokens' must contain only integers. Found: {token_list}"
                        )
                    decoded_text = tokenizer.decode(
                        [int(t) for t in token_list],
                        skip_special_tokens=request.skip_special_tokens,
                    )
                    texts.append(decoded_text)
                text_out = texts
            elif isinstance(request.tokens, list) and not request.tokens:
                text_out = ""
            else:
                return self.create_error_response(
                    f"Invalid tokens type: {type(request.tokens)}. Expected List[int] or List[List[int]]."
                )

            return DetokenizeResponse(text=text_out)
        except Exception as e:
            logger.error("Error during detokenization", exc_info=True)
            if "decode" in str(e).lower():
                return self.create_error_response(
                    f"Error decoding tokens: {e}. Input tokens might be invalid for the model.",
                    err_type="DecodeError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            return self.create_error_response(
                f"Internal server error during detokenization: {e}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def check_utility_items(request: Union[TokenizeRequest, DetokenizeRequest], limits):
    if isinstance(request, DetokenizeRequest):
        tokens = request.tokens
        batch = tokens if tokens and isinstance(tokens[0], list) else None
    else:
        tokens = None
        batch = request.prompt if request.prompt is not None else request.messages
    if isinstance(batch, list) and len(batch) > limits.max_batch_size:
        raise UtilityError("Tokenizer utility batch exceeds item limit", 413)
    # Count the complete input, including extra Chat template/schema fields.
    # Iterators avoid another copy of large token arrays or message dictionaries.
    pending = [iter((request,))]
    count = 0
    while pending:
        try:
            value = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        count += 1
        if isinstance(value, BaseModel):
            pending.append(iter(value.__dict__.values()))
            if value.__pydantic_extra__:
                pending.append(iter(value.__pydantic_extra__.values()))
        elif isinstance(value, dict):
            pending.append(iter(value.values()))
        elif isinstance(value, (list, tuple)):
            if value is tokens:
                # Pydantic validated this field as list[int] or list[list[int]].
                # Its children are either IDs or rows; each row also owns its IDs.
                count += len(tokens)
                if batch is not None and count <= limits.max_input_items:
                    count += sum(map(len, batch))
            else:
                pending.append(iter(value))
        if count > limits.max_input_items:
            raise UtilityError("Tokenizer utility input exceeds item limit", 413)


def copy_tokenizer_for_worker(tokenizer):
    if isinstance(tokenizer, TiktokenTokenizer):
        # Rebuild the native encoding, its bound encode method and Jinja template.
        private = type(tokenizer)(tokenizer.tokenizer.name)
    elif getattr(tokenizer, "_mistral_common_patched", False) is True:
        from sglang.srt.utils.hf_transformers.mistral_utils import (
            patch_mistral_common_tokenizer,
        )

        # The compatibility wrappers close over the original instance. Install
        # them on a fresh backend loaded from the same resolved local file.
        private = type(tokenizer)(
            tokenizer._tokenizer_path,
            mode=tokenizer.mode,
            model_max_length=tokenizer.model_max_length,
            padding_side=tokenizer.padding_side,
            truncation_side=tokenizer.truncation_side,
            model_input_names=copy.deepcopy(tokenizer.model_input_names),
            clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
        )
        private.pad_token = tokenizer.pad_token
        patch_mistral_common_tokenizer(private)
    else:
        private = copy.deepcopy(tokenizer)
        if tokenizer is not None and private is tokenizer:
            raise RuntimeError("Utility tokenizer copy must be an independent instance")
        return private

    # Template selection and stop tokens can be configured after loading.
    private.chat_template = copy.deepcopy(tokenizer.chat_template)
    if hasattr(tokenizer, "additional_stop_token_ids"):
        private.additional_stop_token_ids = copy.deepcopy(
            tokenizer.additional_stop_token_ids
        )
    return private


def make_utility_backend_factory(manager, template_manager):
    """Capture only text-serving inputs, never manager sockets or GPU state.

    Copy during startup, before generation can mutate the source tokenizer.
    Each one-thread executor exclusively owns its tokenizer and Chat handler.
    A non-copyable custom tokenizer fails startup explicitly instead of silently
    sharing mutable tokenizer state with generation.
    """
    tokenizer = manager.tokenizer
    config = {
        name: manager.config_value(name)
        for name in ("tool_call_parser", "reasoning_parser")
    }
    metadata = dict(
        server_args=manager.server_args,
        model_config=manager.model_config,
        model_path=manager.model_path,
        served_model_name=manager.served_model_name,
        image_token_id=getattr(manager, "image_token_id", None),
    )

    def factory():
        try:
            private_tokenizer = copy_tokenizer_for_worker(tokenizer)
            private_template = copy.deepcopy(template_manager)
        except Exception as exc:
            raise RuntimeError(
                "Cannot create a private utility tokenizer/template"
            ) from exc
        versions = {}
        for package in ("transformers", "tokenizers"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not installed"
        logger.info(
            "Utility tokenizer initialized: pid=%s class=%s is_fast=%s versions=%s",
            os.getpid(),
            type(private_tokenizer).__qualname__,
            getattr(private_tokenizer, "is_fast", None),
            versions,
        )
        view = SimpleNamespace(
            **metadata, tokenizer=private_tokenizer, config_value=config.__getitem__
        )
        tokenize = OpenAIServingTokenize(view, private_template)
        detokenize = OpenAIServingDetokenize(view)

        def run(kind, request):
            handler = tokenize if kind == "tokenize" else detokenize
            result = handler._run_non_streaming_sync(request)
            # Preserve the existing flat errors; successful models are encoded
            # here too, so FastAPI does not serialize a large token list on-loop.
            if isinstance(result, Response):
                return result
            mark_phase("serialize_begin")
            response = ORJSONResponse(result.model_dump())
            mark_phase("serialize_end")
            return response

        return run

    return factory
