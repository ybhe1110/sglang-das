import copy
import logging
import os
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

logger = logging.getLogger(__name__)


class _OpenAIServingUtility(OpenAIServingBase):
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
            return await self.utility_service.run(ticket, kind, request)
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
                if not all(isinstance(t, int) for t in request.tokens):
                    return self.create_error_response(
                        "Invalid input: 'tokens' must be a list of integers."
                    )
                tokens_to_decode = [int(t) for t in request.tokens]
                text = tokenizer.decode(
                    tokens_to_decode, skip_special_tokens=request.skip_special_tokens
                )
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


def _check_utility_items(request, limits):
    batch = getattr(request, "prompt", None)
    if batch is None:
        batch = getattr(request, "messages", None)
    tokens = getattr(request, "tokens", None)
    if tokens and isinstance(tokens[0], list):
        batch = tokens
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
        if count > limits.max_input_items:
            raise UtilityError("Tokenizer utility input exceeds item limit", 413)
        if isinstance(value, BaseModel):
            pending.append(iter(value.__dict__.values()))
            if value.__pydantic_extra__:
                pending.append(iter(value.__pydantic_extra__.values()))
        elif isinstance(value, dict):
            pending.append(iter(value.values()))
        elif isinstance(value, (list, tuple)):
            pending.append(iter(value))


def make_utility_backend_factory(manager, template_manager, limits):
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
            private_tokenizer = copy.deepcopy(tokenizer)
            private_template = copy.deepcopy(template_manager)
        except Exception as exc:
            raise RuntimeError(
                "Cannot create a private utility tokenizer/template"
            ) from exc
        if tokenizer is not None and private_tokenizer is tokenizer:
            raise RuntimeError("Utility tokenizer copy must be an independent instance")
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
            _check_utility_items(request, limits)
            handler = tokenize if kind == "tokenize" else detokenize
            result = handler._run_non_streaming_sync(request)
            # Preserve the existing flat errors; successful models are encoded
            # here too, so FastAPI does not serialize a large token list on-loop.
            if isinstance(result, Response):
                return result
            return ORJSONResponse(result.model_dump())

        return run

    return factory
