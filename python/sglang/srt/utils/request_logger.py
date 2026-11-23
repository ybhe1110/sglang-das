# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from sglang.srt.environ import envs
from sglang.srt.sampling.watermarking.core import redact_watermark_secrets
from sglang.srt.utils.log_utils import create_log_targets, format_json_log

if TYPE_CHECKING:
    import fastapi

    from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)

_DEFAULT_WHITELISTED_HEADERS = ["x-smg-routing-key"]
WHITELISTED_HEADERS = _DEFAULT_WHITELISTED_HEADERS + [
    h.lower() for h in envs.SGLANG_LOG_REQUEST_HEADERS.get()
]


def _extract_whitelisted_headers(
    request: Optional[fastapi.Request],
) -> Optional[Dict[str, str]]:
    if request is None:
        return None
    return {h: v for h in WHITELISTED_HEADERS if (v := request.headers.get(h))}


class _RequestLogWriter:
    """One FIFO per HTTP process. Only the writer thread takes sink locks."""

    MAX_RECORDS = 128
    MAX_BYTES = 8 * 1024 * 1024

    def __init__(self):
        self._ready = threading.Condition()
        self._queue = deque()
        self._bytes = 0
        self._closing = False
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="sgl-request-log", daemon=True
        )
        self._thread.start()

    def write(self, targets, message, wait):
        size = sys.getsizeof(message)
        with self._ready:
            if (
                self._closing
                or len(self._queue) >= self.MAX_RECORDS
                or self._bytes + size > self.MAX_BYTES
            ):
                self.dropped += 1
                return
            done = threading.Event() if wait else None
            self._bytes += size
            self._queue.append((targets, message, size, done))
            self._ready.notify()
        if done is not None:
            done.wait()

    def _run(self):
        reported = 0
        while True:
            with self._ready:
                self._ready.wait_for(lambda: self._queue or self._closing)
                if not self._queue:
                    break
                targets, message, size, done = self._queue.popleft()
            try:
                for target in targets:
                    target.info(message)
            except Exception:
                logger.exception("Failed to write request log")
            finally:
                with self._ready:
                    self._bytes -= size
                    dropped = self.dropped
                if done is not None:
                    done.set()
            if dropped > reported:
                logger.warning("Dropped %d request log records", dropped - reported)
                reported = dropped
            # Release the last message before waiting for another record.
            del targets, message
        if self.dropped > reported:
            logger.warning("Dropped %d request log records", self.dropped - reported)

    def close(self):
        with self._ready:
            self._closing = True
            self._ready.notify()
        self._thread.join(timeout=1.0)


class RequestLogger:
    def __init__(
        self,
        log_requests: bool,
        log_requests_level: int,
        log_requests_format: str,
        log_requests_target: Optional[List[str]],
    ):
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self.log_requests_format = log_requests_format
        self.log_requests_target = log_requests_target

        self.metadata: Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]] = (
            self._compute_metadata()
        )
        self.targets = self._setup_targets()
        self._writer = None

        self.log_exceeded_ms = envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.get()

    def _setup_targets(self) -> List[logging.Logger]:
        return create_log_targets(
            targets=self.log_requests_target, name_prefix=__name__
        )

    @asynccontextmanager
    async def background_writes(self):
        writer = self._writer = _RequestLogWriter()
        try:
            yield writer
        finally:
            try:
                await asyncio.to_thread(writer.close)
            finally:
                self._writer = None

    def configure(
        self,
        log_requests: Optional[bool] = None,
        log_requests_level: Optional[int] = None,
        log_requests_format: Optional[str] = None,
        log_requests_target: Optional[List[str]] = None,
    ) -> None:
        if log_requests is not None:
            self.log_requests = log_requests
        if log_requests_level is not None:
            self.log_requests_level = log_requests_level
        if log_requests_format is not None:
            self.log_requests_format = log_requests_format
        if log_requests_target is not None:
            self.log_requests_target = log_requests_target

        self.metadata = self._compute_metadata()
        self.targets = self._setup_targets()

    def log_received_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenizer: Any = None,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        if not self.log_requests:
            return

        max_length, skip_names, _ = self.metadata
        headers = _extract_whitelisted_headers(request)
        if self.log_requests_format == "json":
            log_data = {
                "rid": obj.rid,
                "obj": _transform_data_for_logging(obj, max_length, skip_names),
            }
            if headers:
                log_data["headers"] = headers
            self._log(format_json_log("request.received", log_data))
        else:
            headers_str = f", headers={headers}" if headers else ""
            self._log(
                f"Receive: obj={_dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}{headers_str}"
            )

        # FIXME: This is a temporary fix to get the text from the input ids.
        # We should remove this once we have a proper way.
        if (
            self.log_requests_level >= 2
            and obj.text is None
            and obj.input_ids is not None
            and tokenizer is not None
        ):
            if obj.input_ids and isinstance(obj.input_ids[0], list):
                # Prefill node warmup while PD disaggregated.
                decoded = [
                    tokenizer.decode(_input_ids, skip_special_tokens=False)
                    for _input_ids in obj.input_ids
                ]
            else:
                decoded = tokenizer.decode(obj.input_ids, skip_special_tokens=False)
            obj.text = decoded

    def log_openai_received_request(
        self,
        obj: Any,
        request: Optional[fastapi.Request] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        wait_for_write: bool = False,
    ) -> None:
        """Log the raw OpenAI request payload before request adaptation/tokenization."""
        max_length, _, _ = self.metadata
        max_length = max_length if max_length is not None else 2048
        if headers is None:
            headers = _extract_whitelisted_headers(request)

        if hasattr(obj, "model_dump"):
            obj_to_log = obj.model_dump(exclude_none=True)
        else:
            obj_to_log = obj

        if self.log_requests_format == "json":
            log_data = {
                "obj": _transform_data_for_logging(obj_to_log, max_length=max_length),
            }
            if headers:
                log_data["headers"] = headers
            message = format_json_log("request.received.openai", log_data)
        else:
            headers_str = f", headers={headers}" if headers else ""
            message = f"Receive OpenAI: obj={_dataclass_to_string_truncated(obj_to_log, max_length)}{headers_str}"
        self._log(message, wait_for_write=wait_for_write)

    def log_finished_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        out: Any,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        if not self.log_requests:
            return

        e2e_latency_ms = out["meta_info"].get("e2e_latency", 0) * 1000
        if self.log_exceeded_ms > 0 and e2e_latency_ms < self.log_exceeded_ms:
            return

        max_length, skip_names, out_skip_names = self.metadata
        headers = _extract_whitelisted_headers(request)
        if self.log_requests_format == "json":
            log_data = {
                "rid": obj.rid,
                "obj": _transform_data_for_logging(obj, max_length, skip_names),
            }
            if headers:
                log_data["headers"] = headers
            log_data["out"] = _transform_data_for_logging(
                out, max_length, out_skip_names
            )
            self._log(format_json_log("request.finished", log_data))
        else:
            obj_str = _dataclass_to_string_truncated(
                obj, max_length, skip_names=skip_names
            )
            out_str = f", out={_dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
            headers_str = f", headers={headers}" if headers else ""
            self._log(f"Finish: obj={obj_str}{headers_str}{out_str}")

    def _compute_metadata(
        self,
    ) -> Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]]:
        max_length: Optional[int] = None
        skip_names: Optional[Set[str]] = None
        out_skip_names: Optional[Set[str]] = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "lora_path",
                    "sampling_params",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "lora_path",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
            else:
                raise ValueError(
                    f"Invalid --log-requests-level: {self.log_requests_level=}"
                )
        return max_length, skip_names, out_skip_names

    def _log(self, msg: str, *, wait_for_write: bool = False) -> None:
        writer = self._writer
        if writer is not None:
            writer.write(self.targets, msg, wait_for_write)
        else:
            for target in self.targets:
                target.info(msg)


# TODO unify this w/ `_transform_data_for_logging` if we find performance enough
def _dataclass_to_string_truncated(
    data: Any, max_length: int = 2048, skip_names: Optional[Set[str]] = None
) -> str:
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return (
                str(redact_watermark_secrets(data[:half_length]))
                + " ... "
                + str(redact_watermark_secrets(data[-half_length:]))
            )
        else:
            return str(redact_watermark_secrets(data))
    elif isinstance(data, dict):
        data = redact_watermark_secrets(data)
        return (
            "{"
            + ", ".join(
                f"'{k}': {_dataclass_to_string_truncated(v, max_length)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        values = redact_watermark_secrets(
            {f.name: getattr(data, f.name) for f in fields}
        )
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={_dataclass_to_string_truncated(values[f.name], max_length)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


def _transform_data_for_logging(
    data: Any, max_length: int = 2048, skip_names: Optional[Set[str]] = None
) -> Any:
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return data[:half_length] + "..." + data[-half_length:]
        return data
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return list(data[:half_length]) + ["..."] + list(data[-half_length:])
        return [_transform_data_for_logging(v, max_length) for v in data]
    elif isinstance(data, dict):
        data = redact_watermark_secrets(data)
        return {
            k: _transform_data_for_logging(v, max_length)
            for k, v in data.items()
            if k not in skip_names
        }
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        values = redact_watermark_secrets(
            {f.name: getattr(data, f.name) for f in fields}
        )
        return {
            f.name: _transform_data_for_logging(values[f.name], max_length)
            for f in fields
            if f.name not in skip_names
        }
    elif isinstance(data, (int, float, bool, type(None))):
        return data
    else:
        return str(data)
