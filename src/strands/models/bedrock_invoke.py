"""Bedrock provider using the native InvokeModel APIs.

Use in place of :class:`~strands.models.bedrock.BedrockModel` when the target model does not
support Converse (Custom Model Import, etc.). Request format auto-detects from the model id:
``anthropic.*``/``*claude*`` use the Anthropic Messages API, everything else uses the OpenAI
Chat Completions API. Override with ``model_family``.
"""

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, TypeVar, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing_extensions import Unpack, override

from .._exception_notes import add_exception_note
from ..event_loop import streaming
from ..tools import convert_pydantic_to_tool_spec
from ..types.content import Messages, SystemContentBlock
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .bedrock import (
    BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES,
    DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_BEDROCK_REGION,
    DEFAULT_READ_TIMEOUT,
)
from .model import BaseModelConfig, Model

logger = logging.getLogger(__name__)

ModelFamily = Literal["anthropic", "openai"]
T = TypeVar("T", bound=BaseModel)


_BLOCK_STOP: StreamEvent = {"contentBlockStop": {}}
_TEXT_START: StreamEvent = {"contentBlockStart": {"start": {}}}


def _text_delta(t: str) -> StreamEvent:
    return {"contentBlockDelta": {"delta": {"text": t}}}


def _tool_use_start(tool_use_id: str, name: str) -> StreamEvent:
    return {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}}


def _tool_use_delta(partial_json: str) -> StreamEvent:
    return {"contentBlockDelta": {"delta": {"toolUse": {"input": partial_json}}}}


def _metadata(in_tok: int, out_tok: int, total: int | None = None) -> StreamEvent:
    return {
        "metadata": {
            "usage": {"inputTokens": in_tok, "outputTokens": out_tok, "totalTokens": total or in_tok + out_tok},
            "metrics": {"latencyMs": 0},
        }
    }


class BedrockModelInvoke(Model):
    """AWS Bedrock model provider using ``InvokeModel`` / ``InvokeModelWithResponseStream``."""

    class BedrockInvokeConfig(BaseModelConfig, total=False):
        """Configuration options for ``BedrockModelInvoke``. ``model_family`` overrides id-based detection."""

        model_id: str
        model_family: ModelFamily | None
        max_tokens: int | None
        streaming: bool | None
        temperature: float | None
        top_p: float | None
        top_k: int | None
        stop_sequences: list[str] | None

    def __init__(
        self,
        *,
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        **model_config: Unpack[BedrockInvokeConfig],
    ):
        """Initialize the provider. ``boto_session`` and ``region_name`` are mutually exclusive."""
        if region_name and boto_session:
            raise ValueError("Cannot specify both `region_name` and `boto_session`.")

        validate_config_keys(model_config, self.BedrockInvokeConfig)

        session = boto_session or boto3.Session()
        resolved_region = region_name or session.region_name or os.environ.get("AWS_REGION") or DEFAULT_BEDROCK_REGION

        config: BedrockModelInvoke.BedrockInvokeConfig = {
            "model_id": model_config.get("model_id", DEFAULT_BEDROCK_MODEL_ID),
            "streaming": model_config.get("streaming", True),
        }
        config.update({k: v for k, v in model_config.items() if k != "model_id"})  # type: ignore[typeddict-item]
        self.config = config
        logger.debug("config=<%s> | initializing", self.config)

        if boto_client_config:
            extra = getattr(boto_client_config, "user_agent_extra", None)
            ua = f"{extra} strands-agents" if extra else "strands-agents"
            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=ua))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents", read_timeout=DEFAULT_READ_TIMEOUT)

        self.client = session.client(
            service_name="bedrock-runtime",
            config=client_config,
            endpoint_url=endpoint_url,
            region_name=resolved_region,
        )
        logger.debug("region=<%s> | bedrock client created", self.client.meta.region_name)

    @override
    def update_config(self, **model_config: Unpack[BedrockInvokeConfig]) -> None:  # type: ignore[override]
        """Update the model configuration."""
        validate_config_keys(model_config, self.BedrockInvokeConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> BedrockInvokeConfig:
        """Return the current configuration."""
        return self.config

    def _get_model_family(self) -> ModelFamily:
        """Detect the request/response format from the configured model id."""
        if family := self.config.get("model_family"):
            return family
        model_id = self.config["model_id"].lower()
        return "anthropic" if "anthropic" in model_id or "claude" in model_id else "openai"

    # ----- request formatting

    @staticmethod
    def _media_type(image_format: str) -> str:
        return image_format if image_format.startswith("image/") else f"image/{image_format}"

    @staticmethod
    def _system_text(blocks: list[SystemContentBlock] | None) -> str:
        return " ".join(b.get("text", "") for b in (blocks or []) if "text" in b)

    @staticmethod
    def _to_tool_choice(tool_choice: ToolChoice | None, family: ModelFamily) -> Any:
        """Translate Strands ``ToolChoice`` to the family-specific tool_choice shape."""
        if not tool_choice:
            return None
        c = cast(dict[str, Any], tool_choice)
        if family == "anthropic":
            if "tool" in c:
                return {"type": "tool", "name": c["tool"]["name"]}
            return {"type": "any"} if "any" in c else {"type": "auto"} if "auto" in c else None
        if "tool" in c:
            return {"type": "function", "function": {"name": c["tool"]["name"]}}
        return "required" if "any" in c else "auto" if "auto" in c else None

    def _format_anthropic_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.get("max_tokens", 4096),
            "messages": [],
        }
        if system := self._system_text(system_prompt_content):
            request["system"] = system

        for msg in messages:
            content: list[dict[str, Any]] = []
            for block in msg["content"]:
                if "text" in block:
                    content.append({"type": "text", "text": block["text"]})
                elif "image" in block:
                    img = block["image"]
                    source = {
                        "type": "base64",
                        "media_type": self._media_type(img["format"]),
                        "data": base64.b64encode(img["source"]["bytes"]).decode("utf-8"),
                    }
                    content.append({"type": "image", "source": source})
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    content.append(
                        {"type": "tool_use", "id": tu["toolUseId"], "name": tu["name"], "input": tu["input"]}
                    )
                elif "toolResult" in block:
                    tr = block["toolResult"]
                    rc: list[dict[str, Any]] = [
                        {"type": "text", "text": rb["text"] if "text" in rb else json.dumps(rb["json"])}
                        for rb in tr["content"]
                        if "text" in rb or "json" in rb
                    ]
                    entry: dict[str, Any] = {"type": "tool_result", "tool_use_id": tr["toolUseId"], "content": rc}
                    if tr.get("status") == "error":
                        entry["is_error"] = True
                    content.append(entry)
            if content:
                request["messages"].append({"role": msg["role"], "content": content})

        if tool_specs:
            request["tools"] = [
                {"name": s["name"], "description": s["description"], "input_schema": s["inputSchema"]}
                for s in tool_specs
            ]
        if (tc := self._to_tool_choice(tool_choice, "anthropic")) is not None:
            request["tool_choice"] = tc

        if self.config.get("temperature") is not None:
            request["temperature"] = self.config["temperature"]
        if self.config.get("top_p") is not None:
            request["top_p"] = self.config["top_p"]
        if self.config.get("top_k") is not None:
            request["top_k"] = self.config["top_k"]
        if self.config.get("stop_sequences"):
            request["stop_sequences"] = self.config["stop_sequences"]
        return request

    def _format_openai_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config["model_id"],
            "messages": [],
            "max_tokens": self.config.get("max_tokens", 4096),
            "stream": self.config.get("streaming", True),
        }
        if system := self._system_text(system_prompt_content):
            request["messages"].append({"role": "system", "content": system})

        for msg in messages:
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for block in msg["content"]:
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    fn = {"name": tu["name"], "arguments": json.dumps(tu["input"])}
                    tool_calls.append({"id": tu["toolUseId"], "type": "function", "function": fn})
                elif "toolResult" in block:
                    tr = block["toolResult"]
                    chunks = [c["text"] if "text" in c else json.dumps(c.get("json", "")) for c in tr["content"]]
                    tool_results.append({"role": "tool", "tool_call_id": tr["toolUseId"], "content": "".join(chunks)})
                # Images are dropped on the OpenAI path; use model_family="anthropic" for multimodal input.
            if tool_calls or text_parts:
                entry: dict[str, Any] = {"role": msg["role"]}
                if text_parts:
                    entry["content"] = "".join(text_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                    entry.setdefault("content", None)
                request["messages"].append(entry)
            request["messages"].extend(tool_results)

        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {"name": s["name"], "description": s["description"], "parameters": s["inputSchema"]},
                }
                for s in tool_specs
            ]
            if (tc := self._to_tool_choice(tool_choice, "openai")) is not None:
                request["tool_choice"] = tc

        if self.config.get("temperature") is not None:
            request["temperature"] = self.config["temperature"]
        if self.config.get("top_p") is not None:
            request["top_p"] = self.config["top_p"]
        if self.config.get("stop_sequences"):
            request["stop"] = self.config["stop_sequences"]
        return request

    def _format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any]:
        if self._get_model_family() == "anthropic":
            return self._format_anthropic_request(messages, tool_specs, system_prompt_content, tool_choice)
        return self._format_openai_request(messages, tool_specs, system_prompt_content, tool_choice)

    # ----- response translation

    _ANTHROPIC_STOP = {"tool_use": "tool_use", "max_tokens": "max_tokens", "stop_sequence": "stop_sequence"}
    _OPENAI_STOP = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}

    @classmethod
    def _map_anthropic_stop(cls, reason: str | None) -> str:
        return cls._ANTHROPIC_STOP.get(reason or "", "end_turn")

    @classmethod
    def _map_openai_stop(cls, reason: str | None) -> str:
        return cls._OPENAI_STOP.get(reason or "", "end_turn")

    def _emit_anthropic_chunks(self, body: Any, callback: Callable[..., None]) -> None:
        """Translate an Anthropic Messages stream into Strands ``StreamEvent``s."""
        callback({"messageStart": {"role": "assistant"}})
        stop_reason: str | None = None
        in_toks = out_toks = 0
        active: str | None = None

        for event in body:
            chunk = json.loads(event["chunk"]["bytes"])
            t = chunk.get("type")
            logger.debug("anthropic_chunk_type=<%s>", t)
            if t == "message_start":
                u = (chunk.get("message") or {}).get("usage") or {}
                in_toks = u.get("input_tokens", in_toks)
                out_toks = u.get("output_tokens", out_toks)
            elif t == "content_block_start":
                cb = chunk.get("content_block") or {}
                if cb.get("type") == "tool_use":
                    active = "tool_use"
                    callback(_tool_use_start(cb["id"], cb["name"]))
                else:
                    active = "text"
                    callback(_TEXT_START)
            elif t == "content_block_delta":
                d = chunk.get("delta") or {}
                if "text" in d:
                    callback(_text_delta(d["text"]))
                elif d.get("type") == "input_json_delta" and "partial_json" in d:
                    callback(_tool_use_delta(d["partial_json"]))
            elif t == "content_block_stop":
                if active is not None:
                    callback(_BLOCK_STOP)
                    active = None
            elif t == "message_delta":
                d = chunk.get("delta") or {}
                if "stop_reason" in d:
                    stop_reason = d["stop_reason"]
                u = chunk.get("usage") or {}
                if "output_tokens" in u:
                    out_toks = u["output_tokens"]
            # message_stop carries no payload of interest.

        if active is not None:
            callback(_BLOCK_STOP)
        callback({"messageStop": {"stopReason": self._map_anthropic_stop(stop_reason)}})
        callback(_metadata(in_toks, out_toks))

    def _emit_openai_chunks(self, body: Any, callback: Callable[..., None]) -> None:
        """Translate an OpenAI Chat Completions stream into Strands ``StreamEvent``s.

        Tool calls are keyed by ``index`` and emitted lazily once an id or function name appears.
        """
        callback({"messageStart": {"role": "assistant"}})
        text_open = False
        started: set[int] = set()
        stop_reason: str | None = None
        usage: dict[str, Any] | None = None

        for event in body:
            chunk = json.loads(event["chunk"]["bytes"])
            if choices := chunk.get("choices"):
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if not text_open:
                        callback(_TEXT_START)
                        text_open = True
                    callback(_text_delta(delta["content"]))
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if idx not in started and (tc.get("id") or fn.get("name")):
                        if text_open:
                            callback(_BLOCK_STOP)
                            text_open = False
                        callback(_tool_use_start(tc.get("id") or f"call_{idx}", fn.get("name", "")))
                        started.add(idx)
                    if (args := fn.get("arguments")) and idx in started:
                        callback(_tool_use_delta(args))
                if finish := choices[0].get("finish_reason"):
                    stop_reason = finish
            if chunk.get("usage"):
                usage = chunk["usage"]

        if text_open:
            callback(_BLOCK_STOP)
        for _ in started:
            callback(_BLOCK_STOP)
        callback({"messageStop": {"stopReason": self._map_openai_stop(stop_reason)}})
        if usage:
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            callback(_metadata(inp, out, usage.get("total_tokens", inp + out)))

    def _emit_anthropic_non_streaming(self, body: dict[str, Any], callback: Callable[..., None]) -> None:
        """Translate a non-streaming Anthropic Messages response into events."""
        callback({"messageStart": {"role": "assistant"}})
        for block in body.get("content") or []:
            bt = block.get("type")
            if bt == "text":
                callback(_TEXT_START)
                callback(_text_delta(block.get("text", "")))
                callback(_BLOCK_STOP)
            elif bt == "tool_use":
                callback(_tool_use_start(block["id"], block["name"]))
                callback(_tool_use_delta(json.dumps(block.get("input", {}))))
                callback(_BLOCK_STOP)
        callback({"messageStop": {"stopReason": self._map_anthropic_stop(body.get("stop_reason"))}})
        if u := body.get("usage"):
            callback(_metadata(u.get("input_tokens", 0), u.get("output_tokens", 0)))

    def _emit_openai_non_streaming(self, body: dict[str, Any], callback: Callable[..., None]) -> None:
        """Translate a non-streaming OpenAI Chat Completions response into events."""
        callback({"messageStart": {"role": "assistant"}})
        choices = body.get("choices") or []
        finish: str | None = None
        if choices:
            choice = choices[0]
            msg = choice.get("message") or {}
            finish = choice.get("finish_reason")
            if content := msg.get("content"):
                callback(_TEXT_START)
                callback(_text_delta(content))
                callback(_BLOCK_STOP)
            for idx, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                callback(_tool_use_start(tc.get("id") or f"call_{idx}", fn.get("name", "")))
                callback(_tool_use_delta(fn.get("arguments", "")))
                callback(_BLOCK_STOP)
        callback({"messageStop": {"stopReason": self._map_openai_stop(finish)}})
        if u := body.get("usage"):
            inp = u.get("prompt_tokens", 0)
            out = u.get("completion_tokens", 0)
            callback(_metadata(inp, out, u.get("total_tokens", inp + out)))

    # ----- public API

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a turn through Bedrock InvokeModel.

        Raises:
            ContextWindowOverflowException: Input exceeded the model's context window.
            ModelThrottledException: Bedrock throttled the request.
        """

        def callback(event: StreamEvent | None = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        thread = asyncio.to_thread(self._stream, callback, messages, tool_specs, system_prompt_content, tool_choice)
        task = asyncio.create_task(thread)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            await task

    def _stream(
        self,
        callback: Callable[..., None],
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
    ) -> None:
        """Run the InvokeModel call on a worker thread and stream events."""
        try:
            family = self._get_model_family()
            request = self._format_request(messages, tool_specs, system_prompt_content, tool_choice)
            logger.debug("family=<%s> request=<%s>", family, request)

            common_kwargs = {
                "modelId": self.config["model_id"],
                "body": json.dumps(request),
                "contentType": "application/json",
                "accept": "application/json",
            }

            if self.config.get("streaming", True):
                response = self.client.invoke_model_with_response_stream(**common_kwargs)
                emit = self._emit_anthropic_chunks if family == "anthropic" else self._emit_openai_chunks
                emit(response["body"], callback)
            else:
                response = self.client.invoke_model(**common_kwargs)
                body = json.loads(response["body"].read())
                logger.debug("response_body=<%s>", body)
                emit = self._emit_anthropic_non_streaming if family == "anthropic" else self._emit_openai_non_streaming
                emit(body, callback)

        except ClientError as e:
            msg = str(e)
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "throttlingException"):
                raise ModelThrottledException(msg) from e
            if any(o in msg for o in BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES):
                logger.warning("bedrock threw context window overflow error")
                raise ContextWindowOverflowException(e) from e
            add_exception_note(e, f"└ Bedrock region: {self.client.meta.region_name}")
            add_exception_note(e, f"└ Model id: {self.config.get('model_id')}")
            if code == "AccessDeniedException" and "You don't have access to the model" in msg:
                add_exception_note(
                    e,
                    "└ For more information see "
                    "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#model-access-issue",
                )
            raise
        finally:
            callback()
            logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Constrain the model to a Pydantic ``BaseModel`` via a forced tool call."""
        tool_spec = convert_pydantic_to_tool_spec(output_model)
        response = self.stream(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            tool_choice=cast(ToolChoice, {"any": {}}),
            **kwargs,
        )

        last: dict[str, Any] | None = None
        async for event in streaming.process_stream(response):
            last = event
            yield event

        if last is None or "stop" not in last:
            raise ValueError("Stream ended without a stop event.")
        stop_reason, message, _, _ = last["stop"]
        if stop_reason != "tool_use":
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "tool_use".')
        for block in message["content"]:
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                yield {"output": output_model(**block["toolUse"]["input"])}
                return
        raise ValueError(f"No tool use found for {tool_spec['name']}")
