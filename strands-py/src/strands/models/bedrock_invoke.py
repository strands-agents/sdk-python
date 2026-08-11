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
import time
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..event_loop import streaming
from ..tools import convert_pydantic_to_tool_spec
from ..types.content import Messages, SystemContentBlock
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._defaults import resolve_config_metadata
from ._validation import validate_config_keys
from .bedrock import BedrockModel, _suppress_task_exception
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


def _unsupported_block(block: Mapping[str, Any]) -> TypeError:
    """Build the error for a content block this provider cannot put on the wire."""
    return TypeError(f"content_type=<{next(iter(block), None)}> | unsupported type")


def _latency_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _metadata(in_tok: int, out_tok: int, latency_ms: int, total: int | None = None) -> StreamEvent:
    return {
        "metadata": {
            "usage": {"inputTokens": in_tok, "outputTokens": out_tok, "totalTokens": total or in_tok + out_tok},
            "metrics": {"latencyMs": latency_ms},
        }
    }


class BedrockInvokeModel(BedrockModel):
    """AWS Bedrock model provider using ``InvokeModel`` / ``InvokeModelWithResponseStream``.

    Subclasses :class:`~strands.models.bedrock.BedrockModel`, matching the shape of other
    provider-family subclasses (e.g. ``SageMakerAIModel(OpenAIModel)``), while replacing request
    formatting, streaming, and response translation to talk to the native InvokeModel APIs.
    """

    class BedrockInvokeConfig(BaseModelConfig, total=False):
        """Configuration options for ``BedrockInvokeModel``. ``model_family`` overrides id-based detection.

        ``params`` is splatted onto the formatted request last, so it both reaches wire fields this config
        does not model (``thinking``, ``anthropic_beta``, ...) and overrides any field computed above it.
        """

        model_id: str
        model_family: ModelFamily | None
        max_tokens: int | None
        streaming: bool | None
        temperature: float | None
        top_p: float | None
        top_k: int | None
        stop_sequences: list[str] | None
        params: dict[str, Any] | None

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
        self.client, resolved_region = self._create_bedrock_runtime_client(
            boto_session=boto_session,
            boto_client_config=boto_client_config,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

        validate_config_keys(model_config, self.BedrockInvokeConfig)

        config: BedrockInvokeModel.BedrockInvokeConfig = {
            "model_id": self._get_default_model_with_warning(resolved_region, model_config),
            "streaming": model_config.get("streaming", True),
        }
        config.update({k: v for k, v in model_config.items() if k != "model_id"})  # type: ignore[typeddict-item]
        self.config: BedrockInvokeModel.BedrockInvokeConfig = config  # type: ignore[assignment]

        logger.debug("config=<%s> | initializing", self.config)
        logger.debug("region=<%s> | bedrock client created", self.client.meta.region_name)

    @override
    def update_config(self, **model_config: Unpack[BedrockInvokeConfig]) -> None:  # type: ignore[override]
        """Update the model configuration."""
        validate_config_keys(model_config, self.BedrockInvokeConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> BedrockInvokeConfig:  # type: ignore[override]
        """Return the current configuration, with model metadata resolved from the built-in lookup tables."""
        return resolve_config_metadata(self.config, self.config.get("model_id", ""))

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

    def _apply_sampling_params(self, request: dict[str, Any], stop_key: str, include_top_k: bool = False) -> None:
        """Copy configured temperature/top_p/(top_k)/stop_sequences onto ``request`` when set."""
        config = cast(dict[str, Any], self.config)
        keys = ("temperature", "top_p", "top_k") if include_top_k else ("temperature", "top_p")
        for key in keys:
            if config.get(key) is not None:
                request[key] = config[key]
        if config.get("stop_sequences"):
            request[stop_key] = config["stop_sequences"]

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
        """Build an Anthropic Messages request body.

        Raises:
            TypeError: If a message contains a content block type this provider cannot format.
        """
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
                else:
                    raise _unsupported_block(block)
            if content:
                request["messages"].append({"role": msg["role"], "content": content})

        if tool_specs:
            request["tools"] = [
                {"name": s["name"], "description": s["description"], "input_schema": s["inputSchema"]}
                for s in tool_specs
            ]
        if (tc := self._to_tool_choice(tool_choice, "anthropic")) is not None:
            request["tool_choice"] = tc

        self._apply_sampling_params(request, "stop_sequences", include_top_k=True)
        request.update(self.config.get("params") or {})
        return request

    def _format_openai_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any]:
        """Build an OpenAI Chat Completions request body.

        Raises:
            TypeError: If a message contains a content block type this provider cannot format. Images are
                only formattable on the Anthropic path; use ``model_family="anthropic"`` for multimodal input.
        """
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
                else:
                    raise _unsupported_block(block)
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

        self._apply_sampling_params(request, "stop")
        request.update(self.config.get("params") or {})
        return request

    def _format_invoke_request(
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

    def _emit_anthropic_chunks(self, body: Any, callback: Callable[..., None], start_time: float) -> None:
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
        callback(_metadata(in_toks, out_toks, _latency_ms(start_time)))

    def _emit_openai_chunks(self, body: Any, callback: Callable[..., None], start_time: float) -> None:
        """Translate an OpenAI Chat Completions stream into Strands ``StreamEvent``s.

        Tool calls are keyed by ``index`` and emitted lazily once an id or function name appears. At most
        one content block is open at a time, since the consumer keys its in-progress tool use off the most
        recent ``contentBlockStart``; a block must therefore be closed before the next one opens.
        """
        callback({"messageStart": {"role": "assistant"}})
        active: str | None = None
        active_index: int | None = None
        started: set[int] = set()
        stop_reason: str | None = None
        usage: dict[str, Any] | None = None

        for event in body:
            chunk = json.loads(event["chunk"]["bytes"])
            if choices := chunk.get("choices"):
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if active != "text":
                        if active is not None:
                            callback(_BLOCK_STOP)
                        callback(_TEXT_START)
                        active, active_index = "text", None
                    callback(_text_delta(delta["content"]))
                for tool_call in delta.get("tool_calls") or []:
                    index = tool_call.get("index", 0)
                    fn = tool_call.get("function") or {}
                    if index not in started and (tool_call.get("id") or fn.get("name")):
                        if active is not None:
                            callback(_BLOCK_STOP)
                        callback(_tool_use_start(tool_call.get("id") or f"call_{index}", fn.get("name", "")))
                        started.add(index)
                        active, active_index = "tool_use", index
                    if args := fn.get("arguments"):
                        if active == "tool_use" and index == active_index:
                            callback(_tool_use_delta(args))
                        else:
                            logger.warning("tool_call_index=<%s> | dropping arguments for a closed tool call", index)
                if finish := choices[0].get("finish_reason"):
                    stop_reason = finish
            if chunk.get("usage"):
                usage = chunk["usage"]

        if active is not None:
            callback(_BLOCK_STOP)
        callback({"messageStop": {"stopReason": self._map_openai_stop(stop_reason)}})
        if usage:
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            callback(_metadata(inp, out, _latency_ms(start_time), usage.get("total_tokens", inp + out)))

    def _emit_anthropic_non_streaming(
        self, body: dict[str, Any], callback: Callable[..., None], start_time: float
    ) -> None:
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
            callback(_metadata(u.get("input_tokens", 0), u.get("output_tokens", 0), _latency_ms(start_time)))

    def _emit_openai_non_streaming(
        self, body: dict[str, Any], callback: Callable[..., None], start_time: float
    ) -> None:
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
            callback(_metadata(inp, out, _latency_ms(start_time), u.get("total_tokens", inp + out)))

    # ----- public API

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Reject Converse-shaped request formatting, which this provider never sends.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt_content: Structured system prompt content blocks.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "BedrockInvokeModel sends Anthropic Messages / OpenAI Chat Completions requests, not Bedrock Converse "
            "requests; format_request() is not supported — use BedrockModel for Converse-shaped access."
        )

    @override
    def convert_non_streaming_to_streaming(self, response: dict[str, Any], **kwargs: Any) -> Iterable[StreamEvent]:
        """Reject Converse-shaped response translation, which this provider never receives.

        Args:
            response: The non-streaming response from the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "BedrockInvokeModel receives Anthropic Messages / OpenAI Chat Completions responses, not Bedrock Converse "
            "responses; convert_non_streaming_to_streaming() is not supported — use BedrockModel for Converse-shaped "
            "access."
        )

    @override
    async def count_tokens(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> int:
        """Estimate token count with the local heuristic.

        Bedrock's native CountTokens API only accepts Converse-shaped input, so it cannot count the
        InvokeModel request bodies this provider sends. This bypasses ``BedrockModel.count_tokens``
        and always uses the base heuristic.

        Args:
            messages: List of message objects to count tokens for.
            tool_specs: List of tool specifications to include in the count.
            system_prompt: Plain string system prompt. Ignored if system_prompt_content is provided.
            system_prompt_content: Structured system prompt content blocks.

        Returns:
            Estimated input token count.
        """
        return await Model.count_tokens(self, messages, tool_specs, system_prompt, system_prompt_content)

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
        """Stream a turn through Bedrock InvokeModel."""

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
            await task
        except BaseException:
            # Don't block cancellation on the in-flight blocking boto3 call; consume its exception later instead.
            task.add_done_callback(_suppress_task_exception)
            raise

    def _stream(  # type: ignore[override]
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
            request = self._format_invoke_request(messages, tool_specs, system_prompt_content, tool_choice)
            logger.debug("family=<%s> request=<%s>", family, request)

            common_kwargs = {
                "modelId": self.config["model_id"],
                "body": json.dumps(request),
                "contentType": "application/json",
                "accept": "application/json",
            }

            start_time = time.perf_counter()
            if self.config.get("streaming", True):
                response = self.client.invoke_model_with_response_stream(**common_kwargs)
                emit = self._emit_anthropic_chunks if family == "anthropic" else self._emit_openai_chunks
                emit(response["body"], callback, start_time)
            else:
                response = self.client.invoke_model(**common_kwargs)
                body = json.loads(response["body"].read())
                logger.debug("response_body=<%s>", body)
                emit = self._emit_anthropic_non_streaming if family == "anthropic" else self._emit_openai_non_streaming
                emit(body, callback, start_time)

        except ClientError as error:
            self._raise_translated_client_error(error)
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
