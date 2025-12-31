"""vLLM model provider (OpenAI-compatible).

This provider is implemented as a first-class Strands `Model` (not a subclass of `OpenAIModel`).

It targets vLLM's OpenAI-compatible server and supports:
- **token-out**: `prompt_token_ids`, `token_ids`, logprobs (when the server includes them)
- **token-in**: request-scoped `prompt_token_ids` via `extra_body`
- **tools**: via `/v1/chat/completions` (tool calling)

vLLM exposes provider-specific fields that are not part of the official OpenAI API schema. We send
those fields via `extra_body` to avoid OpenAI SDK validation errors, and we preserve them back onto
`messageStop.additionalModelResponseFields` for downstream consumers.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, AsyncIterable, Optional, Type, TypedDict, TypeVar, Union, cast

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.event_loop import StopReason
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import MessageStopEvent, StreamEvent
from ..types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from ._validation import validate_config_keys
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class VLLMModel(Model):
    """OpenAI-compatible vLLM provider with token-in/out helpers."""

    class VLLMConfig(TypedDict, total=False):
        """Configuration options for vLLM OpenAI-compatible models.

        Attributes:
            model_id: Model ID to pass to the server (e.g., "meta-llama/Llama-3.1-8B-Instruct").
            params: Base request params merged into every request (e.g., max_tokens, temperature).
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        client_args: Optional[dict[str, Any]] = None,
        *,
        return_token_ids: bool = False,
        **model_config: Unpack[VLLMConfig],
    ) -> None:
        """Create a vLLM OpenAI-compatible model client."""
        validate_config_keys(model_config, self.VLLMConfig)
        self.config = dict(model_config)
        self.client_args = client_args or {}
        self._return_token_ids_default = bool(return_token_ids)

    @override
    def update_config(self, **model_config: Unpack[VLLMConfig]) -> None:  # type: ignore[override]
        validate_config_keys(model_config, self.VLLMConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> VLLMConfig:
        return cast(VLLMModel.VLLMConfig, self.config)

    @staticmethod
    def _safe_model_dump(obj: Any) -> dict[str, Any]:
        model_dump = getattr(obj, "model_dump", None)
        if not callable(model_dump):
            return {}
        try:
            dumped = model_dump()
        except Exception:
            return {}
        return dumped if isinstance(dumped, dict) else {}

    @staticmethod
    def _choice0_dump(dumped: dict[str, Any]) -> dict[str, Any] | None:
        choices = dumped.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        c0 = choices[0]
        return c0 if isinstance(c0, dict) else None

    @staticmethod
    def _extend_token_ids_from_choice_dump(token_ids: list[int], choice_dump: dict[str, Any] | None) -> None:
        if not choice_dump:
            return
        maybe = choice_dump.get("token_ids")
        if isinstance(maybe, list) and maybe and all(isinstance(x, int) for x in maybe):
            token_ids.extend(cast(list[int], maybe))

    @staticmethod
    def _extend_logprobs_from_choice_dump(completion_logprobs: list[Any], choice_dump: dict[str, Any] | None) -> None:
        if not choice_dump:
            return
        lp = choice_dump.get("logprobs")
        if lp is None:
            return
        if isinstance(lp, dict):
            content = lp.get("content")
            if isinstance(content, list):
                completion_logprobs.extend(content)
                return
            completion_logprobs.append(lp)
            return
        completion_logprobs.append(lp)

    @staticmethod
    def _stream_switch_content(next_type: str, prev_type: str | None) -> tuple[list[StreamEvent], str]:
        chunks: list[StreamEvent] = []
        if prev_type != next_type:
            if prev_type is not None:
                chunks.append({"contentBlockStop": {}})
            chunks.append({"contentBlockStart": {"start": {}}})
        return chunks, next_type

    @classmethod
    def _format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        if "document" in content:
            mime_type = mimetypes.types_map.get(f".{content['document']['format']}", "application/octet-stream")
            file_data = base64.b64encode(content["document"]["source"]["bytes"]).decode("utf-8")
            return {
                "file": {
                    "file_data": f"data:{mime_type};base64,{file_data}",
                    "filename": content["document"]["name"],
                },
                "type": "file",
            }

        if "image" in content:
            mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
            image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8")
            return {
                "image_url": {
                    "detail": "auto",
                    "format": mime_type,
                    "url": f"data:{mime_type};base64,{image_data}",
                },
                "type": "image_url",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def _format_request_message_tool_call(cls, tool_use: ToolUse) -> dict[str, Any]:
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def _format_request_tool_message(cls, tool_result: ToolResult) -> dict[str, Any]:
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )
        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [cls._format_request_message_content(content) for content in contents],
        }

    @classmethod
    def _format_request_tool_choice(cls, tool_choice: ToolChoice | None) -> dict[str, Any]:
        if not tool_choice:
            return {}

        match tool_choice:
            case {"auto": _}:
                return {"tool_choice": "auto"}
            case {"any": _}:
                return {"tool_choice": "required"}
            case {"tool": {"name": tool_name}}:
                return {"tool_choice": {"type": "function", "function": {"name": tool_name}}}
            case _:
                return {"tool_choice": "auto"}

    @classmethod
    def _format_system_messages(
        cls,
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        return [
            {"role": "system", "content": content["text"]}
            for content in system_prompt_content or []
            if "text" in content
        ]

    @classmethod
    def _format_regular_messages(cls, messages: Messages, **_kwargs: Any) -> list[dict[str, Any]]:
        formatted_messages: list[dict[str, Any]] = []

        for message in messages:
            contents = message["content"]

            if any("reasoningContent" in content for content in contents):
                logger.warning(
                    "reasoningContent is not supported in multi-turn conversations with the Chat Completions API."
                )

            formatted_contents = [
                cls._format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse", "reasoningContent"])
            ]
            formatted_tool_calls = [
                cls._format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls._format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return formatted_messages

    @classmethod
    def _format_request_messages(
        cls,
        messages: Messages,
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        formatted_messages = cls._format_system_messages(
            system_prompt,
            system_prompt_content=system_prompt_content,
            **kwargs,
        )
        formatted_messages.extend(cls._format_regular_messages(messages, **kwargs))
        return [message for message in formatted_messages if message.get("content") or "tool_calls" in message]

    def _format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "messages": self._format_request_messages(
                messages,
                system_prompt,
                system_prompt_content=system_prompt_content,
            ),
            "model": self.config["model_id"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **(self._format_request_tool_choice(tool_choice)),
            **cast(dict[str, Any], self.config.get("params", {})),
        }

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **_kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        async with openai.AsyncOpenAI(**self.client_args) as client:
            try:
                response: ParsedChatCompletion = await client.beta.chat.completions.parse(
                    model=self.get_config()["model_id"],
                    messages=self._format_request(prompt, system_prompt=system_prompt)["messages"],
                    response_format=output_model,
                )
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except openai.RateLimitError as e:
                raise ModelThrottledException(str(e)) from e

        parsed: T | None = None
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the OpenAI response.")

        for choice in response.choices:
            if isinstance(choice.message.parsed, output_model):
                parsed = choice.message.parsed
                break

        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid tool use or tool use input was found in the OpenAI response.")

    async def _stream_completions_token_in(
        self,
        *,
        prompt_token_ids: list[int],
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Token-in streaming (no messages) via vLLM `/v1/completions`.

        This bypasses chat message formatting and sends the already-tokenized prompt to vLLM.
        """
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or not all(isinstance(x, int) for x in prompt_token_ids)
        ):
            raise TypeError("prompt_token_ids must be a non-empty list[int].")

        req_kwargs = dict(kwargs)
        req_kwargs["prompt_token_ids"] = prompt_token_ids
        req_kwargs = self._merge_vllm_extra_body(kwargs=req_kwargs)

        extra_body = cast(dict[str, Any], req_kwargs.get("extra_body") or {})
        if self._return_token_ids_default and "return_token_ids" not in extra_body:
            extra_body["return_token_ids"] = True
        req_kwargs["extra_body"] = extra_body

        if max_tokens is not None:
            req_kwargs["max_tokens"] = max_tokens

        # vLLM completions validates that `prompt` is non-empty (even if prompt_token_ids is provided),
        # so provide a harmless placeholder and rely on prompt_token_ids for the actual tokens.
        request: dict[str, Any] = {
            "model": self.get_config()["model_id"],
            "prompt": " ",
            "stream": True,
            **(self.get_config().get("params") or {}),
            **req_kwargs,
        }

        async with openai.AsyncOpenAI(**self.client_args) as client:
            response = await client.completions.create(**request)

            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}

            token_ids: list[int] = []
            finish_reason: str | None = None

            async for event in response:
                if not getattr(event, "choices", None):
                    continue

                dumped = self._safe_model_dump(event)
                self._extend_token_ids_from_choice_dump(token_ids, self._choice0_dump(dumped))

                choice0 = event.choices[0]
                if getattr(choice0, "text", None):
                    yield {"contentBlockDelta": {"delta": {"text": choice0.text}}}
                if getattr(choice0, "finish_reason", None):
                    finish_reason = choice0.finish_reason
                    break

            yield {"contentBlockStop": {}}

            additional: dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
            if token_ids:
                additional["token_ids"] = token_ids

            stop_reason: StopReason = "end_turn" if finish_reason in (None, "stop") else "max_tokens"
            yield {
                "messageStop": {
                    "stopReason": stop_reason,
                    "additionalModelResponseFields": additional,
                }
            }

    def _merge_vllm_extra_body(self, *, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge vLLM-specific request fields into `extra_body` and remove them from kwargs.

        We keep vLLM-only fields inside `extra_body` to avoid OpenAI SDK validation errors.
        """
        extra_body = kwargs.get("extra_body")
        if extra_body is None:
            extra_body_dict: dict[str, Any] = {}
        else:
            if not isinstance(extra_body, dict):
                raise TypeError("extra_body must be a dict when provided.")
            extra_body_dict = dict(extra_body)

        # Allow per-request override via kwarg while keeping OpenAIModel compatible:
        # - `return_token_ids` is a vLLM extension.
        if "return_token_ids" in kwargs:
            extra_body_dict["return_token_ids"] = bool(kwargs.pop("return_token_ids"))
        elif self._return_token_ids_default and "return_token_ids" not in extra_body_dict:
            extra_body_dict["return_token_ids"] = True

        # Token-in: pass the fully formatted prompt token IDs.
        # This is a vLLM extension; keep it in extra_body.
        if "prompt_token_ids" in kwargs:
            prompt_token_ids = kwargs.pop("prompt_token_ids")
            if prompt_token_ids is not None:
                if not isinstance(prompt_token_ids, list) or not all(isinstance(x, int) for x in prompt_token_ids):
                    raise TypeError("prompt_token_ids must be a list[int] when provided.")
                extra_body_dict["prompt_token_ids"] = prompt_token_ids

        # vLLM logprobs: allow passing an int (e.g. 1) without OpenAI SDK type constraints.
        if "logprobs" in kwargs:
            extra_body_dict["logprobs"] = kwargs.pop("logprobs")

        kwargs["extra_body"] = extra_body_dict
        return kwargs

    async def _stream_chat_vllm(
        self,
        *,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Chat-completions streaming with vLLM-specific token/logprobs extraction."""
        req_kwargs = self._merge_vllm_extra_body(kwargs=dict(kwargs))
        request_prompt_token_ids: list[int] | None = None
        extra_body = req_kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            pti = extra_body.get("prompt_token_ids")
            if isinstance(pti, list) and all(isinstance(x, int) for x in pti):
                request_prompt_token_ids = cast(list[int], pti)

        request = self._format_request(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
        )
        if req_kwargs:
            request.update(req_kwargs)

        async with openai.AsyncOpenAI(**self.client_args) as client:
            response = await client.chat.completions.create(**request)

            yield {"messageStart": {"role": "assistant"}}

            tool_calls: dict[int, list[Any]] = {}
            data_type: str | None = None
            finish_reason: str | None = None
            event = None

            prompt_token_ids: list[int] | None = None
            prompt_logprobs: Any = None
            token_ids: list[int] = []
            completion_logprobs: list[Any] = []

            async for event in response:
                if not getattr(event, "choices", None):
                    continue

                dumped = self._safe_model_dump(event)

                if prompt_token_ids is None and dumped.get("prompt_token_ids") is not None:
                    prompt_token_ids = cast(list[int], dumped.get("prompt_token_ids"))
                if prompt_logprobs is None and dumped.get("prompt_logprobs") is not None:
                    prompt_logprobs = dumped.get("prompt_logprobs")

                choice = event.choices[0]
                choice0_dump = self._choice0_dump(dumped)
                self._extend_token_ids_from_choice_dump(token_ids, choice0_dump)
                self._extend_logprobs_from_choice_dump(completion_logprobs, choice0_dump)

                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield {
                        "contentBlockDelta": {"delta": {"reasoningContent": {"text": choice.delta.reasoning_content}}}
                    }

                if choice.delta.content:
                    chunks, data_type = self._stream_switch_content("text", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield {"contentBlockDelta": {"delta": {"text": choice.delta.content}}}

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    break

            if data_type is not None:
                yield {"contentBlockStop": {}}

            for tool_deltas in tool_calls.values():
                first = tool_deltas[0]
                yield {
                    "contentBlockStart": {"start": {"toolUse": {"toolUseId": first.id, "name": first.function.name}}}
                }
                for td in tool_deltas:
                    yield {"contentBlockDelta": {"delta": {"toolUse": {"input": td.function.arguments or ""}}}}
                yield {"contentBlockStop": {}}

            if tool_calls and finish_reason in (None, "stop"):
                finish_reason = "tool_calls"

            additional: dict[str, Any] = {}
            if request_prompt_token_ids is not None:
                additional["prompt_token_ids"] = request_prompt_token_ids
            elif prompt_token_ids is not None:
                additional["prompt_token_ids"] = prompt_token_ids
            if prompt_logprobs is not None:
                additional["prompt_logprobs"] = prompt_logprobs
            if token_ids:
                additional["token_ids"] = token_ids
            if completion_logprobs:
                additional["logprobs"] = completion_logprobs

            stop_reason = (
                "tool_use"
                if finish_reason == "tool_calls"
                else ("max_tokens" if finish_reason == "length" else "end_turn")
            )
            message_stop: MessageStopEvent = {"stopReason": cast(StopReason, stop_reason)}
            if additional:
                message_stop["additionalModelResponseFields"] = additional
            yield {"messageStop": message_stop}

            async for event in response:
                _ = event
            if event and hasattr(event, "usage") and event.usage:
                yield {
                    "metadata": {
                        "usage": {
                            "inputTokens": event.usage.prompt_tokens,
                            "outputTokens": event.usage.completion_tokens,
                            "totalTokens": event.usage.total_tokens,
                        },
                        "metrics": {"latencyMs": 0},
                    }
                }

    @override
    def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        prompt_token_ids = kwargs.pop("prompt_token_ids", None)
        if prompt_token_ids is not None:
            token_in_endpoint = kwargs.pop("token_in_endpoint", "auto")
            if token_in_endpoint not in ("auto", "chat", "completions"):
                raise ValueError("token_in_endpoint must be one of: 'auto', 'chat', 'completions'.")

            if token_in_endpoint == "auto":
                token_in_endpoint = "chat" if (tool_specs or tool_choice) else "completions"

            if token_in_endpoint == "completions":
                if tool_specs is not None or tool_choice is not None:
                    raise TypeError("tool_specs/tool_choice are not supported in token-only mode.")
                if system_prompt is not None or system_prompt_content is not None:
                    raise TypeError("system_prompt/system_prompt_content are not supported in token-only mode.")
                max_tokens = kwargs.pop("max_tokens", None)
                return self._stream_completions_token_in(
                    prompt_token_ids=cast(list[int], prompt_token_ids),
                    max_tokens=max_tokens,
                    **kwargs,
                )

            return self._stream_chat_vllm(
                messages=[{"role": "user", "content": [{"text": ""}]}],
                tool_specs=tool_specs,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                system_prompt_content=system_prompt_content,
                prompt_token_ids=cast(list[int], prompt_token_ids),
                **kwargs,
            )

        return self._stream_chat_vllm(
            messages=messages,
            tool_specs=tool_specs,
            system_prompt=system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        )
