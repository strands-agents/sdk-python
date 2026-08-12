"""Anthropic Claude model provider.

- Docs: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
"""

import base64
import json
import logging
import mimetypes
from collections.abc import AsyncGenerator
from typing import Any, TypeVar, cast

import anthropic
from pydantic import BaseModel
from typing_extensions import Required, Unpack, override

from ..event_loop.streaming import process_stream
from ..tools.structured_output.structured_output_utils import convert_pydantic_to_tool_spec
from ..types.content import ContentBlock, Message, Messages, SystemContentBlock
from ..types.event_loop import Usage
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolChoiceToolDict, ToolSpec
from ._defaults import resolve_config_metadata
from ._validation import _has_location_source, validate_config_keys
from .model import BaseModelConfig, CacheConfig, CacheToolsConfig, Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_IMAGE_MEDIA_TYPES = {
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

# Anthropic accepts ``cache_control`` on these block types only. A breakpoint on any other block  is rejected.
# https://docs.claude.com/en/docs/build-with-claude/prompt-caching
_CACHEABLE_BLOCK_TYPES = frozenset({"document", "image", "text", "tool_result", "tool_use"})

# ``ephemeral`` is the only cache type the Anthropic API supports
_ANTHROPIC_CACHE_TYPE = "ephemeral"


class AnthropicModel(Model):
    """Anthropic model provider implementation."""

    EVENT_TYPES = {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_stop",
    }

    OVERFLOW_MESSAGES = {
        "prompt is too long:",
        "input is too long",
        "input length exceeds context window",
        "input and output tokens exceed your context limit",
    }

    class AnthropicConfig(BaseModelConfig, total=False):
        """Configuration options for Anthropic models.

        Attributes:
            cache_config: Configuration for prompt caching. Use ``CacheConfig(strategy="auto")`` to add a
                cache breakpoint to the last user message. Caching is off when unset.
            cache_tools: Caches the tool definitions.
            max_tokens: Maximum number of tokens to generate.
            model_id: Calude model ID (e.g., "claude-3-7-sonnet-latest").
                For a complete list of supported models, see
                https://docs.anthropic.com/en/docs/about-claude/models/all-models.
            params: Additional model parameters (e.g., temperature).
                For a complete list of supported parameters, see https://docs.anthropic.com/en/api/messages.
            use_native_token_count: Whether to use the native Anthropic count_tokens API.
                When True, count_tokens() calls the Anthropic API for accurate counts.
                When False (default), skips the API call and uses the local estimator.
        """

        cache_config: CacheConfig | None
        cache_tools: str | CacheToolsConfig | None
        max_tokens: Required[int]
        model_id: Required[str]
        params: dict[str, Any] | None
        use_native_token_count: bool

    def __init__(self, *, client_args: dict[str, Any] | None = None, **model_config: Unpack[AnthropicConfig]):
        """Initialize provider instance.

        Args:
            client_args: Arguments for the underlying Anthropic client (e.g., api_key).
                For a complete list of supported arguments, see https://docs.anthropic.com/en/api/client-sdks.
            **model_config: Configuration options for the Anthropic model.
        """
        validate_config_keys(model_config, self.AnthropicConfig)
        self.config = AnthropicModel.AnthropicConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = anthropic.AsyncAnthropic(**client_args)

    @override
    def update_config(self, **model_config: Unpack[AnthropicConfig]) -> None:  # type: ignore[override]
        """Update the Anthropic model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.AnthropicConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> AnthropicConfig:
        """Get the Anthropic model configuration.

        Returns:
            The Anthropic model configuration.
        """
        return resolve_config_metadata(self.config, self.config["model_id"])

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format an Anthropic content block.

        Args:
            content: Message content.

        Returns:
            Anthropic formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to an Anthropic-compatible format.
        """
        if "document" in content:
            mime_type = mimetypes.types_map.get(f".{content['document']['format']}", "application/octet-stream")
            return {
                "source": {
                    "data": (
                        content["document"]["source"]["bytes"].decode("utf-8")
                        if mime_type == "text/plain"
                        else base64.b64encode(content["document"]["source"]["bytes"]).decode("utf-8")
                    ),
                    "media_type": mime_type,
                    "type": "text" if mime_type == "text/plain" else "base64",
                },
                "title": content["document"]["name"],
                "type": "document",
            }

        if "image" in content:
            image_format = content["image"]["format"]
            return {
                "source": {
                    "data": base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8"),
                    "media_type": _IMAGE_MEDIA_TYPES.get(
                        image_format,
                        mimetypes.types_map.get(f".{image_format}", "application/octet-stream"),
                    ),
                    "type": "base64",
                },
                "type": "image",
            }

        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"]["signature"],
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        if "toolUse" in content:
            return {
                "id": content["toolUse"]["toolUseId"],
                "input": content["toolUse"]["input"],
                "name": content["toolUse"]["name"],
                "type": "tool_use",
            }

        if "toolResult" in content:
            return {
                "content": [
                    self._format_request_message_content(
                        {"text": json.dumps(tool_result_content["json"], ensure_ascii=False)}
                        if "json" in tool_result_content
                        else cast(ContentBlock, tool_result_content)
                    )
                    for tool_result_content in content["toolResult"]["content"]
                ],
                "is_error": content["toolResult"]["status"] == "error",
                "tool_use_id": content["toolResult"]["toolUseId"],
                "type": "tool_result",
            }

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_request_messages(self, messages: Messages, cache_target_idx: int | None = None) -> list[dict[str, Any]]:
        """Format an Anthropic messages array.

        Args:
            messages: List of message objects to be processed by the model.
            cache_target_idx: Index of the message that owns the managed cache breakpoint while
                ``cache_config`` is set. Automatic placement applies to that message only when nothing in
                it already carries the breakpoint.

        Returns:
            An Anthropic messages array.
        """
        cache_config = self.config.get("cache_config")
        configured_ttl = cache_config.ttl if cache_config else None
        formatted_messages = []

        for message_idx, message in enumerate(messages):
            formatted_contents: list[dict[str, Any]] = []
            marked = False

            for content in message["content"]:
                if "cachePoint" in content:
                    ttl = content["cachePoint"].get("ttl")
                    if not ttl and message_idx == cache_target_idx:
                        ttl = configured_ttl

                    if self._attach_cache_control(formatted_contents, ttl):
                        marked = True
                    elif message_idx == cache_target_idx:
                        logger.warning(
                            "msg_idx=<%d> | nothing ahead of the placed cache point accepts a breakpoint, "
                            "falling back to automatic placement",
                            message_idx,
                        )
                    else:
                        logger.warning("no preceding block accepts a cache breakpoint | skipped cache point")
                    continue

                # Check for location sources in image, document, or video content
                if _has_location_source(content):
                    logger.warning("Location sources are not supported by Anthropic | skipping content block")
                    continue

                formatted_contents.append(self._format_request_message_content(content))

            # Automatic placement runs once the whole message is formatted, so the breakpoint lands on a
            # block that survived translation. It is skipped when a caller-placed point already marked one.
            if message_idx == cache_target_idx and not marked:
                if self._attach_cache_control(formatted_contents, configured_ttl):
                    logger.debug("msg_idx=<%d> | added cache point to last user message", message_idx)
                else:
                    logger.debug("msg_idx=<%d> | no cacheable content block, skipped cache point", message_idx)

            if formatted_contents:
                formatted_messages.append({"content": formatted_contents, "role": message["role"]})

        return formatted_messages

    @classmethod
    def _attach_cache_control(cls, formatted_contents: list[dict[str, Any]], ttl: str | None) -> bool:
        """Mark the most recent already-formatted block that the API accepts ``cache_control`` on.

        A cache point marks the preceding content as a cache breakpoint. The block immediately before it
        is not always a valid carrier: it may have been dropped in translation, or be a type the API
        rejects a breakpoint on (a ``thinking`` block, for example). Scanning backwards keeps the
        breakpoint on the nearest valid block instead of emitting a request the API refuses.

        Args:
            formatted_contents: Blocks formatted so far for the current message. Mutated in place.
            ttl: Optional TTL duration carried by the cache point.

        Returns:
            True when a block was marked, False when none of the blocks can carry a breakpoint. 
        """
        for block in reversed(formatted_contents):
            if block.get("type") in _CACHEABLE_BLOCK_TYPES:
                block["cache_control"] = cls._format_cache_control(ttl)
                return True

        return False

    def _caching_enabled(self) -> bool:
        """Whether ``cache_config`` asks for a cache breakpoint on the last user message.

        Both documented strategies enable caching on this provider. ``"auto"`` carries a model-support
        check on Bedrock, but the Anthropic API supports prompt caching on every active Claude model, so
        there is nothing for that check to decide here and the two strategies coincide.

        Returns:
            True if a cache point should be injected into the messages.
        """
        cache_config = self.config.get("cache_config")
        if not cache_config:
            return False

        if cache_config.strategy not in ("auto", "anthropic"):
            logger.warning("strategy=<%s> | unknown cache strategy, prompt caching disabled", cache_config.strategy)
            return False

        return True

    @staticmethod
    def _format_cache_control(ttl: str | None) -> dict[str, Any]:
        """Build an Anthropic ``cache_control`` value.

        Args:
            ttl: TTL duration (e.g. "5m", "1h"). A falsy value is omitted, leaving the API default.

        Returns:
            An Anthropic cache_control dict.
        """
        cache_control: dict[str, Any] = {"type": _ANTHROPIC_CACHE_TYPE}
        if ttl:
            cache_control["ttl"] = ttl
        return cache_control

    def _manage_cache_points(self, messages: Messages) -> tuple[Messages, int | None]:
        """Return a copy of messages carrying at most one message cache point, and the message that owns it.

        A cache point the caller placed in the last user message is honored where it sits. Callers place one
        to mark where their reusable prefix ends, ahead of content rebuilt on every call (retrieved context,
        a timestamp); moving that boundary would put the per-call content inside the cached prefix, so every
        request would write a new cache entry and none would ever read one.

        Extra points in that message, and points in earlier messages, are stripped so they cannot accumulate
        one per turn against the API's shared breakpoint budget.

        Args:
            messages: List of message objects to manage cache points for.

        Returns:
            A new list of messages and the index of the message that owns the managed breakpoint, or None
            when no user message can carry one. The input is never modified.
        """
        target_idx = next(
            (
                idx
                for idx in reversed(range(len(messages)))
                if messages[idx]["role"] == "user"
                and any("cachePoint" not in block for block in messages[idx]["content"])
            ),
            None,
        )
        if target_idx is None:
            logger.debug("no user message with content | skipped cache point")

        copied: list[Message] = []
        stripped = 0
        for msg_idx, message in enumerate(messages):
            content: list[ContentBlock] = []
            honored = False
            for block in message["content"]:
                if "cachePoint" not in block:
                    content.append(block)
                elif msg_idx == target_idx and not honored:
                    honored = True
                    content.append(block)
                else:
                    stripped += 1
            copied.append({"role": message["role"], "content": content})

        if stripped:
            # Warn rather than debug: discarding a cache point the caller placed can silently *cost* them
            # caching, and a request carries one message breakpoint either way. BedrockModel warns on the
            # same strip, so this keeps the two providers equally loud about it.
            logger.warning(
                "count=<%d> | stripped extra cache points, cache_config keeps the first cache point in the "
                "last user message; unset cache_config to keep every cache point",
                stripped,
            )

        return copied, target_idx

    def _format_request_tools(self, tool_specs: list[ToolSpec] | None) -> list[dict[str, Any]]:
        """Format tool definitions, caching them when ``cache_tools`` is configured.

        A ``cache_control`` on the final tool caches the whole tool block, so one breakpoint is enough.

        Args:
            tool_specs: List of tool specifications to make available to the model.

        Returns:
            An Anthropic tools array.
        """
        tools: list[dict[str, Any]] = [
            {
                "name": tool_spec["name"],
                "description": tool_spec["description"],
                "input_schema": tool_spec["inputSchema"]["json"],
            }
            for tool_spec in tool_specs or []
        ]

        cache_tools = self.config.get("cache_tools")
        if cache_tools and tools:
            ttl = cache_tools.ttl if isinstance(cache_tools, CacheToolsConfig) else None
            tools[-1]["cache_control"] = self._format_cache_control(ttl)

        return tools

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format an Anthropic streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Returns:
            An Anthropic streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to an Anthropic-compatible
                format.
        """
        cache_target_idx: int | None = None
        if self._caching_enabled():
            messages, cache_target_idx = self._manage_cache_points(messages)

        request = {
            "max_tokens": self.config["max_tokens"],
            "messages": self._format_request_messages(messages, cache_target_idx),
            "model": self.config["model_id"],
            "tools": self._format_request_tools(tool_specs),
            **(self._format_tool_choice(tool_choice)),
            **({"system": system_prompt} if system_prompt else {}),
            **(self.config.get("params") or {}),
        }

        return request

    @staticmethod
    def _format_tool_choice(tool_choice: ToolChoice | None) -> dict:
        if tool_choice is None:
            return {}

        if "any" in tool_choice:
            return {"tool_choice": {"type": "any"}}
        elif "auto" in tool_choice:
            return {"tool_choice": {"type": "auto"}}
        elif "tool" in tool_choice:
            return {"tool_choice": {"type": "tool", "name": cast(ToolChoiceToolDict, tool_choice)["tool"]["name"]}}
        else:
            return {}

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Anthropic response events into standardized message chunks.

        Args:
            event: A response event from the Anthropic model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as we control chunk_type in the stream method.
        """
        match event["type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_block_start":
                content = event["content_block"]

                if content["type"] == "tool_use":
                    return {
                        "contentBlockStart": {
                            "contentBlockIndex": event["index"],
                            "start": {
                                "toolUse": {
                                    "name": content["name"],
                                    "toolUseId": content["id"],
                                }
                            },
                        }
                    }

                return {"contentBlockStart": {"contentBlockIndex": event["index"], "start": {}}}

            case "content_block_delta":
                delta = event["delta"]

                match delta["type"]:
                    case "signature_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "reasoningContent": {
                                        "signature": delta["signature"],
                                    },
                                },
                            },
                        }

                    case "thinking_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "reasoningContent": {
                                        "text": delta["thinking"],
                                    },
                                },
                            },
                        }

                    case "input_json_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "toolUse": {
                                        "input": delta["partial_json"],
                                    },
                                },
                            },
                        }

                    case "text_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "text": delta["text"],
                                },
                            },
                        }

                    case _:
                        raise RuntimeError(
                            f"event_type=<content_block_delta>, delta_type=<{delta['type']}> | unknown type"
                        )

            case "content_block_stop":
                return {"contentBlockStop": {"contentBlockIndex": event["index"]}}

            case "message_stop":
                message = event["message"]

                return {"messageStop": {"stopReason": message["stop_reason"]}}

            case "metadata":
                usage = event["usage"]
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
                cache_read = usage.get("cache_read_input_tokens") or 0
                cache_write = usage.get("cache_creation_input_tokens") or 0
                usage_chunk: Usage = {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                }
                if cache_read:
                    usage_chunk["cacheReadInputTokens"] = cache_read
                if cache_write:
                    usage_chunk["cacheWriteInputTokens"] = cache_write

                return {
                    "metadata": {
                        "usage": usage_chunk,
                        "metrics": {
                            "latencyMs": 0,  # TODO
                        },
                    }
                }

            case _:
                raise RuntimeError(f"event_type=<{event['type']} | unknown type")

    @override
    async def count_tokens(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> int:
        """Count tokens using Anthropic's native count_tokens API.

        Uses the same message format as the Messages API to get accurate token counts
        directly from the Anthropic service.

        Args:
            messages: List of message objects to count tokens for.
            tool_specs: List of tool specifications to include in the count.
            system_prompt: Plain string system prompt. Ignored if system_prompt_content is provided.
            system_prompt_content: Structured system prompt content blocks.

        Returns:
            Total input token count.
        """
        if self.config.get("use_native_token_count") is not True:
            return await super().count_tokens(messages, tool_specs, system_prompt, system_prompt_content)

        try:
            # system_prompt_content is not used; this provider only accepts system_prompt as a plain string,
            # matching the behavior of stream(). The caller always provides system_prompt alongside
            # system_prompt_content, so the plain string is always available.
            request = self.format_request(messages, tool_specs, system_prompt)
            # Keep only fields accepted by count_tokens; strip inference params (max_tokens, temperature, etc.)
            count_tokens_fields = {"model", "messages", "tools", "tool_choice", "system"}
            request = {k: request[k] for k in request.keys() & count_tokens_fields}

            response = await self.client.messages.count_tokens(**request)
            total_tokens: int = response.input_tokens

            logger.debug(
                "model_id=<%s>, total_tokens=<%d> | native token count",
                self.config["model_id"],
                total_tokens,
            )
            return total_tokens
        except Exception as e:
            logger.debug(
                "model_id=<%s>, error=<%s> | native token counting failed, falling back to estimation",
                self.config["model_id"],
                e,
            )
            return await super().count_tokens(messages, tool_specs, system_prompt, system_prompt_content)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Anthropic model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by Anthropic.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            async with self.client.messages.stream(**request) as stream:
                logger.debug("got response from model")
                async for event in stream:
                    if event.type in AnthropicModel.EVENT_TYPES:
                        if event.type == "message_stop":
                            # Build dict directly to avoid Pydantic serialization warnings
                            # when the message contains ParsedTextBlock objects (issue #1746)
                            yield self.format_chunk(
                                {
                                    "type": "message_stop",
                                    "message": {"stop_reason": event.message.stop_reason},
                                }
                            )
                        elif event.type == "content_block_stop":
                            yield self.format_chunk({"type": "content_block_stop", "index": event.index})
                        else:
                            yield self.format_chunk(event.model_dump())

                try:
                    message_snapshot = await stream.get_final_message()
                except AssertionError as e:
                    logger.warning("error=<%s> | failed to retrieve message snapshot, usage metadata unavailable", e)
                else:
                    yield self.format_chunk({"type": "metadata", "usage": message_snapshot.usage.model_dump()})

        except anthropic.RateLimitError as error:
            raise ModelThrottledException(str(error)) from error

        except anthropic.BadRequestError as error:
            if any(overflow_message in str(error).lower() for overflow_message in AnthropicModel.OVERFLOW_MESSAGES):
                raise ContextWindowOverflowException(str(error)) from error

            raise error

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        tool_spec = convert_pydantic_to_tool_spec(output_model)

        response = self.stream(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            tool_choice=cast(ToolChoice, {"any": {}}),
            **kwargs,
        )
        async for event in process_stream(response):
            yield event

        stop_reason, messages, _, _ = event["stop"]

        if stop_reason != "tool_use":
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "tool_use".')

        content = messages["content"]
        output_response: dict[str, Any] | None = None
        for block in content:
            # if the tool use name doesn't match the tool spec name, skip, and if the block is not a tool use, skip.
            # if the tool use name never matches, raise an error.
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                output_response = block["toolUse"]["input"]
            else:
                continue

        if output_response is None:
            raise ValueError("No valid tool use or tool use input was found in the Anthropic response.")

        yield {"output": output_model(**output_response)}
