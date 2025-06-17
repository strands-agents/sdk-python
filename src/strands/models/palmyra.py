"""Palmyra model provider.

- Docs: https://dev.writer.com/home/introduction
"""

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union, cast

import writerai
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ModelThrottledException
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse

logger = logging.getLogger(__name__)


class PalmyraModel(Model):
    """Palmyra API model provider implementation."""

    class PalmyraConfig(TypedDict, total=False):
        """Configuration options for Palmyra API.

        Attributes:
            model: Model name to use (e.g. palmyra-x5, palmyra-x4, etc.).
            logprobs: Return logprobs or not.
            max_tokens: Maximum number of tokens to generate.
            n: Number of chat completions to generate for each prompt.
            response_format: The response format to use for the chat completion.
            stop: Default stop sequences.
            stream_options: Additional options for streaming.
            tool_choice: Functions calling mode.
            temperature: What sampling temperature to use.
            top_p: Threshold for 'nucleus sampling'
        """

        model: str
        logprobs: bool
        max_tokens: Optional[int]
        n: Optional[int]
        response_format: Dict[str, Any]
        stop: Optional[Union[str, List[str]]]
        stream_options: Dict[str, Any]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[PalmyraConfig]):
        """Initialize provider instance.

        Args:
            client_args: Arguments for the Palmyra client (e.g., api_key, base_url, timeout, etc.).
            **model_config: Configuration options for the Palmyra model.
        """
        self.config = PalmyraModel.PalmyraConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = writerai.Client(**client_args)

    @override
    def update_config(self, **model_config: Unpack[PalmyraConfig]) -> None:  # type: ignore[override]
        """Update the Palmyra Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> PalmyraConfig:
        """Get the Palmyra model configuration.

        Returns:
            The Palmyra model configuration.
        """
        return self.config

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a Palmyra content block.

        - NOTE: "reasoningContent", "video" and "image" are not supported currently.

        Args:
            content: Message content.

        Returns:
            Palmyra formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a Palmyra-compatible format.
        """
        if "text" in content:
            return {"text": content["text"], "type": "text"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_request_message_tool_call(self, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Palmyra tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Palmyra formatted tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    def _format_request_tool_message(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Palmyra tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            Palmyra formatted tool message.
        """
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
            "content": [self._format_request_message_content(content) for content in contents],
        }

    def _format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format a Palmyra compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            Palmyra compatible messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                self._format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse"])
            ]
            formatted_tool_calls = [
                self._format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                self._format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents if len(formatted_contents) > 0 else "",
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> Any:
        """Format a streaming request to the underlying model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            The formatted request.
        """
        return {
            **{k: v for k, v in self.config.items()},
            "messages": self._format_request_messages(messages, system_prompt),
            "stream": True,
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
        }

    @override
    def format_chunk(self, event: Any) -> StreamEvent:
        """Format the model response events into standardized message chunks.

        Args:
            event: A response event from the model.

        Returns:
            The formatted chunk.
        """
        match event.get("chunk_type", ""):
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_block_start":
                if event["data_type"] == "text":
                    return {"contentBlockStart": {"start": {}}}

                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": event["data"].function.name,
                                "toolUseId": event["data"].id,
                            }
                        }
                    }
                }

            case "content_block_delta":
                if event["data_type"] == "text":
                    return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments}}}}

            case "content_block_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"].prompt_tokens,
                            "outputTokens": event["data"].completion_tokens,
                            "totalTokens": event["data"].total_tokens,
                        },
                        "metrics": {
                            "latencyMs": 0,  # No data
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @override
    def stream(self, request: Any) -> Iterable[Any]:
        """Send the request to the model and get a streaming response.

        Args:
            request: The formatted request to send to the model.

        Returns:
            The model's response.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        try:
            response = self.client.chat.chat(**request)
        except writerai.RateLimitError as e:
            raise ModelThrottledException(str(e)) from e

        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_block_start", "data_type": "text"}

        tool_calls: dict[int, list[Any]] = {}

        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]

            if choice.delta.content:
                yield {"chunk_type": "content_block_delta", "data_type": "text", "data": choice.delta.content}

            for tool_call in choice.delta.tool_calls or []:
                tool_calls.setdefault(tool_call.index, []).append(tool_call)

            if choice.finish_reason:
                break

        yield {"chunk_type": "content_block_stop", "data_type": "text"}

        for tool_deltas in tool_calls.values():
            tool_start, tool_deltas = tool_deltas[0], tool_deltas[1:]
            yield {"chunk_type": "content_block_start", "data_type": "tool", "data": tool_start}

            for tool_delta in tool_deltas:
                yield {"chunk_type": "content_block_delta", "data_type": "tool", "data": tool_delta}

            yield {"chunk_type": "content_block_stop", "data_type": "tool"}

        yield {"chunk_type": "message_stop", "data": choice.finish_reason}

        for chunk in response:
            _ = chunk

        yield {"chunk_type": "metadata", "data": chunk.usage}
