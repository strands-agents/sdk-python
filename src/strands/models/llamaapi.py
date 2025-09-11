# Copyright (c) Meta Platforms, Inc. and affiliates
"""Llama API model provider.

- Docs: https://llama.developer.meta.com/
"""

import base64
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, Optional, Type, TypeVar, Union, cast

import llama_api_client
from llama_api_client import LlamaAPIClient
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StreamEvent, Usage
from ..types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from ._validation import validate_config_keys, warn_on_tool_choice_not_supported
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LlamaAPIModel(Model):
    """Llama API model provider implementation."""

    class LlamaConfig(TypedDict, total=False):
        """Configuration options for Llama API models.

        Attributes:
            model_id: Model ID (e.g., "Llama-4-Maverick-17B-128E-Instruct-FP8").
            repetition_penalty: Repetition penalty.
            temperature: Temperature.
            top_p: Top-p.
            max_completion_tokens: Maximum completion tokens.
            top_k: Top-k.
        """

        model_id: str
        repetition_penalty: Optional[float]
        temperature: Optional[float]
        top_p: Optional[float]
        max_completion_tokens: Optional[int]
        top_k: Optional[int]

    def __init__(
        self,
        *,
        client_args: Optional[dict[str, Any]] = None,
        **model_config: Unpack[LlamaConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the Llama API client.
            **model_config: Configuration options for the Llama API model.
        """
        validate_config_keys(model_config, self.LlamaConfig)
        self.config = LlamaAPIModel.LlamaConfig(**model_config)
        logger.debug("config=<%s> | initializing", self.config)

        if not client_args:
            self.client = LlamaAPIClient()
        else:
            self.client = LlamaAPIClient(**client_args)

    @override
    def update_config(self, **model_config: Unpack[LlamaConfig]) -> None:  # type: ignore
        """Update the Llama API Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.LlamaConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> LlamaConfig:
        """Get the Llama API model configuration.

        Returns:
            The Llama API model configuration.
        """
        return self.config

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a LlamaAPI content block.

        - NOTE: "reasoningContent" and "video" are not supported currently.

        Args:
            content: Message content.

        Returns:
            LllamaAPI formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a LlamaAPI-compatible format.
        """
        if "image" in content:
            mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
            image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8")

            return {
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}",
                },
                "type": "image_url",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_request_message_tool_call(self, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Llama API tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Llama API formatted tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
        }

    def _format_request_tool_message(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Llama API tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            Llama API formatted tool message.
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
        """Format a LlamaAPI compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An LlamaAPI compatible messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            formatted_contents: list[dict[str, Any]] | dict[str, Any] | str = ""
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

            if message["role"] == "assistant":
                formatted_contents = formatted_contents[0] if formatted_contents else ""

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents if len(formatted_contents) > 0 else "",
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a Llama API chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An Llama API chat streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to a LlamaAPI-compatible
                format.
        """
        request = {
            "messages": self._format_request_messages(messages, system_prompt),
            "model": self.config["model_id"],
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
        if "temperature" in self.config:
            request["temperature"] = self.config["temperature"]
        if "top_p" in self.config:
            request["top_p"] = self.config["top_p"]
        if "repetition_penalty" in self.config:
            request["repetition_penalty"] = self.config["repetition_penalty"]
        if "max_completion_tokens" in self.config:
            request["max_completion_tokens"] = self.config["max_completion_tokens"]
        if "top_k" in self.config:
            request["top_k"] = self.config["top_k"]

        return request

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Llama API model response events into standardized message chunks.

        Args:
            event: A response event from the model.

        Returns:
            The formatted chunk.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
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

            case "content_delta":
                if event["data_type"] == "text":
                    return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments}}}}

            case "content_stop":
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
                usage = {}
                for metrics in event["data"]:
                    if metrics.metric == "num_prompt_tokens":
                        usage["inputTokens"] = metrics.value
                    elif metrics.metric == "num_completion_tokens":
                        usage["outputTokens"] = metrics.value
                    elif metrics.metric == "num_total_tokens":
                        usage["totalTokens"] = metrics.value

                usage_type = Usage(
                    inputTokens=usage["inputTokens"],
                    outputTokens=usage["outputTokens"],
                    totalTokens=usage["totalTokens"],
                )
                return {
                    "metadata": {
                        "usage": usage_type,
                        "metrics": {
                            "latencyMs": 0,  # TODO
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the LlamaAPI model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation. **Note: This parameter is accepted for
                interface consistency but is currently ignored for this model provider.**
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        warn_on_tool_choice_not_supported(tool_choice)

        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            response = self.client.chat.completions.create(**request)
        except llama_api_client.RateLimitError as e:
            raise ModelThrottledException(str(e)) from e

        logger.debug("got response from model")
        yield self.format_chunk({"chunk_type": "message_start"})

        stop_reason = None
        tool_calls: dict[Any, list[Any]] = {}
        curr_tool_call_id = None

        metrics_event = None
        for chunk in response:
            if chunk.event.event_type == "start":
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
            elif chunk.event.event_type in ["progress", "complete"] and chunk.event.delta.type == "text":
                yield self.format_chunk(
                    {"chunk_type": "content_delta", "data_type": "text", "data": chunk.event.delta.text}
                )
            else:
                if chunk.event.delta.type == "tool_call":
                    if chunk.event.delta.id:
                        curr_tool_call_id = chunk.event.delta.id

                    if curr_tool_call_id not in tool_calls:
                        tool_calls[curr_tool_call_id] = []
                    tool_calls[curr_tool_call_id].append(chunk.event.delta)
                elif chunk.event.event_type == "metrics":
                    metrics_event = chunk.event.metrics
                else:
                    yield self.format_chunk(chunk)

            if stop_reason is None:
                stop_reason = chunk.event.stop_reason

            # stopped generation
            if stop_reason:
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        for tool_deltas in tool_calls.values():
            tool_start, tool_deltas = tool_deltas[0], tool_deltas[1:]
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_start})

            for tool_delta in tool_deltas:
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        yield self.format_chunk({"chunk_type": "message_stop", "data": stop_reason})

        # we may have a metrics event here
        if metrics_event:
            yield self.format_chunk({"chunk_type": "metadata", "data": metrics_event})

        logger.debug("finished streaming response from model")

    @override
    def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            NotImplementedError: Structured output is not currently supported for LlamaAPI models.
        """
        # response_format: ResponseFormat = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": output_model.__name__,
        #         "schema": output_model.model_json_schema(),
        #     },
        # }
        # response = self.client.chat.completions.create(
        #     model=self.config["model_id"],
        #     messages=self.format_request(prompt)["messages"],
        #     response_format=response_format,
        # )
        raise NotImplementedError("Strands sdk-python does not implement this in the Llama API Preview.")
