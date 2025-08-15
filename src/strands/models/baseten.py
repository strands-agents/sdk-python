"""Baseten model provider.

- Docs: https://docs.baseten.co/
"""

import base64
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, AsyncIterable, Optional, Protocol, Type, TypedDict, TypeVar, Union, cast

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Client(Protocol):
    """Protocol defining the OpenAI-compatible interface for the underlying provider client."""

    @property
    # pragma: no cover
    def chat(self) -> Any:
        """Chat completions interface."""
        ...


class BasetenModel(Model):
    """Baseten model provider implementation."""

    client: Client

    class BasetenConfig(TypedDict, total=False):
        """Configuration options for Baseten models.

        Attributes:
            model_id: Model ID for the Baseten model.
                For Model APIs, use model slugs like "deepseek-ai/DeepSeek-R1-0528" or "meta-llama/Llama-4-Maverick-17B-128E-Instruct".
                For dedicated deployments, use the deployment ID.
            base_url: Base URL for the Baseten API.
                For Model APIs: https://inference.baseten.co/v1
                For dedicated deployments: https://model-xxxxxxx.api.baseten.co/environments/production/sync/v1
            params: Model parameters (e.g., max_tokens).
                For a complete list of supported parameters, see
                https://platform.openai.com/docs/api-reference/chat/create.
        """

        model_id: str
        base_url: Optional[str]
        params: Optional[dict[str, Any]]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[BasetenConfig]) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the Baseten client.
                For a complete list of supported arguments, see https://pypi.org/project/openai/.
            **model_config: Configuration options for the Baseten model.
        """
        self.config = dict(model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        
        # Set default base URL for Model APIs if not provided
        if "base_url" not in client_args and "base_url" not in self.config:
            client_args["base_url"] = "https://inference.baseten.co/v1"
        elif "base_url" in self.config:
            client_args["base_url"] = self.config["base_url"]
        
        self.client = openai.AsyncOpenAI(**client_args)

    @override
    def update_config(self, **model_config: Unpack[BasetenConfig]) -> None:  # type: ignore[override]
        """Update the Baseten model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> BasetenConfig:
        """Get the Baseten model configuration.

        Returns:
            The Baseten model configuration.
        """
        return cast(BasetenModel.BasetenConfig, self.config)

    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        """Format a Baseten compatible content block.

        Args:
            content: Message content.

        Returns:
            Baseten compatible content block.

        Raises:
            TypeError: If the content block type cannot be converted to a Baseten-compatible format.
        """
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
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Baseten compatible tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Baseten compatible tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Baseten compatible tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            Baseten compatible tool message.
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
            "content": [cls.format_request_message_content(content) for content in contents],
        }

    def format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format a Baseten compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A Baseten compatible messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        # Check if this is a dedicated deployment (empty model_id)
        is_dedicated_deployment = not self.get_config()["model_id"]

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                self.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse"])
            ]
            formatted_tool_calls = [
                self.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                self.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            # For dedicated deployments, convert content to string if it's a single text block
            if is_dedicated_deployment and formatted_contents and len(formatted_contents) == 1 and "text" in formatted_contents[0]:
                content_value = formatted_contents[0]["text"]
            else:
                content_value = formatted_contents

            formatted_message = {
                "role": message["role"],
                "content": content_value,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a Baseten compatible request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A Baseten compatible request dictionary.
        """
        request = {
            "messages": self.format_request_messages(messages, system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
            **cast(dict[str, Any], self.config.get("params", {})),
        }
        
        # Only include tools if tool_specs is provided
        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs
            ]
        
        # Include model field - use actual model_id for Model APIs, placeholder for dedicated deployments
        model_id = self.get_config()["model_id"]
        if model_id:
            request["model"] = model_id
        else:
            # For dedicated deployments, use a placeholder model name
            request["model"] = "placeholder"
            
        return request

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format a Baseten response event into a standardized message chunk.

        Args:
            event: A response event from the Baseten model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "tool":
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

                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}
                    }

                if event["data_type"] == "reasoning_content":
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}

                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

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
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"].prompt_tokens,
                            "outputTokens": event["data"].completion_tokens,
                            "totalTokens": event["data"].total_tokens,
                        },
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
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Baseten model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("formatted request=<%s>", request)

        logger.debug("invoking model")
        response = await self.client.chat.completions.create(**request)

        logger.debug("got response from model")
        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        tool_calls: dict[int, list[Any]] = {}

        async for event in response:
            # Defensive: skip events with empty or missing choices
            if not getattr(event, "choices", None):
                continue
            choice = event.choices[0]

            if choice.delta.content:
                yield self.format_chunk(
                    {"chunk_type": "content_delta", "data_type": "text", "data": choice.delta.content}
                )

            if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                yield self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "reasoning_content",
                        "data": choice.delta.reasoning_content,
                    }
                )

            for tool_call in choice.delta.tool_calls or []:
                tool_calls.setdefault(tool_call.index, []).append(tool_call)

            if choice.finish_reason:
                break

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        for tool_deltas in tool_calls.values():
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

            for tool_delta in tool_deltas:
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

        # Skip remaining events as we don't have use for anything except the final usage payload
        async for event in response:
            _ = event

        if event.usage:
            yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        model_id = self.get_config()["model_id"]
        parse_kwargs = {
            "messages": self.format_request_messages(prompt),
            "response_format": output_model,
        }
        
        # Only include model field if model_id is not empty (for dedicated deployments)
        if model_id:
            parse_kwargs["model"] = model_id
            
        response: ParsedChatCompletion = await self.client.beta.chat.completions.parse(  # type: ignore
            **parse_kwargs
        )

        parsed: T | None = None
        # Find the first choice with tool_calls
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the Baseten response.")

        for choice in response.choices:
            if isinstance(choice.message.parsed, output_model):
                parsed = choice.message.parsed
                break

        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid tool use or tool use input was found in the Baseten response.") 