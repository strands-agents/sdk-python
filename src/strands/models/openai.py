"""OpenAI model provider.

- Docs: https://platform.openai.com/docs/overview
"""

import base64
import json
import logging
import mimetypes
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Optional, Protocol, Type, TypedDict, TypeVar, Union, cast

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from ._validation import validate_config_keys
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


class OpenAIModel(Model):
    """OpenAI model provider implementation."""

    client: Client

    class OpenAIConfig(TypedDict, total=False):
        """Configuration options for OpenAI models.

        Attributes:
            model_id: Model ID (e.g., "gpt-4o").
                For a complete list of supported models, see https://platform.openai.com/docs/models.
            params: Model parameters (e.g., max_tokens).
                For a complete list of supported parameters, see
                https://platform.openai.com/docs/api-reference/chat/create.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        client: Optional[Client] = None,
        client_args: Optional[dict[str, Any]] = None,
        **model_config: Unpack[OpenAIConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            client: Pre-configured OpenAI-compatible client to reuse across requests.
                When provided, this client will be reused for all requests and will NOT be closed
                by the model. The caller is responsible for managing the client lifecycle.
                This is useful for:
                - Injecting custom client wrappers (e.g., GuardrailsAsyncOpenAI)
                - Reusing connection pools within a single event loop/worker
                - Centralizing observability, retries, and networking policy
                - Pointing to custom model gateways
                Note: The client should not be shared across different asyncio event loops.
            client_args: Arguments for the OpenAI client (legacy approach).
                For a complete list of supported arguments, see https://pypi.org/project/openai/.
            **model_config: Configuration options for the OpenAI model.

        Raises:
            ValueError: If both `client` and `client_args` are provided.
        """
        validate_config_keys(model_config, self.OpenAIConfig)
        self.config = dict(model_config)

        # Validate that only one client configuration method is provided
        if client is not None and client_args is not None and len(client_args) > 0:
            raise ValueError("Only one of 'client' or 'client_args' should be provided, not both.")

        self._custom_client = client
        self.client_args = client_args or {}

        logger.debug("config=<%s> | initializing", self.config)

    @override
    def update_config(self, **model_config: Unpack[OpenAIConfig]) -> None:  # type: ignore[override]
        """Update the OpenAI model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.OpenAIConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> OpenAIConfig:
        """Get the OpenAI model configuration.

        Returns:
            The OpenAI model configuration.
        """
        return cast(OpenAIModel.OpenAIConfig, self.config)

    @classmethod
    def format_request_message_content(cls, content: ContentBlock, **kwargs: Any) -> dict[str, Any]:
        """Format an OpenAI compatible content block.

        Args:
            content: Message content.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            OpenAI compatible content block.

        Raises:
            TypeError: If the content block type cannot be converted to an OpenAI-compatible format.
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
    def format_request_message_tool_call(cls, tool_use: ToolUse, **kwargs: Any) -> dict[str, Any]:
        """Format an OpenAI compatible tool call.

        Args:
            tool_use: Tool use requested by the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            OpenAI compatible tool call.
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
    def format_request_tool_message(cls, tool_result: ToolResult, **kwargs: Any) -> dict[str, Any]:
        """Format an OpenAI compatible tool message.

        Args:
            tool_result: Tool result collected from a tool execution.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            OpenAI compatible tool message.
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

    @classmethod
    def _split_tool_message_images(
        cls, tool_message: dict[str, Any]
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        """Split a tool message into text-only tool message and optional user message with images.

        OpenAI API restricts images to user role messages only. This method extracts any image
        content from a tool message and returns it separately as a user message.

        Args:
            tool_message: A formatted tool message that may contain images.

        Returns:
            A tuple of (tool_message_without_images, user_message_with_images_or_None).
        """
        if tool_message.get("role") != "tool":
            return tool_message, None

        content = tool_message.get("content", [])
        if not isinstance(content, list):
            return tool_message, None

        # Separate image and non-image content
        text_content = []
        image_content = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_content.append(item)
            else:
                text_content.append(item)

        # If no images found, return original message
        if not image_content:
            return tool_message, None

        # Let the user know that we are modifying the messages for OpenAI compatibility
        logger.warning(
            "tool_call_id=<%s> | Moving image from tool message to a new user message for OpenAI compatibility",
            tool_message["tool_call_id"],
        )

        # Append a message to the text content to inform the model about the upcoming image
        text_content.append(
            {
                "type": "text",
                "text": (
                    "Tool successfully returned an image. The image is being provided in the following user message."
                ),
            }
        )

        # Create the clean tool message with the updated text content
        tool_message_clean = {
            "role": "tool",
            "tool_call_id": tool_message["tool_call_id"],
            "content": text_content,
        }

        # Create user message with only images
        user_message_with_images = {"role": "user", "content": image_content}

        return tool_message_clean, user_message_with_images

    @classmethod
    def _format_request_tool_choice(cls, tool_choice: ToolChoice | None) -> dict[str, Any]:
        """Format a tool choice for OpenAI compatibility.

        Args:
            tool_choice: Tool choice configuration in Bedrock format.

        Returns:
            OpenAI compatible tool choice format.
        """
        if not tool_choice:
            return {}

        match tool_choice:
            case {"auto": _}:
                return {"tool_choice": "auto"}  # OpenAI SDK doesn't define constants for these values
            case {"any": _}:
                return {"tool_choice": "required"}
            case {"tool": {"name": tool_name}}:
                return {"tool_choice": {"type": "function", "function": {"name": tool_name}}}
            case _:
                # This should not happen with proper typing, but handle gracefully
                return {"tool_choice": "auto"}

    @classmethod
    def _format_system_messages(
        cls,
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format system messages for OpenAI-compatible providers.

        Args:
            system_prompt: System prompt to provide context to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted system messages.
        """
        # Handle backward compatibility: if system_prompt is provided but system_prompt_content is None
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        # TODO: Handle caching blocks https://github.com/strands-agents/sdk-python/issues/1140
        return [
            {"role": "system", "content": content["text"]}
            for content in system_prompt_content or []
            if "text" in content
        ]

    @classmethod
    def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
        """Format regular messages for OpenAI-compatible providers.

        Args:
            messages: List of message objects to be processed by the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted messages.
        """
        formatted_messages = []

        for message in messages:
            contents = message["content"]

            # Check for reasoningContent and warn user
            if any("reasoningContent" in content for content in contents):
                logger.warning(
                    "reasoningContent is not supported in multi-turn conversations with the Chat Completions API."
                )

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse", "reasoningContent"])
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)

            # Process tool messages to extract images into separate user messages
            # OpenAI API requires images to be in user role messages only
            for tool_msg in formatted_tool_messages:
                tool_msg_clean, user_msg_with_images = cls._split_tool_message_images(tool_msg)
                formatted_messages.append(tool_msg_clean)
                if user_msg_with_images:
                    formatted_messages.append(user_msg_with_images)

        return formatted_messages

    @classmethod
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format an OpenAI compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            An OpenAI compatible messages array.
        """
        formatted_messages = cls._format_system_messages(system_prompt, system_prompt_content=system_prompt_content)
        formatted_messages.extend(cls._format_regular_messages(messages))

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format an OpenAI compatible chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            An OpenAI compatible chat streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to an OpenAI-compatible
                format.
        """
        return {
            "messages": self.format_request_messages(
                messages, system_prompt, system_prompt_content=system_prompt_content
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

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Format an OpenAI response event into a standardized message chunk.

        Args:
            event: A response event from the OpenAI compatible model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as chunk_type is controlled in the stream method.
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

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[Any]:
        """Get an OpenAI client for making requests.

        This context manager handles client lifecycle management:
        - If an injected client was provided during initialization, it yields that client
          without closing it (caller manages lifecycle).
        - Otherwise, creates a new AsyncOpenAI client from client_args and automatically
          closes it when the context exits.

        Note: We create a new client per request to avoid connection sharing in the underlying
        httpx client, as the asyncio event loop does not allow connections to be shared.
        For more details, see https://github.com/encode/httpx/discussions/2959.

        Yields:
            Client: An OpenAI-compatible client instance.
        """
        if self._custom_client is not None:
            # Use the injected client (caller manages lifecycle)
            yield self._custom_client
        else:
            # Create a new client from client_args
            # We initialize an OpenAI context on every request so as to avoid connection sharing in the underlying
            # httpx client. The asyncio event loop does not allow connections to be shared. For more details, please
            # refer to https://github.com/encode/httpx/discussions/2959.
            async with openai.AsyncOpenAI(**self.client_args) as client:
                yield client

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the OpenAI model.

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
            ModelThrottledException: If the request is throttled by OpenAI (rate limits).
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("formatted request=<%s>", request)

        logger.debug("invoking model")

        # We initialize an OpenAI context on every request so as to avoid connection sharing in the underlying httpx
        # client. The asyncio event loop does not allow connections to be shared. For more details, please refer to
        # https://github.com/encode/httpx/discussions/2959.
        async with self._get_client() as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                # Check if this is a context length exceeded error
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    logger.warning("OpenAI threw context window overflow error")
                    raise ContextWindowOverflowException(str(e)) from e
                # Re-raise other BadRequestError exceptions
                raise
            except openai.RateLimitError as e:
                # All rate limit errors should be treated as throttling, not context overflow
                # Rate limits (including TPM) require waiting/retrying, not context reduction
                logger.warning("OpenAI threw rate limit error")
                raise ModelThrottledException(str(e)) from e

            logger.debug("got response from model")
            yield self.format_chunk({"chunk_type": "message_start"})
            tool_calls: dict[int, list[Any]] = {}
            data_type = None
            finish_reason = None  # Store finish_reason for later use
            event = None  # Initialize for scope safety

            async for event in response:
                # Defensive: skip events with empty or missing choices
                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]

                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": data_type,
                            "data": choice.delta.reasoning_content,
                        }
                    )

                if choice.delta.content:
                    chunks, data_type = self._stream_switch_content("text", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {"chunk_type": "content_delta", "data_type": data_type, "data": choice.delta.content}
                    )

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason  # Store for use outside loop
                    if data_type:
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                    break

            for tool_deltas in tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason or "end_turn"})

            # Skip remaining events as we don't have use for anything except the final usage payload
            async for event in response:
                _ = event

            if event and hasattr(event, "usage") and event.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})

        logger.debug("finished streaming response from model")

    def _stream_switch_content(self, data_type: str, prev_data_type: str | None) -> tuple[list[StreamEvent], str]:
        """Handle switching to a new content stream.

        Args:
            data_type: The next content data type.
            prev_data_type: The previous content data type.

        Returns:
            Tuple containing:
            - Stop block for previous content and the start block for the next content.
            - Next content data type.
        """
        chunks = []
        if data_type != prev_data_type:
            if prev_data_type is not None:
                chunks.append(self.format_chunk({"chunk_type": "content_stop", "data_type": prev_data_type}))
            chunks.append(self.format_chunk({"chunk_type": "content_start", "data_type": data_type}))

        return chunks, data_type

    @override
    async def structured_output(
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
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by OpenAI (rate limits).
        """
        # We initialize an OpenAI context on every request so as to avoid connection sharing in the underlying httpx
        # client. The asyncio event loop does not allow connections to be shared. For more details, please refer to
        # https://github.com/encode/httpx/discussions/2959.
        async with self._get_client() as client:
            try:
                response: ParsedChatCompletion = await client.beta.chat.completions.parse(
                    model=self.get_config()["model_id"],
                    messages=self.format_request(prompt, system_prompt=system_prompt)["messages"],
                    response_format=output_model,
                )
            except openai.BadRequestError as e:
                # Check if this is a context length exceeded error
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    logger.warning("OpenAI threw context window overflow error")
                    raise ContextWindowOverflowException(str(e)) from e
                # Re-raise other BadRequestError exceptions
                raise
            except openai.RateLimitError as e:
                # All rate limit errors should be treated as throttling, not context overflow
                # Rate limits (including TPM) require waiting/retrying, not context reduction
                logger.warning("OpenAI threw rate limit error")
                raise ModelThrottledException(str(e)) from e

        parsed: T | None = None
        # Find the first choice with tool_calls
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
