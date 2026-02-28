"""LiteLLM model provider.

- Docs: https://docs.litellm.ai/
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, TypeVar, cast

import litellm
from litellm.exceptions import ContextWindowExceededError
from litellm.utils import supports_response_schema
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..tools import convert_pydantic_to_tool_spec
from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.event_loop import Usage
from ..types.exceptions import ContextWindowOverflowException
from ..types.streaming import MetadataEvent, StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .openai import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LiteLLMModel(OpenAIModel):
    """LiteLLM model provider implementation."""

    class LiteLLMConfig(TypedDict, total=False):
        """Configuration options for LiteLLM models.

        Attributes:
            model_id: Model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet").
                For a complete list of supported models, see https://docs.litellm.ai/docs/providers.
            params: Model parameters (e.g., max_tokens).
                For a complete list of supported parameters, see
                https://docs.litellm.ai/docs/completion/input#input-params-1.
        """

        model_id: str
        params: dict[str, Any] | None

    def __init__(self, client_args: dict[str, Any] | None = None, **model_config: Unpack[LiteLLMConfig]) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the LiteLLM client.
                For a complete list of supported arguments, see
                https://github.com/BerriAI/litellm/blob/main/litellm/main.py.
            **model_config: Configuration options for the LiteLLM model.
        """
        self.client_args = client_args or {}
        validate_config_keys(model_config, self.LiteLLMConfig)
        self.config = dict(model_config)
        self._apply_proxy_prefix()

        logger.debug("config=<%s> | initializing", self.config)

    @override
    def update_config(self, **model_config: Unpack[LiteLLMConfig]) -> None:  # type: ignore[override]
        """Update the LiteLLM model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.LiteLLMConfig)
        self.config.update(model_config)
        self._apply_proxy_prefix()

    @override
    def get_config(self) -> LiteLLMConfig:
        """Get the LiteLLM model configuration.

        Returns:
            The LiteLLM model configuration.
        """
        return cast(LiteLLMModel.LiteLLMConfig, self.config)

    @override
    @classmethod
    def format_request_message_content(cls, content: ContentBlock, **kwargs: Any) -> dict[str, Any]:
        """Format a LiteLLM content block.

        Args:
            content: Message content.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            LiteLLM formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a LiteLLM-compatible format.
        """
        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"]["signature"],
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "video" in content:
            return {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": content["video"]["source"]["bytes"],
                },
            }

        return super().format_request_message_content(content)

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
    @classmethod
    def _format_system_messages(
        cls,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format system messages for LiteLLM with cache point support.

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

        system_content: list[dict[str, Any]] = []
        for block in system_prompt_content or []:
            if "text" in block:
                system_content.append({"type": "text", "text": block["text"]})
            elif "cachePoint" in block and block["cachePoint"].get("type") == "default":
                # Apply cache control to the immediately preceding content block
                # for LiteLLM/Anthropic compatibility
                if system_content:
                    system_content[-1]["cache_control"] = {"type": "ephemeral"}

        # Create single system message with content array rather than mulitple system messages
        return [{"role": "system", "content": system_content}] if system_content else []

    @override
    @classmethod
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format a LiteLLM compatible messages array with cache point support.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model (for legacy compatibility).
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A LiteLLM compatible messages array.
        """
        formatted_messages = cls._format_system_messages(system_prompt, system_prompt_content=system_prompt_content)
        formatted_messages.extend(cls._format_regular_messages(messages))

        return [message for message in formatted_messages if "content" in message or "tool_calls" in message]

    @override
    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Format a LiteLLM response event into a standardized message chunk.

        This method overrides OpenAI's format_chunk to handle the metadata case
        with prompt caching support. All other chunk types use the parent implementation.

        Args:
            event: A response event from the LiteLLM model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        # Handle metadata case with prompt caching support
        if event["chunk_type"] == "metadata":
            usage_data: Usage = {
                "inputTokens": event["data"].prompt_tokens,
                "outputTokens": event["data"].completion_tokens,
                "totalTokens": event["data"].total_tokens,
            }

            # Only LiteLLM over Anthropic supports cache write tokens
            # Waiting until a more general approach is available to set cacheWriteInputTokens
            if tokens_details := getattr(event["data"], "prompt_tokens_details", None):
                if cached := getattr(tokens_details, "cached_tokens", None):
                    usage_data["cacheReadInputTokens"] = cached
            if creation := getattr(event["data"], "cache_creation_input_tokens", None):
                usage_data["cacheWriteInputTokens"] = creation

            return StreamEvent(
                metadata=MetadataEvent(
                    metrics={
                        "latencyMs": 0,  # TODO
                    },
                    usage=usage_data,
                )
            )
        # For all other cases, use the parent implementation
        return super().format_chunk(event)

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
        """Stream conversation with the LiteLLM model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.
        """
        logger.debug("formatting request")
        request = self.format_request(
            messages, tool_specs, system_prompt, tool_choice, system_prompt_content=system_prompt_content
        )
        logger.debug("request=<%s>", request)

        # Check if streaming is disabled in the params
        config = self.get_config()
        params = config.get("params") or {}
        is_streaming = params.get("stream", True)

        litellm_request = {**request}

        litellm_request["stream"] = is_streaming

        logger.debug("invoking model with stream=%s", litellm_request.get("stream"))

        try:
            if is_streaming:
                async for chunk in self._handle_streaming_response(litellm_request):
                    yield chunk
            else:
                async for chunk in self._handle_non_streaming_response(litellm_request):
                    yield chunk
        except ContextWindowExceededError as e:
            logger.warning("litellm client raised context window overflow")
            raise ContextWindowOverflowException(e) from e

        logger.debug("finished processing response from model")

    @override
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Some models do not support native structured output via response_format.
        In cases of proxies, we may not have a way to determine support, so we
        fallback to using tool calling to achieve structured output.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        if supports_response_schema(self.get_config()["model_id"]):
            logger.debug("structuring output using response schema")
            result = await self._structured_output_using_response_schema(output_model, prompt, system_prompt)
        else:
            logger.debug("model does not support response schema, structuring output using tool approach")
            result = await self._structured_output_using_tool(output_model, prompt, system_prompt)

        yield {"output": result}

    async def _structured_output_using_response_schema(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None
    ) -> T:
        """Get structured output using native response_format support."""
        response = await litellm.acompletion(
            **self.client_args,
            model=self.get_config()["model_id"],
            messages=self.format_request(prompt, system_prompt=system_prompt)["messages"],
            response_format=output_model,
        )

        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the response.")
        if not response.choices:
            raise ValueError("No choices found in response")

        choice = response.choices[0]
        try:
            # Parse the message content as JSON
            tool_call_data = json.loads(choice.message.content)
            # Instantiate the output model with the parsed data
            return output_model(**tool_call_data)
        except ContextWindowExceededError as e:
            logger.warning("litellm client raised context window overflow in structured_output")
            raise ContextWindowOverflowException(e) from e
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to parse or load content into model: {e}") from e

    async def _structured_output_using_tool(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None
    ) -> T:
        """Get structured output using tool calling fallback."""
        tool_spec = convert_pydantic_to_tool_spec(output_model)
        request = self.format_request(prompt, [tool_spec], system_prompt, cast(ToolChoice, {"any": {}}))
        args = {**self.client_args, **request, "stream": False}
        response = await litellm.acompletion(**args)

        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the response.")
        if not response.choices or response.choices[0].finish_reason != "tool_calls":
            raise ValueError("No tool_calls found in response")

        choice = response.choices[0]
        try:
            # Parse the tool call content as JSON
            tool_call = choice.message.tool_calls[0]
            tool_call_data = json.loads(tool_call.function.arguments)
            # Instantiate the output model with the parsed data
            return output_model(**tool_call_data)
        except ContextWindowExceededError as e:
            logger.warning("litellm client raised context window overflow in structured_output")
            raise ContextWindowOverflowException(e) from e
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to parse or load content into model: {e}") from e

    async def _process_choice_content(
        self, choice: Any, data_type: str | None, tool_calls: dict[int, list[Any]], is_streaming: bool = True
    ) -> AsyncGenerator[tuple[str | None, StreamEvent], None]:
        """Process content from a choice object (streaming or non-streaming).

        Args:
            choice: The choice object from the response.
            data_type: Current data type being processed.
            tool_calls: Dictionary to collect tool calls.
            is_streaming: Whether this is from a streaming response.

        Yields:
            Tuples of (updated_data_type, stream_event).
        """
        # Get the content source - this is the only difference between streaming/non-streaming
        # We use duck typing here: both choice.delta and choice.message have the same interface
        # (reasoning_content, content, tool_calls attributes) but different object structures
        content_source = choice.delta if is_streaming else choice.message

        # Process reasoning content
        if hasattr(content_source, "reasoning_content") and content_source.reasoning_content:
            chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
            for chunk in chunks:
                yield data_type, chunk
            chunk = self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "reasoning_content",
                    "data": content_source.reasoning_content,
                }
            )
            yield data_type, chunk

        # Process reasoning signature from LiteLLM's thinking attribute
        thinking = getattr(content_source, "thinking", None)
        if thinking is not None:
            sig = getattr(thinking, "signature", None)
            if isinstance(sig, str) and sig:
                chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                for chunk in chunks:
                    yield data_type, chunk
                yield data_type, {
                    "contentBlockDelta": {"delta": {"reasoningContent": {"signature": sig}}}
                }

        # Process text content
        if hasattr(content_source, "content") and content_source.content:
            chunks, data_type = self._stream_switch_content("text", data_type)
            for chunk in chunks:
                yield data_type, chunk
            chunk = self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "text",
                    "data": content_source.content,
                }
            )
            yield data_type, chunk

        # Process tool calls
        if hasattr(content_source, "tool_calls") and content_source.tool_calls:
            if is_streaming:
                # Streaming: tool calls have index attribute for out-of-order delivery
                for tool_call in content_source.tool_calls:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)
            else:
                # Non-streaming: tool calls arrive in order, use enumerated index
                for i, tool_call in enumerate(content_source.tool_calls):
                    tool_calls.setdefault(i, []).append(tool_call)

    async def _process_tool_calls(self, tool_calls: dict[int, list[Any]]) -> AsyncGenerator[StreamEvent, None]:
        """Process and yield tool call events.

        Args:
            tool_calls: Dictionary of tool calls indexed by their position.

        Yields:
            Formatted tool call chunks.
        """
        for tool_deltas in tool_calls.values():
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

            for tool_delta in tool_deltas:
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

    async def _handle_non_streaming_response(
        self, litellm_request: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle non-streaming response from LiteLLM.

        Args:
            litellm_request: The formatted request for LiteLLM.

        Yields:
            Formatted message chunks from the model.
        """
        response = await litellm.acompletion(**self.client_args, **litellm_request)

        logger.debug("got non-streaming response from model")
        yield self.format_chunk({"chunk_type": "message_start"})

        tool_calls: dict[int, list[Any]] = {}
        data_type: str | None = None
        finish_reason: str | None = None

        if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
            choice = response.choices[0]

            if hasattr(choice, "message") and choice.message:
                # Process content using shared logic
                async for updated_data_type, chunk in self._process_choice_content(
                    choice, data_type, tool_calls, is_streaming=False
                ):
                    data_type = updated_data_type
                    yield chunk

            if hasattr(choice, "finish_reason"):
                finish_reason = choice.finish_reason

        # Stop the current content block if we have one
        if data_type:
            yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})

        # Process tool calls
        async for chunk in self._process_tool_calls(tool_calls):
            yield chunk

        yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

        # Add usage information if available
        if hasattr(response, "usage"):
            yield self.format_chunk({"chunk_type": "metadata", "data": response.usage})

    async def _handle_streaming_response(self, litellm_request: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        """Handle streaming response from LiteLLM.

        Args:
            litellm_request: The formatted request for LiteLLM.

        Yields:
            Formatted message chunks from the model.
        """
        # For streaming, use the streaming API
        response = await litellm.acompletion(**self.client_args, **litellm_request)

        logger.debug("got response from model")
        yield self.format_chunk({"chunk_type": "message_start"})

        tool_calls: dict[int, list[Any]] = {}
        data_type: str | None = None
        finish_reason: str | None = None

        async for event in response:
            # Defensive: skip events with empty or missing choices
            if not getattr(event, "choices", None):
                continue
            choice = event.choices[0]

            # Process content using shared logic
            async for updated_data_type, chunk in self._process_choice_content(
                choice, data_type, tool_calls, is_streaming=True
            ):
                data_type = updated_data_type
                yield chunk

            if choice.finish_reason:
                finish_reason = choice.finish_reason
                if data_type:
                    yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                break

        # Process tool calls
        async for chunk in self._process_tool_calls(tool_calls):
            yield chunk

        yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

        # Skip remaining events as we don't have use for anything except the final usage payload
        async for event in response:
            _ = event
            if usage := getattr(event, "usage", None):
                yield self.format_chunk({"chunk_type": "metadata", "data": usage})

        logger.debug("finished streaming response from model")

    def _apply_proxy_prefix(self) -> None:
        """Apply litellm_proxy/ prefix to model_id when use_litellm_proxy is True.

        This is a workaround for https://github.com/BerriAI/litellm/issues/13454
        where use_litellm_proxy parameter is not honored.
        """
        if self.client_args.get("use_litellm_proxy") and "model_id" in self.config:
            model_id = self.get_config()["model_id"]
            if not model_id.startswith("litellm_proxy/"):
                self.config["model_id"] = f"litellm_proxy/{model_id}"
