"""Google Gemini model provider.

- Docs: https://ai.google.dev/api
"""

import base64
import json
import logging
import mimetypes
import os
import time
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union, cast

from google import genai
from google.genai import types
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.event_loop import StopReason
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiModel(Model):
    """Google Gemini model provider implementation."""

    # Safety-related finish reasons that indicate content was blocked
    # These map to the values defined in the Gemini API: https://ai.google.dev/api/generate-content#FinishReason
    SAFETY_FINISH_REASONS = {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.IMAGE_SAFETY,
        types.FinishReason.SPII,
    }

    FINISH_REASON_MAP = {
        **{reason: "content_filtered" for reason in SAFETY_FINISH_REASONS},
        types.FinishReason.MAX_TOKENS: "max_tokens",
        types.FinishReason.STOP: "end_turn",
    }

    class GeminiConfig(TypedDict, total=False):
        """Configuration options for Gemini models."""

        model_id: str
        params: Optional[dict[str, Any]]
        response_schema: Optional[dict[str, Any]]
        response_mime_type: Optional[str]

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client_args: Optional[dict[str, Any]] = None,
        **model_config: Unpack[GeminiConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            api_key: Google AI API key. If not provided, will use GOOGLE_API_KEY env var.
            client_args: Additional arguments for the Gemini client configuration.
            **model_config: Configuration options for the Gemini model.

        GenerationConfig params: https://ai.google.dev/api/generate-content#generationconfig
        """
        self.config = GeminiModel.GeminiConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_config = {"api_key": api_key or os.environ.get("GOOGLE_API_KEY"), **(client_args or {})}

        self.client = genai.Client(**client_config)

    @override
    def update_config(self, **model_config: Unpack[GeminiConfig]) -> None:  # type: ignore[override]
        """Update the Gemini model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> GeminiConfig:
        """Get the Gemini model configuration.

        Returns:
            The Gemini model configuration.
        """
        return self.config

    def _format_inline_data_part(self, data: dict[str, Any], default_mime: str) -> dict[str, Any]:
        """Formats an inline data part (image or document).

        Converts Strands content format to Gemini's inlineData format with base64 encoding.
        Automatically detects MIME type based on file format extension.

        Args:
            data: Content data containing format and source with bytes.
            default_mime: Default MIME type to use if format detection fails.

        Returns:
            Gemini formatted inline data with mimeType and base64 encoded data.

        Examples:
            Image data: {"format": "png", "source": {"bytes": b"..."}}
            Document data: {"format": "pdf", "source": {"bytes": b"..."}}
        """
        file_format = data["format"]
        source_bytes = data["source"]["bytes"]
        mime_type = mimetypes.types_map.get(f".{file_format}", default_mime)

        return {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(source_bytes).decode("utf-8")}}

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a Gemini content block.

        Args:
            content: Message content.

        Returns:
            Gemini formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a Gemini-compatible format.
        """
        if "text" in content:
            return {"text": content["text"]}

        if "image" in content:
            return self._format_inline_data_part(cast(dict[str, Any], content["image"]), "image/png")

        if "document" in content:
            return self._format_inline_data_part(cast(dict[str, Any], content["document"]), "application/octet-stream")

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_function_call(self, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Gemini function call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Gemini formatted function call.

        Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
        """
        return {"functionCall": {"name": tool_use["name"], "args": tool_use["input"]}}

    def _format_function_response(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Gemini function response.

        Args:
            tool_result: Tool result from execution.

        Returns:
            Gemini formatted function response.

        Note:
            Based on API testing, Gemini function responses support inlineData format
            for images and other media. The model can actually process visual content.

        Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
        """
        response_parts = []
        for content in tool_result["content"]:
            if "json" in content:
                response_parts.append(json.dumps(content["json"]))
            elif "text" in content:
                response_parts.append(content["text"])
            elif "image" in content:
                image_data = content["image"]
                formatted_image = self._format_inline_data_part(cast(dict[str, Any], image_data), "image/png")
                return {"functionResponse": {"name": tool_result["toolUseId"], "response": {"image": formatted_image}}}
            elif "document" in content:
                document_data = content["document"]
                formatted_doc = self._format_inline_data_part(cast(dict[str, Any], document_data), "application/pdf")
                return {"functionResponse": {"name": tool_result["toolUseId"], "response": {"document": formatted_doc}}}
            else:
                # Handle other types as text descriptions
                content_type = next(iter(content.keys()), "unknown")
                response_parts.append(f"[{content_type.upper()}: content returned]")

        return {
            "functionResponse": {"name": tool_result["toolUseId"], "response": {"content": "\n".join(response_parts)}}
        }

    def _format_request_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format messages for Gemini API.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            Gemini formatted messages array.

        Gemini StreamGenerateContent Request Body: https://ai.google.dev/api/generate-content#request-body_1
        """
        formatted_messages = []

        for message in messages:
            role = "user" if message["role"] == "user" else "model"

            text_parts = []
            media_parts = []
            function_calls = []
            function_responses = []

            for content in message["content"]:
                if "text" in content:
                    text_parts.append(content["text"])
                elif "image" in content or "document" in content:
                    media_parts.append(self._format_request_message_content(content))
                elif "toolUse" in content:
                    function_calls.append(self._format_function_call(content["toolUse"]))
                elif "toolResult" in content:
                    function_responses.append(self._format_function_response(content["toolResult"]))

            parts = []

            if text_parts:
                parts.append({"text": "\n\n".join(text_parts)})

            if media_parts:
                parts.extend(media_parts)

            parts.extend(function_calls)
            parts.extend(function_responses)

            if parts:
                formatted_messages.append({"role": role, "parts": parts})

        return formatted_messages

    async def _process_chunk(
        self, chunk: Any, output_text_buffer: list[str], tool_calls: dict[str, str]
    ) -> AsyncGenerator[Union[StreamEvent, bool, types.FinishReason], None]:
        """Process a streaming response chunk from Gemini API.

        Extracts text content, function calls, and finish reasons from response chunks.
        Generates appropriate stream events and tracks tool usage.

        Args:
            chunk: Raw response chunk from Gemini streaming API.
            output_text_buffer: Buffer to accumulate text content across chunks.
            tool_calls: Dictionary tracking tool use instances by unique IDs.

        Yields:
            StreamEvent: Standard format events (content_block_start, content_block_delta, etc.)
            bool: True if chunk contains function calls, False otherwise.
            types.FinishReason: Gemini finish reason if chunk indicates completion.
        """
        has_function_call = False
        finish_reason = None

        if chunk.candidates:
            candidate = chunk.candidates[0]

            if candidate.finish_reason:
                finish_reason = candidate.finish_reason

            # Process content parts
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle text content
                    if part.text:
                        output_text_buffer.append(part.text)
                        yield self._format_event("content_block_delta", part.text)

                    # Handle function calls
                    elif hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        has_function_call = True

                        # Generate and store a unique tool ID for this specific tool use instance
                        tool_id = f"tool_{len(tool_calls) + 1}_{function_call.name}"
                        tool_calls[tool_id] = function_call.name

                        # Emit content_block_start for each new tool use
                        yield self._format_event(
                            "content_block_start", {"function_call": function_call, "tool_id": tool_id}
                        )

                        # If there are args, emit the delta event with tool input
                        if function_call.args:
                            yield self._format_event(
                                "content_block_delta", {"function_call": function_call, "args": function_call.args}
                            )

        yield has_function_call
        if finish_reason:
            yield finish_reason

    async def _count_tokens_safely(self, model_id: str, contents: list[dict[str, Any]]) -> int:
        """Safely count tokens with fallback to 0 on error.

        Args:
            model_id: The Gemini model ID
            contents: The content to count tokens for

        Returns:
            Token count, or 0 if counting fails

        Gemini Token Counting: https://ai.google.dev/api/tokens
        """
        try:
            token_count = await self.client.aio.models.count_tokens(model=model_id, contents=contents)
            if hasattr(token_count, "total_tokens"):
                return int(token_count.total_tokens or 0)
            return 0
        except Exception:
            return 0

    def _format_tools(self, tool_specs: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        """Format tool specifications for Gemini.

        Args:
            tool_specs: List of tool specifications.

        Returns:
            Gemini formatted tools array.

        Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
        """
        if not tool_specs:
            return None

        function_declarations = []
        for tool_spec in tool_specs:
            function_declarations.append(
                {
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "parameters": tool_spec["inputSchema"]["json"],
                }
            )

        return [{"function_declarations": function_declarations}]

    def _format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Format a Gemini streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            config_override: Additional configuration options including response_schema for structured output.

        Returns:
            A Gemini streaming request.

        Streaming Content Request Body: https://ai.google.dev/api/generate-content#request-body_1
        """
        generation_config: dict[str, Any] = {}

        params = self.config.get("params")
        if params:
            generation_config.update(params)

        if "response_schema" in self.config:
            generation_config["response_schema"] = self.config["response_schema"]
            generation_config["response_mime_type"] = self.config.get("response_mime_type", "application/json")

        if config_override:
            generation_config.update(config_override)

        request = {
            "contents": self._format_request_messages(messages),
            "generation_config": generation_config,
            "stream": True,
        }

        if system_prompt:
            request["system_instruction"] = {"parts": [{"text": system_prompt}]}

        tools = self._format_tools(tool_specs)
        if tools:
            request["tools"] = tools

        return request

    def _format_event(self, event_type: str, data: Any = None) -> StreamEvent:
        """Format a Gemini event into a standardized message chunk.

        Args:
            event_type: Type of event to format
            data: Data associated with the event

        Returns:
            The formatted event

        Raises:
            RuntimeError: If event_type is not recognized
        """
        match event_type:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_block_start":
                if data and "function_call" in data:
                    function_call = data["function_call"]
                    return {
                        "contentBlockStart": {
                            "start": {"toolUse": {"name": function_call.name, "toolUseId": data["tool_id"]}}
                        }
                    }
                return {"contentBlockStart": {"start": {}}}

            case "content_block_delta":
                if data and "function_call" in data:
                    args = data.get("args", {})
                    return {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}

                return {"contentBlockDelta": {"delta": {"text": data}}}

            case "content_block_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                if isinstance(data, str) and not isinstance(data, types.FinishReason):
                    stop_reason = cast(StopReason, data)
                else:
                    stop_reason = cast(StopReason, self.FINISH_REASON_MAP.get(data, "end_turn"))
                return {"messageStop": {"stopReason": stop_reason}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": data.get("usage", {}),
                        "metrics": {
                            "latencyMs": data.get("latency_ms", 0),
                        },
                    },
                }

            case _:
                raise RuntimeError(f"event_type=<{event_type}> | unknown type")

    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Gemini model.

        Provides real-time streaming of model responses with support for text generation,
        function calling, and comprehensive error handling. Includes token counting and
        performance metrics.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional configuration options including response_schema for structured output.

        Yields:
            StreamEvent: Standardized streaming events including:
                - messageStart: Conversation initiation
                - contentBlockStart: Beginning of content or tool use
                - contentBlockDelta: Incremental content updates
                - contentBlockStop: End of content block
                - messageStop: Conversation completion with stop reason
                - metadata: Usage statistics and performance metrics

        Raises:
            ModelThrottledException: If the model is being rate-limited or quota exceeded.
            RuntimeError: If the request is invalid, blocked by safety settings, or server error occurs.

        Generate Stream Content Response: https://ai.google.dev/api/generate-content#method:-models.streamgeneratecontent
        """
        start_time = time.perf_counter()
        model_id = self.config.get("model_id", "gemini-2.5-flash")

        request = self._format_request(messages, tool_specs, system_prompt, kwargs)

        # Build config directly from the formatted request
        cfg = types.GenerateContentConfig(
            system_instruction=request.get("system_instruction"),
            tools=request.get("tools"),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            )
            if tool_specs
            else None,
            **(request.get("generation_config") or {}),
        )

        logger.debug("invoking gemini model %s", model_id)

        # Pre-flight check for metrics
        input_tokens = await self._count_tokens_safely(model_id, request["contents"])

        # Start the conversation
        yield self._format_event("message_start")
        output_text_buffer: list[str] = []

        try:
            response = await self.client.aio.models.generate_content_stream(
                model=model_id,
                contents=request["contents"],
                config=cfg,
            )

            tool_calls: dict[str, str] = {}
            has_function_call = False
            content_started = False

            async for chunk in response:
                chunk_finish_reason = None
                async for event in self._process_chunk(chunk, output_text_buffer, tool_calls):
                    if isinstance(event, bool):
                        if event:
                            has_function_call = True
                        continue
                    elif isinstance(event, types.FinishReason):
                        chunk_finish_reason = event
                        continue

                    # Handle contentBlockStart events for both text and tool use
                    if "contentBlockStart" in event:
                        content_started = True
                        yield event

                    # Handle contentBlockDelta events
                    elif "contentBlockDelta" in event:
                        # Check if this is text content
                        if "delta" in event["contentBlockDelta"] and "text" in event["contentBlockDelta"]["delta"]:
                            if not content_started:
                                yield self._format_event("content_block_start")
                                content_started = True
                        yield event

                    else:
                        yield event

                # Handle early termination
                if chunk_finish_reason:
                    if chunk_finish_reason in self.SAFETY_FINISH_REASONS:
                        logger.debug("Content blocked due to safety: %s", chunk_finish_reason)
                        yield self._format_event("content_block_stop")
                        yield self._format_event("message_stop", chunk_finish_reason)
                        break
                    elif chunk_finish_reason == types.FinishReason.MAX_TOKENS:
                        logger.debug("Content blocked due to max_tokens reached: %s", chunk_finish_reason)
                        yield self._format_event("content_block_stop")
                        yield self._format_event("message_stop", chunk_finish_reason)
                        break

            # Final events
            if content_started or has_function_call:
                yield self._format_event("content_block_stop")

            if has_function_call:
                yield self._format_event("message_stop", "tool_use")
            else:
                yield self._format_event("message_stop", "end_turn")

            # Usage metadata
            output_tokens = 0
            generated_text = "".join(output_text_buffer)
            if generated_text:
                output_tokens = await self._count_tokens_safely(
                    model_id, [{"role": "model", "parts": [{"text": generated_text}]}]
                )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            yield self._format_event(
                "metadata",
                {
                    "usage": {
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                        "totalTokens": input_tokens + output_tokens,
                    },
                    "latency_ms": latency_ms,
                },
            )

        except genai.errors.ClientError as e:
            # Handle client errors (4xx) - rate limiting, quota, auth, etc.
            if e.status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"):
                raise ModelThrottledException(f"Rate limit or quota exceeded: {e}") from e
            else:
                # Handle other client errors as RuntimeError
                raise RuntimeError(f"Client error from Gemini: {e}") from e

        except genai.errors.ServerError as e:
            # Handle server errors (5xx)
            raise ModelThrottledException(f"Server error: {e}") from e

        except genai.errors.UnknownApiResponseError as e:
            raise RuntimeError(f"Unparseable response from Gemini: {e}") from e

        except Exception as e:
            logger.error("unexpected error during streaming: %s", str(e))
            raise RuntimeError(f"Unexpected error streaming from Gemini: {e}") from e

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model using Gemini's native structured output.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            ValueError: If the model doesn't return valid structured output.

        Gemini Structured Output: https://ai.google.dev/gemini-api/docs/structured-output
        """
        schema = output_model.model_json_schema() if hasattr(output_model, "model_json_schema") else output_model

        structured_config = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        if "config" in kwargs:
            structured_config.update(kwargs.pop("config"))

        logger.debug("Using Gemini's native structured output with schema: %s", output_model.__name__)

        structured_config.pop("tool_specs", None)
        kwargs.pop("tool_specs", None)
        async_response = self.stream(
            messages=prompt, tool_specs=None, system_prompt=system_prompt, **structured_config, **kwargs
        )

        accumulated_text = []
        stop_reason = None

        async for event in async_response:
            # Don't yield streaming events, only collect the final result
            if "messageStop" in event and "stopReason" in event["messageStop"]:
                stop_reason = event["messageStop"]["stopReason"]

            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    accumulated_text.append(delta["text"])

        full_response = "".join(accumulated_text)

        if not full_response.strip():
            logger.error("Empty response from model when generating structured output")
            raise ValueError("Empty response from model when generating structured output")

        if stop_reason != "end_turn":
            logger.error("Model returned unexpected stop_reason: %s", stop_reason)
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "end_turn"')

        try:
            result = output_model.model_validate_json(full_response)
            yield {"output": result}

        except Exception as e:
            logger.error("Failed to create output model from JSON response: %s", str(e))
            raise ValueError(f"Failed to create structured output from Gemini response: {str(e)}") from e
