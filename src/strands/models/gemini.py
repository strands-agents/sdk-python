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
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiModel(Model):
    """Google Gemini model provider implementation."""

    SAFETY_MESSAGES = {"safety", "harmful", "content policy", "blocked due to safety"}

    QUOTA_MESSAGES = {"quota", "limit", "rate limit", "exceeded"}

    class GeminiConfig(TypedDict, total=False):
        """Configuration options for Gemini models."""

        model_id: str
        params: Optional[dict[str, Any]]

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
        """Formats an inline data part (image or document)."""
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
        """
        return {"functionCall": {"name": tool_use["name"], "args": tool_use["input"]}}

    def _format_function_response(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Gemini function response.

        Args:
            tool_result: Tool result from execution.

        Returns:
            Gemini formatted function response.
        """
        response_parts = []
        for content in tool_result["content"]:
            if "json" in content:
                response_parts.append(json.dumps(content["json"]))
            elif "text" in content:
                response_parts.append(content["text"])

        return {
            "functionResponse": {"name": tool_result["toolUseId"], "response": {"content": "\n".join(response_parts)}}
        }

    def _format_request_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format messages for Gemini API.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            Gemini formatted messages array.
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
    ) -> AsyncGenerator[Union[StreamEvent, bool], None]:
        """Process a single chunk from the streaming response."""
        has_function_call = False

        if hasattr(chunk, "candidates") and chunk.candidates:
            for candidate in chunk.candidates:
                if not hasattr(candidate, "content") or not candidate.content:
                    continue

                for part in candidate.content.parts:
                    if part.text:
                        output_text_buffer.append(part.text)
                        yield self.format_event("content_block_delta", part.text)

                    # Handle function calls
                    elif hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        has_function_call = True

                        if function_call.name not in tool_calls:
                            yield self.format_event("content_block_stop")

                            tool_id = f"tool_{len(tool_calls) + 1}"
                            tool_calls[function_call.name] = tool_id

                            yield self.format_event(
                                "content_block_start", {"function_call": function_call, "tool_id": tool_id}
                            )

                        if hasattr(function_call, "args") and function_call.args:
                            args = self._extract_function_args(function_call)
                            yield self.format_event(
                                "content_block_delta", {"function_call": function_call, "args": args}
                            )

        elif hasattr(chunk, "text") and chunk.text:
            output_text_buffer.append(chunk.text)
            yield self.format_event("content_block_delta", chunk.text)

        yield has_function_call

    def _extract_function_args(self, function_call: Any) -> dict[str, Any]:
        """Extract function arguments from various formats."""
        if not hasattr(function_call, "args"):
            return {}

        args = function_call.args

        # Handle Struct type (protobuf)
        if hasattr(args, "fields"):
            return self._struct_to_dict(args)

        # Handle JSON string
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": args}
            except json.JSONDecodeError:
                return {"value": args}

        if isinstance(args, dict):
            return dict(args)

        return {}

    def _struct_to_dict(self, struct_value: Any) -> dict[str, Any]:
        """Convert protobuf Struct to dict."""
        result = {}
        for key, value in struct_value.fields.items():
            if hasattr(value, "string_value"):
                result[key] = value.string_value
            elif hasattr(value, "number_value"):
                result[key] = value.number_value
            elif hasattr(value, "bool_value"):
                result[key] = value.bool_value
            elif hasattr(value, "list_value"):
                result[key] = [self._value_to_python(v) for v in value.list_value.values]
            elif hasattr(value, "struct_value"):
                result[key] = self._struct_to_dict(value.struct_value)
            else:
                result[key] = str(value)
        return result

    def _value_to_python(self, value: Any) -> Any:
        """Convert protobuf Value to Python type."""
        if hasattr(value, "string_value"):
            return value.string_value
        elif hasattr(value, "number_value"):
            return value.number_value
        elif hasattr(value, "bool_value"):
            return value.bool_value
        elif hasattr(value, "struct_value"):
            return self._struct_to_dict(value.struct_value)
        else:
            return str(value)

    async def _count_tokens_safely(self, model_id: str, contents: list[dict[str, Any]]) -> int:
        """Safely count tokens with fallback to 0 on error.

        Args:
            model_id: The Gemini model ID
            contents: The content to count tokens for

        Returns:
            Token count, or 0 if counting fails
        """
        try:
            token_count = await self.client.aio.models.count_tokens(model=model_id, contents=contents)
            if hasattr(token_count, "total_tokens"):
                return int(token_count.total_tokens or 0)
            return 0
        except Exception as e:
            logger.debug("Could not count tokens: %s", str(e))
            return 0

    def _format_tools(self, tool_specs: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        """Format tool specifications for Gemini.

        Args:
            tool_specs: List of tool specifications.

        Returns:
            Gemini formatted tools array.
        """
        if not tool_specs:
            return None

        tools = []
        for tool_spec in tool_specs:
            tools.append(
                {
                    "function_declarations": [
                        {
                            "name": tool_spec["name"],
                            "description": tool_spec["description"],
                            "parameters": tool_spec["inputSchema"]["json"],
                        }
                    ]
                }
            )

        return tools

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Format a Gemini streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            config: Additional configuration options including response_schema for structured output.

        Returns:
            A Gemini streaming request.
        """
        generation_config: dict[str, Any] = {}

        params = self.config.get("params")
        if params:
            generation_config.update(params)

        if config:
            if "response_schema" in config:
                generation_config["response_schema"] = config["response_schema"]
                generation_config["response_mime_type"] = config.get("response_mime_type", "application/json")

            config_params = config.get("params")
            if config_params:
                generation_config.update(config_params)

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

    def format_event(self, event_type: str, data: Any = None) -> StreamEvent:
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
                    return {"contentBlockDelta": {"delta": {"toolUse": {"input": args}}}}

                return {"contentBlockDelta": {"delta": {"text": data}}}

            case "content_block_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match data:
                    case "SAFETY" | "RECITATION":
                        return {"messageStop": {"stopReason": "content_filtered"}}
                    case "MAX_TOKENS":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case "tool_use":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

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
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Gemini model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications. Enables function calling when provided.
            system_prompt: System prompt to provide context to the model.
            config: Additional configuration options including response_schema for structured output.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            StreamEvents: messageStart, contentBlockDelta, contentBlockStop, messageStop, metadata

        Raises:
            ModelThrottledException: If the model is being rate-limited by Gemini API.
            RuntimeError: If an error occurs during streaming or response parsing.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, config)
        logger.debug("formatted request=<%s>", request)

        start_time = time.perf_counter()

        model_id = self.config.get("model_id", "gemini-2.5-flash")

        tool_config = None
        if tool_specs:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            )

        cfg = types.GenerateContentConfig(
            system_instruction=request.get("system_instruction"),
            tools=request.get("tools"),  # Use the formatted tools from format_request
            tool_config=tool_config,
            **(request.get("generation_config") or {}),
        )

        logger.debug("invoking gemini model %s", model_id)

        # Pre-flight check for metrics
        input_tokens = await self._count_tokens_safely(model_id, request["contents"])

        # Start the conversation
        yield self.format_event("message_start")

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

            logger.debug("streaming response from model")

            async for chunk in response:
                async for event in self._process_chunk(chunk, output_text_buffer, tool_calls):
                    if isinstance(event, bool):
                        if event:
                            has_function_call = True
                        continue

                    if "contentBlockDelta" in event and not content_started:
                        yield self.format_event("content_block_start")
                        content_started = True
                    yield event

                if hasattr(chunk, "finish_reason") and isinstance(chunk.finish_reason, str):
                    break

            if content_started or has_function_call:
                yield self.format_event("content_block_stop")

            if has_function_call:
                yield self.format_event("message_stop", "tool_use")
            else:
                yield self.format_event("message_stop", "end_turn")

            output_tokens = 0
            generated_text = "".join(output_text_buffer)
            if generated_text:
                output_tokens = await self._count_tokens_safely(
                    model_id, [{"role": "model", "parts": [{"text": generated_text}]}]
                )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            usage_data = {
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                "metrics": {
                    "latencyMs": latency_ms,
                },
            }

            yield self.format_event("metadata", usage_data)

            logger.debug("finished streaming response from model")

        except genai.errors.ClientError as e:
            error_msg = str(e).lower()

            if any(msg in error_msg for msg in self.SAFETY_MESSAGES):
                logger.warning("safety error: %s", str(e))
                yield self.format_event("content_block_delta", "Response was blocked due to safety concerns.")
                yield self.format_event("content_block_stop")
                yield self.format_event("message_stop", "SAFETY")
            elif any(msg in error_msg for msg in self.QUOTA_MESSAGES):
                logger.warning("quota or rate limit error: %s", str(e))
                yield self.format_event("content_block_stop")
                yield self.format_event("message_stop", "MAX_TOKENS")
                raise ModelThrottledException(f"Rate limit or quota exceeded: {str(e)}") from e
            else:
                logger.warning("client error (other): %s", str(e))
                yield self.format_event("content_block_delta", "Request could not be processed.")
                yield self.format_event("content_block_stop")
                yield self.format_event("message_stop", "SAFETY")

        except genai.errors.UnknownApiResponseError as e:
            logger.warning("incomplete or unparseable response: %s", str(e))
            yield self.format_event("content_block_stop")
            yield self.format_event("message_stop", "SAFETY")
            raise RuntimeError(f"Incomplete response from Gemini: {str(e)}") from e

        except genai.errors.ServerError as e:
            logger.warning("server error: %s", str(e))
            yield self.format_event("content_block_stop")
            yield self.format_event("message_stop", "MAX_TOKENS")
            error_message = str(e)
            raise ModelThrottledException(f"Server error: {error_message}") from e

        except Exception as e:
            logger.error("unexpected error during streaming: %s", str(e))
            raise RuntimeError(f"Error streaming from Gemini: {str(e)}") from e

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
            ModelThrottledException: If the model is being rate-limited.
            genai.errors.ClientError: If the request is invalid or blocked by safety settings.
            genai.errors.ServerError: If the server encounters an error processing the request.
        """
        schema = output_model.model_json_schema() if hasattr(output_model, "model_json_schema") else output_model

        config = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        if "config" in kwargs:
            config.update(kwargs.pop("config"))

        logger.debug("Using Gemini's native structured output with schema: %s", output_model.__name__)

        async_response = self.stream(messages=prompt, system_prompt=system_prompt, config=config, **kwargs)

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

        if stop_reason not in ["end_turn"]:
            logger.error("Model returned unexpected stop_reason: %s", stop_reason)
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "end_turn"')

        try:
            result = output_model.model_validate_json(full_response)
            yield {"output": result}

        except Exception as e:
            logger.error("Failed to create output model from JSON response: %s", str(e))
            raise ValueError(f"Failed to create structured output from Gemini response: {str(e)}") from e
