"""AI21 Jamba model provider.

- Docs: https://docs.ai21.com/docs/sdk
- Github: https://github.com/AI21Labs/ai21-python
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union, cast

from ai21 import AsyncAI21Client
from ai21.errors import TooManyRequestsError
from ai21.models.chat import AssistantMessage, ChatMessage, ToolMessage
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..event_loop.streaming import process_stream
from ..tools import convert_pydantic_to_tool_spec
from ..types.content import Messages
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AI21Model(Model):
    """AI21 Jamba model provider implementation."""

    class AI21Config(TypedDict, total=False):
        """Configuration options for AI21 models.

        Attributes:
            model_id: Model ID (e.g., "jamba-mini", "jamba-large").
                For a complete list of supported models, see https://docs.ai21.com/docs/sdk.
            params: Model parameters (e.g., max_tokens, temperature).
                For a complete list of supported parameters, see
                https://docs.ai21.com/docs/sdk.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(self, *, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[AI21Config]) -> None:
        """Initialize AI21 provider instance.

        Args:
            client_args: Arguments for the AI21 client (e.g., api_key).
                For a complete list of supported arguments, see https://docs.ai21.com/docs/sdk.
            **model_config: Configuration options for the AI21 model.
        """
        self.config = AI21Model.AI21Config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        self.client_args = client_args or {}
        self.client = AsyncAI21Client(**self.client_args)

    @override
    def update_config(self, **model_config: Unpack[AI21Config]) -> None:  # type: ignore[override]
        """Update the AI21 model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> AI21Config:
        """Get the AI21 model configuration.

        Returns:
            The AI21 model configuration.
        """
        return self.config

    def _format_request_messages(
        self, messages: Messages, system_prompt: Optional[str] = None
    ) -> list[ChatMessage | AssistantMessage | ToolMessage]:
        """Format strands messages to AI21 ChatMessage format.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            List of AI21 ChatMessage, AssistantMessage, or ToolMessage objects.

        Raises:
            ValueError: If toolResult content is empty or contains no valid text/json data.
            TypeError: If content type is unsupported.
        """
        formatted_messages = []

        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append(ChatMessage(role="system", content=system_prompt))

        # Convert strands messages to AI21 format
        for message in messages:
            if isinstance(message, dict):
                role = message["role"]
                content_blocks = message["content"]

                # Separate content types using helper methods for consistency
                text_content_parts = []
                tool_calls = []
                tool_messages = []

                for content_block in content_blocks:
                    if "cachePoint" in content_block:
                        continue  # Skip cache points
                    elif "text" in content_block:
                        # Regular text content
                        content_part = self._format_request_message_content(cast(dict[str, Any], content_block))
                        if content_part:
                            text_content_parts.append(content_part)
                    elif "toolUse" in content_block:
                        # Tool calls using helper method
                        tool_call_data = self._format_request_message_tool_call(
                            cast(dict[str, Any], content_block["toolUse"])
                        )
                        tool_calls.append(tool_call_data)
                    elif "toolResult" in content_block:
                        # Tool messages using helper method
                        tool_message_data = self._format_request_message_tool_message(
                            cast(dict[str, Any], content_block["toolResult"])
                        )
                        tool_messages.append(tool_message_data)

                # Create the appropriate AI21 message types
                if text_content_parts or tool_calls:
                    combined_content = "\n".join(text_content_parts) if text_content_parts else ""

                    if tool_calls and role == "assistant":
                        # Assistant message with tool calls - role must be "assistant" for AssistantMessage
                        ai21_message: Union[ChatMessage, AssistantMessage] = AssistantMessage(
                            role="assistant",
                            content=combined_content if combined_content else None,
                            tool_calls=cast(list, tool_calls),  # Cast to satisfy type checker
                        )
                    else:
                        # Regular message without tool calls or non-assistant role
                        ai21_message = ChatMessage(role=role, content=combined_content)
                    formatted_messages.append(ai21_message)

                # Create ToolMessage instances (can coexist with text/tool_calls)
                if tool_messages:
                    for tool_message_data in tool_messages:
                        formatted_messages.append(ToolMessage(**tool_message_data))

        return formatted_messages

    def _format_request_message_tool_call(self, tool_use: dict[str, Any]) -> dict[str, Any]:
        """Format a single tool call for AI21.

        Args:
            tool_use: A strands toolUse content block.

        Returns:
            AI21-formatted tool call.
        """
        return {
            "id": tool_use["toolUseId"],
            "type": "function",
            "function": {"name": tool_use["name"], "arguments": json.dumps(tool_use["input"])},
        }

    def _format_request_message_tool_message(self, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Format a tool message for AI21.

        Args:
            tool_result: A strands toolResult content block.

        Returns:
            AI21-formatted tool message data.
        """
        # Extract content directly for ToolMessage
        result_parts = []
        for result_content in tool_result.get("content", []):
            if isinstance(result_content, dict):
                if "text" in result_content:
                    result_parts.append(result_content["text"])
                elif "json" in result_content:
                    result_parts.append(json.dumps(result_content["json"]))

        content = " ".join(result_parts) if result_parts else ""

        return {"role": "tool", "content": content, "tool_call_id": tool_result["toolUseId"]}

    def _format_request_tool_spec(self, tool_spec: ToolSpec) -> dict[str, Any]:
        """Format a single strands tool specification to AI21 format.

        Args:
            tool_spec: A strands tool specification.

        Returns:
            AI21-formatted tool specification.
        """
        return {
            "type": "function",
            "function": {
                "name": tool_spec["name"],
                "description": tool_spec["description"],
                "parameters": tool_spec["inputSchema"]["json"],
            },
        }

    def _format_request_message_content(self, content: dict[str, Any]) -> str:
        """Format content block to AI21-compatible text string.

        Args:
            content: A strands content block.

        Returns:
            Formatted text content string.

        Raises:
            TypeError: If content type is unsupported.
        """
        if "text" in content:
            return str(content["text"])

        # toolUse and toolResult are both handled by separate helper methods
        # in _format_request_messages for consistency

        # Handle unsupported content types
        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> Any:
        """Format a request for the AI21 model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            The formatted request for AI21 API.
        """
        request = {
            "model": self.config.get("model_id"),
            "messages": self._format_request_messages(messages, system_prompt),
            "stream": True,
        }

        if tool_specs:
            # Type ignore for AI21 API compatibility - expects different format than type checker assumes
            request["tools"] = [self._format_request_tool_spec(spec) for spec in tool_specs]  # type: ignore[misc]

        params = self.config.get("params")
        if params is not None:
            request.update(params)

        return request

    def format_chunk(self, event: dict[str, Any], start_time: Optional[float] = None) -> StreamEvent:
        """Format AI21 response events into standardized message chunks.

        Args:
            event: A response event from the AI21 model.
            start_time: Optional start time for latency calculation.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "text":
                    return {
                        "contentBlockStart": {
                            "start": {},
                            **({} if event.get("index") is None else {"contentBlockIndex": event["index"]}),
                        }
                    }

                # Handle tool use content start - AI21 streaming always returns dict format
                tool_data = event["data"]
                tool_name = tool_data.get("function", {}).get("name")
                tool_id = tool_data.get("id")

                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": tool_name,
                                "toolUseId": tool_id,
                            }
                        },
                        **({} if event.get("index") is None else {"contentBlockIndex": event["index"]}),
                    }
                }

            case "content_delta":
                if event["data_type"] == "text":
                    return {
                        "contentBlockDelta": {
                            "delta": {"text": event["data"]},
                            **({} if event.get("index") is None else {"contentBlockIndex": event["index"]}),
                        }
                    }

                # Handle tool use input arguments - AI21 streaming always returns dict format
                tool_data = event["data"]
                try:
                    function_dict = tool_data.get("function", {})
                    arguments = function_dict.get("arguments", None)

                    # Ensure arguments is always a string to avoid concatenation errors
                    # None should become empty string for proper concatenation
                    if arguments is None:
                        arguments = ""

                    # Skip empty argument chunks to avoid issues with streaming processor
                    if arguments == "":
                        return {
                            "contentBlockDelta": {
                                "delta": {},
                                **({} if event.get("index") is None else {"contentBlockIndex": event["index"]}),
                            }
                        }

                    # Defensive JSON validation - check if arguments form valid JSON fragment
                    # This prevents EventLoopException from malformed JSON during streaming
                    if arguments and isinstance(arguments, str):
                        # Replace problematic characters that could break streaming
                        if any(char in arguments for char in ["\x00", "\r\n\r\n"]):
                            logger.warning(
                                "Potentially malformed tool arguments detected, sanitizing: %s", repr(arguments)
                            )
                            arguments = arguments.replace("\x00", "").replace("\r\n\r\n", " ")

                    return {
                        "contentBlockDelta": {
                            "delta": {"toolUse": {"input": arguments}},
                            **({} if event.get("index") is None else {"contentBlockIndex": event["index"]}),
                        }
                    }

                except Exception as e:
                    # Catch any unexpected errors in tool argument processing to prevent EventLoopException
                    logger.error("Error processing tool arguments, returning empty delta to prevent crash: %s", e)
                    return {
                        "contentBlockDelta": {
                            "delta": {},
                            **({} if event.get("index") is None else {"contentBlockIndex": event["index"]}),
                        }
                    }

            case "content_stop":
                return {
                    "contentBlockStop": {
                        **({} if event.get("index") is None else {"contentBlockIndex": event["index"]})
                    }
                }

            case "message_stop":
                match event["data"]:
                    case "stop":
                        return {"messageStop": {"stopReason": "end_turn"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
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
                            "latencyMs": int((time.time() - start_time) * 1000) if start_time else 0,
                        },
                    }
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    def _handle_text_content(
        self, choice: Any, content_started: bool, current_content_index: int, start_time: float
    ) -> tuple[bool, list[StreamEvent]]:
        """Handle text content from a response chunk."""
        events_to_yield = []

        if choice.delta and choice.delta.content:
            if not content_started:
                events_to_yield.append(
                    self.format_chunk(
                        {"chunk_type": "content_start", "data_type": "text", "index": current_content_index}, start_time
                    )
                )
                content_started = True

            events_to_yield.append(
                self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "text",
                        "data": choice.delta.content,
                        "index": current_content_index,
                    },
                    start_time,
                )
            )

        return content_started, events_to_yield

    def _process_tool_arguments(
        self,
        tool_call: dict[str, Any],
        arguments: str,
        tool_index: int,
        tool_argument_chunks: dict[int, list],
        tool_complete_json: dict[int, str],
    ) -> bool:
        """Process tool arguments with AI21-specific JSON handling."""
        # Initialize buffer for this tool if not exists
        if tool_index not in tool_argument_chunks:
            tool_argument_chunks[tool_index] = []

        # Check if this is a complete valid JSON chunk
        if arguments.strip().startswith("{") and arguments.strip().endswith("}") and len(arguments.strip()) > 10:
            try:
                json.loads(arguments.strip())
                # This is complete JSON - store it
                tool_complete_json[tool_index] = arguments.strip()
                return True  # Signal that this is complete JSON
            except json.JSONDecodeError:
                # Not complete JSON, treat as incremental
                pass

        # Buffer this incremental chunk
        tool_argument_chunks[tool_index].append((tool_call, arguments))
        return False

    def _finalize_buffered_tool_arguments(
        self,
        tool_argument_chunks: dict[int, list],
        tool_complete_json: dict[int, str],
        tool_calls: dict[int, str],
        tool_metadata: dict[str, dict[str, Any]],
        current_content_index: int,
        start_time: float,
    ) -> list[StreamEvent]:
        """Process all buffered tool arguments at the end of streaming."""
        events_to_yield = []

        for tool_index in tool_argument_chunks:
            if tool_index in tool_complete_json:
                # Send the complete JSON only
                current_tool_id = tool_calls.get(tool_index)
                if current_tool_id and current_tool_id in tool_metadata:
                    complete_tool_call = {
                        "function": {
                            "arguments": tool_complete_json[tool_index],
                            "name": tool_metadata[current_tool_id]["name"],
                        },
                        "id": tool_metadata[current_tool_id]["id"],
                        "index": tool_index,
                    }
                    events_to_yield.append(
                        self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "tool",
                                "data": complete_tool_call,
                                "index": current_content_index,
                            },
                            start_time,
                        )
                    )
            else:
                # Send all incremental chunks for this tool
                for buffered_tool_call, buffered_arguments in tool_argument_chunks[tool_index]:
                    if buffered_arguments:  # Skip empty arguments
                        events_to_yield.append(
                            self.format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": "tool",
                                    "data": buffered_tool_call,
                                    "index": current_content_index,
                                },
                                start_time,
                            )
                        )

        return events_to_yield

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the AI21 model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: If the request is throttled by AI21.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            response = await self.client.chat.completions.create(**request)
            logger.debug("got response from model")

            start_time = time.time()
            yield self.format_chunk({"chunk_type": "message_start"}, start_time)

            content_started = False
            tool_calls: dict[int, str] = {}  # Map tool index to tool_id
            tool_metadata: dict[str, dict] = {}  # Store tool name/id for each tool_id
            tool_argument_chunks: dict[int, list] = {}  # Buffer all argument chunks for each tool
            tool_complete_json: dict[int, str] = {}  # Store complete JSON for each tool if found
            current_content_index = 0

            async for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # Handle text content
                content_started, text_events = self._handle_text_content(
                    choice, content_started, current_content_index, start_time
                )
                for event in text_events:
                    yield event

                # Handle tool calls
                if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        # AI21 uses tool index to track tool calls, not always tool ID
                        tool_index = (
                            tool_call.get("index", 0) if isinstance(tool_call, dict) else getattr(tool_call, "index", 0)
                        )
                        tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)

                        # Check if this chunk has tool metadata (name/id) - indicates start of new tool
                        has_tool_metadata = (
                            isinstance(tool_call, dict)
                            and tool_call.get("function", {}).get("name")
                            and tool_call.get("id")
                        )

                        # Store tool metadata when we first see this tool
                        if (
                            has_tool_metadata
                            and tool_id is not None
                            and isinstance(tool_id, str)
                            and tool_id not in tool_metadata
                        ):
                            if content_started:
                                yield self.format_chunk(
                                    {"chunk_type": "content_stop", "index": current_content_index}, start_time
                                )
                                content_started = False
                                current_content_index += 1

                            # Extract and store tool metadata
                            tool_name = tool_call.get("function", {}).get("name")
                            tool_metadata[tool_id] = {"name": tool_name, "id": tool_id, "index": tool_index}

                            yield self.format_chunk(
                                {
                                    "chunk_type": "content_start",
                                    "data_type": "tool",
                                    "data": tool_call,
                                    "index": current_content_index,
                                },
                                start_time,
                            )
                            tool_calls[tool_index] = tool_id
                        else:
                            # For subsequent chunks (incremental or complete), use the stored metadata
                            current_tool_id = tool_calls.get(tool_index)
                            if (
                                current_tool_id is not None
                                and isinstance(current_tool_id, str)
                                and current_tool_id in tool_metadata
                            ):
                                # Enrich the tool_call with stored metadata
                                if isinstance(tool_call, dict):
                                    enriched_tool_call = tool_call.copy()
                                    if "function" not in enriched_tool_call:
                                        enriched_tool_call["function"] = {}
                                    if "name" not in enriched_tool_call.get("function", {}):
                                        tool_name = tool_metadata[current_tool_id].get("name")
                                        if tool_name is not None:
                                            enriched_tool_call["function"]["name"] = str(tool_name)
                                    if "id" not in enriched_tool_call:
                                        tool_id_value = tool_metadata[current_tool_id].get("id")
                                        if tool_id_value is not None:
                                            enriched_tool_call["id"] = str(tool_id_value)
                                    tool_call = enriched_tool_call

                                # Process tool call arguments
                                arguments = tool_call.get("function", {}).get("arguments", "")
                                if arguments:  # Only process if there are actual arguments
                                    is_complete = self._process_tool_arguments(
                                        tool_call, arguments, tool_index, tool_argument_chunks, tool_complete_json
                                    )
                                    if is_complete:
                                        continue  # Skip yielding for complete JSON, handle at end

                if choice.finish_reason:
                    break

            # Process all buffered tool arguments
            buffered_events = self._finalize_buffered_tool_arguments(
                tool_argument_chunks, tool_complete_json, tool_calls, tool_metadata, current_content_index, start_time
            )
            for event in buffered_events:
                yield event

            # Close any open content blocks
            if content_started:
                yield self.format_chunk({"chunk_type": "content_stop", "index": current_content_index}, start_time)

            if tool_calls:
                yield self.format_chunk({"chunk_type": "content_stop", "index": current_content_index}, start_time)

            # Message stop
            finish_reason = getattr(choice, "finish_reason", "stop") if "choice" in locals() else "stop"
            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason}, start_time)

            # Usage metadata
            if "chunk" in locals() and hasattr(chunk, "usage") and chunk.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": chunk.usage}, start_time)

        except TooManyRequestsError as e:
            raise ModelThrottledException(str(e)) from e

        logger.debug("finished streaming response from model")

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
            ValueError: If no valid tool use or tool use input was found in the AI21 response.
        """
        tool_spec = convert_pydantic_to_tool_spec(output_model)
        response = self.stream(messages=prompt, tool_specs=[tool_spec], system_prompt=system_prompt, **kwargs)
        async for event in process_stream(response):
            yield event

        stop_reason, messages, _, _ = event["stop"]

        content = messages["content"]

        for block in content:
            if block.get("toolUse"):
                if block["toolUse"]["name"] == tool_spec["name"]:
                    yield {"output": output_model(**block["toolUse"]["input"])}
                    return

        # If no tool use found, check stop_reason and provide helpful error
        if stop_reason != "tool_use":
            raise ValueError(
                f"AI21 did not use tools (stop_reason: '{stop_reason}'). "
                f"AI21 may have responded with text instead of calling the tool."
            )
        else:
            raise ValueError("No valid tool use or tool use input was found in the AI21 response.")
