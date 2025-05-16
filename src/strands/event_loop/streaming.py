"""Utilities for handling streaming responses from language models."""

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..types.content import ContentBlock, Message, Messages
from ..types.models import Model
from ..types.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStart,
    ContentBlockStartEvent,
    MessageStartEvent,
    MessageStopEvent,
    MetadataEvent,
    Metrics,
    RedactContentEvent,
    StopReason,
    StreamEvent,
    Usage,
)
from ..types.tools import ToolConfig, ToolUse

logger = logging.getLogger(__name__)


def remove_blank_messages_content_text(messages: Messages) -> Messages:
    """Remove or replace blank text in message content.

    Args:
        messages: Conversation messages to update.

    Returns:
        Updated messages.
    """
    removed_blank_message_content_text = False
    replaced_blank_message_content_text = False

    for message in messages:
        # only modify assistant messages
        if "role" in message and message["role"] != "assistant":
            continue

        if "content" in message:
            content = message["content"]
            has_tool_use = any("toolUse" in item for item in content)

            if has_tool_use:
                # Remove blank 'text' items for assistant messages
                before_len = len(content)
                content[:] = [item for item in content if "text" not in item or item["text"].strip()]
                if not removed_blank_message_content_text and before_len != len(content):
                    removed_blank_message_content_text = True
            else:
                # Replace blank 'text' with '[blank text]' for assistant messages
                for item in content:
                    if "text" in item and not item["text"].strip():
                        replaced_blank_message_content_text = True
                        item["text"] = "[blank text]"

    if removed_blank_message_content_text:
        logger.debug("removed blank message context text")
    if replaced_blank_message_content_text:
        logger.debug("replaced blank message context text")

    return messages


def handle_message_start(event: MessageStartEvent, message: Message) -> Message:
    """Handles the start of a message by setting the role in the message dictionary.

    Args:
        event: A message start event.
        message: The message dictionary being constructed.

    Returns:
        Updated message dictionary with the role set.
    """
    message["role"] = event["role"]
    return message


def handle_content_block_start(event: ContentBlockStartEvent) -> Dict[str, Any]:
    """Handles the start of a content block by extracting tool usage information if any.

    Args:
        event: Start event.

    Returns:
        Dictionary with tool use id and name if tool use request, empty dictionary otherwise.
    """
    start: ContentBlockStart = event["start"]
    current_tool_use = {}

    if "toolUse" in start and start["toolUse"]:
        tool_use_data = start["toolUse"]
        current_tool_use["toolUseId"] = tool_use_data["toolUseId"]
        current_tool_use["name"] = tool_use_data["name"]
        current_tool_use["input"] = ""

    return current_tool_use


def handle_content_block_delta(
    event: ContentBlockDeltaEvent, state: Dict[str, Any], callback_handler: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Handles content block delta updates by appending text, tool input, or reasoning content to the state.

    Args:
        event: Delta event.
        state: The current state of message processing.
        callback_handler: Callback for processing events as they happen.
        **kwargs: Additional keyword arguments to pass to the callback handler.

    Returns:
        Updated state with appended text or tool input.
    """
    delta_content = event["delta"]

    if "toolUse" in delta_content:
        if "input" not in state["current_tool_use"]:
            state["current_tool_use"]["input"] = ""

        state["current_tool_use"]["input"] += delta_content["toolUse"]["input"]
        callback_handler(delta=delta_content, current_tool_use=state["current_tool_use"], **kwargs)

    elif "text" in delta_content:
        state["text"] += delta_content["text"]
        callback_handler(data=delta_content["text"], delta=delta_content, **kwargs)

    elif "reasoningContent" in delta_content:
        if "text" in delta_content["reasoningContent"]:
            if "reasoningText" not in state:
                state["reasoningText"] = ""

            state["reasoningText"] += delta_content["reasoningContent"]["text"]
            callback_handler(
                reasoningText=delta_content["reasoningContent"]["text"],
                delta=delta_content,
                reasoning=True,
                **kwargs,
            )

        elif "signature" in delta_content["reasoningContent"]:
            if "signature" not in state:
                state["signature"] = ""

            state["signature"] += delta_content["reasoningContent"]["signature"]
            callback_handler(
                reasoning_signature=delta_content["reasoningContent"]["signature"],
                delta=delta_content,
                reasoning=True,
                **kwargs,
            )

    return state


def handle_content_block_stop(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the end of a content block by finalizing tool usage, text content, or reasoning content.

    Args:
        state: The current state of message processing.

    Returns:
        Updated state with finalized content block.
    """
    content: List[ContentBlock] = state["content"]

    current_tool_use = state["current_tool_use"]
    text = state["text"]
    reasoning_text = state["reasoningText"]

    if current_tool_use:
        if "input" not in current_tool_use:
            current_tool_use["input"] = ""

        try:
            current_tool_use["input"] = json.loads(current_tool_use["input"])
        except ValueError:
            current_tool_use["input"] = {}

        tool_use_id = current_tool_use["toolUseId"]
        tool_use_name = current_tool_use["name"]

        tool_use = ToolUse(
            toolUseId=tool_use_id,
            name=tool_use_name,
            input=current_tool_use["input"],
        )
        content.append({"toolUse": tool_use})
        state["current_tool_use"] = {}

    elif text:
        content.append({"text": text})
        state["text"] = ""

    elif reasoning_text:
        content.append(
            {
                "reasoningContent": {
                    "reasoningText": {
                        "text": state["reasoningText"],
                        "signature": state["signature"],
                    }
                }
            }
        )
        state["reasoningText"] = ""

    return state


def handle_message_stop(event: MessageStopEvent) -> StopReason:
    """Handles the end of a message by returning the stop reason.

    Args:
        event: Stop event.

    Returns:
        The reason for stopping the stream.
    """
    return event["stopReason"]


def handle_redact_content(event: RedactContentEvent, messages: Messages, state: Dict[str, Any]) -> None:
    """Handles redacting content from the input or output.

    Args:
        event: Redact Content Event.
        messages: Agent messages.
        state: The current state of message processing.
    """
    if event.get("redactUserContentMessage") is not None:
        messages[-1]["content"] = [{"text": event["redactUserContentMessage"]}]  # type: ignore

    if event.get("redactAssistantContentMessage") is not None:
        state["message"]["content"] = [{"text": event["redactAssistantContentMessage"]}]


def extract_usage_metrics(event: MetadataEvent) -> Tuple[Usage, Metrics]:
    """Extracts usage metrics from the metadata chunk.

    Args:
        event: metadata.

    Returns:
        The extracted usage metrics and latency.
    """
    usage = Usage(**event["usage"])
    metrics = Metrics(**event["metrics"])

    return usage, metrics


def process_stream(
    chunks: Iterable[StreamEvent],
    callback_handler: Any,
    messages: Messages,
    **kwargs: Any,
) -> Tuple[StopReason, Message, Usage, Metrics, Any]:
    """Processes the response stream from the API, constructing the final message and extracting usage metrics.

    Args:
        chunks: The chunks of the response stream from the model.
        callback_handler: Callback for processing events as they happen.
        messages: The agents messages.
        **kwargs: Additional keyword arguments that will be passed to the callback handler.
            And also returned in the request_state.

    Returns:
        The reason for stopping, the constructed message, the usage metrics, and the updated request state.
    """
    stop_reason: StopReason = "end_turn"

    state: Dict[str, Any] = {
        "message": {"role": "assistant", "content": []},
        "text": "",
        "current_tool_use": {},
        "reasoningText": "",
        "signature": "",
    }
    state["content"] = state["message"]["content"]

    usage: Usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
    metrics: Metrics = Metrics(latencyMs=0)

    kwargs.setdefault("request_state", {})

    for chunk in chunks:
        # Callback handler call here allows each event to be visible to the caller
        callback_handler(event=chunk)

        if "messageStart" in chunk:
            state["message"] = handle_message_start(chunk["messageStart"], state["message"])
        elif "contentBlockStart" in chunk:
            state["current_tool_use"] = handle_content_block_start(chunk["contentBlockStart"])
        elif "contentBlockDelta" in chunk:
            state = handle_content_block_delta(chunk["contentBlockDelta"], state, callback_handler, **kwargs)
        elif "contentBlockStop" in chunk:
            state = handle_content_block_stop(state)
        elif "messageStop" in chunk:
            stop_reason = handle_message_stop(chunk["messageStop"])
        elif "metadata" in chunk:
            usage, metrics = extract_usage_metrics(chunk["metadata"])
        elif "redactContent" in chunk:
            handle_redact_content(chunk["redactContent"], messages, state)

    return stop_reason, state["message"], usage, metrics, kwargs["request_state"]


def stream_messages(
    model: Model,
    system_prompt: Optional[str],
    messages: Messages,
    tool_config: Optional[ToolConfig],
    callback_handler: Any,
    **kwargs: Any,
) -> Tuple[StopReason, Message, Usage, Metrics, Any]:
    """Streams messages to the model and processes the response.

    Args:
        model: Model provider.
        system_prompt: The system prompt to send.
        messages: List of messages to send.
        tool_config: Configuration for the tools to use.
        callback_handler: Callback for processing events as they happen.
        **kwargs: Additional keyword arguments that will be passed to the callback handler.
            And also returned in the request_state.

    Returns:
        The reason for stopping, the final message, the usage metrics, and updated request state.
    """
    logger.debug("model=<%s> | streaming messages", model)

    messages = remove_blank_messages_content_text(messages)
    tool_specs = [tool["toolSpec"] for tool in tool_config.get("tools", [])] or None if tool_config else None

    chunks = model.converse(messages, tool_specs, system_prompt)
    return process_stream(chunks, callback_handler, messages, **kwargs)
