"""Utilities for handling streaming responses from language models."""

import json
import logging
import time
import warnings
from typing import Any, AsyncGenerator, AsyncIterable, Optional

from ..models.model import Model
from ..tools import InvalidToolUseNameException
from ..tools.tools import validate_tool_use_name
from ..types._events import (
    CitationStreamEvent,
    ModelStopReason,
    ModelStreamChunkEvent,
    ModelStreamEvent,
    ReasoningRedactedContentStreamEvent,
    ReasoningSignatureStreamEvent,
    ReasoningTextStreamEvent,
    TextStreamEvent,
    ToolUseStreamEvent,
    TypedEvent,
)
from ..types.citations import CitationsContentBlock
from ..types.content import ContentBlock, Message, Messages, SystemContentBlock
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
from ..types.tools import ToolSpec, ToolUse

logger = logging.getLogger(__name__)


def _normalize_messages(messages: Messages) -> Messages:
    """Remove or replace blank text in message content.

    Args:
        messages: Conversation messages to update.

    Returns:
        Updated messages.
    """
    removed_blank_message_content_text = False
    replaced_blank_message_content_text = False
    replaced_tool_names = False

    for message in messages:
        # only modify assistant messages
        if "role" in message and message["role"] != "assistant":
            continue
        if "content" in message:
            content = message["content"]
            if len(content) == 0:
                content.append({"text": "[blank text]"})
                continue

            has_tool_use = False

            # Ensure the tool-uses always have valid names before sending
            # https://github.com/strands-agents/sdk-python/issues/1069
            for item in content:
                if "toolUse" in item:
                    has_tool_use = True
                    tool_use: ToolUse = item["toolUse"]

                    try:
                        validate_tool_use_name(tool_use)
                    except InvalidToolUseNameException:
                        tool_use["name"] = "INVALID_TOOL_NAME"
                        replaced_tool_names = True

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
    if replaced_tool_names:
        logger.debug("replaced invalid tool name")

    return messages


def remove_blank_messages_content_text(messages: Messages) -> Messages:
    """Remove or replace blank text in message content.

    !!deprecated!!
        This function is deprecated and will be removed in a future version.

    Args:
        messages: Conversation messages to update.

    Returns:
        Updated messages.
    """
    warnings.warn(
        "remove_blank_messages_content_text is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    removed_blank_message_content_text = False
    replaced_blank_message_content_text = False

    for message in messages:
        # only modify assistant messages
        if "role" in message and message["role"] != "assistant":
            continue
        if "content" in message:
            content = message["content"]
            has_tool_use = any("toolUse" in item for item in content)
            if len(content) == 0:
                content.append({"text": "[blank text]"})
                continue

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


def handle_content_block_start(event: ContentBlockStartEvent) -> dict[str, Any]:
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
    event: ContentBlockDeltaEvent, state: dict[str, Any]
) -> tuple[dict[str, Any], ModelStreamEvent]:
    """Handles content block delta updates by appending text, tool input, or reasoning content to the state.

    Args:
        event: Delta event.
        state: The current state of message processing.

    Returns:
        Updated state with appended text or tool input.
    """
    delta_content = event["delta"]

    typed_event: ModelStreamEvent = ModelStreamEvent({})

    if "toolUse" in delta_content:
        if "input" not in state["current_tool_use"]:
            state["current_tool_use"]["input"] = ""

        state["current_tool_use"]["input"] += delta_content["toolUse"]["input"]
        typed_event = ToolUseStreamEvent(delta_content, state["current_tool_use"])

    elif "text" in delta_content:
        state["text"] += delta_content["text"]
        typed_event = TextStreamEvent(text=delta_content["text"], delta=delta_content)

    elif "citation" in delta_content:
        if "citationsContent" not in state:
            state["citationsContent"] = []

        state["citationsContent"].append(delta_content["citation"])
        typed_event = CitationStreamEvent(delta=delta_content, citation=delta_content["citation"])

    elif "reasoningContent" in delta_content:
        if "text" in delta_content["reasoningContent"]:
            if "reasoningText" not in state:
                state["reasoningText"] = ""

            state["reasoningText"] += delta_content["reasoningContent"]["text"]
            typed_event = ReasoningTextStreamEvent(
                reasoning_text=delta_content["reasoningContent"]["text"],
                delta=delta_content,
            )

        elif "signature" in delta_content["reasoningContent"]:
            if "signature" not in state:
                state["signature"] = ""

            state["signature"] += delta_content["reasoningContent"]["signature"]
            typed_event = ReasoningSignatureStreamEvent(
                reasoning_signature=delta_content["reasoningContent"]["signature"],
                delta=delta_content,
            )

        elif redacted_content := delta_content["reasoningContent"].get("redactedContent"):
            state["redactedContent"] = state.get("redactedContent", b"") + redacted_content
            typed_event = ReasoningRedactedContentStreamEvent(redacted_content=redacted_content, delta=delta_content)

    return state, typed_event


def handle_content_block_stop(state: dict[str, Any]) -> dict[str, Any]:
    """Handles the end of a content block by finalizing tool usage, text content, or reasoning content.

    Args:
        state: The current state of message processing.

    Returns:
        Updated state with finalized content block.
    """
    content: list[ContentBlock] = state["content"]

    current_tool_use = state["current_tool_use"]
    text = state["text"]
    reasoning_text = state["reasoningText"]
    citations_content = state["citationsContent"]
    redacted_content = state.get("redactedContent")

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
        if citations_content:
            citations_block: CitationsContentBlock = {"citations": citations_content, "content": [{"text": text}]}
            content.append({"citationsContent": citations_block})
            state["citationsContent"] = []
        else:
            content.append({"text": text})
        state["text"] = ""

    elif reasoning_text:
        content_block: ContentBlock = {
            "reasoningContent": {
                "reasoningText": {
                    "text": state["reasoningText"],
                }
            }
        }

        if "signature" in state:
            content_block["reasoningContent"]["reasoningText"]["signature"] = state["signature"]

        content.append(content_block)
        state["reasoningText"] = ""
    elif redacted_content:
        content.append({"reasoningContent": {"redactedContent": redacted_content}})
        state["redactedContent"] = b""

    return state


def handle_message_stop(event: MessageStopEvent) -> StopReason:
    """Handles the end of a message by returning the stop reason.

    Args:
        event: Stop event.

    Returns:
        The reason for stopping the stream.
    """
    return event["stopReason"]


def handle_redact_content(event: RedactContentEvent, state: dict[str, Any]) -> None:
    """Handles redacting content from the input or output.

    Args:
        event: Redact Content Event.
        state: The current state of message processing.
    """
    if event.get("redactAssistantContentMessage") is not None:
        state["message"]["content"] = [{"text": event["redactAssistantContentMessage"]}]


def extract_usage_metrics(event: MetadataEvent, time_to_first_byte_ms: int | None = None) -> tuple[Usage, Metrics]:
    """Extracts usage metrics from the metadata chunk.

    Args:
        event: metadata.
        time_to_first_byte_ms: time to get the first byte from the model in milliseconds

    Returns:
        The extracted usage metrics and latency.
    """
    # MetadataEvent has total=False, making all fields optional, but Usage and Metrics types
    # have Required fields. Provide defaults to handle cases where custom models don't
    # provide usage/metrics (e.g., when latency info is unavailable).
    usage = Usage(**{"inputTokens": 0, "outputTokens": 0, "totalTokens": 0, **event.get("usage", {})})
    metrics = Metrics(**{"latencyMs": 0, **event.get("metrics", {})})
    if time_to_first_byte_ms:
        metrics["timeToFirstByteMs"] = time_to_first_byte_ms

    return usage, metrics


async def process_stream(
    chunks: AsyncIterable[StreamEvent], start_time: float | None = None
) -> AsyncGenerator[TypedEvent, None]:
    """Processes the response stream from the API, constructing the final message and extracting usage metrics.

    Args:
        chunks: The chunks of the response stream from the model.
        start_time: Time when the model request is initiated

    Yields:
        The reason for stopping, the constructed message, and the usage metrics.
    """
    stop_reason: StopReason = "end_turn"
    first_byte_time = None

    state: dict[str, Any] = {
        "message": {"role": "assistant", "content": []},
        "text": "",
        "current_tool_use": {},
        "reasoningText": "",
        "citationsContent": [],
    }
    state["content"] = state["message"]["content"]

    usage: Usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
    metrics: Metrics = Metrics(latencyMs=0, timeToFirstByteMs=0)

    async for chunk in chunks:
        # Track first byte time when we get first content
        if first_byte_time is None and ("contentBlockDelta" in chunk or "contentBlockStart" in chunk):
            first_byte_time = time.time()
        yield ModelStreamChunkEvent(chunk=chunk)

        if "messageStart" in chunk:
            state["message"] = handle_message_start(chunk["messageStart"], state["message"])
        elif "contentBlockStart" in chunk:
            state["current_tool_use"] = handle_content_block_start(chunk["contentBlockStart"])
        elif "contentBlockDelta" in chunk:
            state, typed_event = handle_content_block_delta(chunk["contentBlockDelta"], state)
            yield typed_event
        elif "contentBlockStop" in chunk:
            state = handle_content_block_stop(state)
        elif "messageStop" in chunk:
            stop_reason = handle_message_stop(chunk["messageStop"])
        elif "metadata" in chunk:
            time_to_first_byte_ms = (
                int(1000 * (first_byte_time - start_time)) if (start_time and first_byte_time) else None
            )
            usage, metrics = extract_usage_metrics(chunk["metadata"], time_to_first_byte_ms)
        elif "redactContent" in chunk:
            handle_redact_content(chunk["redactContent"], state)

    yield ModelStopReason(stop_reason=stop_reason, message=state["message"], usage=usage, metrics=metrics)


async def stream_messages(
    model: Model,
    system_prompt: Optional[str],
    messages: Messages,
    tool_specs: list[ToolSpec],
    *,
    tool_choice: Optional[Any] = None,
    system_prompt_content: Optional[list[SystemContentBlock]] = None,
    **kwargs: Any,
) -> AsyncGenerator[TypedEvent, None]:
    """Streams messages to the model and processes the response.

    Args:
        model: Model provider.
        system_prompt: The system prompt string, used for backwards compatibility with models that expect it.
        messages: List of messages to send.
        tool_specs: The list of tool specs.
        tool_choice: Optional tool choice constraint for forcing specific tool usage.
        system_prompt_content: The authoritative system prompt content blocks that always contains the
            system prompt data.
        **kwargs: Additional keyword arguments for future extensibility.

    Yields:
        The reason for stopping, the final message, and the usage metrics
    """
    logger.debug("model=<%s> | streaming messages", model)

    messages = _normalize_messages(messages)
    start_time = time.time()

    chunks = model.stream(
        messages,
        tool_specs if tool_specs else None,
        system_prompt,
        tool_choice=tool_choice,
        system_prompt_content=system_prompt_content,
    )

    async for event in process_stream(chunks, start_time):
        yield event
