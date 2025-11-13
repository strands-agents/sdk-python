"""Conversion functions between Strands and A2A types."""

from typing import TypeAlias, cast
from uuid import uuid4

from a2a.types import Message as A2AMessage
from a2a.types import Part, Role, Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent, TextPart

from ...agent.agent_result import AgentResult
from ...telemetry.metrics import EventLoopMetrics
from ...types.agent import AgentInput
from ...types.content import ContentBlock, Message

A2AResponse: TypeAlias = tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None] | A2AMessage


def convert_input_to_message(prompt: AgentInput) -> A2AMessage:
    """Convert AgentInput to A2A Message.

    Args:
        prompt: Input in various formats (string, message list, or content blocks).

    Returns:
        A2AMessage ready to send to the remote agent.

    Raises:
        ValueError: If prompt format is unsupported.
    """
    message_id = uuid4().hex

    if isinstance(prompt, str):
        return A2AMessage(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(kind="text", text=prompt))],
            message_id=message_id,
        )

    if isinstance(prompt, list) and prompt and (isinstance(prompt[0], dict)):
        if "role" in prompt[0]:
            for msg in reversed(prompt):
                if msg.get("role") == "user":
                    content = cast(list[ContentBlock], msg.get("content", []))
                    parts = convert_content_blocks_to_parts(content)
                    return A2AMessage(
                        kind="message",
                        role=Role.user,
                        parts=parts,
                        message_id=message_id,
                    )
        else:
            parts = convert_content_blocks_to_parts(cast(list[ContentBlock], prompt))
            return A2AMessage(
                kind="message",
                role=Role.user,
                parts=parts,
                message_id=message_id,
            )

    raise ValueError(f"Unsupported input type: {type(prompt)}")


def convert_content_blocks_to_parts(content_blocks: list[ContentBlock]) -> list[Part]:
    """Convert Strands ContentBlocks to A2A Parts.

    Args:
        content_blocks: List of Strands content blocks.

    Returns:
        List of A2A Part objects.
    """
    parts = []
    for block in content_blocks:
        if "text" in block:
            parts.append(Part(TextPart(kind="text", text=block["text"])))
    return parts


def convert_response_to_agent_result(response: A2AResponse) -> AgentResult:
    """Convert A2A response to AgentResult.

    Args:
        response: A2A response (either A2AMessage or tuple of task and update event).

    Returns:
        AgentResult with extracted content and metadata.
    """
    content: list[ContentBlock] = []

    if isinstance(response, tuple) and len(response) == 2:
        task, update_event = response
        if update_event is None and task and hasattr(task, "artifacts") and task.artifacts is not None:
            for artifact in task.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            content.append({"text": part.root.text})
    elif isinstance(response, A2AMessage):
        for part in response.parts:
            if hasattr(part, "root") and hasattr(part.root, "text"):
                content.append({"text": part.root.text})

    message: Message = {
        "role": "assistant",
        "content": content,
    }

    return AgentResult(
        stop_reason="end_turn",
        message=message,
        metrics=EventLoopMetrics(),
        state={},
    )
