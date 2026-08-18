"""Conversion functions between Strands and A2A types."""

from collections.abc import Sequence
from typing import cast
from uuid import uuid4

from a2a.types import Message as A2AMessage
from a2a.types import Part, Role, TaskState

from ...agent.agent_result import AgentResult
from ...telemetry.metrics import EventLoopMetrics
from ...types.a2a import A2AResponse
from ...types.agent import AgentInput
from ...types.content import ContentBlock, Message
from ...types.event_loop import StopReason

# Mapping from A2A TaskState to Strands stop_reason
_STATE_TO_STOP_REASON: dict[TaskState, StopReason] = {
    TaskState.TASK_STATE_COMPLETED: "end_turn",
    TaskState.TASK_STATE_FAILED: "end_turn",
    TaskState.TASK_STATE_CANCELED: "end_turn",
    TaskState.TASK_STATE_REJECTED: "end_turn",
    TaskState.TASK_STATE_INPUT_REQUIRED: "interrupt",
    TaskState.TASK_STATE_AUTH_REQUIRED: "interrupt",
}


def _task_state_to_str(task_state: TaskState) -> str:
    """Render a TaskState as the kebab-case wire string used by the A2A v0.3 JSON spec.

    v1.0's protobuf enum names are SCREAMING_SNAKE_CASE (e.g. ``TASK_STATE_INPUT_REQUIRED``);
    this keeps ``AgentResult.state["a2a_task_state"]`` stable for existing Strands callers.
    """
    name: str = TaskState.Name(task_state)  # type: ignore[attr-defined]
    return name.removeprefix("TASK_STATE_").lower().replace("_", "-")


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
            role=Role.ROLE_USER,
            parts=[Part(text=prompt)],
            message_id=message_id,
        )

    if isinstance(prompt, list) and prompt and (isinstance(prompt[0], dict)):
        # Check for interrupt responses - not supported in A2A
        if "interruptResponse" in prompt[0]:
            raise ValueError("InterruptResponseContent is not supported for A2AAgent")

        if "role" in prompt[0]:
            for msg in reversed(prompt):
                if msg.get("role") == "user":
                    content = cast(list[ContentBlock], msg.get("content", []))
                    parts = convert_content_blocks_to_parts(content)
                    return A2AMessage(
                        role=Role.ROLE_USER,
                        parts=parts,
                        message_id=message_id,
                    )
        else:
            parts = convert_content_blocks_to_parts(cast(list[ContentBlock], prompt))
            return A2AMessage(
                role=Role.ROLE_USER,
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
            parts.append(Part(text=block["text"]))
    return parts


def _extract_task_state(response: A2AResponse) -> TaskState | None:
    """Extract the task state carried by a single A2A StreamResponse, if any.

    Args:
        response: A single StreamResponse from the A2A event stream.

    Returns:
        The TaskState if this response carries one (a ``task`` or ``status_update``), else None.
    """
    if response.HasField("status_update"):
        return response.status_update.status.state
    if response.HasField("task"):
        return response.task.status.state
    return None


def _parts_to_content(parts: Sequence[Part]) -> list[ContentBlock]:
    """Convert a sequence of A2A text Parts into Strands ContentBlocks, dropping non-text parts."""
    return [{"text": part.text} for part in parts if part.HasField("text")]


def convert_responses_to_agent_result(responses: Sequence[A2AResponse]) -> AgentResult:
    """Convert the full sequence of A2A StreamResponse events from one call into an AgentResult.

    A2A v1.0 streams flat ``StreamResponse`` events (``task`` | ``message`` | ``status_update`` |
    ``artifact_update``) rather than pairing a cumulative Task with each update, so the final
    content is reconstructed across the whole stream:
    - ``artifact_update`` parts accumulate: a compliant-streaming server sends incremental deltas,
      while a non-compliant server that sends the full text once still works, as that is then the
      only chunk.
    - ``status_update`` messages are progress narration on non-compliant-streaming servers and
      would duplicate the artifact content above, so they are only used when no artifact content
      was found (e.g. a status-only terminal event such as a rejection or failure message).
    - a bare ``task`` or ``message`` response (no separate update events) supplies content
      directly.

    Maps A2A task lifecycle states to appropriate Strands stop_reasons:
    - completed → end_turn
    - failed → end_turn (with error content)
    - canceled → end_turn (with cancellation info)
    - rejected → end_turn (with rejection info)
    - input_required → interrupt (agent needs user input)
    - auth_required → interrupt (agent needs authentication)

    Args:
        responses: All StreamResponse events observed for one ``send_message`` call, in order.

    Returns:
        AgentResult with extracted content and metadata.
    """
    artifact_content: list[ContentBlock] = []
    status_message_content: list[ContentBlock] = []
    task_content: list[ContentBlock] = []
    message_content: list[ContentBlock] = []
    task_state: TaskState | None = None

    for response in responses:
        state = _extract_task_state(response)
        if state is not None:
            task_state = state

        if response.HasField("artifact_update"):
            artifact_content.extend(_parts_to_content(response.artifact_update.artifact.parts))
        elif response.HasField("status_update"):
            if response.status_update.status.HasField("message"):
                status_message_content = _parts_to_content(response.status_update.status.message.parts)
        elif response.HasField("task"):
            task_content = [
                content for artifact in response.task.artifacts for content in _parts_to_content(artifact.parts)
            ]
        elif response.HasField("message"):
            message_content = _parts_to_content(response.message.parts)

    content = artifact_content or status_message_content or task_content or message_content
    stop_reason: StopReason = _STATE_TO_STOP_REASON.get(task_state, "end_turn") if task_state else "end_turn"

    message: Message = {
        "role": "assistant",
        "content": content,
    }

    # Build state dict with A2A metadata
    state_dict: dict[str, str] = {}
    if task_state is not None:
        state_dict["a2a_task_state"] = _task_state_to_str(task_state)

    return AgentResult(
        stop_reason=stop_reason,
        message=message,
        metrics=EventLoopMetrics(),
        state=state_dict,
    )
