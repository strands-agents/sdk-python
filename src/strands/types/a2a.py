"""Additional A2A types."""

from typing import TypeAlias

from a2a.types import Message, Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent

from ._events import TypedEvent

A2AResponse: TypeAlias = tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None] | Message


class A2AStreamEvent(TypedEvent):
    """Event that wraps streamed A2A types."""

    def __init__(self, a2a_event: A2AResponse) -> None:
        """Initialize with A2A event.

        Args:
            a2a_event: The original A2A event (Task tuple or Message)
        """
        super().__init__(
            {
                "type": "a2a_stream",
                "event": a2a_event,  # Nest A2A event to avoid field conflicts
            }
        )
