"""Hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

from dataclasses import dataclass

from ..types.content import Message
from .registry import HookEvent


@dataclass
class AgentInitializedEvent(HookEvent):
    """Event triggered when an agent has finished initialization.

    This event is fired after the agent has been fully constructed and all
    built-in components have been initialized. Hook providers can use this
    event to perform setup tasks that require a fully initialized agent.
    """

    pass


@dataclass
class BeforeInvocationEvent(HookEvent):
    """Event triggered at the beginning of a new agent request.

    This event is fired before the agent begins processing a new user request,
    before any model inference or tool execution occurs. Hook providers can
    use this event to perform request-level setup, logging, or validation.

    This event is triggered at the beginning of the following api calls:
      - Agent.__call__
      - Agent.stream_async
      - Agent.structured_output
    """

    pass


@dataclass
class AfterInvocationEvent(HookEvent):
    """Event triggered at the end of an agent request.

    This event is fired after the agent has completed processing a request,
    regardless of whether it completed successfully or encountered an error.
    Hook providers can use this event for cleanup, logging, or state persistence.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    This event is triggered at the end of the following api calls:
      - Agent.__call__
      - Agent.stream_async
      - Agent.structured_output
    """

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class MessageAddedEvent(HookEvent):
    """Event triggered when a message is added to the agent's conversation.

    This event is fired whenever the agent adds a new message to its internal
    message history, including user messages, assistant responses, and tool
    results. Hook providers can use this event for logging, monitoring, or
    implementing custom message processing logic.

    Note: This event is only triggered for messages added by the framework
    itself, not for messages manually added by tools or external code.

    Attributes:
        message: The message that was added to the conversation history.
    """

    message: Message
