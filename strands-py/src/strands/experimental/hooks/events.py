"""Experimental hook events emitted as part of invoking Agents and BidiAgents.

This module defines the events that are emitted as Agents and BidiAgents run through the lifecycle of a request.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...hooks.events import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent
from ...hooks.registry import BaseHookEvent
from ...types.content import Message

if TYPE_CHECKING:
    from ..bidi.agent.agent import BidiAgent
    from ..bidi.models import BidiModelTimeoutError

# Deprecated aliases - warning emitted on access via __getattr__
_DEPRECATED_ALIASES = {
    "BeforeToolInvocationEvent": BeforeToolCallEvent,
    "AfterToolInvocationEvent": AfterToolCallEvent,
    "BeforeModelInvocationEvent": BeforeModelCallEvent,
    "AfterModelInvocationEvent": AfterModelCallEvent,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        warnings.warn(
            f"{name} has been moved to production with an updated name. "
            f"Use {_DEPRECATED_ALIASES[name].__name__} from strands.hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# BidiAgent Hook Events


@dataclass
class BidiHookEvent(BaseHookEvent):
    """Base class for BidiAgent hook events.

    Attributes:
        agent: The BidiAgent instance that triggered this event.
    """

    agent: "BidiAgent"


@dataclass
class BidiAgentInitializedEvent(BidiHookEvent):
    """Event triggered when a BidiAgent has finished initialization.

    This event is fired after the BidiAgent has been fully constructed and all
    built-in components have been initialized. Hook providers can use this
    event to perform setup tasks that require a fully initialized agent.
    """

    pass


@dataclass
class BidiBeforeInvocationEvent(BidiHookEvent):
    """Event triggered when BidiAgent starts a streaming session.

    This event is fired before the BidiAgent begins a streaming session,
    before any model connection or audio processing occurs. Hook providers can
    use this event to perform session-level setup, logging, or validation.

    This event is triggered at the beginning of agent.start().
    """

    pass


@dataclass
class BidiAfterInvocationEvent(BidiHookEvent):
    """Event triggered when BidiAgent ends a streaming session.

    This event is fired after the BidiAgent has completed a streaming session,
    regardless of whether it completed successfully or encountered an error.
    Hook providers can use this event for cleanup, logging, or state persistence.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    This event is triggered at the end of agent.stop().
    """

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BidiMessageAddedEvent(BidiHookEvent):
    """Event triggered when BidiAgent adds a message to the conversation.

    This event is fired whenever the BidiAgent adds a new message to its internal
    message history, including user messages (from transcripts), assistant responses,
    and tool results. Hook providers can use this event for logging, monitoring, or
    implementing custom message processing logic.

    Note: This event is only triggered for messages added by the framework
    itself, not for messages manually added by tools or external code.

    Attributes:
        message: The message that was added to the conversation history.
    """

    message: Message


@dataclass
class BidiInterruptionEvent(BidiHookEvent):
    """Event triggered when model generation is interrupted.

    This event is fired when the user interrupts the assistant (e.g., by speaking
    during the assistant's response) or when an error causes interruption. This is
    specific to bidirectional streaming and doesn't exist in standard agents.

    Hook providers can use this event to log interruptions, implement custom
    interruption handling, or trigger cleanup logic.

    Attributes:
        reason: The reason for the interruption ("user_speech" or "error").
        interrupted_response_id: Optional ID of the response that was interrupted.
    """

    reason: Literal["user_speech", "error"]
    interrupted_response_id: str | None = None


@dataclass
class BidiBeforeConnectionRestartEvent(BidiHookEvent):
    """Event emitted before the agent restarts the model connection.

    A restart is triggered either reactively, after the model reports a timeout, or
    proactively, when the reconnect timer fires ahead of the provider's limit.

    Attributes:
        reason: What triggered the restart ("timeout" reactively, "scheduled" proactively).
        timeout_error: The model's timeout error on the reactive path; None when scheduled.
    """

    reason: Literal["timeout", "scheduled"]
    timeout_error: "BidiModelTimeoutError | None" = None


@dataclass
class BidiAfterConnectionRestartEvent(BidiHookEvent):
    """Event emitted after the agent attempts to restart the model connection.

    Attributes:
        reason: What triggered the restart ("timeout" reactively, "scheduled" proactively).
        exception: Populated if an exception was raised during the restart. None means success.
    """

    reason: Literal["timeout", "scheduled"]
    exception: Exception | None = None
