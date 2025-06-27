"""Hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

from dataclasses import dataclass

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
class StartRequestEvent(HookEvent):
    """Event triggered at the beginning of a new agent request.

    This event is fired when the agent begins processing a new user request,
    before any model inference or tool execution occurs. Hook providers can
    use this event to perform request-level setup, logging, or validation.
    """

    pass


@dataclass
class EndRequestEvent(HookEvent):
    """Event triggered at the end of an agent request.

    This event is fired after the agent has completed processing a request,
    regardless of whether it completed successfully or encountered an error.
    Hook providers can use this event for cleanup, logging, or state persistence.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.
    """

    @property
    def should_reverse_callbacks(self) -> bool:
        """Return True to invoke callbacks in reverse order for proper cleanup.

        Returns:
            True, indicating callbacks should be invoked in reverse order.
        """
        return True
