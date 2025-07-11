"""Typed hook system for extending agent functionality.

This module provides a composable mechanism for building objects that can hook
into specific events during the agent lifecycle. The hook system enables both
built-in SDK components and user code to react to or modify agent behavior
through strongly-typed event callbacks.

Example Usage:
    ```python
    from strands.hooks import HookProvider, HookRegistry
    from strands.hooks.events import StartRequestEvent, EndRequestEvent

    class LoggingHooks(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(StartRequestEvent, self.log_start)
            registry.add_callback(EndRequestEvent, self.log_end)

        def log_start(self, event: StartRequestEvent) -> None:
            print(f"Request started for {event.agent.name}")

        def log_end(self, event: EndRequestEvent) -> None:
            print(f"Request completed for {event.agent.name}")

    # Use with agent
    agent = Agent(hooks=[LoggingHooks()])
    ```

This replaces the older callback_handler approach with a more composable,
type-safe system that supports multiple subscribers per event type.
"""

from .events import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    MessageAddedEvent,
)
from .registry import HookCallback, HookEvent, HookProvider, HookRegistry

__all__ = [
    "AgentInitializedEvent",
    "BeforeInvocationEvent",
    "AfterInvocationEvent",
    "MessageAddedEvent",
    "HookEvent",
    "HookProvider",
    "HookCallback",
    "HookRegistry",
]
