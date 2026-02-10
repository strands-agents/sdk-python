"""Typed hook system for extending agent functionality.

This module provides a composable mechanism for building objects that can hook
into specific events during the agent lifecycle. The hook system enables both
built-in SDK components and user code to react to or modify agent behavior
through strongly-typed event callbacks.

Example Usage with Class-Based Hooks:
    ```python
    from strands.hooks import HookProvider, HookRegistry
    from strands.hooks.events import BeforeInvocationEvent, AfterInvocationEvent

    class LoggingHooks(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(BeforeInvocationEvent, self.log_start)
            registry.add_callback(AfterInvocationEvent, self.log_end)

        def log_start(self, event: BeforeInvocationEvent) -> None:
            print(f"Request started for {event.agent.name}")

        def log_end(self, event: AfterInvocationEvent) -> None:
            print(f"Request completed for {event.agent.name}")

    # Use with agent
    agent = Agent(hooks=[LoggingHooks()])
    ```

Example Usage with Decorator-Based Hooks:
    ```python
    from strands import Agent, hook
    from strands.hooks import BeforeToolCallEvent

    @hook
    def log_tool_calls(event: BeforeToolCallEvent) -> None:
        '''Log all tool calls before execution.'''
        print(f"Tool: {event.tool_use}")

    agent = Agent(hooks=[log_tool_calls])
    ```

This module supports both the class-based HookProvider approach and the newer
decorator-based @hook approach for maximum flexibility.
"""

from .decorator import DecoratedFunctionHook, hook
from .events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    # Multiagent hook events
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    AfterToolCallEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
    MultiAgentInitializedEvent,
)
from .registry import BaseHookEvent, HookCallback, HookEvent, HookProvider, HookRegistry

__all__ = [
    # Events
    "AgentInitializedEvent",
    "BeforeInvocationEvent",
    "BeforeToolCallEvent",
    "AfterToolCallEvent",
    "BeforeModelCallEvent",
    "AfterModelCallEvent",
    "AfterInvocationEvent",
    "MessageAddedEvent",
    # Multiagent events
    "AfterMultiAgentInvocationEvent",
    "AfterNodeCallEvent",
    "BeforeMultiAgentInvocationEvent",
    "BeforeNodeCallEvent",
    "MultiAgentInitializedEvent",
    # Registry
    "HookEvent",
    "HookProvider",
    "HookCallback",
    "HookRegistry",
    "BaseHookEvent",
    # Decorator
    "hook",
    "DecoratedFunctionHook",
]
