"""Typed hook system for BidiAgent.

This module provides hook events specifically for BidiAgent, enabling
composable extension of bidirectional streaming agent functionality.

Example Usage:
    ```python
    from strands.experimental.bidi.hooks import (
        BidiBeforeInvocationEvent,
        BidiInterruptionEvent,
        HookProvider,
        HookRegistry
    )

    class BidiLoggingHooks(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(BidiBeforeInvocationEvent, self.log_session_start)
            registry.add_callback(BidiInterruptionEvent, self.log_interruption)

        def log_session_start(self, event: BidiBeforeInvocationEvent) -> None:
            print(f"BidiAgent {event.agent.name} starting session")

        def log_interruption(self, event: BidiInterruptionEvent) -> None:
            print(f"Interrupted: {event.reason}")

    # Use with BidiAgent
    agent = BidiAgent(hooks=[BidiLoggingHooks()])
    ```
"""

from ....hooks import HookCallback, HookProvider, HookRegistry
from .events import (
    BidiAfterInvocationEvent,
    BidiAfterToolCallEvent,
    BidiAgentInitializedEvent,
    BidiBeforeInvocationEvent,
    BidiBeforeToolCallEvent,
    BidiHookEvent,
    BidiInterruptionEvent,
    BidiMessageAddedEvent,
)

__all__ = [
    "BidiAgentInitializedEvent",
    "BidiBeforeInvocationEvent",
    "BidiAfterInvocationEvent",
    "BidiBeforeToolCallEvent",
    "BidiAfterToolCallEvent",
    "BidiMessageAddedEvent",
    "BidiInterruptionEvent",
    "BidiHookEvent",
    "HookProvider",
    "HookCallback",
    "HookRegistry",
]
