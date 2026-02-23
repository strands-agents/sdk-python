"""Plugin system for extending agent functionality.

This module provides a composable mechanism for building objects that can
extend agent behavior through automatic hook and tool registration.

Example Usage with Decorators (recommended):
    ```python
    from strands.plugins import Plugin, hook
    from strands.hooks import BeforeModelCallEvent
    from strands import tool

    class LoggingPlugin(Plugin):
        name = "logging"

        @hook
        def on_model_call(self, event: BeforeModelCallEvent) -> None:
            print(f"Model called for {event.agent.name}")

        @tool
        def log_message(self, message: str) -> str:
            '''Log a message.'''
            print(message)
            return "Logged"
    ```

Example Usage with Custom Initialization:
    ```python
    from strands.plugins import Plugin
    from strands.hooks import BeforeModelCallEvent

    class LoggingPlugin(Plugin):
        name = "logging"

        def init_agent(self, agent: Agent) -> None:
            # Custom initialization - no super() needed
            # Decorated hooks/tools are auto-registered by the registry
            agent.hooks.add_callback(BeforeModelCallEvent, self.on_model_call)

        def on_model_call(self, event: BeforeModelCallEvent) -> None:
            print(f"Model called for {event.agent.name}")
    ```
"""

from .decorator import hook
from .plugin import Plugin

__all__ = [
    "Plugin",
    "hook",
]
