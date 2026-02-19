"""Plugin system for extending agent functionality.

This module provides a composable mechanism for building objects that can
extend agent behavior through automatic hook and tool registration.

Example Usage with Decorators (recommended):
    ```python
    from strands.plugins import Plugin, hook
    from strands.hooks import BeforeModelCallEvent

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

Example Usage with Manual Registration:
    ```python
    from strands.plugins import Plugin
    from strands.hooks import BeforeModelCallEvent

    class LoggingPlugin(Plugin):
        name = "logging"

        def init_plugin(self, agent: Agent) -> None:
            super().init_plugin(agent)  # Register decorated methods
            # Add additional manual hooks
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
