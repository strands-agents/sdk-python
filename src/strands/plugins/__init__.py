"""Plugin system for extending agent functionality.

This module provides a composable mechanism for building objects that can
extend agent behavior through a standardized initialization pattern.

Example Usage:
    ```python
    from strands.plugins import Plugin

    class LoggingPlugin(Plugin):
        name = "logging"

        def init_plugin(self, agent: Agent) -> None:
            agent.add_hook(self.on_model_call, BeforeModelCallEvent)

        def on_model_call(self, event: BeforeModelCallEvent) -> None:
            print(f"Model called for {event.agent.name}")
    ```
"""

from .plugin import Plugin

__all__ = [
    "Plugin",
]
