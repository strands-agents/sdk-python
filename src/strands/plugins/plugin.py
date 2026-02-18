"""Plugin protocol for extending agent functionality.

This module defines the Plugin Protocol, which provides a composable way to
add behavior changes to agents through a standardized initialization pattern.
"""

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..agent import Agent


@runtime_checkable
class Plugin(Protocol):
    """Protocol for objects that extend agent functionality.

    Plugins provide a composable way to add behavior changes to agents.
    They are initialized with an agent instance and can register hooks,
    modify agent attributes, or perform other setup tasks.

    Example:
        ```python
        class MyPlugin:
            name = "my-plugin"

            def init_plugin(self, agent: Agent) -> None:
                agent.add_hook(self.on_model_call, BeforeModelCallEvent)
        ```
    """

    name: str

    def init_plugin(self, agent: "Agent") -> None | Awaitable[None]:
        """Initialize the plugin with an agent instance.

        Args:
            agent: The agent instance to extend.
        """
        ...
