"""Plugin base class for extending agent functionality.

This module defines the Plugin base class, which provides a composable way to
add behavior changes to agents through a standardized initialization pattern.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


class Plugin(ABC):
    """Base class for objects that extend agent functionality.

    Plugins provide a composable way to add behavior changes to agents.
    They can register hooks, modify agent attributes, or perform other
    setup tasks on an agent instance.

    Attributes:
        name: A stable string identifier for the plugin

    Example:
        ```python
        class MyPlugin(Plugin):
            name = "my-plugin"

            def init_agent(self, agent: Agent) -> None:
                agent.add_hook(self.on_model_call, BeforeModelCallEvent)
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A stable string identifier for the plugin."""
        ...

    @abstractmethod
    def init_agent(self, agent: "Agent") -> None | Awaitable[None]:
        """Initialize the agent instance.

        Args:
            agent: The agent instance to initialize.
        """
        ...
