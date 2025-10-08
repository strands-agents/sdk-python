"""Tool provider interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from ...types.tools import AgentTool


class ToolProvider(ABC):
    """Interface for providing tools with lifecycle management.

    Provides a way to load a collection of tools and clean them up
    when done, with lifecycle managed by the agent.
    """

    @abstractmethod
    async def load_tools(self, **kwargs: Any) -> Sequence["AgentTool"]:
        """Load and return the tools in this provider.

        Args:
            **kwargs: Additional arguments for future compatibility.

        Returns:
            List of tools that are ready to use.
        """
        pass

    @abstractmethod
    async def add_provider_consumer(self, id: Any, **kwargs: Any) -> None:
        """Add a consumer to this tool provider.

        Args:
            id: Unique identifier for the consumer.
            **kwargs: Additional arguments for future compatibility.
        """
        pass

    @abstractmethod
    async def remove_provider_consumer(self, id: Any, **kwargs: Any) -> None:
        """Remove a consumer from this tool provider.

        Args:
            id: Unique identifier for the consumer.
            **kwargs: Additional arguments for future compatibility.

        Provider may clean up resources when no consumers remain.
        """
        pass
