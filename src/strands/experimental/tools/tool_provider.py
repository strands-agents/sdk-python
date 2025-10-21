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
    def add_consumer(self, id: Any, **kwargs: Any) -> None:
        """Add a consumer to this tool provider.

        This method is synchronous to avoid deadlocks during garbage collection.
        When Agent finalizers run during GC, they need to clean up tool providers
        without using run_async() which can deadlock due to GIL restrictions.

        Args:
            id: Unique identifier for the consumer.
            **kwargs: Additional arguments for future compatibility.
        """
        pass

    @abstractmethod
    def remove_consumer(self, id: Any, **kwargs: Any) -> None:
        """Remove a consumer from this tool provider.

        This method is synchronous to avoid deadlocks during garbage collection.
        When Agent finalizers run during GC, they need to clean up tool providers
        without using run_async() which can deadlock due to GIL restrictions.

        Provider may clean up resources when no consumers remain.

        Args:
            id: Unique identifier for the consumer.
            **kwargs: Additional arguments for future compatibility.
        """
        pass
