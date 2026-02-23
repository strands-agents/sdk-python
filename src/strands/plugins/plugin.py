"""Plugin base class for extending agent functionality.

This module defines the Plugin base class, which provides a composable way to
add behavior changes to agents through automatic hook and tool registration.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING

from ..tools.decorator import DecoratedFunctionTool
from .decorator import _WrappedHookCallable

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for objects that extend agent functionality.

    Plugins provide a composable way to add behavior changes to agents.
    They support automatic discovery and registration of methods decorated
    with @hook and @tool decorators.

    Attributes:
        name: A stable string identifier for the plugin (must be provided by subclass)
        hooks: Tuple of discovered @hook decorated methods (read-only)
        tools: Tuple of discovered @tool decorated methods (read-only)

    Example using decorators (recommended):
        ```python
        from strands.plugins import Plugin, hook
        from strands.hooks import BeforeModelCallEvent
        from strands import tool

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_model_call(self, event: BeforeModelCallEvent):
                print(f"Model called: {event}")

            @tool
            def my_tool(self, param: str) -> str:
                '''A tool that does something.'''
                return f"Result: {param}"
        ```

    Example with custom initialization:
        ```python
        class MyPlugin(Plugin):
            name = "my-plugin"

            def init_agent(self, agent: Agent) -> None:
                # Custom initialization logic - no super() needed
                # Decorated hooks/tools are auto-registered by the plugin registry
                agent.hooks.add_callback(BeforeModelCallEvent, self.custom_hook)

            def custom_hook(self, event: BeforeModelCallEvent):
                print(event)
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A stable string identifier for the plugin."""
        ...

    def __init__(self) -> None:
        """Initialize the plugin and discover decorated methods.

        Scans the class for methods decorated with @hook and @tool and stores
        references for later registration when the plugin is attached to an agent.
        """
        self._hooks: list[_WrappedHookCallable] = []
        self._tools: list[DecoratedFunctionTool] = []
        self._discover_decorated_methods()

    @property
    def hooks(self) -> tuple[_WrappedHookCallable, ...]:
        """Discovered @hook decorated methods.

        Returns a tuple of hook callbacks that will be auto-registered
        when the plugin is attached to an agent.
        """
        return tuple(self._hooks)

    @property
    def tools(self) -> tuple[DecoratedFunctionTool, ...]:
        """Discovered @tool decorated methods.

        Returns a tuple of tools that will be auto-registered
        when the plugin is attached to an agent.
        """
        return tuple(self._tools)

    def _discover_decorated_methods(self) -> None:
        """Scan class for @hook and @tool decorated methods."""
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
            except Exception:
                # Skip attributes that can't be accessed
                continue

            # Check for @hook decorated methods
            if hasattr(attr, "_hook_event_types") and callable(attr):
                self._hooks.append(attr)
                logger.debug("plugin=<%s>, hook=<%s> | discovered hook method", self.name, attr_name)

            # Check for @tool decorated methods (DecoratedFunctionTool instances)
            if isinstance(attr, DecoratedFunctionTool):
                self._tools.append(attr)
                logger.debug("plugin=<%s>, tool=<%s> | discovered tool method", self.name, attr_name)

    def init_agent(self, agent: "Agent") -> None | Awaitable[None]:
        """Initialize the agent instance.

        Override this method to add custom initialization logic. Decorated
        hooks and tools are automatically registered by the plugin registry,
        so there's no need to call super().init_agent(agent).

        Args:
            agent: The agent instance to initialize.
        """
        return None
