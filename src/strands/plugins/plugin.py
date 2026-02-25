"""Plugin base class for extending agent functionality.

This module defines the Plugin base class, which provides a composable way to
add behavior changes to agents through a standardized initialization pattern.
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
        _hooks: List of discovered @hook decorated methods (populated in __init__)
        _tools: List of discovered @tool decorated methods (populated in __init__)

    Example using decorators (recommended):
        ```python
        from strands.plugins import Plugin, hook
        from strands.hooks import BeforeModelCallEvent

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

    Example with manual registration:
        ```python
        class MyPlugin(Plugin):
            name = "my-plugin"

            def init_plugin(self, agent: Agent) -> None:
                super().init_plugin(agent)  # Register decorated methods
                # Add additional manual hooks if needed
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
        references for later registration when init_plugin is called.
        """
        self._hooks: list[_WrappedHookCallable] = []
        self._tools: list[DecoratedFunctionTool] = []
        self._discover_decorated_methods()

    def _discover_decorated_methods(self) -> None:
        """Scan class for @hook and @tool decorated methods."""
        for name in dir(self):
            try:
                attr = getattr(self, name)
            except Exception:
                # Skip attributes that can't be accessed
                continue

            # Check for @hook decorated methods
            if hasattr(attr, "_hook_event_types") and callable(attr):
                self._hooks.append(attr)
                logger.debug("plugin=<%s>, hook=<%s> | discovered hook method", self.name, name)

            # Check for @tool decorated methods (DecoratedFunctionTool instances)
            if isinstance(attr, DecoratedFunctionTool):
                self._tools.append(attr)
                logger.debug("plugin=<%s>, tool=<%s> | discovered tool method", self.name, name)

    def init_plugin(self, agent: "Agent") -> None | Awaitable[None]:
        """Initialize the plugin with an agent instance.

        Default implementation that registers all discovered @hook methods
        with the agent's hook registry and adds all discovered @tool methods
        to the agent's tools list.

        Subclasses can override this method and call super().init_plugin(agent)
        to retain automatic registration while adding custom initialization logic.

        Args:
            agent: The agent instance to extend.
        """
        # Register discovered hooks with the agent's hook registry
        for hook_callback in self._hooks:
            event_types = getattr(hook_callback, "_hook_event_types", [])
            for event_type in event_types:
                agent.add_hook(hook_callback, event_type)
                logger.debug(
                    "plugin=<%s>, hook=<%s>, event_type=<%s> | registered hook",
                    self.name,
                    getattr(hook_callback, "__name__", repr(hook_callback)),
                    event_type.__name__,
                )

        # Register discovered tools with the agent's tool registry
        if self._tools:
            agent.tool_registry.process_tools(self._tools)
            for tool in self._tools:
                logger.debug(
                    "plugin=<%s>, tool=<%s> | registered tool",
                    self.name,
                    tool.tool_name,
                )

        return None
