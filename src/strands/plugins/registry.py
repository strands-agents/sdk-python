"""Plugin registry for managing plugins attached to an agent.

This module provides the _PluginRegistry class for tracking and managing
plugins that have been initialized with an agent instance.
"""

import inspect
import logging
from typing import TYPE_CHECKING

from .._async import run_async
from .plugin import Plugin

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


class _PluginRegistry:
    """Registry for managing plugins attached to an agent.

    The _PluginRegistry tracks plugins that have been initialized with an agent,
    providing methods to add plugins and invoke their initialization.

    Example:
        ```python
        registry = _PluginRegistry(agent)

        class MyPlugin:
            name = "my-plugin"

            def init_plugin(self, agent: Agent) -> None:
                pass

        plugin = MyPlugin()
        registry.add_plugin(plugin)
        ```
    """

    def __init__(self, agent: "Agent") -> None:
        """Initialize a plugin registry with an agent reference.

        Args:
            agent: The agent instance that plugins will be initialized with.
        """
        self._agent = agent
        self._plugins: dict[str, Plugin] = {}

    def add_plugin(self, plugin: Plugin) -> None:
        """Add and initialize a plugin with the agent.

        This method registers the plugin and calls its init_plugin method.
        Handles both sync and async init_plugin implementations automatically.

        Args:
            plugin: The plugin to add and initialize.

        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        if plugin.name in self._plugins:
            raise ValueError(f"plugin_name=<{plugin.name}> | plugin already registered")

        logger.debug("plugin_name=<%s> | registering and initializing plugin", plugin.name)
        self._plugins[plugin.name] = plugin

        if inspect.iscoroutinefunction(plugin.init_plugin):
            run_async(lambda: plugin.init_plugin(self._agent))
        else:
            plugin.init_plugin(self._agent)

