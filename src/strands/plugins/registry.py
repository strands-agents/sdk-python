"""Plugin registry for managing plugins attached to an agent.

This module provides the PluginRegistry class for tracking and managing
plugins that have been initialized with an agent instance.
"""

import logging
from typing import TYPE_CHECKING

from .plugin import Plugin

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugins attached to an agent.

    The PluginRegistry tracks plugins that have been initialized with an agent,
    providing methods to add, retrieve, and check for plugins by name.

    Example:
        ```python
        registry = PluginRegistry()

        class MyPlugin:
            name = "my-plugin"

            def init_plugin(self, agent: Agent) -> None:
                pass

        plugin = MyPlugin()
        registry.add_plugin(plugin, agent)

        # Check if plugin is registered
        if registry.has_plugin("my-plugin"):
            retrieved = registry.get_plugin("my-plugin")
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty plugin registry."""
        self._plugins: dict[str, Plugin] = {}

    def add_plugin(self, plugin: Plugin, agent: "Agent") -> None:
        """Add and initialize a plugin with the given agent.

        Args:
            plugin: The plugin to add and initialize.
            agent: The agent instance to initialize the plugin with.

        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        if plugin.name in self._plugins:
            raise ValueError(f"plugin_name=<{plugin.name}> | plugin already registered")

        logger.debug("plugin_name=<%s> | registering plugin", plugin.name)
        self._plugins[plugin.name] = plugin

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a plugin by name.

        Args:
            name: The name of the plugin to retrieve.

        Returns:
            The plugin if found, None otherwise.
        """
        return self._plugins.get(name)

    def has_plugin(self, name: str) -> bool:
        """Check if a plugin with the given name is registered.

        Args:
            name: The name of the plugin to check.

        Returns:
            True if the plugin is registered, False otherwise.
        """
        return name in self._plugins

    def list_plugins(self) -> list[str]:
        """Get a list of all registered plugin names.

        Returns:
            A list of plugin names in registration order.
        """
        return list(self._plugins.keys())
