"""Tests for the plugin system."""

import unittest.mock

import pytest

from strands.plugins import Plugin, PluginRegistry

# Plugin Protocol Tests


def test_plugin_protocol_is_runtime_checkable():
    """Test that Plugin Protocol is runtime checkable with isinstance."""

    class MyPlugin:
        name = "my-plugin"

        def init_plugin(self, agent):
            pass

    plugin = MyPlugin()
    assert isinstance(plugin, Plugin)


def test_plugin_protocol_sync_implementation():
    """Test Plugin Protocol works with synchronous init_plugin."""

    class SyncPlugin:
        name = "sync-plugin"

        def init_plugin(self, agent):
            agent.custom_attribute = "initialized by plugin"

    plugin = SyncPlugin()
    mock_agent = unittest.mock.Mock()

    # Verify the plugin matches the protocol
    assert isinstance(plugin, Plugin)
    assert plugin.name == "sync-plugin"

    # Execute init_plugin synchronously
    plugin.init_plugin(mock_agent)
    assert mock_agent.custom_attribute == "initialized by plugin"


@pytest.mark.asyncio
async def test_plugin_protocol_async_implementation():
    """Test Plugin Protocol works with asynchronous init_plugin."""

    class AsyncPlugin:
        name = "async-plugin"

        async def init_plugin(self, agent):
            agent.custom_attribute = "initialized by async plugin"

    plugin = AsyncPlugin()
    mock_agent = unittest.mock.Mock()

    # Verify the plugin matches the protocol
    assert isinstance(plugin, Plugin)
    assert plugin.name == "async-plugin"

    # Execute init_plugin asynchronously
    await plugin.init_plugin(mock_agent)
    assert mock_agent.custom_attribute == "initialized by async plugin"


def test_plugin_protocol_requires_name():
    """Test that Plugin Protocol requires a name property."""

    class PluginWithoutName:
        def init_plugin(self, agent):
            pass

    plugin = PluginWithoutName()
    # A class without 'name' should not pass isinstance check
    assert not isinstance(plugin, Plugin)


def test_plugin_protocol_requires_init_plugin_method():
    """Test that Plugin Protocol requires an init_plugin method."""

    class PluginWithoutInitPlugin:
        name = "incomplete-plugin"

    plugin = PluginWithoutInitPlugin()
    # A class without 'init_plugin' should not pass isinstance check
    assert not isinstance(plugin, Plugin)


def test_plugin_protocol_with_class_attribute_name():
    """Test Plugin Protocol works when name is a class attribute."""

    class PluginWithClassAttribute:
        name: str = "class-attr-plugin"

        def init_plugin(self, agent):
            pass

    plugin = PluginWithClassAttribute()
    assert isinstance(plugin, Plugin)
    assert plugin.name == "class-attr-plugin"


def test_plugin_protocol_with_property_name():
    """Test Plugin Protocol works when name is a property."""

    class PluginWithProperty:
        @property
        def name(self):
            return "property-plugin"

        def init_plugin(self, agent):
            pass

    plugin = PluginWithProperty()
    assert isinstance(plugin, Plugin)
    assert plugin.name == "property-plugin"


# PluginRegistry Tests


@pytest.fixture
def registry():
    """Create a fresh PluginRegistry for each test."""
    return PluginRegistry()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return unittest.mock.Mock()


def test_plugin_registry_add_plugin(registry, mock_agent):
    """Test adding a plugin to the registry."""

    class TestPlugin:
        name = "test-plugin"

        def init_plugin(self, agent):
            pass

    plugin = TestPlugin()
    registry.add_plugin(plugin, mock_agent)

    assert registry.has_plugin("test-plugin")
    assert registry.get_plugin("test-plugin") is plugin


def test_plugin_registry_add_duplicate_raises_error(registry, mock_agent):
    """Test that adding a duplicate plugin raises an error."""

    class TestPlugin:
        name = "test-plugin"

        def init_plugin(self, agent):
            pass

    plugin1 = TestPlugin()
    plugin2 = TestPlugin()

    registry.add_plugin(plugin1, mock_agent)

    with pytest.raises(ValueError, match="plugin_name=<test-plugin> | plugin already registered"):
        registry.add_plugin(plugin2, mock_agent)


def test_plugin_registry_get_plugin_not_found(registry):
    """Test getting a plugin that doesn't exist returns None."""
    assert registry.get_plugin("nonexistent") is None


def test_plugin_registry_has_plugin_false(registry):
    """Test has_plugin returns False for unregistered plugins."""
    assert not registry.has_plugin("nonexistent")


def test_plugin_registry_list_plugins(registry, mock_agent):
    """Test listing all registered plugins."""

    class Plugin1:
        name = "plugin-1"

        def init_plugin(self, agent):
            pass

    class Plugin2:
        name = "plugin-2"

        def init_plugin(self, agent):
            pass

    class Plugin3:
        name = "plugin-3"

        def init_plugin(self, agent):
            pass

    registry.add_plugin(Plugin1(), mock_agent)
    registry.add_plugin(Plugin2(), mock_agent)
    registry.add_plugin(Plugin3(), mock_agent)

    plugin_names = registry.list_plugins()
    assert plugin_names == ["plugin-1", "plugin-2", "plugin-3"]


def test_plugin_registry_list_plugins_empty(registry):
    """Test listing plugins when registry is empty."""
    assert registry.list_plugins() == []
