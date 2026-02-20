"""Tests for the plugin system."""

import unittest.mock

import pytest

from strands.plugins import Plugin
from strands.plugins.registry import _PluginRegistry

# Plugin Tests


def test_plugin_class_requires_inheritance():
    """Test that Plugin class requires inheritance."""

    class MyPlugin(Plugin):
        name = "my-plugin"

        def init_plugin(self, agent):
            pass

    plugin = MyPlugin()
    assert isinstance(plugin, Plugin)


def test_plugin_class_sync_implementation():
    """Test Plugin class works with synchronous init_plugin."""

    class SyncPlugin(Plugin):
        name = "sync-plugin"

        def init_plugin(self, agent):
            agent.custom_attribute = "initialized by plugin"

    plugin = SyncPlugin()
    mock_agent = unittest.mock.Mock()

    # Verify the plugin is an instance of Plugin
    assert isinstance(plugin, Plugin)
    assert plugin.name == "sync-plugin"

    # Execute init_plugin synchronously
    plugin.init_plugin(mock_agent)
    assert mock_agent.custom_attribute == "initialized by plugin"


@pytest.mark.asyncio
async def test_plugin_class_async_implementation():
    """Test Plugin class works with asynchronous init_plugin."""

    class AsyncPlugin(Plugin):
        name = "async-plugin"

        async def init_plugin(self, agent):
            agent.custom_attribute = "initialized by async plugin"

    plugin = AsyncPlugin()
    mock_agent = unittest.mock.Mock()

    # Verify the plugin is an instance of Plugin
    assert isinstance(plugin, Plugin)
    assert plugin.name == "async-plugin"

    # Execute init_plugin asynchronously
    await plugin.init_plugin(mock_agent)
    assert mock_agent.custom_attribute == "initialized by async plugin"


def test_plugin_class_requires_name():
    """Test that Plugin class requires a name property."""

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):

        class PluginWithoutName(Plugin):
            def init_plugin(self, agent):
                pass

        PluginWithoutName()


def test_plugin_class_requires_init_plugin_method():
    """Test that Plugin class requires an init_plugin method."""

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):

        class PluginWithoutInitPlugin(Plugin):
            name = "incomplete-plugin"

        PluginWithoutInitPlugin()


def test_plugin_class_with_class_attribute_name():
    """Test Plugin class works when name is a class attribute."""

    class PluginWithClassAttribute(Plugin):
        name: str = "class-attr-plugin"

        def init_plugin(self, agent):
            pass

    plugin = PluginWithClassAttribute()
    assert isinstance(plugin, Plugin)
    assert plugin.name == "class-attr-plugin"


def test_plugin_class_with_property_name():
    """Test Plugin class works when name is a property."""

    class PluginWithProperty(Plugin):
        @property
        def name(self):
            return "property-plugin"

        def init_plugin(self, agent):
            pass

    plugin = PluginWithProperty()
    assert isinstance(plugin, Plugin)
    assert plugin.name == "property-plugin"


# _PluginRegistry Tests


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return unittest.mock.Mock()


@pytest.fixture
def registry(mock_agent):
    """Create a fresh _PluginRegistry for each test."""
    return _PluginRegistry(mock_agent)


def test_plugin_registry_add_and_init_calls_init_plugin(registry, mock_agent):
    """Test adding a plugin calls its init_plugin method."""

    class TestPlugin(Plugin):
        name = "test-plugin"

        def __init__(self):
            self.initialized = False

        def init_plugin(self, agent):
            self.initialized = True
            agent.plugin_initialized = True

    plugin = TestPlugin()
    registry.add_and_init(plugin)

    assert plugin.initialized
    assert mock_agent.plugin_initialized


def test_plugin_registry_add_duplicate_raises_error(registry, mock_agent):
    """Test that adding a duplicate plugin raises an error."""

    class TestPlugin(Plugin):
        name = "test-plugin"

        def init_plugin(self, agent):
            pass

    plugin1 = TestPlugin()
    plugin2 = TestPlugin()

    registry.add_and_init(plugin1)

    with pytest.raises(ValueError, match="plugin_name=<test-plugin> | plugin already registered"):
        registry.add_and_init(plugin2)


def test_plugin_registry_add_and_init_with_async_plugin(registry, mock_agent):
    """Test that add_and_init handles async plugins using run_async."""

    class AsyncPlugin(Plugin):
        name = "async-plugin"

        def __init__(self):
            self.initialized = False

        async def init_plugin(self, agent):
            self.initialized = True
            agent.async_plugin_initialized = True

    plugin = AsyncPlugin()
    registry.add_and_init(plugin)

    assert plugin.initialized
    assert mock_agent.async_plugin_initialized
