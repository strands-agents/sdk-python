"""Tests for the plugin system."""

import unittest.mock

import pytest

from strands.plugins import Plugin
from strands.plugins.registry import _PluginRegistry

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


# _PluginRegistry Tests


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return unittest.mock.Mock()


@pytest.fixture
def registry(mock_agent):
    """Create a fresh _PluginRegistry for each test."""
    return _PluginRegistry(mock_agent)


def test_plugin_registry_add_plugin_calls_init_plugin(registry, mock_agent):
    """Test adding a plugin calls its init_plugin method."""

    class TestPlugin:
        name = "test-plugin"

        def __init__(self):
            self.initialized = False

        def init_plugin(self, agent):
            self.initialized = True
            agent.plugin_initialized = True

    plugin = TestPlugin()
    registry.add_plugin(plugin)

    assert plugin.initialized
    assert mock_agent.plugin_initialized


def test_plugin_registry_add_duplicate_raises_error(registry, mock_agent):
    """Test that adding a duplicate plugin raises an error."""

    class TestPlugin:
        name = "test-plugin"

        def init_plugin(self, agent):
            pass

    plugin1 = TestPlugin()
    plugin2 = TestPlugin()

    registry.add_plugin(plugin1)

    with pytest.raises(ValueError, match="plugin_name=<test-plugin> | plugin already registered"):
        registry.add_plugin(plugin2)


def test_plugin_registry_add_plugin_async_raises_runtime_error(registry):
    """Test that add_plugin raises RuntimeError for async plugins."""

    class AsyncPlugin:
        name = "async-plugin"

        async def init_plugin(self, agent):
            pass

    plugin = AsyncPlugin()

    with pytest.raises(RuntimeError, match="use add_plugin_async instead"):
        registry.add_plugin(plugin)


@pytest.mark.asyncio
async def test_plugin_registry_add_plugin_async_with_sync_plugin(mock_agent):
    """Test add_plugin_async works with sync plugins."""
    registry = _PluginRegistry(mock_agent)

    class SyncPlugin:
        name = "sync-plugin"

        def __init__(self):
            self.initialized = False

        def init_plugin(self, agent):
            self.initialized = True

    plugin = SyncPlugin()
    await registry.add_plugin_async(plugin)

    assert plugin.initialized


@pytest.mark.asyncio
async def test_plugin_registry_add_plugin_async_with_async_plugin(mock_agent):
    """Test add_plugin_async works with async plugins."""
    registry = _PluginRegistry(mock_agent)

    class AsyncPlugin:
        name = "async-plugin"

        def __init__(self):
            self.initialized = False

        async def init_plugin(self, agent):
            self.initialized = True

    plugin = AsyncPlugin()
    await registry.add_plugin_async(plugin)

    assert plugin.initialized


@pytest.mark.asyncio
async def test_plugin_registry_add_plugin_async_duplicate_raises_error(mock_agent):
    """Test that add_plugin_async raises error for duplicate plugins."""
    registry = _PluginRegistry(mock_agent)

    class TestPlugin:
        name = "test-plugin"

        async def init_plugin(self, agent):
            pass

    plugin1 = TestPlugin()
    plugin2 = TestPlugin()

    await registry.add_plugin_async(plugin1)

    with pytest.raises(ValueError, match="plugin_name=<test-plugin> | plugin already registered"):
        await registry.add_plugin_async(plugin2)
