"""Tests for the Plugin base class with auto-discovery."""

import unittest.mock

import pytest

from strands.hooks import BeforeInvocationEvent, BeforeModelCallEvent, HookRegistry
from strands.plugins import Plugin, hook
from strands.tools.decorator import tool


def _configure_mock_agent_with_hooks():
    """Helper to create a mock agent with working add_hook."""
    mock_agent = unittest.mock.MagicMock()
    mock_agent.hooks = HookRegistry()
    mock_agent.add_hook.side_effect = lambda callback, event_type=None: mock_agent.hooks.add_callback(
        event_type, callback
    )
    return mock_agent


class TestPluginBaseClass:
    """Tests for Plugin base class basics."""

    def test_plugin_is_class_not_protocol(self):
        """Test that Plugin is now a class, not a Protocol."""

        class MyPlugin(Plugin):
            name = "my-plugin"

        plugin = MyPlugin()
        assert isinstance(plugin, Plugin)

    def test_plugin_requires_name_attribute(self):
        """Test that Plugin subclass must have name attribute."""

        class MyPlugin(Plugin):
            name = "my-plugin"

        plugin = MyPlugin()
        assert plugin.name == "my-plugin"

    def test_plugin_name_as_property(self):
        """Test that Plugin name can be a property."""

        class MyPlugin(Plugin):
            @property
            def name(self) -> str:
                return "property-plugin"

        plugin = MyPlugin()
        assert plugin.name == "property-plugin"


class TestPluginAutoDiscovery:
    """Tests for automatic discovery of decorated methods."""

    def test_plugin_discovers_hook_decorated_methods(self):
        """Test that Plugin.__init__ discovers @hook decorated methods."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_before_model(self, event: BeforeModelCallEvent):
                pass

        plugin = MyPlugin()
        assert len(plugin._hooks) == 1
        assert plugin._hooks[0].__name__ == "on_before_model"

    def test_plugin_discovers_multiple_hooks(self):
        """Test that Plugin discovers multiple @hook decorated methods."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def hook1(self, event: BeforeModelCallEvent):
                pass

            @hook
            def hook2(self, event: BeforeInvocationEvent):
                pass

        plugin = MyPlugin()
        assert len(plugin._hooks) == 2
        hook_names = {h.__name__ for h in plugin._hooks}
        assert "hook1" in hook_names
        assert "hook2" in hook_names

    def test_plugin_discovers_tool_decorated_methods(self):
        """Test that Plugin.__init__ discovers @tool decorated methods."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @tool
            def my_tool(self, param: str) -> str:
                """A test tool."""
                return param

        plugin = MyPlugin()
        assert len(plugin._tools) == 1
        assert plugin._tools[0].tool_name == "my_tool"

    def test_plugin_discovers_both_hooks_and_tools(self):
        """Test that Plugin discovers both @hook and @tool decorated methods."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def my_hook(self, event: BeforeModelCallEvent):
                pass

            @tool
            def my_tool(self, param: str) -> str:
                """A test tool."""
                return param

        plugin = MyPlugin()
        assert len(plugin._hooks) == 1
        assert len(plugin._tools) == 1

    def test_plugin_ignores_non_decorated_methods(self):
        """Test that Plugin doesn't discover non-decorated methods."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            def regular_method(self):
                pass

            @hook
            def decorated_hook(self, event: BeforeModelCallEvent):
                pass

        plugin = MyPlugin()
        assert len(plugin._hooks) == 1
        assert plugin._hooks[0].__name__ == "decorated_hook"


class TestPluginInitPlugin:
    """Tests for Plugin.init_plugin() auto-registration."""

    def test_init_plugin_registers_hooks_with_agent(self):
        """Test that init_plugin registers discovered hooks with agent."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_before_model(self, event: BeforeModelCallEvent):
                pass

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()

        plugin.init_plugin(mock_agent)

        # Verify hook was registered
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1

    def test_init_plugin_registers_tools_with_agent(self):
        """Test that init_plugin adds discovered tools to agent's tools."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @tool
            def my_tool(self, param: str) -> str:
                """A test tool."""
                return param

        plugin = MyPlugin()
        mock_agent = unittest.mock.MagicMock()
        mock_agent.hooks = HookRegistry()
        mock_agent.tool_registry = unittest.mock.MagicMock()

        plugin.init_plugin(mock_agent)

        # Verify tool was added to agent
        mock_agent.tool_registry.process_tools.assert_called_once()

    def test_init_plugin_registers_both_hooks_and_tools(self):
        """Test that init_plugin registers both hooks and tools."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def my_hook(self, event: BeforeModelCallEvent):
                pass

            @tool
            def my_tool(self, param: str) -> str:
                """A test tool."""
                return param

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()
        mock_agent.tool_registry = unittest.mock.MagicMock()

        plugin.init_plugin(mock_agent)

        # Verify both registered
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1
        mock_agent.tool_registry.process_tools.assert_called_once()


class TestPluginHookWithUnionTypes:
    """Tests for Plugin hooks with union types."""

    def test_init_plugin_registers_hook_for_union_types(self):
        """Test that hooks with union types are registered for all event types."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_model_events(self, event: BeforeModelCallEvent | BeforeInvocationEvent):
                pass

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()

        plugin.init_plugin(mock_agent)

        # Verify hook was registered for both event types
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeInvocationEvent, [])) == 1


class TestPluginMultipleAgents:
    """Tests for plugin reuse with multiple agents."""

    def test_plugin_can_be_attached_to_multiple_agents(self):
        """Test that the same plugin instance can be used with multiple agents."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_before_model(self, event: BeforeModelCallEvent):
                pass

        plugin = MyPlugin()

        mock_agent1 = _configure_mock_agent_with_hooks()
        mock_agent2 = _configure_mock_agent_with_hooks()

        plugin.init_plugin(mock_agent1)
        plugin.init_plugin(mock_agent2)

        # Verify both agents have the hook registered
        assert len(mock_agent1.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1
        assert len(mock_agent2.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1


class TestPluginSubclassOverride:
    """Tests for subclass overriding init_plugin."""

    def test_subclass_can_override_init_plugin(self):
        """Test that subclass can override init_plugin and call super()."""
        custom_init_called = False

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_before_model(self, event: BeforeModelCallEvent):
                pass

            def init_plugin(self, agent):
                nonlocal custom_init_called
                custom_init_called = True
                super().init_plugin(agent)

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()

        plugin.init_plugin(mock_agent)

        assert custom_init_called
        # Verify auto-registration still happened via super()
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1

    def test_subclass_can_add_manual_hooks(self):
        """Test that subclass can manually add hooks in addition to decorated ones."""
        manual_hook_added = False

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def auto_hook(self, event: BeforeModelCallEvent):
                pass

            def manual_hook(self, event: BeforeInvocationEvent):
                pass

            def init_plugin(self, agent):
                nonlocal manual_hook_added
                super().init_plugin(agent)
                # Add manual hook
                agent.hooks.add_callback(BeforeInvocationEvent, self.manual_hook)
                manual_hook_added = True

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()

        plugin.init_plugin(mock_agent)

        assert manual_hook_added
        # Verify both hooks registered
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeInvocationEvent, [])) == 1


class TestPluginAsyncInitPlugin:
    """Tests for async init_plugin support."""

    @pytest.mark.asyncio
    async def test_async_init_plugin_supported(self):
        """Test that async init_plugin is supported."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_before_model(self, event: BeforeModelCallEvent):
                pass

            async def init_plugin(self, agent):
                # Just call super synchronously - async is for custom logic
                super().init_plugin(agent)

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()

        await plugin.init_plugin(mock_agent)

        # Verify hook was registered
        assert len(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, [])) == 1


class TestPluginBoundMethods:
    """Tests for bound method registration."""

    def test_hooks_are_bound_to_instance(self):
        """Test that registered hooks are bound to the plugin instance."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            def __init__(self):
                super().__init__()
                self.events_received = []

            @hook
            def on_before_model(self, event: BeforeModelCallEvent):
                self.events_received.append(event)

        plugin = MyPlugin()
        mock_agent = _configure_mock_agent_with_hooks()

        plugin.init_plugin(mock_agent)

        # Call the registered hook and verify it accesses the correct instance
        mock_event = unittest.mock.MagicMock(spec=BeforeModelCallEvent)
        callbacks = list(mock_agent.hooks._registered_callbacks.get(BeforeModelCallEvent, []))
        callbacks[0](mock_event)

        assert len(plugin.events_received) == 1
        assert plugin.events_received[0] is mock_event

    def test_tools_are_bound_to_instance(self):
        """Test that registered tools are bound to the plugin instance."""

        class MyPlugin(Plugin):
            name = "my-plugin"

            def __init__(self):
                super().__init__()
                self.tool_called = False

            @tool
            def my_tool(self, param: str) -> str:
                """A test tool."""
                self.tool_called = True
                return param

        plugin = MyPlugin()
        mock_agent = unittest.mock.MagicMock()
        mock_agent.hooks = HookRegistry()
        mock_agent.tool_registry = unittest.mock.MagicMock()

        plugin.init_plugin(mock_agent)

        # Get the tool that was registered and call it
        call_args = mock_agent.tool_registry.process_tools.call_args
        registered_tools = call_args[0][0]
        assert len(registered_tools) == 1

        # Call the tool - it should be bound to the instance
        result = registered_tools[0]("test")
        assert plugin.tool_called
        assert result == "test"
