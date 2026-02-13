"""Tests for the Plugin base class auto-discovery and properties."""

import unittest.mock

from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent
from strands.hooks.decorator import DecoratedFunctionHook, hook
from strands.plugins.plugin import Plugin
from strands.tools.decorator import DecoratedFunctionTool, tool

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class SimplePlugin(Plugin):
    """Plugin with one tool and one hook."""

    name = "simple"

    @tool
    def greet(self, who: str) -> str:
        """Say hello."""
        return f"Hello, {who}!"

    @hook
    def on_invocation(self, event: BeforeInvocationEvent) -> None:
        """Track invocations."""


class ToolOnlyPlugin(Plugin):
    """Plugin with tools but no hooks."""

    name = "tool-only"

    @tool
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


class HookOnlyPlugin(Plugin):
    """Plugin with hooks but no tools."""

    name = "hook-only"

    @hook
    def before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Before invocation hook."""

    @hook
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Before tool call hook."""


class EmptyPlugin(Plugin):
    """Plugin with nothing decorated."""

    name = "empty"

    def regular_method(self) -> str:
        return "not a tool"


class MultiEventHookPlugin(Plugin):
    """Plugin with a hook that listens to multiple event types."""

    name = "multi-event"

    @hook
    def on_event(self, event: BeforeInvocationEvent | BeforeToolCallEvent) -> None:
        """Handle multiple events."""


class InheritedPlugin(SimplePlugin):
    """Plugin that inherits from SimplePlugin and adds its own tool."""

    name = "inherited"

    @tool
    def farewell(self, who: str) -> str:
        """Say goodbye."""
        return f"Goodbye, {who}!"


class InitPlugin(Plugin):
    """Plugin that overrides init_plugin."""

    name = "init-test"
    initialized_with: "unittest.mock.Mock | None" = None

    def init_plugin(self, agent: "unittest.mock.Mock") -> None:  # type: ignore[override]
        self.initialized_with = agent


# ---------------------------------------------------------------------------
# Tests: auto-discovery
# ---------------------------------------------------------------------------


class TestPluginAutoDiscovery:
    """Test that @tool and @hook methods are discovered correctly."""

    def test_discovers_tools(self) -> None:
        plugin = SimplePlugin()
        assert len(plugin.tools) == 1
        assert plugin.tools[0].tool_name == "greet"

    def test_discovers_hooks(self) -> None:
        plugin = SimplePlugin()
        assert len(plugin.hooks) == 1
        hook_provider = plugin.hooks[0]
        assert isinstance(hook_provider, DecoratedFunctionHook)
        assert hook_provider.name == "on_invocation"

    def test_tool_only_plugin(self) -> None:
        plugin = ToolOnlyPlugin()
        assert len(plugin.tools) == 2
        tool_names = {t.tool_name for t in plugin.tools}
        assert tool_names == {"add", "multiply"}
        assert len(plugin.hooks) == 0

    def test_hook_only_plugin(self) -> None:
        plugin = HookOnlyPlugin()
        assert len(plugin.tools) == 0
        assert len(plugin.hooks) == 2
        hook_names = {h.name for h in plugin.hooks}
        assert hook_names == {"before_invocation", "before_tool_call"}

    def test_empty_plugin(self) -> None:
        plugin = EmptyPlugin()
        assert len(plugin.tools) == 0
        assert len(plugin.hooks) == 0

    def test_multi_event_hook(self) -> None:
        plugin = MultiEventHookPlugin()
        assert len(plugin.hooks) == 1
        hook_provider = plugin.hooks[0]
        assert isinstance(hook_provider, DecoratedFunctionHook)
        assert set(hook_provider.event_types) == {BeforeInvocationEvent, BeforeToolCallEvent}

    def test_inherited_plugin_discovers_parent_and_child(self) -> None:
        plugin = InheritedPlugin()
        tool_names = {t.tool_name for t in plugin.tools}
        assert "greet" in tool_names
        assert "farewell" in tool_names

        hook_names = {h.name for h in plugin.hooks}
        assert "on_invocation" in hook_names


# ---------------------------------------------------------------------------
# Tests: tools are properly bound
# ---------------------------------------------------------------------------


class TestPluginToolBinding:
    """Test that discovered tools are properly bound to the plugin instance."""

    def test_tool_is_callable(self) -> None:
        plugin = SimplePlugin()
        result = plugin.tools[0]("world")
        assert result == "Hello, world!"

    def test_each_instance_gets_its_own_tools(self) -> None:
        p1 = SimplePlugin()
        p2 = SimplePlugin()
        # Each instance should have distinct tool lists
        assert p1.tools is not p2.tools

    def test_tool_is_decorated_function_tool(self) -> None:
        plugin = SimplePlugin()
        assert isinstance(plugin.tools[0], DecoratedFunctionTool)


# ---------------------------------------------------------------------------
# Tests: hooks are properly bound
# ---------------------------------------------------------------------------


class TestPluginHookBinding:
    """Test that discovered hooks are properly bound to the plugin instance."""

    def test_hook_is_hook_provider(self) -> None:
        plugin = SimplePlugin()
        from strands.hooks.registry import HookProvider

        assert isinstance(plugin.hooks[0], HookProvider)

    def test_hook_is_decorated_function_hook(self) -> None:
        plugin = SimplePlugin()
        assert isinstance(plugin.hooks[0], DecoratedFunctionHook)


# ---------------------------------------------------------------------------
# Tests: property setters (filtering)
# ---------------------------------------------------------------------------


class TestPluginPropertySetters:
    """Test that tools and hooks lists can be replaced for filtering."""

    def test_tools_setter(self) -> None:
        plugin = ToolOnlyPlugin()
        assert len(plugin.tools) == 2

        # Filter down to just 'add'
        plugin.tools = [t for t in plugin.tools if t.tool_name == "add"]
        assert len(plugin.tools) == 1
        assert plugin.tools[0].tool_name == "add"

    def test_hooks_setter(self) -> None:
        plugin = HookOnlyPlugin()
        assert len(plugin.hooks) == 2

        # Filter down to just 'before_invocation'
        plugin.hooks = [h for h in plugin.hooks if h.name == "before_invocation"]
        assert len(plugin.hooks) == 1

    def test_tools_setter_empty(self) -> None:
        plugin = SimplePlugin()
        plugin.tools = []
        assert len(plugin.tools) == 0

    def test_hooks_setter_empty(self) -> None:
        plugin = SimplePlugin()
        plugin.hooks = []
        assert len(plugin.hooks) == 0


# ---------------------------------------------------------------------------
# Tests: init_plugin
# ---------------------------------------------------------------------------


class TestPluginInit:
    """Test the init_plugin lifecycle callback."""

    def test_init_plugin_default_is_noop(self) -> None:
        plugin = SimplePlugin()
        # Should not raise
        plugin.init_plugin(unittest.mock.Mock())

    def test_init_plugin_receives_agent(self) -> None:
        plugin = InitPlugin()
        mock_agent = unittest.mock.Mock()
        plugin.init_plugin(mock_agent)
        assert plugin.initialized_with is mock_agent


# ---------------------------------------------------------------------------
# Tests: name attribute
# ---------------------------------------------------------------------------


class TestPluginName:
    """Test the name attribute."""

    def test_name_from_class_attribute(self) -> None:
        plugin = SimplePlugin()
        assert plugin.name == "simple"

    def test_default_name_is_empty(self) -> None:
        class Unnamed(Plugin):
            pass

        plugin = Unnamed()
        assert plugin.name == ""


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------


class TestPluginRepr:
    """Test the __repr__ method."""

    def test_repr_includes_name(self) -> None:
        plugin = SimplePlugin()
        r = repr(plugin)
        assert "simple" in r

    def test_repr_includes_tool_names(self) -> None:
        plugin = SimplePlugin()
        r = repr(plugin)
        assert "greet" in r

    def test_repr_includes_hook_names(self) -> None:
        plugin = SimplePlugin()
        r = repr(plugin)
        assert "on_invocation" in r
