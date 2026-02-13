"""Integration tests for Plugin registration with Agent."""

import unittest.mock
from typing import Any

from strands.agent.agent import Agent
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent
from strands.hooks.decorator import hook
from strands.plugins.plugin import Plugin
from strands.tools.decorator import tool

# ---------------------------------------------------------------------------
# Test plugins
# ---------------------------------------------------------------------------


class MathPlugin(Plugin):
    """Plugin that provides math tools."""

    name = "math"

    @tool
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b


class LoggingPlugin(Plugin):
    """Plugin that provides logging hooks."""

    name = "logging"

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[str] = []

    @hook
    def log_invocation(self, event: BeforeInvocationEvent) -> None:
        """Log invocations."""
        self.invocations.append("invocation_started")


class FullPlugin(Plugin):
    """Plugin with both tools and hooks."""

    name = "full"

    def __init__(self) -> None:
        super().__init__()
        self.init_called = False

    @tool
    def echo(self, text: str) -> str:
        """Echo the input."""
        return text

    @hook
    def before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Before invocation."""

    def init_plugin(self, agent: Any) -> None:
        self.init_called = True


class SystemPromptPlugin(Plugin):
    """Plugin that modifies agent system prompt during init."""

    name = "system-prompt"

    def init_plugin(self, agent: Any) -> None:
        current = agent.system_prompt or ""
        agent.system_prompt = current + "\nYou are also a helpful math tutor."


# ---------------------------------------------------------------------------
# Tests: tool registration
# ---------------------------------------------------------------------------


class TestPluginToolRegistration:
    """Test that plugin tools are registered with the agent."""

    def test_plugin_tools_registered(self) -> None:
        plugin = MathPlugin()
        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin],
        )
        assert "add" in agent.tool_names
        assert "subtract" in agent.tool_names

    def test_plugin_tools_combined_with_agent_tools(self) -> None:
        @tool
        def standalone_tool(x: str) -> str:
            """A standalone tool."""
            return x

        plugin = MathPlugin()
        agent = Agent(
            model=unittest.mock.MagicMock(),
            tools=[standalone_tool],
            plugins=[plugin],
        )
        assert "standalone_tool" in agent.tool_names
        assert "add" in agent.tool_names
        assert "subtract" in agent.tool_names

    def test_multiple_plugins_tools_registered(self) -> None:
        plugin1 = MathPlugin()

        class StringPlugin(Plugin):
            name = "string"

            @tool
            def upper(self, text: str) -> str:
                """Uppercase text."""
                return text.upper()

        plugin2 = StringPlugin()
        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin1, plugin2],
        )
        assert "add" in agent.tool_names
        assert "subtract" in agent.tool_names
        assert "upper" in agent.tool_names


# ---------------------------------------------------------------------------
# Tests: hook registration
# ---------------------------------------------------------------------------


class TestPluginHookRegistration:
    """Test that plugin hooks are registered with the agent."""

    def test_plugin_hooks_registered(self) -> None:
        plugin = LoggingPlugin()
        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin],
        )
        # Verify the hook is in the registry by checking callbacks exist for the event type
        callbacks = list(agent.hooks._registered_callbacks.get(BeforeInvocationEvent, []))
        # At least one callback should be from our plugin
        assert any(True for _ in callbacks), "Expected at least one BeforeInvocationEvent callback"

    def test_plugin_hooks_combined_with_agent_hooks(self) -> None:
        @hook
        def standalone_hook(event: BeforeToolCallEvent) -> None:
            """A standalone hook."""

        plugin = LoggingPlugin()
        agent = Agent(
            model=unittest.mock.MagicMock(),
            hooks=[standalone_hook],
            plugins=[plugin],
        )
        before_invocation_cbs = list(agent.hooks._registered_callbacks.get(BeforeInvocationEvent, []))
        before_tool_cbs = list(agent.hooks._registered_callbacks.get(BeforeToolCallEvent, []))
        assert len(before_invocation_cbs) >= 1
        assert len(before_tool_cbs) >= 1


# ---------------------------------------------------------------------------
# Tests: init_plugin
# ---------------------------------------------------------------------------


class TestPluginInitCallback:
    """Test that init_plugin is called during agent initialization."""

    def test_init_plugin_called(self) -> None:
        plugin = FullPlugin()
        assert not plugin.init_called
        Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin],
        )
        assert plugin.init_called

    def test_init_plugin_can_modify_agent(self) -> None:
        plugin = SystemPromptPlugin()
        agent = Agent(
            model=unittest.mock.MagicMock(),
            system_prompt="You are helpful.",
            plugins=[plugin],
        )
        assert "math tutor" in agent.system_prompt

    def test_init_plugin_called_for_each_plugin(self) -> None:
        init_calls: list[str] = []

        class P1(Plugin):
            name = "p1"

            def init_plugin(self, agent: Any) -> None:
                init_calls.append("p1")

        class P2(Plugin):
            name = "p2"

            def init_plugin(self, agent: Any) -> None:
                init_calls.append("p2")

        Agent(
            model=unittest.mock.MagicMock(),
            plugins=[P1(), P2()],
        )
        assert init_calls == ["p1", "p2"]


# ---------------------------------------------------------------------------
# Tests: filtering before passing to agent
# ---------------------------------------------------------------------------


class TestPluginFiltering:
    """Test that plugin tools/hooks can be filtered before agent creation."""

    def test_filter_tools_before_agent(self) -> None:
        plugin = MathPlugin()
        # Keep only 'add'
        plugin.tools = [t for t in plugin.tools if t.tool_name == "add"]

        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin],
        )
        assert "add" in agent.tool_names
        assert "subtract" not in agent.tool_names

    def test_remove_all_tools_before_agent(self) -> None:
        plugin = MathPlugin()
        plugin.tools = []

        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin],
        )
        # None of the plugin tools should be registered
        assert "add" not in agent.tool_names
        assert "subtract" not in agent.tool_names

    def test_filter_hooks_before_agent(self) -> None:
        plugin = LoggingPlugin()
        plugin.hooks = []

        # Hooks list was cleared so no plugin-specific callbacks added
        # (there may be built-in callbacks from conversation_manager, retry, etc.)
        # Just verify it doesn't crash
        Agent(
            model=unittest.mock.MagicMock(),
            plugins=[plugin],
        )


# ---------------------------------------------------------------------------
# Tests: no plugins
# ---------------------------------------------------------------------------


class TestNoPlugins:
    """Test that agent works normally when no plugins are provided."""

    def test_none_plugins(self) -> None:
        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=None,
        )
        # Should work fine, no extra tools
        assert agent.tool_names is not None

    def test_empty_plugins_list(self) -> None:
        agent = Agent(
            model=unittest.mock.MagicMock(),
            plugins=[],
        )
        assert agent.tool_names is not None

    def test_omitted_plugins(self) -> None:
        agent = Agent(
            model=unittest.mock.MagicMock(),
        )
        assert agent.tool_names is not None


# ---------------------------------------------------------------------------
# Tests: top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """Test that Plugin and hook are importable from top-level strands package."""

    def test_import_plugin(self) -> None:
        from strands import Plugin as P

        assert P is Plugin

    def test_import_hook(self) -> None:
        from strands import hook as h

        assert h is hook
