from unittest.mock import ANY, MagicMock, Mock

import pytest

import strands
from strands.experimental.hooks import AfterToolInvocation, BeforeToolInvocation, HookRegistry
from strands.handlers.tool_handler import AgentToolHandler
from tests.fixtures.mock_hook_provider import MockHookProvider


@pytest.fixture
def tool_times_2():
    @strands.tools.tool
    def multiply_by_2(x: int) -> int:
        return x * 2

    return multiply_by_2


@pytest.fixture
def tool_times_5():
    @strands.tools.tool
    def multiply_by_5(x: int) -> int:
        return x * 5

    return multiply_by_5


@pytest.fixture
def tool_registry(tool_times_2, tool_times_5):
    registry = strands.tools.registry.ToolRegistry()
    registry.register_tool(tool_times_2)
    registry.register_tool(tool_times_5)
    return registry


@pytest.fixture
def mock_provider():
    return MockHookProvider(event_types=[BeforeToolInvocation, AfterToolInvocation])


@pytest.fixture
def hook_registry(mock_provider):
    hook_registry = HookRegistry()
    hook_registry.add_hook(mock_provider)
    return hook_registry


@pytest.fixture
def agent(hook_registry):
    agent = MagicMock()
    agent._hooks = hook_registry
    return agent


@pytest.fixture
def tool_handler(agent, tool_registry):
    return strands.handlers.tool_handler.AgentToolHandler(agent, tool_registry)


@pytest.fixture
def tool_use_identity(tool_registry):
    @strands.tools.tool
    def identity(a: int) -> int:
        return a

    tool_registry.register_tool(identity)

    return {"toolUseId": "identity", "name": "identity", "input": {"a": 1}}


def test_process(tool_handler, tool_use_identity, generate):
    process = tool_handler.process(
        tool_use_identity,
        model=Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    _, tru_result = generate(process)
    exp_result = {"toolUseId": "identity", "status": "success", "content": [{"text": "1"}]}

    assert tru_result == exp_result


def test_process_missing_tool(tool_handler, generate):
    process = tool_handler.process(
        tool={"toolUseId": "missing", "name": "missing", "input": {}},
        model=Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    _, tru_result = generate(process)
    exp_result = {
        "toolUseId": "missing",
        "status": "error",
        "content": [{"text": "Unknown tool: missing"}],
    }

    assert tru_result == exp_result


def test_process_hooks(agent, tool_handler, generate, hook_registry, tool_times_2, mock_provider):
    """Test that the correct hooks are emitted."""

    process = tool_handler.process(
        tool={"toolUseId": "test", "name": tool_times_2.tool_name, "input": {"x": 5}},
        model=Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    _, result = generate(process)

    assert len(mock_provider.events_received) == 2

    assert mock_provider.events_received[0] == BeforeToolInvocation(
        agent=agent,
        selected_tool=tool_times_2,
        tool_use={"input": {"x": 5}, "name": "multiply_by_2", "toolUseId": "test"},
        kwargs=ANY,
    )

    assert mock_provider.events_received[1] == AfterToolInvocation(
        agent=agent,
        selected_tool=tool_times_2,
        exception=None,
        tool_use={"toolUseId": "test", "name": tool_times_2.tool_name, "input": {"x": 5}},
        result={"toolUseId": "test", "status": "success", "content": [{"text": "10"}]},
        kwargs=ANY,
    )


def test_process_hook_after_tool_invocation_on_exception(agent, tool_registry, generate, mock_provider):
    """Test that AfterToolInvocation hook is invoked even when tool throws exception."""
    error = ValueError("Tool failed")

    failing_tool = MagicMock()
    failing_tool.tool_name = "failing_tool"

    failing_tool.invoke.side_effect = error

    tool_registry.register_tool(failing_tool)
    tool_handler = AgentToolHandler(agent, tool_registry)

    process = tool_handler.process(
        tool={"toolUseId": "test", "name": "failing_tool", "input": {"x": 5}},
        model=Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    _, result = generate(process)

    assert mock_provider.events_received[1] == AfterToolInvocation(
        agent=agent,
        selected_tool=failing_tool,
        tool_use={"input": {"x": 5}, "name": "failing_tool", "toolUseId": "test"},
        kwargs=ANY,
        result={"content": [{"text": "Error: Tool failed"}], "status": "error", "toolUseId": "test"},
        exception=error,
    )


def test_process_hook_before_tool_invocation_updates(
    agent, tool_handler, tool_times_5, generate, hook_registry, mock_provider
):
    """Test that modifying properties on BeforeToolInvocation takes effect."""

    updated_tool_use = {"toolUseId": "modified", "name": "replacement_tool", "input": {"x": 3}}

    def modify_hook(event: BeforeToolInvocation):
        # Modify selected_tool to use replacement_tool
        event.selected_tool = tool_times_5
        # Modify tool_use to change toolUseId
        event.tool_use = updated_tool_use

    hook_registry.add_callback(BeforeToolInvocation, modify_hook)

    process = tool_handler.process(
        tool={"toolUseId": "original", "name": "original_tool", "input": {"x": 1}},
        model=Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    _, result = generate(process)

    # Should use replacement_tool (5 * 3 = 15) instead of original_tool (1 * 2 = 2)
    assert result == {"toolUseId": "modified", "status": "success", "content": [{"text": "15"}]}

    assert mock_provider.events_received[1] == AfterToolInvocation(
        agent=agent,
        selected_tool=tool_times_5,
        tool_use=updated_tool_use,
        kwargs=ANY,
        result={"content": [{"text": "15"}], "status": "success", "toolUseId": "modified"},
        exception=None,
    )


def test_process_hook_after_tool_invocation_updates(agent, tool_handler, tool_times_2, generate, hook_registry):
    """Test that modifying properties on AfterToolInvocation takes effect."""

    updated_result = {"toolUseId": "modified", "status": "success", "content": [{"text": "modified_result"}]}

    def modify_hook(event: AfterToolInvocation):
        # Modify result to change the output
        event.result = updated_result

    hook_registry.add_callback(AfterToolInvocation, modify_hook)

    process = tool_handler.process(
        tool={"toolUseId": "test", "name": tool_times_2.tool_name, "input": {"x": 5}},
        model=Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    _, result = generate(process)

    # Should return modified result instead of original (5 * 2 = 10)
    assert result == updated_result
