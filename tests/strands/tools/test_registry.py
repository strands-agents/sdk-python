"""
Tests for the SDK tool registry module.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import strands
from strands.experimental.tools import ToolProvider
from strands.tools import PythonAgentTool
from strands.tools.decorator import DecoratedFunctionTool, tool
from strands.tools.mcp import MCPClient
from strands.tools.registry import ToolRegistry


def test_load_tool_from_filepath_failure():
    """Test error handling when load_tool fails."""
    tool_registry = ToolRegistry()
    error_message = "Failed to load tool failing_tool: Tool file not found: /path/to/failing_tool.py"

    with pytest.raises(ValueError, match=error_message):
        tool_registry.load_tool_from_filepath("failing_tool", "/path/to/failing_tool.py")


def test_process_tools_with_invalid_path():
    """Test that process_tools raises an exception when a non-path string is passed."""
    tool_registry = ToolRegistry()
    invalid_path = "not a filepath"

    with pytest.raises(
        ValueError,
        match=f'Failed to load tool {invalid_path}: Tool string: "{invalid_path}" is not a valid tool string',
    ):
        tool_registry.process_tools([invalid_path])


def test_register_tool_with_similar_name_raises():
    tool_1 = PythonAgentTool(tool_name="tool-like-this", tool_spec=MagicMock(), tool_func=lambda: None)
    tool_2 = PythonAgentTool(tool_name="tool_like_this", tool_spec=MagicMock(), tool_func=lambda: None)

    tool_registry = ToolRegistry()

    tool_registry.register_tool(tool_1)

    with pytest.raises(ValueError) as err:
        tool_registry.register_tool(tool_2)

    assert (
        str(err.value) == "Tool name 'tool_like_this' already exists as 'tool-like-this'. "
        "Cannot add a duplicate tool which differs by a '-' or '_'"
    )


def test_get_all_tool_specs_returns_right_tool_specs():
    tool_1 = strands.tool(lambda a: a, name="tool_1")
    tool_2 = strands.tool(lambda b: b, name="tool_2")

    tool_registry = ToolRegistry()

    tool_registry.register_tool(tool_1)
    tool_registry.register_tool(tool_2)

    tool_specs = tool_registry.get_all_tool_specs()

    assert tool_specs == [
        tool_1.tool_spec,
        tool_2.tool_spec,
    ]


def test_scan_module_for_tools():
    @tool
    def tool_function_1(a):
        return a

    @tool
    def tool_function_2(b):
        return b

    def tool_function_3(c):
        return c

    def tool_function_4(d):
        return d

    tool_function_4.tool_spec = "invalid"

    mock_module = MagicMock()
    mock_module.tool_function_1 = tool_function_1
    mock_module.tool_function_2 = tool_function_2
    mock_module.tool_function_3 = tool_function_3
    mock_module.tool_function_4 = tool_function_4

    tool_registry = ToolRegistry()

    tools = tool_registry._scan_module_for_tools(mock_module)

    assert len(tools) == 2
    assert all(isinstance(tool, DecoratedFunctionTool) for tool in tools)


def test_process_tools_flattens_lists_and_tuples_and_sets():
    def function() -> str:
        return "done"

    tool_a = tool(name="tool_a")(function)
    tool_b = tool(name="tool_b")(function)
    tool_c = tool(name="tool_c")(function)
    tool_d = tool(name="tool_d")(function)
    tool_e = tool(name="tool_e")(function)
    tool_f = tool(name="tool_f")(function)

    registry = ToolRegistry()

    all_tools = [tool_a, (tool_b, tool_c), [{tool_d, tool_e}, [tool_f]]]

    tru_tool_names = sorted(registry.process_tools(all_tools))
    exp_tool_names = [
        "tool_a",
        "tool_b",
        "tool_c",
        "tool_d",
        "tool_e",
        "tool_f",
    ]
    assert tru_tool_names == exp_tool_names


def test_register_tool_duplicate_name_without_hot_reload():
    """Test that registering a tool with duplicate name raises ValueError when hot reload is not supported."""
    # Create mock tools that don't support hot reload
    tool_1 = MagicMock()
    tool_1.tool_name = "duplicate_tool"
    tool_1.supports_hot_reload = False
    tool_1.is_dynamic = False

    tool_2 = MagicMock()
    tool_2.tool_name = "duplicate_tool"
    tool_2.supports_hot_reload = False
    tool_2.is_dynamic = False

    tool_registry = ToolRegistry()
    tool_registry.register_tool(tool_1)

    with pytest.raises(
        ValueError, match="Tool name 'duplicate_tool' already exists. Cannot register tools with exact same name."
    ):
        tool_registry.register_tool(tool_2)


def test_register_tool_duplicate_name_with_hot_reload():
    """Test that registering a tool with duplicate name succeeds when hot reload is supported."""
    # Create mock tools with hot reload support
    tool_1 = MagicMock(spec=PythonAgentTool)
    tool_1.tool_name = "hot_reload_tool"
    tool_1.supports_hot_reload = True
    tool_1.is_dynamic = False

    tool_2 = MagicMock(spec=PythonAgentTool)
    tool_2.tool_name = "hot_reload_tool"
    tool_2.supports_hot_reload = True
    tool_2.is_dynamic = False

    tool_registry = ToolRegistry()
    tool_registry.register_tool(tool_1)

    tool_registry.register_tool(tool_2)

    # Verify the second tool replaced the first
    assert tool_registry.registry["hot_reload_tool"] == tool_2


def test_register_strands_tools_from_module():
    tool_registry = ToolRegistry()
    tool_registry.process_tools(["tests.fixtures.say_tool"])

    assert len(tool_registry.registry) == 2
    assert "say" in tool_registry.registry
    assert "dont_say" in tool_registry.registry


def test_register_strands_tools_specific_tool_from_module():
    tool_registry = ToolRegistry()
    tool_registry.process_tools(["tests.fixtures.say_tool:say"])

    assert len(tool_registry.registry) == 1
    assert "say" in tool_registry.registry
    assert "dont_say" not in tool_registry.registry


def test_register_strands_tools_specific_tool_from_module_tool_missing():
    tool_registry = ToolRegistry()

    with pytest.raises(ValueError, match="Failed to load tool tests.fixtures.say_tool:nay: "):
        tool_registry.process_tools(["tests.fixtures.say_tool:nay"])


def test_register_strands_tools_specific_tool_from_module_not_a_tool():
    tool_registry = ToolRegistry()

    with pytest.raises(ValueError, match="Failed to load tool tests.fixtures.say_tool:not_a_tool: "):
        tool_registry.process_tools(["tests.fixtures.say_tool:not_a_tool"])


def test_register_strands_tools_with_dict():
    tool_registry = ToolRegistry()
    tool_registry.process_tools([{"path": "tests.fixtures.say_tool"}])

    assert len(tool_registry.registry) == 2
    assert "say" in tool_registry.registry
    assert "dont_say" in tool_registry.registry


def test_register_strands_tools_specific_tool_with_dict():
    tool_registry = ToolRegistry()
    tool_registry.process_tools([{"path": "tests.fixtures.say_tool", "name": "say"}])

    assert len(tool_registry.registry) == 1
    assert "say" in tool_registry.registry


def test_register_strands_tools_specific_tool_with_dict_not_found():
    tool_registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Failed to load tool {'path': 'tests.fixtures.say_tool'"
        ", 'name': 'nay'}: Tool \"nay\" not found in \"tests.fixtures.say_tool\"",
    ):
        tool_registry.process_tools([{"path": "tests.fixtures.say_tool", "name": "nay"}])


def test_register_strands_tools_module_no_spec():
    tool_registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Failed to load tool tests.fixtures.mocked_model_provider: "
        "The module mocked_model_provider is not a valid module",
    ):
        tool_registry.process_tools(["tests.fixtures.mocked_model_provider"])


def test_register_strands_tools_module_no_function():
    tool_registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Failed to load tool tests.fixtures.tool_with_spec_but_no_function: "
        "Module-based tool tool_with_spec_but_no_function missing function tool_with_spec_but_no_function",
    ):
        tool_registry.process_tools(["tests.fixtures.tool_with_spec_but_no_function"])


def test_register_strands_tools_module_non_callable_function():
    tool_registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Failed to load tool tests.fixtures.tool_with_spec_but_non_callable_function:"
        " Tool tool_with_spec_but_non_callable_function function is not callable",
    ):
        tool_registry.process_tools(["tests.fixtures.tool_with_spec_but_non_callable_function"])


def test_tool_registry_cleanup_with_mcp_client():
    """Test that ToolRegistry cleanup properly handles MCP clients without orphaning threads."""
    # Create a mock MCP client that simulates a real tool provider
    mock_transport = MagicMock()
    mock_client = MCPClient(mock_transport)

    # Mock the client to avoid actual network operations
    mock_client.load_tools = AsyncMock(return_value=[])

    registry = ToolRegistry()

    # Use process_tools to properly register the client
    registry.process_tools([mock_client])

    # Verify the client was registered as a consumer
    assert registry._registry_id in mock_client._consumers

    # Test cleanup calls remove_consumer
    registry.cleanup()

    # Verify cleanup was attempted
    assert registry._registry_id not in mock_client._consumers


def test_tool_registry_cleanup_exception_handling():
    """Test that ToolRegistry cleanup attempts all providers even if some fail."""
    # Create mock providers - one that fails, one that succeeds
    failing_provider = MagicMock()
    failing_provider.remove_consumer.side_effect = Exception("Cleanup failed")

    working_provider = MagicMock()

    registry = ToolRegistry()
    registry._tool_providers = [failing_provider, working_provider]

    # Cleanup should attempt both providers and raise the first exception
    with pytest.raises(Exception, match="Cleanup failed"):
        registry.cleanup()

    # Verify both providers were attempted
    failing_provider.remove_consumer.assert_called_once()
    working_provider.remove_consumer.assert_called_once()


def test_tool_registry_cleanup_idempotent():
    """Test that ToolRegistry cleanup is idempotent."""
    provider = MagicMock(spec=ToolProvider)
    provider.load_tools = AsyncMock(return_value=[])

    registry = ToolRegistry()

    # Use process_tools to properly register the provider
    registry.process_tools([provider])

    # First cleanup should call remove_consumer
    registry.cleanup()
    provider.remove_consumer.assert_called_once_with(registry._registry_id)

    # Reset mock call count
    provider.remove_consumer.reset_mock()

    # Second cleanup should call remove_consumer again (not idempotent yet)
    # This test documents current behavior - registry cleanup is not idempotent
    registry.cleanup()
    provider.remove_consumer.assert_called_once_with(registry._registry_id)


def test_tool_registry_process_tools_exception_after_add_consumer():
    """Test that tool provider is still tracked for cleanup even if load_tools fails."""
    # Create a mock tool provider that fails during load_tools
    mock_provider = MagicMock(spec=ToolProvider)
    mock_provider.add_consumer = MagicMock()
    mock_provider.remove_consumer = MagicMock()

    async def failing_load_tools():
        raise Exception("Failed to load tools")

    mock_provider.load_tools = AsyncMock(side_effect=failing_load_tools)

    registry = ToolRegistry()

    # Processing should fail but provider should still be tracked
    with pytest.raises(ValueError, match="Failed to load tool"):
        registry.process_tools([mock_provider])

    # Verify provider was added to registry for cleanup tracking
    assert mock_provider in registry._tool_providers

    # Verify add_consumer was called before the failure
    mock_provider.add_consumer.assert_called_once_with(registry._registry_id)

    # Cleanup should still work
    registry.cleanup()
    mock_provider.remove_consumer.assert_called_once_with(registry._registry_id)


def test_tool_registry_add_consumer_before_load_tools():
    """Test that add_consumer is called before load_tools to ensure cleanup tracking."""
    # Create a mock tool provider that tracks call order
    mock_provider = MagicMock(spec=ToolProvider)
    call_order = []

    def track_add_consumer(*args, **kwargs):
        call_order.append("add_consumer")

    async def track_load_tools(*args, **kwargs):
        call_order.append("load_tools")
        return []

    mock_provider.add_consumer.side_effect = track_add_consumer
    mock_provider.load_tools = AsyncMock(side_effect=track_load_tools)

    registry = ToolRegistry()

    # Process the tool provider
    registry.process_tools([mock_provider])

    # Verify add_consumer was called before load_tools
    assert call_order == ["add_consumer", "load_tools"]

    # Verify the provider was added to the registry for cleanup
    assert mock_provider in registry._tool_providers

    # Verify add_consumer was called with the registry ID
    mock_provider.add_consumer.assert_called_once_with(registry._registry_id)
