"""Unit tests for MCPClient ToolProvider functionality."""

import re
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool
from strands.tools.mcp.mcp_client import ToolFilters
from strands.types import PaginatedList
from strands.types.exceptions import ToolProviderException


@pytest.fixture
def mock_transport():
    """Create a mock transport callable."""

    def transport():
        read_stream = MagicMock()
        write_stream = MagicMock()
        return read_stream, write_stream

    return transport


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    return tool


@pytest.fixture
def mock_agent_tool(mock_mcp_tool):
    """Create a mock MCPAgentTool."""
    agent_tool = MagicMock(spec=MCPAgentTool)
    agent_tool.tool_name = "test_tool"
    agent_tool.mcp_tool = mock_mcp_tool
    return agent_tool


def create_mock_tool(tool_name: str, mcp_tool_name: str | None = None) -> MagicMock:
    """Helper to create mock tools with specific names."""
    tool = MagicMock(spec=MCPAgentTool)
    tool.tool_name = tool_name
    tool.tool_spec = {
        "name": tool_name,
        "description": f"Description for {tool_name}",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    }
    tool.mcp_tool = MagicMock(spec=MCPTool)
    tool.mcp_tool.name = mcp_tool_name or tool_name
    tool.mcp_tool.description = f"Description for {tool_name}"
    return tool


def test_init_with_tool_filters_and_prefix(mock_transport):
    """Test initialization with tool filters and prefix."""
    filters = {"allowed": ["tool1"]}
    prefix = "test_prefix"

    client = MCPClient(mock_transport, tool_filters=filters, prefix=prefix)

    assert client._tool_filters == filters
    assert client._prefix == prefix
    assert client._loaded_tools is None
    assert client._tool_provider_started is False


@pytest.mark.asyncio
async def test_load_tools_starts_client_when_not_started(mock_transport, mock_agent_tool):
    """Test that load_tools starts the client when not already started."""
    client = MCPClient(mock_transport)

    with patch.object(client, "start") as mock_start, patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([mock_agent_tool])

        tools = await client.load_tools()

        mock_start.assert_called_once()
        assert client._tool_provider_started is True
        assert len(tools) == 1
        assert tools[0] is mock_agent_tool


@pytest.mark.asyncio
async def test_load_tools_does_not_start_client_when_already_started(mock_transport, mock_agent_tool):
    """Test that load_tools does not start client when already started."""
    client = MCPClient(mock_transport)
    client._tool_provider_started = True

    with patch.object(client, "start") as mock_start, patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([mock_agent_tool])

        tools = await client.load_tools()

        mock_start.assert_not_called()
        assert len(tools) == 1


@pytest.mark.asyncio
async def test_load_tools_raises_exception_on_client_start_failure(mock_transport):
    """Test that load_tools raises ToolProviderException when client start fails."""
    client = MCPClient(mock_transport)

    with patch.object(client, "start") as mock_start:
        mock_start.side_effect = Exception("Client start failed")

        with pytest.raises(ToolProviderException, match="Failed to start MCP client: Client start failed"):
            await client.load_tools()


@pytest.mark.asyncio
async def test_load_tools_caches_tools(mock_transport, mock_agent_tool):
    """Test that load_tools caches tools and doesn't reload them."""
    client = MCPClient(mock_transport)
    client._tool_provider_started = True

    with patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([mock_agent_tool])

        # First call
        tools1 = await client.load_tools()
        # Second call
        tools2 = await client.load_tools()

        # Client should only be called once
        mock_list_tools.assert_called_once()
        assert tools1 is tools2


@pytest.mark.asyncio
async def test_load_tools_handles_pagination(mock_transport):
    """Test that load_tools handles pagination correctly."""
    tool1 = create_mock_tool("tool1")
    tool2 = create_mock_tool("tool2")

    client = MCPClient(mock_transport)
    client._tool_provider_started = True

    with patch.object(client, "list_tools_sync") as mock_list_tools:
        # Mock pagination: first page returns tool1 with next token, second page returns tool2 with no token
        mock_list_tools.side_effect = [
            PaginatedList([tool1], token="page2"),
            PaginatedList([tool2], token=None),
        ]

        tools = await client.load_tools()

        # Should have called list_tools_sync twice
        assert mock_list_tools.call_count == 2
        # First call with no token, second call with "page2" token
        mock_list_tools.assert_any_call(None, prefix=None)
        mock_list_tools.assert_any_call("page2", prefix=None)

        assert len(tools) == 2
        assert tools[0] is tool1
        assert tools[1] is tool2


@pytest.mark.asyncio
async def test_allowed_filter_string_match(mock_transport):
    """Test allowed filter with string matching."""
    tool1 = create_mock_tool("allowed_tool")
    tool2 = create_mock_tool("rejected_tool")

    filters: ToolFilters = {"allowed": ["allowed_tool"]}
    client = MCPClient(mock_transport, tool_filters=filters)
    client._tool_provider_started = True

    with patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([tool1, tool2])

        tools = await client.load_tools()

        assert len(tools) == 1
        assert tools[0].tool_name == "allowed_tool"


@pytest.mark.asyncio
async def test_allowed_filter_regex_match(mock_transport):
    """Test allowed filter with regex matching."""
    tool1 = create_mock_tool("echo_tool")
    tool2 = create_mock_tool("other_tool")

    filters: ToolFilters = {"allowed": [re.compile(r"echo_.*")]}
    client = MCPClient(mock_transport, tool_filters=filters)
    client._tool_provider_started = True

    with patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([tool1, tool2])

        tools = await client.load_tools()

        assert len(tools) == 1
        assert tools[0].tool_name == "echo_tool"


@pytest.mark.asyncio
async def test_allowed_filter_callable_match(mock_transport):
    """Test allowed filter with callable matching."""
    tool1 = create_mock_tool("short")
    tool2 = create_mock_tool("very_long_tool_name")

    def short_names_only(tool) -> bool:
        return len(tool.tool_name) <= 10

    filters: ToolFilters = {"allowed": [short_names_only]}
    client = MCPClient(mock_transport, tool_filters=filters)
    client._tool_provider_started = True

    with patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([tool1, tool2])

        tools = await client.load_tools()

        assert len(tools) == 1
        assert tools[0].tool_name == "short"


@pytest.mark.asyncio
async def test_rejected_filter_string_match(mock_transport):
    """Test rejected filter with string matching."""
    tool1 = create_mock_tool("good_tool")
    tool2 = create_mock_tool("bad_tool")

    filters: ToolFilters = {"rejected": ["bad_tool"]}
    client = MCPClient(mock_transport, tool_filters=filters)
    client._tool_provider_started = True

    with patch.object(client, "list_tools_sync") as mock_list_tools:
        mock_list_tools.return_value = PaginatedList([tool1, tool2])

        tools = await client.load_tools()

        assert len(tools) == 1
        assert tools[0].tool_name == "good_tool"


@pytest.mark.asyncio
async def test_prefix_renames_tools(mock_transport):
    """Test that prefix properly renames tools."""
    # Create a mock MCP tool (not MCPAgentTool)
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "original_name"

    client = MCPClient(mock_transport, prefix="prefix")
    client._tool_provider_started = True

    # Mock the session active state
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    client._background_thread = mock_thread

    with (
        patch.object(client, "_invoke_on_background_thread") as mock_invoke,
        patch("strands.tools.mcp.mcp_client.MCPAgentTool") as mock_agent_tool_class,
    ):
        # Mock the MCP server response
        mock_list_tools_result = MagicMock()
        mock_list_tools_result.tools = [mock_mcp_tool]
        mock_list_tools_result.nextCursor = None

        mock_future = MagicMock()
        mock_future.result.return_value = mock_list_tools_result
        mock_invoke.return_value = mock_future

        # Mock MCPAgentTool creation
        mock_agent_tool = MagicMock(spec=MCPAgentTool)
        mock_agent_tool.tool_name = "prefix_original_name"
        mock_agent_tool_class.return_value = mock_agent_tool

        # Call list_tools_sync directly to test prefix functionality
        result = client.list_tools_sync(prefix="prefix")

        # Should create MCPAgentTool with prefixed name
        mock_agent_tool_class.assert_called_once_with(mock_mcp_tool, client, name_override="prefix_original_name")

        assert len(result) == 1
        assert result[0] is mock_agent_tool


@pytest.mark.asyncio
async def test_add_consumer(mock_transport):
    """Test adding a provider consumer."""
    client = MCPClient(mock_transport)

    await client.add_consumer("consumer1")

    assert "consumer1" in client._consumers
    assert len(client._consumers) == 1


@pytest.mark.asyncio
async def test_remove_consumer_without_cleanup(mock_transport):
    """Test removing a provider consumer without triggering cleanup."""
    client = MCPClient(mock_transport)
    client._consumers.add("consumer1")
    client._consumers.add("consumer2")
    client._tool_provider_started = True

    await client.remove_consumer("consumer1")

    assert "consumer1" not in client._consumers
    assert "consumer2" in client._consumers
    assert client._tool_provider_started is True  # Should not cleanup yet


@pytest.mark.asyncio
async def test_remove_consumer_with_cleanup(mock_transport):
    """Test removing the last provider consumer triggers cleanup."""
    client = MCPClient(mock_transport)
    client._consumers.add("consumer1")
    client._tool_provider_started = True
    client._loaded_tools = [MagicMock()]

    with patch.object(client, "stop") as mock_stop:
        await client.remove_consumer("consumer1")

        assert len(client._consumers) == 0
        assert client._tool_provider_started is False
        assert client._loaded_tools is None
        mock_stop.assert_called_once_with(None, None, None)


@pytest.mark.asyncio
async def test_remove_consumer_cleanup_failure(mock_transport):
    """Test that remove_consumer raises ToolProviderException when cleanup fails."""
    client = MCPClient(mock_transport)
    client._consumers.add("consumer1")
    client._tool_provider_started = True

    with patch.object(client, "stop") as mock_stop:
        mock_stop.side_effect = Exception("Cleanup failed")

        with pytest.raises(ToolProviderException, match="Failed to cleanup MCP client: Cleanup failed"):
            await client.remove_consumer("consumer1")


def test_mcp_client_reuse_across_multiple_agents(mock_transport):
    """Test that a single MCPClient can be used across multiple agents."""
    from strands import Agent

    tool1 = create_mock_tool(tool_name="shared_echo", mcp_tool_name="echo")
    client = MCPClient(mock_transport, tool_filters={"allowed": ["echo"]}, prefix="shared")

    with (
        patch.object(client, "list_tools_sync") as mock_list_tools,
        patch.object(client, "start") as mock_start,
        patch.object(client, "stop") as mock_stop,
    ):
        mock_list_tools.return_value = PaginatedList([tool1])

        # Create two agents with the same client
        agent_1 = Agent(tools=[client])
        agent_2 = Agent(tools=[client])

        # Both agents should have the same tool
        assert "shared_echo" in agent_1.tool_names
        assert "shared_echo" in agent_2.tool_names
        assert agent_1.tool_names == agent_2.tool_names

        # Client should only be started once
        mock_start.assert_called_once()

        # First agent cleanup - client should remain active
        agent_1.cleanup()
        mock_stop.assert_not_called()  # Should not stop yet

        # Second agent should still work
        assert "shared_echo" in agent_2.tool_names

        # Final cleanup when last agent is removed
        agent_2.cleanup()
        mock_stop.assert_called_once()  # Now it should stop
