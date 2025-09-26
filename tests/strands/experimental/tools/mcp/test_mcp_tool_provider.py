"""Unit tests for MCPToolProvider."""

import re
from unittest.mock import MagicMock, patch

import pytest

from strands.experimental.tools.mcp import MCPToolProvider, ToolFilters
from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool
from strands.types import PaginatedList
from strands.types.exceptions import ToolProviderException


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    client = MagicMock(spec=MCPClient)
    client.start = MagicMock()
    client.stop = MagicMock()
    client.list_tools_sync = MagicMock()
    return client


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    return tool


@pytest.fixture
def mock_agent_tool(mock_mcp_tool, mock_mcp_client):
    """Create a mock MCPAgentTool."""
    agent_tool = MagicMock(spec=MCPAgentTool)
    agent_tool.tool_name = "test_tool"
    agent_tool.mcp_tool = mock_mcp_tool
    agent_tool.mcp_client = mock_mcp_client
    return agent_tool


def create_mock_tool(name: str) -> MagicMock:
    """Helper to create mock tools with specific names."""
    tool = MagicMock(spec=MCPAgentTool)
    tool.tool_name = name
    tool.mcp_tool = MagicMock()
    tool.mcp_tool.name = name
    return tool


def test_init_with_client_only(mock_mcp_client):
    """Test initialization with only client."""
    provider = MCPToolProvider(client=mock_mcp_client)

    assert provider._client is mock_mcp_client
    assert provider._tool_filters is None
    assert provider._prefix is None
    assert provider._tools is None
    assert provider._started is False


def test_init_with_all_parameters(mock_mcp_client):
    """Test initialization with all parameters."""
    filters = {"allowed": ["tool1"]}
    prefix = "test_prefix"

    provider = MCPToolProvider(client=mock_mcp_client, tool_filters=filters, prefix=prefix)

    assert provider._client is mock_mcp_client
    assert provider._tool_filters == filters
    assert provider._prefix == prefix
    assert provider._tools is None
    assert provider._started is False


@pytest.mark.asyncio
async def test_load_tools_starts_client_when_not_started(mock_mcp_client, mock_agent_tool):
    """Test that load_tools starts the client when not already started."""
    mock_mcp_client.list_tools_sync.return_value = PaginatedList([mock_agent_tool])

    provider = MCPToolProvider(client=mock_mcp_client)

    tools = await provider.load_tools()

    mock_mcp_client.start.assert_called_once()
    assert provider._started is True
    assert len(tools) == 1
    assert tools[0] is mock_agent_tool


@pytest.mark.asyncio
async def test_load_tools_does_not_start_client_when_already_started(mock_mcp_client, mock_agent_tool):
    """Test that load_tools does not start client when already started."""
    mock_mcp_client.list_tools_sync.return_value = PaginatedList([mock_agent_tool])

    provider = MCPToolProvider(client=mock_mcp_client)
    provider._started = True

    tools = await provider.load_tools()

    mock_mcp_client.start.assert_not_called()
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_load_tools_raises_exception_on_client_start_failure(mock_mcp_client):
    """Test that load_tools raises ToolProviderException when client start fails."""
    mock_mcp_client.start.side_effect = Exception("Client start failed")

    provider = MCPToolProvider(client=mock_mcp_client)

    with pytest.raises(ToolProviderException, match="Failed to start MCP client: Client start failed"):
        await provider.load_tools()


@pytest.mark.asyncio
async def test_load_tools_caches_tools(mock_mcp_client, mock_agent_tool):
    """Test that load_tools caches tools and doesn't reload them."""
    mock_mcp_client.list_tools_sync.return_value = PaginatedList([mock_agent_tool])

    provider = MCPToolProvider(client=mock_mcp_client)

    # First call
    tools1 = await provider.load_tools()
    # Second call
    tools2 = await provider.load_tools()

    # Client should only be called once
    mock_mcp_client.list_tools_sync.assert_called_once()
    assert tools1 is tools2


@pytest.mark.asyncio
async def test_load_tools_handles_pagination(mock_mcp_client, mock_agent_tool):
    """Test that load_tools handles pagination correctly."""
    tool1 = MagicMock(spec=MCPAgentTool)
    tool1.tool_name = "tool1"
    tool2 = MagicMock(spec=MCPAgentTool)
    tool2.tool_name = "tool2"

    # Mock pagination: first page returns tool1 with next token, second page returns tool2 with no token
    mock_mcp_client.list_tools_sync.side_effect = [
        PaginatedList([tool1], token="page2"),
        PaginatedList([tool2], token=None),
    ]

    provider = MCPToolProvider(client=mock_mcp_client)

    tools = await provider.load_tools()

    # Should have called list_tools_sync twice
    assert mock_mcp_client.list_tools_sync.call_count == 2
    # First call with no token, second call with "page2" token
    mock_mcp_client.list_tools_sync.assert_any_call(None)
    mock_mcp_client.list_tools_sync.assert_any_call("page2")

    assert len(tools) == 2
    assert tools[0] is tool1
    assert tools[1] is tool2


@pytest.mark.asyncio
async def test_allowed_filter_string_match(mock_mcp_client):
    """Test allowed filter with string matching."""
    tool1 = create_mock_tool("allowed_tool")
    tool2 = create_mock_tool("rejected_tool")

    mock_mcp_client.list_tools_sync.return_value = PaginatedList([tool1, tool2])

    filters: ToolFilters = {"allowed": ["allowed_tool"]}
    provider = MCPToolProvider(client=mock_mcp_client, tool_filters=filters)

    tools = await provider.load_tools()

    assert len(tools) == 1
    assert tools[0].tool_name == "allowed_tool"


@pytest.mark.asyncio
async def test_allowed_filter_regex_match(mock_mcp_client):
    """Test allowed filter with regex matching."""
    tool1 = create_mock_tool("echo_tool")
    tool2 = create_mock_tool("other_tool")

    mock_mcp_client.list_tools_sync.return_value = PaginatedList([tool1, tool2])

    filters: ToolFilters = {"allowed": [re.compile(r"echo_.*")]}
    provider = MCPToolProvider(client=mock_mcp_client, tool_filters=filters)

    tools = await provider.load_tools()

    assert len(tools) == 1
    assert tools[0].tool_name == "echo_tool"


@pytest.mark.asyncio
async def test_allowed_filter_callable_match(mock_mcp_client):
    """Test allowed filter with callable matching."""
    tool1 = create_mock_tool("short")
    tool2 = create_mock_tool("very_long_tool_name")

    mock_mcp_client.list_tools_sync.return_value = PaginatedList([tool1, tool2])

    def short_names_only(tool) -> bool:
        return len(tool.tool_name) <= 10

    filters: ToolFilters = {"allowed": [short_names_only]}
    provider = MCPToolProvider(client=mock_mcp_client, tool_filters=filters)

    tools = await provider.load_tools()

    assert len(tools) == 1
    assert tools[0].tool_name == "short"


@pytest.mark.asyncio
async def test_rejected_filter(mock_mcp_client):
    """Test rejected filter functionality."""
    tool1 = create_mock_tool("good_tool")
    tool2 = create_mock_tool("bad_tool")

    mock_mcp_client.list_tools_sync.return_value = PaginatedList([tool1, tool2])

    filters: ToolFilters = {"rejected": ["bad_tool"]}
    provider = MCPToolProvider(client=mock_mcp_client, tool_filters=filters)

    tools = await provider.load_tools()

    assert len(tools) == 1
    assert tools[0].tool_name == "good_tool"


@pytest.mark.asyncio
async def test_prefix_renames_tools(mock_mcp_client):
    """Test that prefix properly renames tools."""
    original_tool = MagicMock(spec=MCPAgentTool)
    original_tool.tool_name = "original_name"
    original_tool.mcp_tool = MagicMock()
    original_tool.mcp_tool.name = "original_name"
    original_tool.mcp_client = mock_mcp_client

    mock_mcp_client.list_tools_sync.return_value = PaginatedList([original_tool])

    with patch("strands.experimental.tools.mcp.mcp_tool_provider.MCPAgentTool") as mock_agent_tool_class:
        new_tool = MagicMock(spec=MCPAgentTool)
        new_tool.tool_name = "prefix_original_name"
        mock_agent_tool_class.return_value = new_tool

        provider = MCPToolProvider(client=mock_mcp_client, prefix="prefix")

        tools = await provider.load_tools()

        # Should create new MCPAgentTool with prefixed name
        mock_agent_tool_class.assert_called_once_with(
            original_tool.mcp_tool, original_tool.mcp_client, agent_facing_tool_name="prefix_original_name"
        )

        assert len(tools) == 1
        assert tools[0] is new_tool


@pytest.mark.asyncio
async def test_cleanup_stops_client_when_started(mock_mcp_client):
    """Test that cleanup stops the client when started."""
    provider = MCPToolProvider(client=mock_mcp_client)
    provider._started = True
    provider._tools = [MagicMock()]

    await provider.cleanup()

    mock_mcp_client.stop.assert_called_once_with(None, None, None)
    assert provider._started is False
    assert provider._tools is None


@pytest.mark.asyncio
async def test_cleanup_does_nothing_when_not_started(mock_mcp_client):
    """Test that cleanup does nothing when not started."""
    provider = MCPToolProvider(client=mock_mcp_client)
    provider._started = False

    await provider.cleanup()

    mock_mcp_client.stop.assert_not_called()
    assert provider._started is False


@pytest.mark.asyncio
async def test_cleanup_raises_exception_on_client_stop_failure(mock_mcp_client):
    """Test that cleanup raises ToolProviderException when client stop fails."""
    mock_mcp_client.stop.side_effect = Exception("Client stop failed")

    provider = MCPToolProvider(client=mock_mcp_client)
    provider._started = True

    with pytest.raises(ToolProviderException, match="Failed to cleanup MCP client: Client stop failed"):
        await provider.cleanup()

    # State is not reset when cleanup fails
    assert provider._started is True
    assert provider._tools is None


@pytest.mark.asyncio
async def test_cleanup_does_not_reset_state_on_exception(mock_mcp_client):
    """Test that cleanup does not reset state when exception occurs."""
    mock_mcp_client.stop.side_effect = Exception("Client stop failed")

    provider = MCPToolProvider(client=mock_mcp_client)
    provider._started = True
    mock_tool = MagicMock()
    provider._tools = [mock_tool]

    with pytest.raises(ToolProviderException):
        await provider.cleanup()

    # State should not be reset when exception occurs
    assert provider._started is True
    assert provider._tools == [mock_tool]


@pytest.mark.asyncio
async def test_load_tools_with_empty_tool_list(mock_mcp_client):
    """Test load_tools with empty tool list from server."""
    mock_mcp_client.list_tools_sync.return_value = PaginatedList([])

    provider = MCPToolProvider(client=mock_mcp_client)

    tools = await provider.load_tools()

    assert len(tools) == 0
    assert provider._started is True


@pytest.mark.asyncio
async def test_load_tools_with_no_filters(mock_mcp_client, mock_agent_tool):
    """Test load_tools with no filters applied."""
    mock_mcp_client.list_tools_sync.return_value = PaginatedList([mock_agent_tool])

    provider = MCPToolProvider(client=mock_mcp_client, tool_filters=None)

    tools = await provider.load_tools()

    assert len(tools) == 1
    assert tools[0] is mock_agent_tool


@pytest.mark.asyncio
async def test_load_tools_with_empty_filters(mock_mcp_client, mock_agent_tool):
    """Test load_tools with empty filters dict."""
    mock_mcp_client.list_tools_sync.return_value = PaginatedList([mock_agent_tool])

    provider = MCPToolProvider(client=mock_mcp_client, tool_filters={})

    tools = await provider.load_tools()

    assert len(tools) == 1
    assert tools[0] is mock_agent_tool
