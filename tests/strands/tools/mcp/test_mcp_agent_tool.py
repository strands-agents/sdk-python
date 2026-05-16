from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPAgentTool, MCPClient
from strands.types._events import ToolResultEvent


@pytest.fixture
def mock_mcp_client():
    mock_server = MagicMock(spec=MCPClient)
    mock_server.call_tool_sync.return_value = {
        "status": "success",
        "toolUseId": "test-123",
        "content": [{"text": "Success result"}],
    }
    # Mock internal methods used by MCPAgentTool.stream()
    mock_server._create_call_tool_coroutine.return_value = MagicMock()
    mock_server._handle_tool_result.return_value = {
        "status": "success",
        "toolUseId": "test-123",
        "content": [{"text": "Success result"}],
    }
    mock_server._handle_tool_execution_error.return_value = {
        "status": "error",
        "toolUseId": "test-123",
        "content": [{"text": "error"}],
        "isError": True,
    }
    return mock_server


@pytest.fixture
def mock_mcp_tool():
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {"type": "object", "properties": {}}
    mock_tool.outputSchema = None  # MCP tools can have optional outputSchema
    return mock_tool


@pytest.fixture
def mcp_agent_tool(mock_mcp_tool, mock_mcp_client):
    return MCPAgentTool(mock_mcp_tool, mock_mcp_client)


def test_tool_name(mcp_agent_tool, mock_mcp_tool):
    assert mcp_agent_tool.tool_name == "test_tool"
    assert mcp_agent_tool.tool_name == mock_mcp_tool.name


def test_tool_type(mcp_agent_tool):
    assert mcp_agent_tool.tool_type == "python"


def test_tool_spec_with_description(mcp_agent_tool, mock_mcp_tool):
    tool_spec = mcp_agent_tool.tool_spec

    assert tool_spec["name"] == "test_tool"
    assert tool_spec["description"] == "A test tool"
    assert tool_spec["inputSchema"]["json"] == {"type": "object", "properties": {}}
    assert "outputSchema" not in tool_spec


def test_tool_spec_without_description(mock_mcp_tool, mock_mcp_client):
    mock_mcp_tool.description = None

    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    tool_spec = agent_tool.tool_spec

    assert tool_spec["description"] == "Tool which performs test_tool"


def test_tool_spec_with_output_schema(mock_mcp_tool, mock_mcp_client):
    mock_mcp_tool.outputSchema = {"type": "object", "properties": {"result": {"type": "string"}}}

    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    tool_spec = agent_tool.tool_spec

    assert "outputSchema" in tool_spec
    assert tool_spec["outputSchema"]["json"] == {"type": "object", "properties": {"result": {"type": "string"}}}


def test_tool_spec_without_output_schema(mock_mcp_tool, mock_mcp_client):
    mock_mcp_tool.outputSchema = None

    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    tool_spec = agent_tool.tool_spec

    assert "outputSchema" not in tool_spec


@pytest.mark.asyncio
async def test_stream(mcp_agent_tool, mock_mcp_client, alist):
    tool_use = {"toolUseId": "test-123", "name": "test_tool", "input": {"param": "value"}}

    mock_result = mock_mcp_client._handle_tool_result.return_value

    with patch("asyncio.wrap_future") as mock_wrap_future:
        # Make wrap_future return a coroutine that resolves to the mock call_tool result
        async def mock_awaitable(_):
            return MagicMock()  # call_tool_result (raw MCP response)

        mock_wrap_future.side_effect = mock_awaitable

        tru_events = await alist(mcp_agent_tool.stream(tool_use, {}))

    assert len(tru_events) == 1
    event = tru_events[0]
    assert event.exception is None
    assert event.tool_result == mock_result
    mock_mcp_client._create_call_tool_coroutine.assert_called_once_with(
        "test_tool", {"param": "value"}, None
    )
    mock_mcp_client._handle_tool_result.assert_called_once()


def test_timeout_initialization(mock_mcp_tool, mock_mcp_client):
    timeout = timedelta(seconds=30)
    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client, timeout=timeout)
    assert agent_tool.timeout == timeout


def test_timeout_default_none(mock_mcp_tool, mock_mcp_client):
    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    assert agent_tool.timeout is None


@pytest.mark.asyncio
async def test_stream_with_timeout(mock_mcp_tool, mock_mcp_client, alist):
    timeout = timedelta(seconds=45)
    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client, timeout=timeout)
    tool_use = {"toolUseId": "test-456", "name": "test_tool", "input": {"param": "value"}}

    mock_result = mock_mcp_client._handle_tool_result.return_value

    with patch("asyncio.wrap_future") as mock_wrap_future:

        async def mock_awaitable(_):
            return MagicMock()

        mock_wrap_future.side_effect = mock_awaitable

        tru_events = await alist(agent_tool.stream(tool_use, {}))

    assert len(tru_events) == 1
    assert tru_events[0].exception is None
    assert tru_events[0].tool_result == mock_result
    mock_mcp_client._create_call_tool_coroutine.assert_called_once_with(
        "test_tool", {"param": "value"}, timeout
    )


@pytest.mark.asyncio
async def test_stream_propagates_exception(mock_mcp_tool, mock_mcp_client, alist):
    """Test that stream() passes the original exception via ToolResultEvent.exception.

    This ensures parity with decorated tools, where the exception is accessible
    via event.exception for debugging and conditional handling.
    """
    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    tool_use = {"toolUseId": "test-123", "name": "test_tool", "input": {"param": "value"}}

    test_exception = RuntimeError("MCP server connection failed")
    with patch("asyncio.wrap_future", side_effect=test_exception):
        mock_error_result = {
            "status": "error", "toolUseId": "test-123",
            "content": [{"text": "Tool execution failed: MCP server connection failed"}],
            "isError": True,
        }
        mock_mcp_client._handle_tool_execution_error.return_value = mock_error_result

        tru_events = await alist(agent_tool.stream(tool_use, {}))

        assert len(tru_events) == 1
        event = tru_events[0]
        assert event.exception is test_exception
        assert event.tool_result == mock_error_result
        mock_mcp_client._handle_tool_execution_error.assert_called_once_with("test-123", test_exception)


@pytest.mark.asyncio
async def test_stream_no_exception_on_success(mock_mcp_tool, mock_mcp_client, alist):
    """Test that stream() sets exception=None on successful execution."""
    agent_tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    tool_use = {"toolUseId": "test-123", "name": "test_tool", "input": {"param": "value"}}

    mock_result = mock_mcp_client._handle_tool_result.return_value

    with patch("asyncio.wrap_future") as mock_wrap_future:

        async def mock_awaitable(_):
            return MagicMock()

        mock_wrap_future.side_effect = mock_awaitable

        tru_events = await alist(agent_tool.stream(tool_use, {}))

    assert len(tru_events) == 1
    assert tru_events[0].exception is None
    assert tru_events[0].tool_result == mock_result
