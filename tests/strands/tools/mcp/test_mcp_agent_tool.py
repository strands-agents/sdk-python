from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPAgentTool, MCPClient
from strands.types._events import ToolResultEvent


@pytest.fixture
def mock_mcp_tool():
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {"type": "object", "properties": {}}
    mock_tool.outputSchema = None  # MCP tools can have optional outputSchema
    return mock_tool


@pytest.fixture
def mock_mcp_client():
    mock_server = MagicMock(spec=MCPClient)
    mock_server.call_tool_sync.return_value = {
        "status": "success",
        "toolUseId": "test-123",
        "content": [{"text": "Success result"}],
    }
    return mock_server


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

    tru_events = await alist(mcp_agent_tool.stream(tool_use, {}))
    exp_events = [ToolResultEvent(mock_mcp_client.call_tool_async.return_value)]

    assert tru_events == exp_events
    mock_mcp_client.call_tool_async.assert_called_once_with(
        tool_use_id="test-123", name="test_tool", arguments={"param": "value"}, read_timeout_seconds=None
    )


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

    tru_events = await alist(agent_tool.stream(tool_use, {}))
    exp_events = [ToolResultEvent(mock_mcp_client.call_tool_async.return_value)]

    assert tru_events == exp_events
    mock_mcp_client.call_tool_async.assert_called_once_with(
        tool_use_id="test-456", name="test_tool", arguments={"param": "value"}, read_timeout_seconds=timeout
    )
