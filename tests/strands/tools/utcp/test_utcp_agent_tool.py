from unittest.mock import AsyncMock, MagicMock

import pytest
from utcp.data.tool import JsonSchema
from utcp.data.tool import Tool as UTCPTool

from strands.tools.utcp import UTCPAgentTool, UTCPClient
from strands.types._events import ToolResultEvent


@pytest.fixture
def mock_utcp_tool():
    """Create a mock UTCP tool for testing."""
    from utcp_http.http_call_template import HttpCallTemplate

    input_schema = JsonSchema(
        type="object",
        properties={
            "location": JsonSchema(type="string", description="Location name"),
            "units": JsonSchema(type="string", enum=["celsius", "fahrenheit"]),
        },
        required=["location"],
        description="Input parameters for weather tool",
    )

    call_template = HttpCallTemplate(
        name="weather_api", call_template_type="http", url="https://api.weather.com/weather"
    )

    mock_tool = UTCPTool(
        name="get_weather",
        description="Get current weather information",
        inputs=input_schema,
        tags=["weather", "forecast"],
        tool_call_template=call_template,
    )
    return mock_tool


@pytest.fixture
def mock_utcp_client():
    """Create a mock UTCP client for testing."""
    mock_client = MagicMock(spec=UTCPClient)
    mock_client.call_tool = AsyncMock(
        return_value={
            "status": "success",
            "toolUseId": "test-123",
            "content": [{"text": "Weather: 22°C, Sunny"}],
        }
    )
    return mock_client


@pytest.fixture
def utcp_agent_tool(mock_utcp_tool, mock_utcp_client):
    """Create a UTCPAgentTool instance for testing."""
    return UTCPAgentTool(mock_utcp_tool, mock_utcp_client)


def test_tool_name(utcp_agent_tool, mock_utcp_tool):
    """Test that tool name is correctly returned."""
    assert utcp_agent_tool.tool_name == "get_weather"
    assert utcp_agent_tool.tool_name == mock_utcp_tool.name


def test_tool_type(utcp_agent_tool):
    """Test that tool type is always 'python'."""
    assert utcp_agent_tool.tool_type == "python"


def test_tool_spec_with_description(utcp_agent_tool, mock_utcp_tool):
    """Test tool spec conversion with description."""
    tool_spec = utcp_agent_tool.tool_spec

    assert tool_spec["name"] == "get_weather"
    assert tool_spec["description"] == "Get current weather information"

    input_schema = tool_spec["inputSchema"]["json"]
    assert input_schema["type"] == "object"
    assert "location" in input_schema["properties"]
    assert "units" in input_schema["properties"]
    assert input_schema["required"] == ["location"]
    assert input_schema["description"] == "Input parameters for weather tool"


def test_tool_spec_without_description(mock_utcp_client):
    """Test tool spec conversion without description."""
    from utcp_http.http_call_template import HttpCallTemplate

    call_template = HttpCallTemplate(name="test_api", call_template_type="http", url="https://api.test.com")

    # Create tool without description
    tool_without_desc = UTCPTool(
        name="test_tool",
        description="",  # Empty description
        tool_call_template=call_template,
    )

    agent_tool = UTCPAgentTool(tool_without_desc, mock_utcp_client)
    tool_spec = agent_tool.tool_spec

    assert tool_spec["description"] == "Tool which performs test_tool"


def test_tool_spec_with_optional_fields(mock_utcp_client):
    """Test tool spec conversion with optional input schema fields."""
    from utcp_http.http_call_template import HttpCallTemplate

    input_schema = JsonSchema(
        type="object",
        properties={"value": JsonSchema(type="number")},
        required=["value"],
        description="Test input",
        title="Test Tool Input",
        minimum=0,
        maximum=100,
        format="float",
        enum=["option1", "option2"],
        items=JsonSchema(type="string"),
    )

    call_template = HttpCallTemplate(name="test_api", call_template_type="http", url="https://api.test.com")

    tool = UTCPTool(
        name="test_tool",
        description="Test tool with optional fields",
        inputs=input_schema,
        tool_call_template=call_template,
    )

    agent_tool = UTCPAgentTool(tool, mock_utcp_client)
    tool_spec = agent_tool.tool_spec

    input_json = tool_spec["inputSchema"]["json"]
    assert input_json["description"] == "Test input"
    assert input_json["title"] == "Test Tool Input"
    assert input_json["minimum"] == "0"
    assert input_json["maximum"] == "100"
    assert input_json["format"] == "float"


@pytest.mark.asyncio
async def test_stream(utcp_agent_tool, mock_utcp_client, alist):
    """Test the stream method calls UTCP client correctly."""
    tool_use = {"toolUseId": "test-123", "name": "get_weather", "input": {"location": "London", "units": "celsius"}}

    tru_events = await alist(utcp_agent_tool.stream(tool_use, {}))
    exp_events = [ToolResultEvent(mock_utcp_client.call_tool_async.return_value)]

    assert tru_events == exp_events
    mock_utcp_client.call_tool_async.assert_called_once_with(
        tool_use_id="test-123", tool_name="get_weather", arguments={"location": "London", "units": "celsius"}
    )


@pytest.mark.asyncio
async def test_stream_with_empty_input(utcp_agent_tool, mock_utcp_client, alist):
    """Test the stream method with empty input."""
    tool_use = {"toolUseId": "test-456", "name": "get_weather", "input": None}

    tru_events = await alist(utcp_agent_tool.stream(tool_use, {}))
    exp_events = [ToolResultEvent(mock_utcp_client.call_tool_async.return_value)]

    assert tru_events == exp_events
    mock_utcp_client.call_tool_async.assert_called_once_with(
        tool_use_id="test-456", tool_name="get_weather", arguments={}
    )


@pytest.mark.asyncio
async def test_stream_with_invocation_state(utcp_agent_tool, mock_utcp_client, alist):
    """Test the stream method with invocation state (should be ignored)."""
    tool_use = {"toolUseId": "test-789", "name": "get_weather", "input": {"location": "Paris"}}

    invocation_state = {"session_id": "abc123", "user_id": "user456"}

    tru_events = await alist(utcp_agent_tool.stream(tool_use, invocation_state))
    exp_events = [ToolResultEvent(mock_utcp_client.call_tool_async.return_value)]

    assert tru_events == exp_events
    mock_utcp_client.call_tool_async.assert_called_once_with(
        tool_use_id="test-789", tool_name="get_weather", arguments={"location": "Paris"}
    )
