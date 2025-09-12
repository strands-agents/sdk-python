import json
from unittest.mock import AsyncMock, patch

import pytest
from utcp.data.tool import JsonSchema
from utcp.data.tool import Tool as UTCPTool
from utcp_http.http_call_template import HttpCallTemplate

from strands.tools.utcp import UTCPAgentTool, UTCPClient
from strands.types.exceptions import UTCPClientInitializationError


@pytest.fixture
def sample_utcp_tools():
    """Create sample UTCP tools for testing."""
    call_template1 = HttpCallTemplate(name="weather_api", call_template_type="http", url="https://weather.com/utcp")
    call_template2 = HttpCallTemplate(name="news_api", call_template_type="http", url="https://news.com/utcp")

    input_schema1 = JsonSchema(type="object", properties={"location": JsonSchema(type="string")}, required=["location"])

    input_schema2 = JsonSchema(type="object", properties={"category": JsonSchema(type="string")}, required=["category"])

    tools = [
        UTCPTool(
            name="weather_api.get_current",
            description="Get current weather",
            inputs=input_schema1,
            tool_call_template=call_template1,
            tags=["weather", "current"],
        ),
        UTCPTool(
            name="news_api.get_headlines",
            description="Get news headlines",
            inputs=input_schema2,
            tool_call_template=call_template2,
            tags=["news", "headlines"],
        ),
    ]
    return tools


def test_utcp_client_init(utcp_config):
    """Test UTCPClient initialization."""
    client = UTCPClient(utcp_config)

    assert client._config == utcp_config
    assert client._utcp_client is None


@pytest.mark.asyncio
async def test_utcp_client_context_manager(utcp_config, mock_utcp_native_client):
    """Test UTCPClient as async context manager."""
    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_utcp_native_client

        async with UTCPClient(utcp_config) as client:
            assert client._utcp_client == mock_utcp_native_client
            mock_create.assert_called_once()

        # After context exit, client should be cleaned up
        assert client._utcp_client is None


@pytest.mark.asyncio
async def test_start_success(utcp_config, mock_utcp_native_client):
    """Test successful client start."""
    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_utcp_native_client

        client = UTCPClient(utcp_config)
        result = await client.start()

        assert result == client
        assert client._utcp_client == mock_utcp_native_client
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_start_failure(utcp_config):
    """Test client start failure."""

    async def mock_create_failure(*args, **kwargs):
        raise Exception("Connection failed")

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = mock_create_failure

        client = UTCPClient(utcp_config)

        with pytest.raises(UTCPClientInitializationError) as exc_info:
            await client.start()

        assert "UTCP client initialization failed" in str(exc_info.value)


def test_list_tools_sync_not_initialized(utcp_config):
    """Test list_tools_sync when client is not initialized."""
    client = UTCPClient(utcp_config)

    with pytest.raises(UTCPClientInitializationError) as exc_info:
        client.list_tools_sync()

    assert "UTCP client is not initialized" in str(exc_info.value)


def test_list_tools_sync_success(utcp_client_sync, sample_utcp_tools):
    """Test successful list_tools_sync."""
    # Setup mock to return sample tools
    utcp_client_sync._utcp_client.search_tools.return_value = sample_utcp_tools

    result = utcp_client_sync.list_tools_sync()

    assert len(result) == 2
    assert all(isinstance(tool, UTCPAgentTool) for tool in result)
    assert result[0].tool_name == "weather_api_get_current"
    assert result[1].tool_name == "news_api_get_headlines"
    assert result.pagination_token is None  # UTCP doesn't support pagination


def test_list_tools_sync_with_pagination_token(utcp_client_sync, sample_utcp_tools):
    """Test list_tools_sync with pagination token (should be ignored)."""
    utcp_client_sync._utcp_client.search_tools.return_value = sample_utcp_tools

    result = utcp_client_sync.list_tools_sync(pagination_token="some_token")

    assert len(result) == 2
    assert result.pagination_token is None  # UTCP doesn't support pagination


@pytest.mark.asyncio
async def test_call_tool_async_not_initialized(utcp_config):
    """Test call_tool_async when client is not initialized."""
    client = UTCPClient(utcp_config)

    with pytest.raises(UTCPClientInitializationError) as exc_info:
        await client.call_tool_async("test-id", "test_tool", {})

    assert "UTCP client is not initialized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_tool_async_success_dict_result(utcp_client):
    """Test successful call_tool_async with dict result."""
    # Setup mock to return a dict result
    mock_result = {"temperature": 22.5, "conditions": "sunny", "humidity": 65}
    utcp_client._utcp_client.call_tool.return_value = mock_result

    result = await utcp_client.call_tool_async("test-123", "weather.get_current", {"location": "London"})

    utcp_client._utcp_client.call_tool.assert_called_once_with(
        tool_name="weather.get_current", tool_args={"location": "London"}
    )

    assert result["status"] == "success"
    assert result["toolUseId"] == "test-123"
    assert len(result["content"]) == 1

    # Check that the dict was converted to JSON
    content_text = result["content"][0]["text"]
    parsed_result = json.loads(content_text)
    assert parsed_result == mock_result


@pytest.mark.asyncio
async def test_call_tool_async_success_string_result(utcp_client):
    """Test successful call_tool_async with string result."""
    mock_result = "The weather in London is sunny with 22°C"
    utcp_client._utcp_client.call_tool.return_value = mock_result

    result = await utcp_client.call_tool_async("test-456", "weather.get_current", {"location": "London"})

    assert result["status"] == "success"
    assert result["toolUseId"] == "test-456"
    assert len(result["content"]) == 1
    assert result["content"][0]["text"] == mock_result


@pytest.mark.asyncio
async def test_call_tool_async_success_list_result(utcp_client):
    """Test successful call_tool_async with list result."""
    mock_result = ["item1", "item2", "item3"]
    utcp_client._utcp_client.call_tool.return_value = mock_result

    result = await utcp_client.call_tool_async("test-789", "list.get_items", {})

    assert result["status"] == "success"
    assert result["toolUseId"] == "test-789"
    assert len(result["content"]) == 1

    # Check that the list was converted to JSON
    content_text = result["content"][0]["text"]
    parsed_result = json.loads(content_text)
    assert parsed_result == mock_result


@pytest.mark.asyncio
async def test_call_tool_async_failure(utcp_client):
    """Test call_tool_async failure."""

    async def mock_call_tool_failure(*args, **kwargs):
        raise Exception("Tool execution failed")

    utcp_client._utcp_client.call_tool.side_effect = mock_call_tool_failure

    result = await utcp_client.call_tool_async("test-error", "failing_tool", {})

    assert result["status"] == "error"
    assert result["toolUseId"] == "test-error"
    assert len(result["content"]) == 1
    assert "UTCP tool execution failed" in result["content"][0]["text"]


def test_call_tool_sync_not_initialized(utcp_config):
    """Test call_tool_sync when client is not initialized."""
    client = UTCPClient(utcp_config)

    with pytest.raises(UTCPClientInitializationError) as exc_info:
        client.call_tool_sync("test-id", "test_tool", {})

    assert "UTCP client is not initialized" in str(exc_info.value)


def test_call_tool_sync_success(utcp_client_sync):
    """Test successful call_tool_sync."""
    formatted_result = {"status": "success", "toolUseId": "test-sync", "content": [{"text": '{"result": "success"}'}]}

    # Mock asyncio.run to return formatted result
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = formatted_result

        result = utcp_client_sync.call_tool_sync("test-sync", "sync_tool", {"param": "value"})

        # Verify asyncio.run was called with the correct coroutine
        mock_run.assert_called_once()

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-sync"


@pytest.mark.asyncio
async def test_search_tools_not_initialized(utcp_config):
    """Test search_tools when client is not initialized."""
    client = UTCPClient(utcp_config)

    with pytest.raises(UTCPClientInitializationError) as exc_info:
        await client.search_tools("weather")

    assert "UTCP client is not initialized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_tools_success(utcp_client_with_search_results):
    """Test successful search_tools."""
    result = await utcp_client_with_search_results.search_tools("weather", max_results=5)

    utcp_client_with_search_results._utcp_client.search_tools.assert_called_once_with(query="weather", limit=5)
    assert len(result) == 1
    assert isinstance(result[0], UTCPAgentTool)
    assert result[0].tool_name == "weather_api_get_current"


@pytest.mark.asyncio
async def test_search_tools_with_max_results(utcp_client_with_limited_search):
    """Test search_tools with max_results limit."""
    result = await utcp_client_with_limited_search.search_tools("test", max_results=3)

    # Should only return first 3 results
    assert len(result) == 3
    assert all(isinstance(tool, UTCPAgentTool) for tool in result)
    utcp_client_with_limited_search._utcp_client.search_tools.assert_called_once_with(query="test", limit=3)


def test_handle_tool_execution_error():
    """Test _handle_tool_execution_error method."""
    client = UTCPClient({})
    exception = Exception("Test error")

    result = client._handle_tool_execution_error("error-123", exception)

    assert result["status"] == "error"
    assert result["toolUseId"] == "error-123"
    assert len(result["content"]) == 1
    assert "UTCP tool execution failed: Test error" in result["content"][0]["text"]


def test_handle_tool_result_with_complex_object():
    """Test _handle_tool_result with complex object."""
    client = UTCPClient({})
    complex_result = {"nested": {"data": [1, 2, 3]}, "string": "value", "number": 42}

    result = client._handle_tool_result("complex-123", complex_result)

    assert result["status"] == "success"
    assert result["toolUseId"] == "complex-123"
    assert len(result["content"]) == 1

    # Verify JSON formatting
    content_text = result["content"][0]["text"]
    parsed = json.loads(content_text)
    assert parsed == complex_result


def test_handle_tool_result_processing_error():
    """Test _handle_tool_result when JSON processing fails."""
    client = UTCPClient({})

    # Create an object that can't be JSON serialized
    class UnserializableObject:
        def __str__(self):
            raise Exception("Cannot convert to string")

    bad_result = UnserializableObject()

    # Mock json.dumps to raise an exception
    with patch("json.dumps", side_effect=Exception("JSON error")):
        result = client._handle_tool_result("json-error", bad_result)

    assert result["status"] == "error"
    assert result["toolUseId"] == "json-error"
    assert "UTCP tool execution failed" in result["content"][0]["text"]
