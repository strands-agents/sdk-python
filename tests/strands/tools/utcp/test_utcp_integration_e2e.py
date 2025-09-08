"""End-to-end integration tests for UTCP integration.

These tests verify the complete integration between UTCP and Strands SDK,
including configuration loading, tool discovery, and execution.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from utcp.data.tool import JsonSchema
from utcp.data.tool import Tool as UTCPTool
from utcp_http.http_call_template import HttpCallTemplate

from strands.tools.utcp import UTCPAgentTool, UTCPClient
from strands.types.exceptions import UTCPClientInitializationError


@pytest.fixture
def sample_providers_config():
    """Create a sample providers configuration."""
    return [
        {
            "name": "weather_service",
            "provider_type": "http",
            "url": "https://api.weather.com/utcp",
            "http_method": "GET",
            "auth": {"auth_type": "api_key", "api_key": "${WEATHER_API_KEY}", "var_name": "X-API-Key"},
        },
        {"name": "local_tools", "provider_type": "cli", "command_name": "my-cli-tool --utcp"},
        {
            "name": "news_service",
            "provider_type": "http",
            "url": "https://api.news.com/utcp",
            "http_method": "POST",
            "content_type": "application/json",
        },
    ]


@pytest.fixture
def temp_providers_file(sample_providers_config):
    """Create a temporary providers.json file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_providers_config, f, indent=2)
        temp_file = Path(f.name)

    yield str(temp_file)

    # Cleanup
    temp_file.unlink()


@pytest.fixture
def sample_utcp_tools():
    """Create comprehensive sample UTCP tools."""
    weather_call_template = HttpCallTemplate(
        name="weather_service", call_template_type="http", url="https://api.weather.com/utcp"
    )
    news_call_template = HttpCallTemplate(
        name="news_service", call_template_type="http", url="https://api.news.com/utcp"
    )

    weather_input = JsonSchema(
        type="object",
        properties={
            "location": JsonSchema(type="string", description="City name"),
            "units": JsonSchema(type="string", enum=["celsius", "fahrenheit"], description="Temperature units"),
        },
        required=["location"],
        description="Weather query parameters",
    )

    news_input = JsonSchema(
        type="object",
        properties={
            "category": JsonSchema(type="string", description="News category"),
            "limit": JsonSchema(type="integer", minimum=1, maximum=100, description="Number of articles"),
        },
        required=["category"],
        description="News query parameters",
    )

    tools = [
        UTCPTool(
            name="weather_service.get_current",
            description="Get current weather for a location",
            inputs=weather_input,
            tool_call_template=weather_call_template,
            tags=["weather", "current", "forecast"],
        ),
        UTCPTool(
            name="weather_service.get_forecast",
            description="Get weather forecast for a location",
            inputs=weather_input,
            tool_call_template=weather_call_template,
            tags=["weather", "forecast", "future"],
        ),
        UTCPTool(
            name="news_service.get_headlines",
            description="Get latest news headlines",
            inputs=news_input,
            tool_call_template=news_call_template,
            tags=["news", "headlines", "current"],
        ),
    ]
    return tools


@pytest.mark.asyncio
async def test_full_integration_workflow(temp_providers_file, sample_utcp_tools):
    """Test the complete UTCP integration workflow."""
    config = {
        "providers_file_path": temp_providers_file,
        "load_variables_from": [{"type": "dotenv", "env_file_path": ".env"}],
    }

    # Mock the native UTCP client with correct async/sync interface
    mock_native_client = MagicMock()  # Base as MagicMock, not AsyncMock
    mock_native_client.tool_repository = MagicMock()
    mock_native_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)

    # Create a mock search function that filters by query
    async def mock_search_tools(query="", limit=100):
        if query == "weather":
            return [tool for tool in sample_utcp_tools if "weather" in tool.tags]
        return sample_utcp_tools

    mock_native_client.search_tools = mock_search_tools
    mock_native_client.call_tool = AsyncMock(return_value={"temperature": 22.5, "conditions": "sunny", "humidity": 65})

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_native_client

        # Test full workflow
        async with UTCPClient(config) as client:
            # 1. Test tool discovery
            all_tools = await client.list_tools()
            assert len(all_tools) == 3
            assert all(isinstance(tool, UTCPAgentTool) for tool in all_tools)

            # Verify tool names
            tool_names = [tool.tool_name for tool in all_tools]
            expected_names = [
                "weather_service_get_current",
                "weather_service_get_forecast",
                "news_service_get_headlines",
            ]
            assert set(tool_names) == set(expected_names)

            # 2. Test tool search
            weather_tools = await client.search_tools("weather", max_results=10)
            assert len(weather_tools) == 2
            assert all("weather" in tool.utcp_tool.tags for tool in weather_tools)

            # 3. Test tool execution
            result = await client.call_tool_async(
                tool_use_id="integration-test-123",
                tool_name="weather_service.get_current",
                arguments={"location": "London", "units": "celsius"},
            )

            assert result["status"] == "success"
            assert result["toolUseId"] == "integration-test-123"
            assert len(result["content"]) == 1

            # Verify the result contains expected data
            content_text = result["content"][0]["text"]
            parsed_result = json.loads(content_text)
            assert parsed_result["temperature"] == 22.5
            assert parsed_result["conditions"] == "sunny"

            # Verify native client was called correctly
            mock_native_client.call_tool.assert_called_with(
                tool_name="weather_service.get_current", tool_args={"location": "London", "units": "celsius"}
            )


@pytest.mark.asyncio
async def test_tool_spec_conversion_integration(temp_providers_file, sample_utcp_tools):
    """Test that UTCP tool specs are correctly converted to Strands format."""
    config = {"providers_file_path": temp_providers_file}

    mock_native_client = MagicMock()  # Base as MagicMock, not AsyncMock
    mock_native_client.tool_repository = MagicMock()
    mock_native_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)
    mock_native_client.search_tools = AsyncMock(return_value=sample_utcp_tools)
    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_native_client

        async with UTCPClient(config) as client:
            tools = await client.list_tools()

            # Test weather tool spec conversion
            weather_tool = next(tool for tool in tools if tool.tool_name == "weather_service_get_current")

            spec = weather_tool.tool_spec
            assert spec["name"] == "weather_service_get_current"  # Sanitized name for Bedrock compatibility
            assert spec["description"] == "Get current weather for a location"

            input_schema = spec["inputSchema"]["json"]
            assert input_schema["type"] == "object"
            assert "location" in input_schema["properties"]
            assert "units" in input_schema["properties"]
            assert input_schema["required"] == ["location"]
            assert input_schema["description"] == "Weather query parameters"

            # Test enum handling
            units_prop = input_schema["properties"]["units"]
            assert units_prop.enum == ["celsius", "fahrenheit"]

            # Test news tool spec conversion
            news_tool = next(tool for tool in tools if tool.tool_name == "news_service_get_headlines")

            news_spec = news_tool.tool_spec
            news_input_schema = news_spec["inputSchema"]["json"]

            # Test integer constraints
            limit_prop = news_input_schema["properties"]["limit"]
            assert limit_prop.minimum == 1
            assert limit_prop.maximum == 100


@pytest.mark.asyncio
async def test_error_handling_integration(temp_providers_file):
    """Test error handling in the integration."""
    config = {"providers_file_path": temp_providers_file}

    # Test initialization failure
    async def mock_create_failure(*args, **kwargs):
        raise Exception("Network error")

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = mock_create_failure

        with pytest.raises(UTCPClientInitializationError) as exc_info:
            async with UTCPClient(config):
                pass

        assert "UTCP client initialization failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_execution_error_handling(temp_providers_file, sample_utcp_tools):
    """Test error handling during tool execution."""
    config = {"providers_file_path": temp_providers_file}

    async def mock_call_tool_failure(*args, **kwargs):
        raise Exception("Tool execution failed")

    mock_native_client = MagicMock()  # Base as MagicMock, not AsyncMock
    mock_native_client.tool_repository = MagicMock()
    mock_native_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)
    mock_native_client.call_tool = AsyncMock(side_effect=mock_call_tool_failure)

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_native_client

        async with UTCPClient(config) as client:
            result = await client.call_tool_async(
                tool_use_id="error-test-123", tool_name="weather_service.get_current", arguments={"location": "London"}
            )

            assert result["status"] == "error"
            assert result["toolUseId"] == "error-test-123"
            assert "UTCP tool execution failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_tool_streaming_integration(temp_providers_file, sample_utcp_tools):
    """Test UTCPAgentTool streaming integration."""
    config = {"providers_file_path": temp_providers_file}

    mock_native_client = MagicMock()  # Base as MagicMock, not AsyncMock
    mock_native_client.tool_repository = MagicMock()
    mock_native_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)

    mock_native_client.search_tools = AsyncMock(return_value=sample_utcp_tools)
    mock_native_client.call_tool = AsyncMock(return_value="Weather data")  # Raw result

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_native_client

        async with UTCPClient(config) as client:
            tools = await client.list_tools()
            weather_tool = next(tool for tool in tools if tool.tool_name == "weather_service_get_current")

            # Expected result after _handle_tool_result processing
            expected_result = {
                "status": "success",
                "toolUseId": "stream-test-123",
                "content": [{"text": "Weather data"}],
            }

            # Test streaming
            tool_use = {
                "toolUseId": "stream-test-123",
                "name": "weather_service.get_current",
                "input": {"location": "Paris", "units": "celsius"},
            }

            results = []
            async for result in weather_tool.stream(tool_use, {}):
                results.append(result)

            assert len(results) == 1
            # The result should be wrapped in a ToolResultEvent
            assert hasattr(results[0], "tool_result")
            assert results[0].tool_result == expected_result

            # Verify the native client was called correctly
            mock_native_client.call_tool.assert_called_once_with(
                tool_name="weather_service.get_current",
                tool_args={"location": "Paris", "units": "celsius"},
            )


def test_configuration_validation(sample_providers_config):
    """Test that configuration is properly validated."""
    # Test with missing providers file
    config = {"providers_file_path": "/nonexistent/file.json"}
    client = UTCPClient(config)

    # The error should occur during start(), not during init
    assert client._config == config
    assert client._utcp_client is None


@pytest.mark.asyncio
async def test_multiple_provider_types_integration(temp_providers_file, sample_utcp_tools):
    """Test integration with multiple provider types."""
    config = {"providers_file_path": temp_providers_file}

    mock_native_client = MagicMock()  # Base as MagicMock, not AsyncMock
    mock_native_client.tool_repository = MagicMock()
    mock_native_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)
    mock_native_client.search_tools = AsyncMock(return_value=sample_utcp_tools)

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_native_client

        async with UTCPClient(config) as client:
            tools = await client.list_tools()

            # Verify we have tools from different provider types
            provider_types = set()
            for tool in tools:
                call_template = tool.utcp_tool.tool_call_template
                provider_types.add(type(call_template).__name__)

            # Should have HttpCallTemplate (all sample tools are HTTP)
            assert "HttpCallTemplate" in provider_types

            # All tools should be HTTP tools in our sample
            assert len(tools) == 3  # All HTTP tools
