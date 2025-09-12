"""
UTCP Client Integration Tests

This module provides comprehensive integration tests for the UTCP (Universal Tool Calling Protocol)
client implementation. These tests validate the complete workflow from tool discovery through
execution using real UTCP servers with the current UTCP API.
"""

import threading
import time

import pytest
import requests

from strands import Agent
from strands.tools.utcp.utcp_client import UTCPClient


def start_utcp_calculator_server(port: int = 8002):
    """
    Initialize and start a UTCP calculator server for integration testing.

    This function creates a FastAPI server that provides calculator tools
    via UTCP protocol. The server uses HTTP transport for communication.
    """
    from tests_integ.utcp_calculator_server import start_calculator_server

    start_calculator_server(port)


def start_utcp_echo_server(port: int = 8003):
    """
    Initialize and start a UTCP echo server for integration testing.

    This function creates a FastAPI server that provides an echo tool
    via UTCP protocol for testing basic communication flow.
    """
    from tests_integ.utcp_echo_server import start_echo_server

    start_echo_server(port)


def wait_for_server(url: str, timeout: int = 10) -> bool:
    """
    Wait for a server to become available.

    Args:
        url: The URL to check
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is available, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.1)
    return False


@pytest.mark.asyncio
async def test_utcp_client():
    """
    Test UTCP client integration with calculator and echo tools.

    This test validates the UTCP integration approach:
    - HTTP-based tool discovery
    - Direct async tool execution
    - Built-in search capabilities
    """
    # Start test servers
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8002}, daemon=True)
    calculator_thread.start()

    echo_thread = threading.Thread(target=start_utcp_echo_server, kwargs={"port": 8003}, daemon=True)
    echo_thread.start()

    # Wait for servers to start
    assert wait_for_server("http://127.0.0.1:8002/utcp"), "Calculator server failed to start"
    assert wait_for_server("http://127.0.0.1:8003/utcp"), "Echo server failed to start"

    # Configure UTCP client with both servers
    config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8002/utcp",
            },
            {"name": "echo", "call_template_type": "http", "http_method": "GET", "url": "http://127.0.0.1:8003/utcp"},
        ]
    }

    # Test UTCP client functionality
    async with UTCPClient(config) as utcp_client:
        # Test tool discovery
        tools = await utcp_client.list_tools()
        assert len(tools) > 0, "No tools discovered"

        # Verify we have calculator tools
        tool_names = [tool.tool_name for tool in tools]
        assert any("add" in name for name in tool_names), "Calculator add tool not found"
        assert any("echo" in name for name in tool_names), "Echo tool not found"

        # Test tool execution - calculator
        result = await utcp_client.call_tool("calculator.add", {"a": 5, "b": 3})
        assert result["result"] == 8, f"Expected 8, got {result}"

        # Test tool execution - echo
        result = await utcp_client.call_tool("echo.echo", {"message": "Hello UTCP!"})
        assert result["echo"] == "Hello UTCP!", f"Expected 'Hello UTCP!', got {result}"


@pytest.mark.asyncio
async def test_can_reuse_utcp_client():
    """
    Test that UTCP client can be reused for multiple operations.
    """
    # Start calculator server
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8004}, daemon=True)
    calculator_thread.start()

    assert wait_for_server("http://127.0.0.1:8004/utcp"), "Calculator server failed to start"

    config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8004/utcp",
            }
        ]
    }

    async with UTCPClient(config) as utcp_client:
        # First operation
        result1 = await utcp_client.call_tool("calculator.add", {"a": 10, "b": 20})
        assert result1["result"] == 30

        # Second operation with same client
        result2 = await utcp_client.call_tool("calculator.multiply", {"a": 4, "b": 5})
        assert result2["result"] == 20

        # Third operation
        result3 = await utcp_client.call_tool("calculator.subtract", {"a": 100, "b": 25})
        assert result3["result"] == 75


@pytest.mark.asyncio
async def test_utcp_tool_search_functionality():
    """
    Test UTCP tool search and filtering capabilities.
    """
    # Start calculator server
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8005}, daemon=True)
    calculator_thread.start()

    assert wait_for_server("http://127.0.0.1:8005/utcp"), "Calculator server failed to start"

    config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8005/utcp",
            }
        ]
    }

    async with UTCPClient(config) as utcp_client:
        # Get all tools
        all_tools = await utcp_client.list_tools()
        assert len(all_tools) >= 4, "Expected at least 4 calculator tools"

        # Verify tool properties
        for tool in all_tools:
            assert hasattr(tool, "tool_name"), "Tool should have tool_name property"
            assert hasattr(tool, "tool_spec"), "Tool should have tool_spec property"
            assert tool.tool_name.startswith("calculator_"), "Tool names should be prefixed"


@pytest.mark.asyncio
async def test_utcp_multiple_provider_types():
    """
    Test UTCP client with multiple provider types and configurations.
    """
    # Start both servers
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8006}, daemon=True)
    calculator_thread.start()

    echo_thread = threading.Thread(target=start_utcp_echo_server, kwargs={"port": 8007}, daemon=True)
    echo_thread.start()

    assert wait_for_server("http://127.0.0.1:8006/utcp"), "Calculator server failed to start"
    assert wait_for_server("http://127.0.0.1:8007/utcp"), "Echo server failed to start"

    config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8006/utcp",
            },
            {"name": "echo", "call_template_type": "http", "http_method": "GET", "url": "http://127.0.0.1:8007/utcp"},
        ]
    }

    async with UTCPClient(config) as utcp_client:
        # Test tools from both providers
        tools = await utcp_client.list_tools()

        # Should have tools from both providers
        tool_names = [tool.tool_name for tool in tools]
        calculator_tools = [name for name in tool_names if "calculator" in name]
        echo_tools = [name for name in tool_names if "echo" in name]

        assert len(calculator_tools) >= 4, "Should have calculator tools"
        assert len(echo_tools) >= 2, "Should have echo tools"

        # Test execution from both providers
        calc_result = await utcp_client.call_tool("calculator.divide", {"a": 15, "b": 3})
        assert calc_result["result"] == 5

        echo_result = await utcp_client.call_tool("echo.echo_upper", {"message": "test"})
        assert echo_result["echo"] == "TEST"


@pytest.mark.asyncio
async def test_utcp_error_handling():
    """
    Test UTCP client error handling for various failure scenarios.
    """
    # Test with invalid configuration
    invalid_config = {
        "manual_call_templates": [
            {
                "name": "nonexistent",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:9999/utcp",  # Non-existent server
            }
        ]
    }

    # Should handle connection errors gracefully
    async with UTCPClient(invalid_config) as utcp_client:
        tools = await utcp_client.list_tools()
        assert len(tools) == 0, "Should return empty list for failed connections"

    # Test with valid server but invalid tool call
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8008}, daemon=True)
    calculator_thread.start()

    assert wait_for_server("http://127.0.0.1:8008/utcp"), "Calculator server failed to start"

    valid_config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8008/utcp",
            }
        ]
    }

    async with UTCPClient(valid_config) as utcp_client:
        # Test invalid tool name
        with pytest.raises(ValueError, match="Tool execution failed"):
            await utcp_client.call_tool("calculator.nonexistent", {"a": 1, "b": 2})

        # Test invalid arguments
        with pytest.raises(ValueError, match="Tool execution failed"):
            await utcp_client.call_tool("calculator.add", {"invalid": "args"})


@pytest.mark.asyncio
async def test_utcp_agent_integration():
    """Test UTCP tools with Strands Agent end-to-end workflow."""
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8009}, daemon=True)
    calculator_thread.start()

    assert wait_for_server("http://127.0.0.1:8009/utcp"), "Calculator server failed to start"

    config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8009/utcp",
            }
        ]
    }

    async with UTCPClient(config) as utcp_client:
        tools = await utcp_client.list_tools()
        assert len(tools) > 0, "Should discover calculator tools"

        # Create Agent with UTCP tools
        agent = Agent(tools=tools)
        result = agent("Calculate 5 + 3 using the add function")

        # Verify agent used UTCP tools
        assert result is not None
        assert "8" in str(result) or "eight" in str(result).lower()


@pytest.mark.asyncio
async def test_utcp_agent_multiple_tools():
    """Test Agent with multiple UTCP tools from different providers."""
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8010}, daemon=True)
    calculator_thread.start()

    echo_thread = threading.Thread(target=start_utcp_echo_server, kwargs={"port": 8011}, daemon=True)
    echo_thread.start()

    assert wait_for_server("http://127.0.0.1:8010/utcp"), "Calculator server failed to start"
    assert wait_for_server("http://127.0.0.1:8011/utcp"), "Echo server failed to start"

    config = {
        "manual_call_templates": [
            {
                "name": "calculator",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8010/utcp",
            },
            {
                "name": "echo",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "http://127.0.0.1:8011/utcp",
            },
        ]
    }

    async with UTCPClient(config) as utcp_client:
        tools = await utcp_client.list_tools()
        assert len(tools) >= 2, "Should have tools from both providers"

        agent = Agent(tools=tools)
        result = agent("Add 10 and 5, then echo the result back to me")

        # Verify both tools were used
        assert result is not None
        assert "15" in str(result)
