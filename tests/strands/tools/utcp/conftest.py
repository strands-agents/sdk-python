"""Shared test fixtures for UTCP integration tests."""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from strands.tools.utcp.utcp_client import UTCPClient


@pytest.fixture
def alist():
    """Fixture to convert async generators to lists for testing."""

    async def _alist(async_gen) -> List:
        """Convert async generator to list."""
        result = []
        async for item in async_gen:
            result.append(item)
        return result

    return _alist


@pytest_asyncio.fixture
async def async_context():
    """Provide an async context for tests that need it."""
    # This fixture can be used for tests that need async setup/teardown
    yield
    # Cleanup code can go here if needed


@pytest.fixture
def utcp_config():
    """Create a sample UTCP configuration."""
    return {
        "providers_file_path": "/tmp/test_providers.json",
        "load_variables_from": [{"type": "dotenv", "env_file_path": ".env"}],
    }


@pytest.fixture
def mock_utcp_native_client():
    """Create a mock native UTCP client that matches the real interface.

    Based on UTCP library analysis:
    - create: ASYNC (class method)
    - call_tool: ASYNC
    - search_tools: ASYNC (updated in v0.1.8)
    - tool_repository.get_tools: ASYNC
    """
    # Use MagicMock as base to avoid making everything async by default
    mock_client = MagicMock()

    # Set up tool_repository with async get_tools - configure return value to avoid unawaited coroutines
    mock_client.tool_repository = MagicMock()
    mock_client.tool_repository.get_tools = AsyncMock(return_value=[])

    # call_tool is async in real UTCP - configure return value to avoid unawaited coroutines
    mock_client.call_tool = AsyncMock(return_value={"status": "success"})

    # search_tools is now async in UTCP v0.1.8+ - use AsyncMock
    mock_client.search_tools = AsyncMock(return_value=[])

    return mock_client


@pytest.fixture
def utcp_client_sync(utcp_config, mock_utcp_native_client):
    """Create a UTCPClient instance for sync testing."""
    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_utcp_native_client

        client = UTCPClient(utcp_config)
        # Manually set the mock client to simulate successful initialization
        client._utcp_client = mock_utcp_native_client
        yield client  # Keep patch context alive during test execution


@pytest.fixture
def utcp_client_with_search_results(utcp_config, sample_utcp_tools):
    """Create a UTCPClient with pre-configured search results to avoid mock modification warnings."""
    # Create mock with pre-configured search results
    mock_client = MagicMock(spec=["tool_repository", "call_tool", "call_tool_async", "search_tools"])
    mock_client.tool_repository = MagicMock()
    mock_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)
    mock_client.call_tool = AsyncMock(return_value={"status": "success"})  # ✅ Added return_value
    mock_client.call_tool_async = MagicMock(return_value={"status": "success"})  # Non-async mock
    # Pre-configure search_tools with weather results (now async)
    weather_tools = [tool for tool in sample_utcp_tools if "weather" in tool.tags]
    mock_client.search_tools = AsyncMock(return_value=weather_tools)

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_client

        client = UTCPClient(utcp_config)
        client._utcp_client = mock_client
        yield client


@pytest.fixture
def utcp_client_with_limited_search(utcp_config, sample_utcp_tools):
    """Create a UTCPClient with search_tools that respects limit parameter."""
    # Create mock with pre-configured search behavior
    mock_client = MagicMock(spec=["tool_repository", "call_tool", "call_tool_async", "search_tools"])
    mock_client.tool_repository = MagicMock()
    mock_client.tool_repository.get_tools = AsyncMock(return_value=sample_utcp_tools)
    mock_client.call_tool = AsyncMock(return_value={"status": "success"})  # ✅ Added return_value
    mock_client.call_tool_async = MagicMock(return_value={"status": "success"})  # Non-async mock
    # Create many tools for testing limits
    many_tools = sample_utcp_tools * 10  # 20 tools total

    # Pre-configure search_tools with proper limit handling (now async)
    async def mock_search_tools(query, limit):
        return many_tools[:limit]

    mock_client.search_tools = AsyncMock(side_effect=mock_search_tools)

    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_client

        client = UTCPClient(utcp_config)
        client._utcp_client = mock_client
        yield client


@pytest_asyncio.fixture
async def utcp_client(utcp_config, mock_utcp_native_client):
    """Create a UTCPClient instance for async testing."""
    with patch("strands.tools.utcp.utcp_client.UtcpClient.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_utcp_native_client

        client = UTCPClient(utcp_config)
        await client.start()
        yield client
        await client.stop()
