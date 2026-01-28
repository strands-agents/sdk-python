"""Shared fixtures and helpers for MCP client tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_transport():
    """Create a mock MCP transport."""
    mock_read_stream = AsyncMock()
    mock_write_stream = AsyncMock()
    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__.return_value = (mock_read_stream, mock_write_stream)
    mock_transport_callable = MagicMock(return_value=mock_transport_cm)

    return {
        "read_stream": mock_read_stream,
        "write_stream": mock_write_stream,
        "transport_cm": mock_transport_cm,
        "transport_callable": mock_transport_callable,
    }


@pytest.fixture
def mock_session():
    """Create a mock MCP session."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    # Default: no task support (get_server_capabilities is sync, not async!)
    mock_session.get_server_capabilities = MagicMock(return_value=None)

    # Create a mock context manager for ClientSession
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session

    # Patch ClientSession to return our mock session
    with patch("strands.tools.mcp.mcp_client.ClientSession", return_value=mock_session_cm):
        yield mock_session


def create_server_capabilities(has_task_support: bool) -> MagicMock:
    """Create mock server capabilities.

    Args:
        has_task_support: Whether the server should advertise task support.

    Returns:
        MagicMock representing server capabilities.
    """
    caps = MagicMock()
    if has_task_support:
        caps.tasks = MagicMock()
        caps.tasks.requests = MagicMock()
        caps.tasks.requests.tools = MagicMock()
        caps.tasks.requests.tools.call = MagicMock()
    else:
        caps.tasks = None
    return caps
