"""Test for MCP client context variable propagation.

This test verifies that context variables set in the main thread are
properly propagated to the MCP client's background thread.

Related: https://github.com/strands-agents/sdk-python/issues/1440
"""

import contextvars
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.tools.mcp import MCPClient


@pytest.fixture
def mock_transport():
    """Create mock MCP transport."""
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
    """Create mock MCP session."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session

    with patch("strands.tools.mcp.mcp_client.ClientSession", return_value=mock_session_cm):
        yield mock_session


# Context variable for testing
test_contextvar: contextvars.ContextVar[str] = contextvars.ContextVar("test_contextvar", default="default_value")


def test_mcp_client_propagates_contextvars_to_background_thread(mock_transport, mock_session):
    """Test that context variables are propagated to the MCP client background thread.

    This verifies the fix for https://github.com/strands-agents/sdk-python/issues/1440
    where context variables set in the main thread were not accessible in the
    MCP client's background thread.
    """
    # Store the value seen in the background thread
    background_thread_value = {}

    # Patch _background_task to capture the contextvar value
    original_background_task = MCPClient._background_task

    def capturing_background_task(self):
        # Capture the contextvar value in the background thread
        background_thread_value["contextvar"] = test_contextvar.get()
        background_thread_value["thread_id"] = threading.current_thread().ident
        # Call the original background task
        return original_background_task(self)

    # Set a specific value in the main thread
    test_contextvar.set("main_thread_value")
    main_thread_id = threading.current_thread().ident

    with patch.object(MCPClient, "_background_task", capturing_background_task):
        with MCPClient(mock_transport["transport_callable"]) as client:
            # Verify the client started successfully
            assert client._background_thread is not None

    # Verify context was propagated to background thread
    assert "contextvar" in background_thread_value, "Background task should have run and captured contextvar"
    assert background_thread_value["contextvar"] == "main_thread_value", (
        f"Context variable should be propagated to background thread. "
        f"Expected 'main_thread_value', got '{background_thread_value['contextvar']}'"
    )
    # Verify it was indeed a different thread
    assert background_thread_value["thread_id"] != main_thread_id, "Background task should run in a different thread"
