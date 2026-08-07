"""Unit tests for the mcp 1.x/2.x compatibility layer."""

from unittest.mock import MagicMock, patch

from strands.tools.mcp import _compat
from strands.tools.mcp._compat import MCPError, streamable_http_transport


def test_mcp_v2_flag_matches_discover_capability():
    """Test that the flag reflects whether ClientSession has the 2.x discover API."""
    from mcp import ClientSession

    assert _compat.MCP_V2 is hasattr(ClientSession, "discover")


def test_mcp_error_resolves_to_installed_exception():
    """Test that MCPError is the mcp package's error type regardless of its spelling."""
    import mcp.shared.exceptions as mcp_exceptions

    installed = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError
    assert MCPError is installed


def test_streamable_http_transport_v1_call_shape():
    """Test that the 1.x transport receives url, headers, and auth as loose kwargs."""
    headers = {"Authorization": "Bearer token"}
    auth = MagicMock()

    with (
        patch.object(_compat, "MCP_V2", False),
        patch("mcp.client.streamable_http.streamablehttp_client", create=True) as mock_client,
    ):
        result = streamable_http_transport("https://example.com/mcp", headers=headers, auth=auth)

        mock_client.assert_called_once_with(url="https://example.com/mcp", headers=headers, auth=auth)
        assert result is mock_client.return_value


def test_streamable_http_transport_v2_call_shape():
    """Test that the 2.x transport receives a pre-configured HTTP client carrying the headers and auth."""
    headers = {"Authorization": "Bearer token"}
    auth = MagicMock()
    http_client = MagicMock()

    with (
        patch.object(_compat, "MCP_V2", True),
        patch("mcp.client.streamable_http.streamable_http_client", create=True) as mock_client,
        patch("mcp.client.streamable_http.create_mcp_http_client", create=True, return_value=http_client) as mock_http,
    ):
        result = streamable_http_transport("https://example.com/mcp", headers=headers, auth=auth)

        mock_http.assert_called_once_with(headers=headers, auth=auth)
        mock_client.assert_called_once_with(url="https://example.com/mcp", http_client=http_client)
        assert result is mock_client.return_value
