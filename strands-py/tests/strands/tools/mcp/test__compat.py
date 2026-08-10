"""Unit tests for the mcp 1.x/2.x compatibility layer.

The branch-specific tests patch real attributes of the installed `mcp`
package (never `create=True`, which would invent missing names and pass
against any spelling), so each test runs only on the line whose names exist
and is skipped on the other.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from strands.tools.mcp import _compat
from strands.tools.mcp._compat import MCPError, streamable_http_transport

requires_mcp_v1 = pytest.mark.skipif(_compat.MCP_V2, reason="exercises the mcp 1.x branch")
requires_mcp_v2 = pytest.mark.skipif(not _compat.MCP_V2, reason="exercises the mcp 2.x branch")


def test_installed_line_exposes_expected_transport_names():
    """Test that the names each branch imports exist on the line the flag selects.

    Directional on purpose: late 1.x releases backport `streamable_http_client`
    as an alias, so only the branch actually taken is asserted against the
    installed module.
    """
    import mcp.client.streamable_http as streamable_http_module

    if _compat.MCP_V2:
        assert hasattr(streamable_http_module, "streamable_http_client")
        assert hasattr(streamable_http_module, "create_mcp_http_client")
    else:
        assert hasattr(streamable_http_module, "streamablehttp_client")


def test_mcp_error_resolves_to_installed_exception():
    """Test that MCPError is the mcp package's error type regardless of its spelling."""
    import mcp.shared.exceptions as mcp_exceptions

    installed = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError
    assert MCPError is installed


@requires_mcp_v1
def test_streamable_http_transport_v1_call_shape():
    """Test that the 1.x transport receives url, headers, and auth as loose kwargs."""
    headers = {"Authorization": "Bearer token"}
    auth = MagicMock()

    with patch("mcp.client.streamable_http.streamablehttp_client") as mock_client:
        result = streamable_http_transport("https://example.com/mcp", headers=headers, auth=auth)

        mock_client.assert_called_once_with(url="https://example.com/mcp", headers=headers, auth=auth)
        assert result is mock_client.return_value


@requires_mcp_v2
@pytest.mark.asyncio
async def test_streamable_http_transport_v2_owns_client_lifecycle():
    """Test that the 2.x transport closes the HTTPX client it creates.

    Guards https://github.com/strands-agents/harness-sdk/pull/3708: 2.x's
    `streamable_http_client` only closes a client it created itself, so the
    adapter must bind the client's lifetime to the transport's.
    """
    lifecycle_events = []
    transport_streams = MagicMock()
    auth = MagicMock()

    @asynccontextmanager
    async def fake_http_client(headers=None, auth=None):
        lifecycle_events.append(("client_enter", headers, auth))
        yield MagicMock()
        lifecycle_events.append(("client_exit", headers, auth))

    @asynccontextmanager
    async def fake_transport(url, http_client):
        lifecycle_events.append(("transport_enter", url))
        yield transport_streams
        lifecycle_events.append(("transport_exit", url))

    headers = {"Authorization": "Bearer token"}
    with (
        patch("mcp.client.streamable_http.create_mcp_http_client", fake_http_client),
        patch("mcp.client.streamable_http.streamable_http_client", fake_transport),
    ):
        async with streamable_http_transport("https://example.com/mcp", headers=headers, auth=auth) as streams:
            assert streams is transport_streams

    assert lifecycle_events == [
        ("client_enter", headers, auth),
        ("transport_enter", "https://example.com/mcp"),
        ("transport_exit", "https://example.com/mcp"),
        ("client_exit", headers, auth),
    ]
