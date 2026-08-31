"""Unit tests for the mcp 1.x/2.x compatibility layer.

The branch-specific tests patch real attributes of the installed `mcp`
package (never `create=True`, which would invent missing names and pass
against any spelling), so each test is gated on the names it patches
actually existing on the installed line. Late 1.x releases backport the
2.x transport names, so the 2.x-branch transport test runs there too with
the `MCP_V2` flag forced.
"""

import importlib
import inspect
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.client.streamable_http as streamable_http_module
import pytest

from strands.tools.mcp import _compat
from strands.tools.mcp._compat import MCPError, negotiate_session, streamable_http_transport

requires_mcp_v1 = pytest.mark.skipif(_compat.MCP_V2, reason="exercises the mcp 1.x branch")
requires_mcp_v2 = pytest.mark.skipif(not _compat.MCP_V2, reason="exercises the mcp 2.x line")
requires_v2_transport_names = pytest.mark.skipif(
    not hasattr(streamable_http_module, "streamable_http_client"),
    reason="installed mcp line lacks the 2.x transport names",
)


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


def test_installed_line_list_methods_accept_params():
    """Test that the session list_* methods accept the `params` keyword on the installed line.

    `MCPClient` passes pagination as `params` only: the `cursor` keyword is
    deprecated on 1.x and removed on 2.x, while `params` exists on both lines
    (added in 1.23.0, the dependency floor).
    """
    from mcp import ClientSession

    for method_name in ("list_tools", "list_prompts", "list_resources", "list_resource_templates"):
        method_parameters = inspect.signature(getattr(ClientSession, method_name)).parameters
        assert "params" in method_parameters, method_name


def test_installed_line_spells_the_accessor_read_fields_as_expected():
    """Test that every model field a `_compat` accessor reads exists on the installed line's models."""
    from mcp.types import ListResourceTemplatesResult, ListToolsResult, ToolExecution

    if _compat.MCP_V2:
        assert "next_cursor" in ListToolsResult.model_fields
        assert "resource_templates" in ListResourceTemplatesResult.model_fields
        assert "task_support" in ToolExecution.model_fields
    else:
        assert "nextCursor" in ListToolsResult.model_fields
        assert "resourceTemplates" in ListResourceTemplatesResult.model_fields
        assert "taskSupport" in ToolExecution.model_fields


def test_next_cursor_v1_reads_the_camel_case_field(monkeypatch):
    """Test that the 1.x branch reads the result's `nextCursor` field."""
    monkeypatch.setattr(_compat, "MCP_V2", False)

    assert _compat.next_cursor(SimpleNamespace(nextCursor="page-2")) == "page-2"


def test_next_cursor_v2_reads_the_snake_case_field(monkeypatch):
    """Test that the 2.x branch reads the result's `next_cursor` field."""
    monkeypatch.setattr(_compat, "MCP_V2", True)

    assert _compat.next_cursor(SimpleNamespace(next_cursor="page-2")) == "page-2"


def test_resource_templates_v1_reads_the_camel_case_field(monkeypatch):
    """Test that the 1.x branch reads the result's `resourceTemplates` field."""
    monkeypatch.setattr(_compat, "MCP_V2", False)

    assert _compat.resource_templates(SimpleNamespace(resourceTemplates=["template"])) == ["template"]


def test_resource_templates_v2_reads_the_snake_case_field(monkeypatch):
    """Test that the 2.x branch reads the result's `resource_templates` field."""
    monkeypatch.setattr(_compat, "MCP_V2", True)

    assert _compat.resource_templates(SimpleNamespace(resource_templates=["template"])) == ["template"]


def test_task_support_v1_reads_the_camel_case_field(monkeypatch):
    """Test that the 1.x branch reads the tool's `execution.taskSupport` field."""
    monkeypatch.setattr(_compat, "MCP_V2", False)

    assert _compat.task_support(SimpleNamespace(execution=SimpleNamespace(taskSupport="optional"))) == "optional"


def test_task_support_v2_reads_the_snake_case_field(monkeypatch):
    """Test that the 2.x branch reads the tool's `execution.task_support` field."""
    monkeypatch.setattr(_compat, "MCP_V2", True)

    assert _compat.task_support(SimpleNamespace(execution=SimpleNamespace(task_support="optional"))) == "optional"


def test_task_support_returns_none_for_a_tool_without_execution():
    """Test that a tool with no execution block declares no task support level."""
    assert _compat.task_support(SimpleNamespace(execution=None)) is None


def test_mcp_error_resolves_to_installed_exception():
    """Test that MCPError is the mcp package's error type regardless of its spelling."""
    import mcp.shared.exceptions as mcp_exceptions

    installed = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError
    assert MCPError is installed


def test_get_session_id_callback_falls_back_to_plain_callable(monkeypatch):
    """Test that `GetSessionIdCallback` degrades to a plain callable alias when the name is absent.

    Reloads `_compat` with the name deleted from the installed transport
    module to exercise the 2.x fallback on either line, then reloads again so
    later tests see the module as built against the real environment.
    """
    monkeypatch.delattr(streamable_http_module, "GetSessionIdCallback", raising=False)
    try:
        reloaded = importlib.reload(_compat)
        assert reloaded.GetSessionIdCallback == Callable[[], str | None]
    finally:
        monkeypatch.undo()
        importlib.reload(_compat)


@requires_mcp_v2
def test_installed_v2_line_exposes_negotiate_auto():
    """Test that the negotiation policy `negotiate_session` imports exists on the 2.x line."""
    from mcp.client.client import negotiate_auto

    assert callable(negotiate_auto)


@requires_mcp_v2
def test_installed_v2_session_exposes_negotiation_attributes():
    """Test that the session attributes the 2.x negotiation branch reads exist on the real ClientSession."""
    from mcp import ClientSession

    assert hasattr(ClientSession, "instructions")
    assert hasattr(ClientSession, "server_capabilities")


@pytest.mark.asyncio
async def test_negotiate_session_v1_runs_the_initialize_handshake(monkeypatch):
    """Test that the 1.x handshake is initialize(), with instructions read from its result."""
    monkeypatch.setattr(_compat, "MCP_V2", False)
    server_capabilities = MagicMock()
    session = MagicMock()
    session.initialize = AsyncMock(return_value=MagicMock(instructions="use the tools"))
    session.get_server_capabilities = MagicMock(return_value=server_capabilities)

    instructions, capabilities = await negotiate_session(session)

    session.initialize.assert_awaited_once_with()
    assert instructions == "use the tools"
    assert capabilities is server_capabilities


@pytest.mark.asyncio
async def test_negotiate_session_v2_negotiates_and_reads_session_properties(monkeypatch):
    """Test that the 2.x handshake delegates to negotiate_auto and reads the session properties.

    The `mcp.client.client` module exists only on the 2.x line, so a stub
    module stands in for it to exercise this branch under the 1.x pin.
    """
    monkeypatch.setattr(_compat, "MCP_V2", True)
    negotiate_auto = AsyncMock()
    client_module = ModuleType("mcp.client.client")
    client_module.negotiate_auto = negotiate_auto  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp.client.client", client_module)
    server_capabilities = MagicMock()
    session = MagicMock()
    session.instructions = "use the tools"
    session.server_capabilities = server_capabilities

    instructions, capabilities = await negotiate_session(session)

    negotiate_auto.assert_awaited_once_with(session)
    session.initialize.assert_not_called()
    assert instructions == "use the tools"
    assert capabilities is server_capabilities


@requires_mcp_v1
def test_streamable_http_transport_v1_call_shape():
    """Test that the 1.x transport receives url, headers, and auth as loose kwargs."""
    headers = {"Authorization": "Bearer token"}
    auth = MagicMock()

    with patch("mcp.client.streamable_http.streamablehttp_client") as mock_client:
        result = streamable_http_transport("https://example.com/mcp", headers=headers, auth=auth)

        mock_client.assert_called_once_with(url="https://example.com/mcp", headers=headers, auth=auth)
        assert result is mock_client.return_value


@requires_v2_transport_names
@pytest.mark.asyncio
async def test_streamable_http_transport_v2_owns_client_lifecycle(monkeypatch):
    """Test that the 2.x transport closes the HTTPX client it creates.

    Guards https://github.com/strands-agents/harness-sdk/pull/3708: 2.x's
    `streamable_http_client` only closes a client it created itself, so the
    adapter must bind the client's lifetime to the transport's.
    """
    monkeypatch.setattr(_compat, "MCP_V2", True)
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
