"""Tests for the vended MCP client tool."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPAgentTool
from strands.types.tools import ToolContext
from strands.vended_tools import make_mcp_client
from strands.vended_tools.mcp_client.mcp_client import (
    MCPClientToolError,
    _canonicalise_url,
    _validate_servers,
)


class _StubAgent:
    """A minimal agent stand-in that supports weakref (SimpleNamespace does not)."""

    def __init__(self, label: str | None = None) -> None:
        self.label = label


def _tool_context(agent: Any | None = None) -> ToolContext:
    """Build a ToolContext with a distinct agent object."""
    if agent is None:
        agent = _StubAgent()
    return ToolContext(
        tool_use={"name": "mcp_client", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={},
    )


def _make_mcp_tool(name: str = "test_tool", description: str = "A test tool", **kwargs) -> MCPTool:
    """Build a real MCPTool; delegates to mcp.types.Tool so tool_spec works correctly."""
    return MCPTool(
        name=name,
        description=description,
        inputSchema=kwargs.pop("inputSchema", {"type": "object", "properties": {}}),
        **kwargs,
    )


def _make_agent_tool(name: str = "test_tool", description: str = "A test tool", **kwargs) -> MCPAgentTool:
    """Wrap a real MCPTool in a real MCPAgentTool backed by a MagicMock client."""
    return MCPAgentTool(mcp_tool=_make_mcp_tool(name=name, description=description, **kwargs), mcp_client=MagicMock())


def _fake_mcp_client_class(
    *,
    list_tools_return: list[MCPAgentTool] | None = None,
    call_tool_return: dict[str, Any] | None = None,
) -> Any:
    """Return a MagicMock-based MCPClient replacement whose start/stop record calls."""
    instance = MagicMock()
    instance.start = MagicMock()
    instance.stop = MagicMock()
    instance._list_all_tools_sync = MagicMock(return_value=list_tools_return or [])
    instance.load_tools = AsyncMock(return_value=list_tools_return or [])

    async def _call(*args: Any, **kwargs: Any) -> Any:
        return call_tool_return or {"status": "success", "content": [{"text": "ok"}]}

    instance.call_tool_async = _call
    return MagicMock(return_value=[instance]), instance


def _mcp_instance(tools: list[MCPAgentTool] | None = None) -> MagicMock:
    """Return a MagicMock that satisfies _build_client_from_config's return contract."""
    instance = MagicMock()
    instance._list_all_tools_sync = MagicMock(return_value=tools or [])
    instance.load_tools = AsyncMock(return_value=tools or [])
    return instance


class TestSessionLifecycle:
    """A full connect -> list_tools -> call_tool -> disconnect flow."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        client_class, instance = _fake_mcp_client_class(
            list_tools_return=[_make_agent_tool(name="echo", description="Echoes input")],
            call_tool_return={
                "status": "success",
                "content": [{"text": "hello world"}],
            },
        )

        agent = _StubAgent(label="a")

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            connect_result = await tool(
                command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            assert "session_id" in connect_result
            assert connect_result["server"] == "https://mcp.example.com/mcp"
            instance.start.assert_called_once()

            list_result = await tool(
                command="list_tools", session_id=connect_result["session_id"], tool_context=_tool_context(agent)
            )
            assert len(list_result) == 1
            assert list_result[0]["name"] == "echo"

            call_result = await tool(
                command="call_tool",
                session_id=connect_result["session_id"],
                tool_name="echo",
                arguments={"msg": "hi"},
                tool_context=_tool_context(agent),
            )
            assert call_result["status"] == "success"
            assert call_result["content"][0]["text"] == "hello world"
            assert "truncated" not in call_result

            disconnect_result = await tool(
                command="disconnect", session_id=connect_result["session_id"], tool_context=_tool_context(agent)
            )
            assert disconnect_result == f"Session successfully disconnected: {connect_result['session_id']}"
            instance.stop.assert_called_once()

            with pytest.raises(MCPClientToolError, match="No active session"):
                await tool(
                    command="list_tools",
                    session_id=connect_result["session_id"],
                    tool_context=_tool_context(agent),
                )


class TestAllowlistEnforcement:
    """Server allowlist rejects anything the developer didn't sign off on."""

    @pytest.mark.asyncio
    async def test_url_not_on_allowlist_is_rejected(self):
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with pytest.raises(MCPClientToolError, match="not on the allowlist"):
            await tool(command="connect", server="https://evil.example.com/mcp", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_allowlist_match_is_scheme_and_host_case_insensitive(self):
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as client_cls:
            instance = _mcp_instance()
            client_cls.return_value = [instance]
            result = await tool(
                command="connect",
                server="HTTPS://MCP.EXAMPLE.COM/mcp",
                tool_context=_tool_context(),
            )
        assert "session_id" in result
        instance.start.assert_called_once()

    def test_empty_allowlist_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="must not be empty"):
            make_mcp_client(servers=[])

    def test_disallowed_scheme_in_allowlist_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="unsupported scheme"):
            make_mcp_client(servers=[{"url": "file:///etc/passwd"}])

    def test_missing_host_in_allowlist_is_rejected(self):
        with pytest.raises(ValueError, match="no host"):
            make_mcp_client(servers=[{"url": "https:///"}])

    def test_stdio_config_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="url.*or.*command|command.*or.*url|must have either"):
            make_mcp_client(servers=[{"args": ["server.js"]}])

    def test_stdio_config_is_accepted_at_construction(self):
        # A stdio config with command+args must be accepted; transport creation happens at connect time.
        tool = make_mcp_client(servers=[{"command": "node", "args": ["server.js"]}])
        assert tool.tool_name == "mcp_client"

    def test_config_with_both_url_and_command_is_rejected(self):
        with pytest.raises(ValueError, match="both 'url'.*and 'command'|both.*url.*command"):
            make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp", "command": "node"}])

    def test_disabled_config_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="disabled"):
            make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp", "disabled": True}])


class TestConnectInputValidation:
    """Connect requires ``server``; call_tool requires ``tool_name``."""

    @pytest.mark.asyncio
    async def test_connect_without_server_is_rejected(self):
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with pytest.raises(MCPClientToolError, match="server"):
            await t(command="connect", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_call_tool_without_name_is_rejected(self):
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        agent = _StubAgent()
        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as client_cls:
            client_cls.return_value = [_mcp_instance()]
            connect_result = await t(
                command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            with pytest.raises(MCPClientToolError, match="tool_name"):
                await t(command="call_tool", session_id=connect_result["session_id"], tool_context=_tool_context(agent))


class TestSessionScoping:
    """Sessions are scoped to the agent that opened them."""

    @pytest.mark.asyncio
    async def test_unknown_session_id_is_rejected(self):
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with pytest.raises(MCPClientToolError, match="No active session"):
            await tool(command="list_tools", session_id="does-not-exist", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_missing_session_id_is_rejected(self):
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with pytest.raises(MCPClientToolError, match="required"):
            await tool(command="list_tools", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_session_id_from_another_agent_is_rejected(self):
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as client_cls:
            client_cls.return_value = [_mcp_instance()]

            agent_a = _StubAgent(label="a")
            agent_b = _StubAgent(label="b")

            connected = await tool(
                command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent_a)
            )
            with pytest.raises(MCPClientToolError, match="No active session"):
                await tool(
                    command="list_tools", session_id=connected["session_id"], tool_context=_tool_context(agent_b)
                )


class TestUrlNormalisation:
    """The URL canonicalisation used for allowlist matching."""

    def test_canonicalise_strips_trailing_slash(self):
        assert _canonicalise_url("https://Example.com/mcp/") == "https://example.com/mcp"

    def test_canonicalise_lowercases_scheme_and_host(self):
        assert _canonicalise_url("HTTPS://EXAMPLE.COM/PATH") == "https://example.com/PATH"

    def test_canonicalise_preserves_non_default_port_and_query(self):
        assert _canonicalise_url("https://Example.com:8443/mcp?token=abc") == "https://example.com:8443/mcp?token=abc"

    def test_canonicalise_drops_default_https_port(self):
        # `https://host` and `https://host:443` must canonicalise identically so an
        # allowlist entry without the port still matches a connect URL that names it.
        assert _canonicalise_url("https://mcp.example.com:443/mcp") == "https://mcp.example.com/mcp"

    def test_canonicalise_drops_default_http_port(self):
        assert _canonicalise_url("http://mcp.example.com:80/mcp") == "http://mcp.example.com/mcp"

    def test_validate_servers_deduplicates(self):
        server_map = _validate_servers(
            [
                {"url": "https://example.com/mcp"},
                {"url": "HTTPS://EXAMPLE.COM/mcp/"},
            ]
        )
        assert set(server_map.keys()) == {"https://example.com/mcp"}

    def test_validate_servers_treats_default_port_as_equal(self):
        server_map = _validate_servers(
            [
                {"url": "https://example.com/mcp"},
                {"url": "https://example.com:443/mcp"},
            ]
        )
        assert set(server_map.keys()) == {"https://example.com/mcp"}


class TestCancelSignal:
    """The agent cancel signal is forwarded to call_tool_async."""

    @pytest.mark.asyncio
    async def test_cancel_signal_forwarded_to_call_tool_async(self):

        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        captured: dict[str, Any] = {}

        async def _call(*args: Any, **kwargs: Any) -> Any:
            captured["cancel_signal"] = kwargs.get("cancel_signal")
            return {"status": "success", "content": []}

        client_class, instance = _fake_mcp_client_class(
            list_tools_return=[_make_agent_tool(name="slow")],
        )
        instance.call_tool_async = _call

        cancel = threading.Event()
        agent = _StubAgent()
        ctx = _tool_context(agent)
        ctx.cancel_signal = cancel

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            connected = await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=ctx)
            await tool(
                command="call_tool",
                session_id=connected["session_id"],
                tool_name="slow",
                tool_context=ctx,
            )

        assert captured["cancel_signal"] is cancel


class TestToolMetadata:
    """The tool exposes a sensible name, description, and input schema."""

    def test_custom_name(self):
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}], name="my_mcp")
        assert t.tool_name == "my_mcp"

    def test_default_description_includes_permitted_servers(self):
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        assert "https://mcp.example.com/mcp" in t.tool_spec["description"]

    def test_schema_exposes_command_field_and_hides_context(self):
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        props = t.tool_spec["inputSchema"]["json"]["properties"]
        assert "command" in props
        assert "server" in props
        assert "session_id" in props
        assert "tool_context" not in props
