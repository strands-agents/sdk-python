"""Tests for the vended MCP client tool."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPAgentTool
from strands.types.tools import ToolContext
from strands.vended_tools import make_mcp_client
from strands.vended_tools.mcp_client.mcp_client import MCPClientToolError


class _StubAgent:
    """A minimal agent stand-in."""

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


def _make_mcp_tool(name: str = "test_tool", description: str = "A test tool", **kwargs: Any) -> MCPTool:
    """Build a real MCPTool; delegates to mcp.types.Tool so tool_spec works correctly."""
    return MCPTool(
        name=name,
        description=description,
        inputSchema=kwargs.pop("inputSchema", {"type": "object", "properties": {}}),
        **kwargs,
    )


def _make_agent_tool(name: str = "test_tool", description: str = "A test tool", **kwargs: Any) -> MCPAgentTool:
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

    async def _call(*args: Any, **kwargs: Any) -> Any:
        return call_tool_return or {"status": "success", "content": [{"text": "ok"}]}

    instance.call_tool_async = MagicMock(side_effect=_call)
    return MagicMock(return_value=[instance]), instance


def _mcp_instance(tools: list[MCPAgentTool] | None = None) -> MagicMock:
    """Return a MagicMock that satisfies load_servers' return contract."""
    instance = MagicMock()
    instance._list_all_tools_sync = MagicMock(return_value=tools or [])
    return instance


class TestServerValidation:
    """_validate_servers rejects invalid configs at construction time."""

    def test_empty_allowlist_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_mcp_client(servers=[])

    def test_disabled_config_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="disabled"):
            make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp", "disabled": True}])

    def test_both_url_and_command_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="both 'url'.*and 'command'|both.*url.*command"):
            make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp", "command": "node"}])

    def test_neither_url_nor_command_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="url.*or.*command|command.*or.*url|must have either"):
            make_mcp_client(servers=[{"args": ["server.js"]}])

    def test_different_configs_colliding_on_same_key_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="two different configs|duplicate"):
            make_mcp_client(
                servers=[
                    {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer A"}},
                    {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer B"}},
                ]
            )

    def test_identical_configs_are_deduplicated(self) -> None:
        config = {"url": "https://mcp.example.com/mcp"}
        tool = make_mcp_client(servers=[config, config])
        assert tool is not None

    def test_stdio_config_without_command_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="url.*or.*command|command.*or.*url|must have either"):
            make_mcp_client(servers=[{"args": ["server.js"]}])

    def test_stdio_config_is_accepted(self) -> None:
        tool = make_mcp_client(servers=[{"command": "node", "args": ["server.js"]}])
        assert tool.tool_name == "mcp_client"
        assert "node server.js" in tool.tool_spec["description"]


class TestConnect:
    """connect enforces the allowlist, validates input, and handles errors."""

    @pytest.mark.asyncio
    async def test_url_not_on_allowlist_is_rejected(self) -> None:
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with pytest.raises(MCPClientToolError, match="not on the allowlist"):
            await tool(command="connect", server="https://evil.example.com/mcp", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_missing_server_is_rejected(self) -> None:
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with pytest.raises(MCPClientToolError, match="server"):
            await t(command="connect", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_agent_isolation(self) -> None:
        """A connection opened by agent A is not visible to agent B."""
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as client_cls:
            client_cls.return_value = [_mcp_instance()]
            agent_a = _StubAgent(label="a")
            agent_b = _StubAgent(label="b")
            await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent_a))
            with pytest.raises(MCPClientToolError, match="No active connection"):
                await tool(command="list_tools", tool_context=_tool_context(agent_b))

    @pytest.mark.asyncio
    async def test_start_failure_stops_client_and_leaves_no_connection(self) -> None:
        """If start() raises, the partially-started client is stopped and no connection is registered."""
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        client_class, instance = _fake_mcp_client_class()
        instance.start = MagicMock(side_effect=RuntimeError("connection refused"))
        agent = _StubAgent()

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            with pytest.raises(RuntimeError, match="connection refused"):
                await t(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))

        instance.stop.assert_called_once()
        with pytest.raises(MCPClientToolError, match="No active connection"):
            await t(command="list_tools", tool_context=_tool_context(agent))


class TestSessionLifecycle:
    """Full connect -> list_tools -> call_tool -> disconnect flow and reconnect behaviour."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self) -> None:
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        client_class, instance = _fake_mcp_client_class(
            list_tools_return=[_make_agent_tool(name="echo", description="Echoes input")],
            call_tool_return={"status": "success", "content": [{"text": "hello world"}]},
        )
        agent = _StubAgent()

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            connect_result = await tool(
                command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            assert connect_result == "Successfully connected to https://mcp.example.com/mcp"
            instance.start.assert_called_once()

            list_result = await tool(command="list_tools", tool_context=_tool_context(agent))
            assert len(list_result) == 1
            assert list_result[0]["name"] == "echo"

            call_result = await tool(
                command="call_tool",
                tool_name="echo",
                arguments={"msg": "hi"},
                tool_context=_tool_context(agent),
            )
            assert call_result["status"] == "success"
            assert call_result["content"][0]["text"] == "hello world"
            call_kwargs = instance.call_tool_async.call_args.kwargs
            assert call_kwargs["name"] == "echo"
            assert call_kwargs["arguments"] == {"msg": "hi"}

            disconnect_result = await tool(command="disconnect", tool_context=_tool_context(agent))
            assert disconnect_result == "Successfully disconnected"
            instance.stop.assert_called_once()

            with pytest.raises(MCPClientToolError, match="No active connection"):
                await tool(command="list_tools", tool_context=_tool_context(agent))

    @pytest.mark.asyncio
    async def test_reconnect_stops_existing_client(self) -> None:
        """connect always stops the existing client and starts a fresh one."""
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        client_class, instance = _fake_mcp_client_class()
        agent = _StubAgent()

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))
            await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))

        assert instance.stop.call_count == 1
        assert instance.start.call_count == 2

    @pytest.mark.asyncio
    async def test_commands_without_active_connection_are_rejected(self) -> None:
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        for command in ("list_tools", "call_tool"):
            with pytest.raises(MCPClientToolError, match="No active connection"):
                await t(command=command, tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_call_tool_without_name_is_rejected(self) -> None:
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        agent = _StubAgent()
        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as client_cls:
            client_cls.return_value = [_mcp_instance()]
            await t(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))
            with pytest.raises(MCPClientToolError, match="tool_name"):
                await t(command="call_tool", tool_context=_tool_context(agent))

    @pytest.mark.asyncio
    async def test_disconnect_when_stop_raises_still_evicts(self) -> None:
        """RuntimeError from stop() must not prevent session eviction."""
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        agent = _StubAgent()
        client_class, instance = _fake_mcp_client_class()
        instance.stop = MagicMock(side_effect=RuntimeError("already closed"))

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            await t(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))
            result = await t(command="disconnect", tool_context=_tool_context(agent))

        assert result == "Successfully disconnected"
        with pytest.raises(MCPClientToolError, match="No active connection"):
            await t(command="list_tools", tool_context=_tool_context(agent))

    @pytest.mark.asyncio
    async def test_cancel_signal_forwarded_to_call_tool_async(self) -> None:
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        captured: dict[str, Any] = {}

        async def _call(*args: Any, **kwargs: Any) -> Any:
            captured["cancel_signal"] = kwargs.get("cancel_signal")
            return {"status": "success", "content": []}

        client_class, instance = _fake_mcp_client_class()
        instance.call_tool_async = _call

        cancel = threading.Event()
        agent = _StubAgent()
        ctx = _tool_context(agent)
        ctx.cancel_signal = cancel

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=ctx)
            await tool(command="call_tool", tool_name="slow", tool_context=ctx)

        assert captured["cancel_signal"] is cancel

    @pytest.mark.asyncio
    async def test_list_tools_returns_server_side_names(self) -> None:
        """list_tools must return mcp_tool.name (server-side) regardless of prefix config."""
        tool = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        agent = _StubAgent()
        agent_tool = MCPAgentTool(
            mcp_tool=_make_mcp_tool(name="echo"),
            mcp_client=MagicMock(),
            name_override="fs_echo",
        )
        client_class, _ = _fake_mcp_client_class(list_tools_return=[agent_tool])
        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers", client_class):
            await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))
            tools = await tool(command="list_tools", tool_context=_tool_context(agent))
        assert tools[0]["name"] == "echo"


class TestConfigForwarding:
    """The matched server config is forwarded correctly to MCPClient."""

    @pytest.mark.asyncio
    async def test_matched_config_reaches_load_servers(self) -> None:
        server_config = {"url": "https://mcp.example.com/mcp", "headers": {"X-Api-Key": "secret"}}
        tool = make_mcp_client(servers=[server_config])
        agent = _StubAgent()

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as mock_load:
            mock_load.return_value = [_mcp_instance()]
            await tool(command="connect", server="https://mcp.example.com/mcp", tool_context=_tool_context(agent))

        mock_load.assert_called_once()
        _, passed_config = mock_load.call_args[0][0].popitem()
        assert passed_config["headers"] == {"X-Api-Key": "secret"}

    @pytest.mark.asyncio
    async def test_stdio_connect_key_round_trips_to_output(self) -> None:
        tool = make_mcp_client(servers=[{"command": "npx", "args": ["-y", "my-server"]}])
        agent = _StubAgent()

        with patch("strands.vended_tools.mcp_client.mcp_client.MCPClient.load_servers") as mock_load:
            mock_load.return_value = [_mcp_instance()]
            result = await tool(command="connect", server="npx -y my-server", tool_context=_tool_context(agent))

        assert "npx -y my-server" in result


class TestToolMetadata:
    """The tool exposes a sensible name, description, and input schema."""

    def test_custom_name(self) -> None:
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}], name="my_mcp")
        assert t.tool_name == "my_mcp"

    def test_default_description_includes_permitted_servers(self) -> None:
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        assert "https://mcp.example.com/mcp" in t.tool_spec["description"]

    def test_schema_exposes_command_field_and_hides_context(self) -> None:
        t = make_mcp_client(servers=[{"url": "https://mcp.example.com/mcp"}])
        props = t.tool_spec["inputSchema"]["json"]["properties"]
        assert "command" in props
        assert "server" in props
        assert "session_id" not in props
        assert "tool_context" not in props
