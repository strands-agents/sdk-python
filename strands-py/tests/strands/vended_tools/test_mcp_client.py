"""Tests for the vended MCP client tool.

The tool is a thin shim over ``strands.tools.mcp.MCPClient``. These tests exercise the
security surface — allowlist enforcement, SSRF guard, session scoping, session cap — with
the real code paths, and cover the connect->list->call->disconnect happy path by patching
``MCPClient`` so no real MCP server is needed.
"""

from __future__ import annotations

import gc
import socket
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands.types.tools import ToolContext
from strands.vended_tools import make_mcp_client
from strands.vended_tools.mcp_client.mcp_client import (
    _canonicalise_url,
    _cap_structured_content,
    _no_redirect_httpx_client_factory,
    _normalise_allowlist,
)


class _StubAgent:
    """A minimal agent stand-in that supports weakref (SimpleNamespace does not)."""

    def __init__(self, label: str | None = None) -> None:
        self.label = label


def _tool_context(agent: Any | None = None) -> ToolContext:
    """Build a ToolContext with a distinct sentinel agent object for identity checks."""
    if agent is None:
        agent = _StubAgent()
    return ToolContext(
        tool_use={"name": "mcp_client", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={},
    )


class _FakeMCPTool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema
        self.outputSchema: dict[str, Any] | None = None


class _FakeAgentTool:
    def __init__(self, tool: _FakeMCPTool) -> None:
        self.mcp_tool = tool


class _FakePaginatedList(list):
    def __init__(self, items: list[Any], token: str | None = None) -> None:
        super().__init__(items)
        self.pagination_token = token


def _fake_mcp_client_class(
    *,
    list_tools_return: list[_FakeAgentTool] | None = None,
    call_tool_return: dict[str, Any] | None = None,
) -> Any:
    """Return a MagicMock-based MCPClient replacement whose start/stop record calls."""
    instance = MagicMock()
    instance.start = MagicMock()
    instance.stop = MagicMock()
    instance.list_tools_sync = MagicMock(return_value=_FakePaginatedList(list_tools_return or [], token=None))

    async def _call(*args: Any, **kwargs: Any) -> Any:
        return call_tool_return or {"status": "success", "content": [{"text": "ok"}]}

    instance.call_tool_async = _call
    return MagicMock(return_value=instance), instance


def _mcp_instance(tools: list[_FakeAgentTool] | None = None) -> MagicMock:
    """Return a MagicMock configured to look enough like an ``MCPClient`` that connect works.

    ``_list_tools_via_client`` reads ``pagination_token`` off the returned page and
    iterates it, so a bare ``MagicMock`` (whose ``pagination_token`` is another
    ``MagicMock`` rather than ``None``) would loop forever.
    """
    instance = MagicMock()
    instance.list_tools_sync = MagicMock(return_value=_FakePaginatedList(tools or [], token=None))
    return instance


class TestAllowlistEnforcement:
    """URL allowlist rejects anything the developer didn't sign off on."""

    @pytest.mark.asyncio
    async def test_url_not_on_allowlist_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with pytest.raises(ValueError, match="not on the developer-set allowlist"):
            await tool(op="connect", server_url="https://evil.example.com/mcp", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_allowlist_match_is_scheme_and_host_case_insensitive(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient") as client_cls,
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            instance = _mcp_instance()
            client_cls.return_value = instance
            result = await tool(
                op="connect",
                server_url="HTTPS://MCP.EXAMPLE.COM/mcp",
                tool_context=_tool_context(),
            )
        assert "session_id" in result
        instance.start.assert_called_once()

    def test_empty_allowlist_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="must not be empty"):
            make_mcp_client(allowed_urls=[])

    def test_disallowed_scheme_in_allowlist_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="unsupported scheme"):
            make_mcp_client(allowed_urls=["file:///etc/passwd"])

    def test_missing_host_in_allowlist_is_rejected(self):
        with pytest.raises(ValueError, match="no host"):
            make_mcp_client(allowed_urls=["https:///"])

    def test_allowlist_entry_with_userinfo_is_rejected(self):
        # A canonicaliser that dropped userinfo would let `https://user:pass@host/x`
        # collide with `https://host/x` on the allowlist. Reject at construction.
        with pytest.raises(ValueError, match="credentials"):
            make_mcp_client(allowed_urls=["https://user:pass@mcp.example.com/mcp"])

    def test_allowlist_entry_with_fragment_is_rejected(self):
        with pytest.raises(ValueError, match="fragment"):
            make_mcp_client(allowed_urls=["https://mcp.example.com/mcp#anchor"])

    @pytest.mark.asyncio
    async def test_connect_url_with_userinfo_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with pytest.raises(ValueError, match="credentials"):
            await tool(
                op="connect",
                server_url="https://user:pass@mcp.example.com/mcp",
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_connect_url_with_fragment_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with pytest.raises(ValueError, match="fragment"):
            await tool(
                op="connect",
                server_url="https://mcp.example.com/mcp#x",
                tool_context=_tool_context(),
            )

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, "8"])
    def test_session_limit_must_be_positive_integer(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"], session_limit=bad)  # type: ignore[arg-type]


class TestSSRFGuard:
    """The SSRF guard rejects hostnames that resolve to non-public addresses."""

    @pytest.mark.asyncio
    async def test_hostname_resolving_to_private_ip_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_hostname_resolving_to_loopback_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_literal_private_ip_in_url_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["http://192.168.1.1/mcp"])
        with pytest.raises(ValueError, match="non-public address"):
            await tool(op="connect", server_url="http://192.168.1.1/mcp", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_link_local_metadata_ip_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["http://169.254.169.254/mcp"])
        with pytest.raises(ValueError, match="metadata address|non-public address"):
            await tool(op="connect", server_url="http://169.254.169.254/mcp", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_cgnat_ip_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["http://100.64.0.1/mcp"])
        with pytest.raises(ValueError, match="non-public address"):
            await tool(op="connect", server_url="http://100.64.0.1/mcp", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_ipv4_mapped_cgnat_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::ffff:100.64.0.1", 0, 0, 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_ipv4_mapped_private_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::ffff:10.0.0.5", 0, 0, 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_ipv4_multicast_is_rejected(self):
        """`is_global` returns True for multicast on CPython — the guard must layer explicit checks."""
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("239.255.0.1", 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_ipv6_multicast_is_rejected(self):
        """`is_global` also returns True for IPv6 multicast (`ff00::/8`) on CPython."""
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("ff02::1", 0, 0, 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_ipv6_site_local_is_rejected(self):
        """`fec0::/10` slips through `is_global` on some Python versions — explicit block."""
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fec0::1", 0, 0, 0))]
            with pytest.raises(ValueError, match="non-public address"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    async def test_unresolvable_host_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai:
            gai.side_effect = socket.gaierror("nope")
            with pytest.raises(ValueError, match="Could not resolve host"):
                await tool(
                    op="connect",
                    server_url="https://mcp.example.com/mcp",
                    tool_context=_tool_context(),
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "host",
        [
            "svc.internal",
            "db.corp",
            "printer.local",
            "example.LOCAL",
            "gateway.home",
            "onboarding.i2p",
            "secret.onion",
            "svc.internal.",  # trailing dot
        ],
    )
    async def test_suffix_denylist_rejects_before_dns(self, host):
        tool = make_mcp_client(allowed_urls=[f"https://{host}/mcp"])
        with pytest.raises(ValueError, match="blocked suffix"):
            await tool(op="connect", server_url=f"https://{host}/mcp", tool_context=_tool_context())


class TestSessionScoping:
    """Sessions are scoped to the agent that opened them."""

    @pytest.mark.asyncio
    async def test_session_id_from_another_agent_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient") as client_cls,
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            client_cls.return_value = _mcp_instance()

            agent_a = _StubAgent(label="a")
            agent_b = _StubAgent(label="b")

            connected = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent_a)
            )
            session_id = connected["session_id"]

            with pytest.raises(ValueError, match="No active session"):
                await tool(op="list_tools", session_id=session_id, tool_context=_tool_context(agent_b))

    @pytest.mark.asyncio
    async def test_gc_of_owning_agent_invalidates_sessions(self):
        """If the owning agent is garbage-collected, its sessions become unreachable.

        The check-that-fresh-agent-cannot-list-sessions branch of this test is not
        specific to GC — cross-agent rejection would fire whether the tool held an
        ``id(agent)`` or a ``weakref.ref(agent)``. To exercise the GC-specific path
        we also keep our own :class:`weakref.ref` to the agent and assert it is
        dead after :func:`gc.collect`; that assertion fails if GC didn't actually
        run (e.g. because something held a strong reference), which is what the
        second-pass reviewer asked us to prove.
        """
        import weakref as _weakref

        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient") as client_cls,
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            client_cls.return_value = _mcp_instance()

            class _Agent:
                pass

            agent = _Agent()
            # Independent weakref: dies iff the agent object is GC'd. Distinct from
            # the tool's internal weakref so it proves GC actually happened.
            gc_witness = _weakref.ref(agent)

            connected = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            session_id = connected["session_id"]

            # Sanity: while a strong reference exists, our witness is alive.
            assert gc_witness() is agent

            # Drop the only strong reference and run GC.
            del agent
            gc.collect()

            # This is the GC-specific assertion. If nothing else holds `agent`, this
            # is None; if it isn't, GC didn't actually clean up and the test fails.
            assert gc_witness() is None

            # And the enforcement path still returns the shared error string.
            with pytest.raises(ValueError, match="No active session"):
                await tool(
                    op="list_tools",
                    session_id=session_id,
                    tool_context=_tool_context(_Agent()),
                )

    @pytest.mark.asyncio
    async def test_unknown_session_id_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with pytest.raises(ValueError, match="No active session"):
            await tool(op="list_tools", session_id="does-not-exist", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_missing_session_id_is_rejected(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with pytest.raises(ValueError, match="required"):
            await tool(op="list_tools", tool_context=_tool_context())


class TestSessionLimit:
    """The tool refuses to open more than ``session_limit`` sessions concurrently."""

    @pytest.mark.asyncio
    async def test_session_limit_rejects_further_connects(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"], session_limit=1)
        # Persistent agents: otherwise the ephemeral `_tool_context()` agents get
        # garbage-collected between calls and the dead-session purge in connect
        # frees the slot before the cap check.
        agent_a = _StubAgent(label="a")
        agent_b = _StubAgent(label="b")
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient") as client_cls,
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            client_cls.return_value = _mcp_instance()
            await tool(op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent_a))
            with pytest.raises(RuntimeError, match="concurrent sessions"):
                await tool(
                    op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent_b)
                )

    @pytest.mark.asyncio
    async def test_dead_sessions_are_purged_before_enforcing_cap(self):
        # If a session's owning agent is garbage-collected, the slot should be
        # reclaimed automatically at the next connect. Otherwise a stream of
        # connect-and-forget agents pins the cap at zero forever.
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"], session_limit=1)
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient") as client_cls,
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            client_cls.return_value = _mcp_instance()

            class _Agent:
                pass

            ephemeral = _Agent()
            await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(ephemeral)
            )
            del ephemeral
            gc.collect()

            # A second connect from a fresh agent should now succeed: the first
            # session is unreachable and its slot has been returned to the pool.
            client_cls.return_value = _mcp_instance()
            connected = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context()
            )
            assert "session_id" in connected


class TestHappyPath:
    """A full connect -> list_tools -> call_tool -> disconnect flow."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        fake_tool = _FakeMCPTool(
            name="echo",
            description="Echoes input",
            input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        )
        client_class, instance = _fake_mcp_client_class(
            list_tools_return=[_FakeAgentTool(fake_tool)],
            call_tool_return={
                "status": "success",
                "content": [{"text": "hello world"}],
            },
        )

        agent = _StubAgent(label="a")

        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient", client_class),
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

            connect_result = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            assert "session_id" in connect_result
            assert connect_result["server_url"] == "https://mcp.example.com/mcp"
            instance.start.assert_called_once()

            list_result = await tool(
                op="list_tools", session_id=connect_result["session_id"], tool_context=_tool_context(agent)
            )
            assert len(list_result["tools"]) == 1
            assert list_result["tools"][0]["name"] == "echo"

            call_result = await tool(
                op="call_tool",
                session_id=connect_result["session_id"],
                tool_name="echo",
                arguments={"msg": "hi"},
                tool_context=_tool_context(agent),
            )
            assert call_result["status"] == "success"
            assert call_result["text"] == "hello world"
            assert "truncated" not in call_result

            disconnect_result = await tool(
                op="disconnect", session_id=connect_result["session_id"], tool_context=_tool_context(agent)
            )
            assert disconnect_result == {"disconnected": True}
            instance.stop.assert_called_once()

            with pytest.raises(ValueError, match="No active session"):
                await tool(op="list_tools", session_id=connect_result["session_id"], tool_context=_tool_context(agent))

    @pytest.mark.asyncio
    async def test_call_tool_rejects_name_not_advertised_by_server(self):
        # Both SDKs cache the tool set at connect and reject unadvertised names
        # locally, so an obvious typo fails with a clear error instead of hitting
        # the server. Mirror-tested on the TypeScript side.
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        fake_tool = _FakeMCPTool(name="echo", description="", input_schema={"type": "object"})
        client_class, _ = _fake_mcp_client_class(list_tools_return=[_FakeAgentTool(fake_tool)])
        agent = _StubAgent()
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient", client_class),
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            connected = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            with pytest.raises(ValueError, match="not exposed by the connected server"):
                await tool(
                    op="call_tool",
                    session_id=connected["session_id"],
                    tool_name="not_a_real_tool",
                    tool_context=_tool_context(agent),
                )


class TestResultTruncation:
    """Oversized tool-call results are truncated before being returned to the model."""

    @pytest.mark.asyncio
    async def test_oversized_text_is_truncated(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        big_text = "x" * 250_000
        fake_tool = _FakeMCPTool(name="anything", description="", input_schema={"type": "object"})
        client_class, _ = _fake_mcp_client_class(
            list_tools_return=[_FakeAgentTool(fake_tool)],
            call_tool_return={"status": "success", "content": [{"text": big_text}]},
        )
        agent = _StubAgent(label="a")

        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient", client_class),
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            connect_result = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            call_result = await tool(
                op="call_tool",
                session_id=connect_result["session_id"],
                tool_name="anything",
                tool_context=_tool_context(agent),
            )
        assert call_result["truncated"] is True
        assert len(call_result["text"]) < len(big_text)

    @pytest.mark.asyncio
    async def test_oversized_structured_content_is_capped(self):
        tool = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        big_structured = {"payload": "y" * 250_000}
        fake_tool = _FakeMCPTool(name="anything", description="", input_schema={"type": "object"})
        client_class, _ = _fake_mcp_client_class(
            list_tools_return=[_FakeAgentTool(fake_tool)],
            call_tool_return={
                "status": "success",
                "content": [{"text": "ok"}],
                "structuredContent": big_structured,
            },
        )
        agent = _StubAgent(label="a")

        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient", client_class),
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            connect_result = await tool(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            call_result = await tool(
                op="call_tool",
                session_id=connect_result["session_id"],
                tool_name="anything",
                tool_context=_tool_context(agent),
            )
        assert call_result["truncated"] is True
        assert call_result["structured_content"] != big_structured
        assert call_result["structured_content"].get("__truncated__") is True


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

    def test_normalise_allowlist_deduplicates(self):
        allowlist = _normalise_allowlist(["https://example.com/mcp", "HTTPS://EXAMPLE.COM/mcp/"])
        assert allowlist == frozenset({"https://example.com/mcp"})

    def test_normalise_allowlist_treats_default_port_as_equal(self):
        allowlist = _normalise_allowlist(["https://example.com/mcp", "https://example.com:443/mcp"])
        assert allowlist == frozenset({"https://example.com/mcp"})


class TestNoRedirect:
    """The httpx client used by MCP refuses to follow redirects."""

    def test_no_redirect_factory_disables_follow_redirects(self):
        # `_no_redirect_httpx_client_factory` is the MCP-shaped hook that guarantees
        # an allowlisted URL cannot be 3xx'd to a private endpoint the SSRF guard
        # never saw. Assert the resulting client is actually configured that way.
        client = _no_redirect_httpx_client_factory()
        assert client.follow_redirects is False


class TestStructuredContentCap:
    """Structured content is size-capped and reports the size in the unit it caps in."""

    def test_non_json_serialisable_structured_content_is_replaced(self):
        # Sets are not JSON-serialisable; the cap must swap the value for a marker
        # rather than smuggle an opaque blob through to the model.
        response: dict[str, Any] = {"status": "success", "text": ""}
        capped = _cap_structured_content({"key": {1, 2, 3}}, response)
        assert response.get("truncated") is True
        assert capped == {"__truncated__": True, "reason": "structured_content was not JSON-serialisable"}

    def test_capped_size_is_reported_in_bytes(self):
        # A payload of multi-byte characters exceeding the byte cap should report
        # the byte length in `size`, matching the unit the cap is measured in.
        multibyte = "é" * 90_000  # each character is two UTF-8 bytes → 180_000 bytes
        response: dict[str, Any] = {"status": "success", "text": ""}
        capped = _cap_structured_content({"payload": multibyte}, response)
        assert response.get("truncated") is True
        assert isinstance(capped, dict)
        assert capped["__truncated__"] is True
        # `size` should be a byte count (>= 180_000), not a character count (~90_000).
        assert capped["size"] > 100_000


class TestToolMetadata:
    """The tool exposes a sensible name, description, and input schema."""

    def test_custom_name(self):
        t = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"], name="my_mcp")
        assert t.tool_name == "my_mcp"

    def test_schema_exposes_op_field_and_hides_context(self):
        t = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        props = t.tool_spec["inputSchema"]["json"]["properties"]
        assert "op" in props
        assert "server_url" in props
        assert "session_id" in props
        assert "tool_context" not in props


class TestConnectInputValidation:
    """Connect requires ``server_url``; call_tool requires ``tool_name``."""

    @pytest.mark.asyncio
    async def test_connect_without_url_is_rejected(self):
        t = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        with pytest.raises(ValueError, match="server_url"):
            await t(op="connect", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_call_tool_without_name_is_rejected(self):
        t = make_mcp_client(allowed_urls=["https://mcp.example.com/mcp"])
        agent = _StubAgent()
        with (
            patch("strands.vended_tools.mcp_client.mcp_client.socket.getaddrinfo") as gai,
            patch("strands.vended_tools.mcp_client.mcp_client.MCPClient") as client_cls,
        ):
            gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            client_cls.return_value = _mcp_instance()
            connect_result = await t(
                op="connect", server_url="https://mcp.example.com/mcp", tool_context=_tool_context(agent)
            )
            with pytest.raises(ValueError, match="tool_name"):
                await t(op="call_tool", session_id=connect_result["session_id"], tool_context=_tool_context(agent))
