"""Tests for the a2a_client vended tool.

Covers the SSRF-facing surface (URL guard: schemes, private IPs, metadata IPs,
blocked suffixes, DNS resolution) first, then oversized-input rejection, then
the happy path against a mocked ``A2AAgent`` and card resolver.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.agent.agent_result import AgentResult
from strands.types.tools import ToolContext
from strands.vended_tools.a2a_client import a2a_client, make_a2a_client
from strands.vended_tools.a2a_client.url_guard import UrlNotAllowedError, validate_url


def _tool_context() -> ToolContext:
    """Build a minimal ToolContext for the a2a_client tool."""
    agent = SimpleNamespace()
    return ToolContext(
        tool_use={"name": "a2a_client", "toolUseId": "id", "input": {}},
        agent=agent,
        invocation_state={},
    )


def _make_agent_card(url: str = "https://example.com", name: str = "remote", description: str = "d") -> MagicMock:
    """Build a mock agent card with just the fields the tool reads."""
    card = MagicMock()
    card.url = url
    card.name = name
    card.description = description
    card.model_dump_json = MagicMock(return_value=f'{{"url":"{url}","name":"{name}"}}')
    return card


def _make_agent_result(text: str = "hello from remote", stop_reason: str = "end_turn") -> AgentResult:
    """Build a minimal AgentResult with a single text block."""
    return AgentResult(
        stop_reason=stop_reason,  # type: ignore[arg-type]
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=MagicMock(),
        state={},
    )


# =====================================================================
# URL guard — SSRF surface
# =====================================================================


class TestUrlGuard:
    """Standalone tests for the ``validate_url`` helper."""

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "gopher://example.com",
            "",
        ],
    )
    def test_rejects_non_http_schemes(self, url):
        with pytest.raises(UrlNotAllowedError):
            validate_url(url)

    def test_rejects_non_string(self):
        with pytest.raises(UrlNotAllowedError):
            validate_url(None)  # type: ignore[arg-type]

    def test_rejects_missing_hostname(self):
        with pytest.raises(UrlNotAllowedError, match="hostname"):
            validate_url("http:///path")

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1",
            "http://127.1.2.3",
            "http://[::1]",
            "http://localhost",
            "https://LOCALHOST",
        ],
    )
    def test_rejects_loopback(self, url):
        with pytest.raises(UrlNotAllowedError):
            validate_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.1",
            "http://192.168.1.1",
            "http://172.16.0.1",
        ],
    )
    def test_rejects_rfc1918_ips(self, url):
        with pytest.raises(UrlNotAllowedError):
            validate_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254",
        ],
    )
    def test_rejects_metadata_ip(self, url):
        with pytest.raises(UrlNotAllowedError):
            validate_url(url)

    def test_rejects_link_local(self):
        with pytest.raises(UrlNotAllowedError, match="link-local"):
            validate_url("http://169.254.1.1")

    @pytest.mark.parametrize(
        "url",
        [
            "http://foo.internal",
            "http://bar.corp",
            "http://svc.local",
            "http://api.home",
        ],
    )
    def test_rejects_blocked_suffixes(self, url):
        with pytest.raises(UrlNotAllowedError):
            validate_url(url)

    def test_allows_public_dns_name(self):
        # Use a hostname that reliably resolves to a public IP.
        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            host = validate_url("https://example.com")
            assert host == "example.com"

    def test_allows_public_ip_literal(self):
        host = validate_url("https://8.8.8.8")
        assert host == "8.8.8.8"

    def test_rejects_when_dns_resolves_to_private_ip(self):
        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("10.0.0.1", 0))]):
            with pytest.raises(UrlNotAllowedError, match="private"):
                validate_url("https://sneaky.example.com")

    def test_rejects_when_dns_returns_no_addresses(self):
        with patch("socket.getaddrinfo", return_value=[]):
            with pytest.raises(UrlNotAllowedError, match="resolved to no addresses"):
                validate_url("https://empty.example.com")

    def test_rejects_when_dns_fails(self):
        import socket as _socket

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("nope")):
            with pytest.raises(UrlNotAllowedError, match="could not resolve"):
                validate_url("https://nx.example.com")

    def test_developer_allowlist_admits_matching_prefix(self):
        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            host = validate_url(
                "https://api.example.com/v1/agent",
                allowed_prefixes=("https://api.example.com/",),
            )
            assert host == "api.example.com"

    def test_developer_allowlist_rejects_non_matching_prefix(self):
        with pytest.raises(UrlNotAllowedError, match="allowlist"):
            validate_url(
                "https://evil.example.com",
                allowed_prefixes=("https://api.example.com/",),
            )

    def test_rejects_cgnat(self):
        with pytest.raises(UrlNotAllowedError, match="CGNAT"):
            validate_url("http://100.64.0.1")

    @pytest.mark.parametrize(
        "url",
        [
            # Python's `is_global` returns True for multicast on 3.10-3.14,
            # so these must be rejected explicitly by the guard's own checks
            # rather than falling through to the `is_global` branch.
            "http://239.255.255.250",  # SSDP
            "http://224.0.0.1",
            "http://[ff02::1]",
        ],
    )
    def test_rejects_multicast_ip_literals(self, url):
        with pytest.raises(UrlNotAllowedError, match="multicast"):
            validate_url(url)

    def test_rejects_bare_metadata_hostname(self):
        # GCP: `http://metadata/` inside a VPC resolves to the metadata server.
        # We reject the label before ever calling getaddrinfo.
        with pytest.raises(UrlNotAllowedError, match="metadata label"):
            validate_url("http://metadata")
        with pytest.raises(UrlNotAllowedError, match="metadata label"):
            validate_url("http://METADATA/some/path")

    @pytest.mark.parametrize(
        "url",
        [
            "http://[::ffff:127.0.0.1]",
            "http://[::ffff:10.0.0.1]",
            "http://[::ffff:169.254.169.254]",
        ],
    )
    def test_rejects_ipv4_mapped_ipv6(self, url):
        with pytest.raises(UrlNotAllowedError):
            validate_url(url)

    def test_strips_trailing_dot_before_suffix_match(self):
        with pytest.raises(UrlNotAllowedError, match="blocked suffix"):
            validate_url("http://foo.internal.")


# =====================================================================
# Tool: input validation + oversize rejection
# =====================================================================


class TestToolInputValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_http_url(self):
        with pytest.raises(ValueError, match="scheme"):
            await a2a_client(
                url="ftp://example.com",
                message="hi",
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_rejects_private_ip(self):
        with pytest.raises(ValueError, match="private"):
            await a2a_client(
                url="http://192.168.1.1",
                message="hi",
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_rejects_metadata_ip(self):
        with pytest.raises(ValueError):
            await a2a_client(
                url="http://169.254.169.254",
                message="hi",
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_rejects_oversized_message(self):
        big = "x" * (64 * 1024 + 1)
        with pytest.raises(ValueError, match="limit is"):
            await a2a_client(
                url="https://example.com",
                message=big,
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_rejects_non_string_message(self):
        with pytest.raises(ValueError, match="string"):
            await a2a_client(
                url="https://example.com",
                message=123,  # type: ignore[arg-type]
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_developer_allowlist_rejects_off_list_url(self):
        tool = make_a2a_client(allowed_url_prefixes=("https://api.example.com/",))
        with pytest.raises(ValueError, match="allowlist"):
            await tool(
                url="https://other.example.com",
                message="hi",
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_rejects_when_multiagent_depth_at_cap(self):
        ctx = _tool_context()
        ctx.invocation_state["multiagent_depth"] = 3
        with pytest.raises(ValueError, match="multiagent_depth=3"):
            await a2a_client(url="https://example.com", message="hi", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_when_parent_cancel_signal_is_set(self):
        import threading

        ctx = _tool_context()
        cancel = threading.Event()
        cancel.set()
        ctx.agent._cancel_signal = cancel
        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with pytest.raises(asyncio.CancelledError):
                await a2a_client(url="https://example.com", message="hi", tool_context=ctx)


# =====================================================================
# Tool: oversize card / response
# =====================================================================


class TestOversizeGuard:
    @pytest.mark.asyncio
    async def test_rejects_oversized_card(self):
        big_card = _make_agent_card(url="https://example.com")
        big_card.model_dump_json = MagicMock(return_value="x" * (300 * 1024))

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=big_card)
                mock_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
                mock_agent_cls.return_value = mock_agent

                with pytest.raises(ValueError, match="agent card is"):
                    await a2a_client(
                        url="https://example.com",
                        message="hi",
                        tool_context=_tool_context(),
                    )

    @pytest.mark.asyncio
    async def test_truncates_oversized_response(self):
        card = _make_agent_card()
        long_text = "y" * (300 * 1024)
        result = _make_agent_result(text=long_text)

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = AsyncMock(return_value=result)
                mock_agent_cls.return_value = mock_agent

                out = await a2a_client(
                    url="https://example.com",
                    message="hi",
                    tool_context=_tool_context(),
                )
                assert out["output"].endswith("... [truncated]")
                assert len(out["output"].encode("utf-8")) <= 256 * 1024

    @pytest.mark.asyncio
    async def test_rejects_when_card_url_points_to_private_host(self):
        card = _make_agent_card(url="http://10.0.0.1")

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
                mock_agent_cls.return_value = mock_agent

                with pytest.raises(ValueError, match="disallowed url"):
                    await a2a_client(
                        url="https://example.com",
                        message="hi",
                        tool_context=_tool_context(),
                    )

    @pytest.mark.asyncio
    async def test_rejects_when_card_url_falls_outside_developer_allowlist(self):
        # Developer pins the tool to one remote; the remote's card advertises a
        # different public host. The allowlist is re-applied to `card.url`, so
        # the send is rejected even though the advertised host would pass the
        # SSRF checks on its own.
        tool = make_a2a_client(allowed_url_prefixes=("https://agents.example.com/",))
        card = _make_agent_card(url="https://other.example.com/")

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
                mock_agent_cls.return_value = mock_agent

                with pytest.raises(ValueError, match="disallowed url"):
                    await tool(
                        url="https://agents.example.com/one",
                        message="hi",
                        tool_context=_tool_context(),
                    )
                # The message must not have been sent.
                mock_agent.invoke_async.assert_not_called()


# =====================================================================
# Tool: happy path
# =====================================================================


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_response_and_agent_info(self):
        card = _make_agent_card(url="https://example.com", name="remote-agent", description="A test agent")
        result = _make_agent_result(text="hello from remote")

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = AsyncMock(return_value=result)
                mock_agent_cls.return_value = mock_agent

                out = await a2a_client(
                    url="https://example.com",
                    message="hi remote",
                    tool_context=_tool_context(),
                )

                assert out["status"] == "success"
                assert out["output"] == "hello from remote"
                assert out["remote_card"] == {
                    "name": "remote-agent",
                    "description": "A test agent",
                    "url": "https://example.com",
                }
                assert isinstance(out["execution_time_ms"], int)
                assert out["execution_time_ms"] >= 0
                mock_agent_cls.assert_called_once_with(endpoint="https://example.com", client_config=None)
                mock_agent.invoke_async.assert_awaited_once_with("hi remote")

    @pytest.mark.asyncio
    async def test_passes_developer_client_config(self):
        card = _make_agent_card()
        result = _make_agent_result()
        sentinel_config = object()  # ClientConfig is not needed for equality; identity is enough
        tool = make_a2a_client(client_config=sentinel_config)  # type: ignore[arg-type]

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = AsyncMock(return_value=result)
                mock_agent_cls.return_value = mock_agent

                await tool(url="https://example.com", message="hi", tool_context=_tool_context())

                mock_agent_cls.assert_called_once_with(
                    endpoint="https://example.com",
                    client_config=sentinel_config,
                )


# =====================================================================
# Tool: timeout + cancellation
# =====================================================================


class TestTimeoutAndCancellation:
    @pytest.mark.asyncio
    async def test_total_timeout_propagates_as_timeout_error(self):
        card = _make_agent_card()

        async def _slow_invoke(*_a, **_kw):
            await asyncio.sleep(1)
            return _make_agent_result()

        tool = make_a2a_client(timeout_seconds=0.05)

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = _slow_invoke
                mock_agent_cls.return_value = mock_agent

                with pytest.raises(TimeoutError, match="timed out"):
                    await tool(
                        url="https://example.com",
                        message="hi",
                        tool_context=_tool_context(),
                    )

    @pytest.mark.asyncio
    async def test_cancels_underlying_invoke(self):
        # Simulate the underlying invoke being cancelled: asyncio.wait_for
        # should surface CancelledError-ish behavior as TimeoutError only after
        # the deadline. Test that if the underlying call raises CancelledError
        # directly, it propagates.
        card = _make_agent_card()

        async def _cancelled_invoke(*_a, **_kw):
            raise asyncio.CancelledError()

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            with patch("strands.agent.a2a_agent.A2AAgent", autospec=True) as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.get_agent_card = AsyncMock(return_value=card)
                mock_agent.invoke_async = _cancelled_invoke
                mock_agent_cls.return_value = mock_agent

                with pytest.raises(asyncio.CancelledError):
                    await a2a_client(
                        url="https://example.com",
                        message="hi",
                        tool_context=_tool_context(),
                    )


# =====================================================================
# Tool metadata
# =====================================================================


class TestToolMetadata:
    def test_default_name(self):
        assert a2a_client.tool_name == "a2a_client"

    def test_custom_name(self):
        assert make_a2a_client(name="remote_agent").tool_name == "remote_agent"

    def test_schema_advertises_url_and_message(self):
        props = a2a_client.tool_spec["inputSchema"]["json"]["properties"]
        assert "url" in props
        assert "message" in props
        # tool_context is injected by the framework, not part of the model-facing schema.
        assert "tool_context" not in props

    def test_schema_omits_dev_only_fields(self):
        # Auth material, allowlists, and caps live on the factory, not on
        # the model-facing schema.
        props = a2a_client.tool_spec["inputSchema"]["json"]["properties"]
        for hidden in ("client_config", "allowed_url_prefixes", "timeout_seconds"):
            assert hidden not in props
