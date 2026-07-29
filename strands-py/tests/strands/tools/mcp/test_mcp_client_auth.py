"""Unit tests for MCPClient OAuth authentication (url/auth/auth_provider constructor path)."""

from unittest.mock import patch

import httpx
import pytest
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider

from strands.tools.mcp import MCPClient, MCPClientCredentials
from strands.tools.mcp.mcp_client import _InMemoryTokenStorage


@pytest.fixture
def streamablehttp_transport():
    """Patch streamablehttp_client as imported into mcp_client."""
    with patch("strands.tools.mcp.mcp_client.streamablehttp_client") as http:
        yield http


# Transport resolution


def test_url_builds_streamable_http_transport(streamablehttp_transport):
    client = MCPClient(url="https://mcp.example.com")

    client._transport_callable()

    streamablehttp_transport.assert_called_once_with(url="https://mcp.example.com", headers=None, auth=None)


def test_transport_callable_passthrough(streamablehttp_transport):
    def transport_callable():
        return None

    client = MCPClient(transport_callable)

    assert client._transport_callable is transport_callable
    streamablehttp_transport.assert_not_called()


def test_headers_passed_to_transport(streamablehttp_transport):
    client = MCPClient(url="https://mcp.example.com", headers={"X-Api-Key": "abc"})

    client._transport_callable()

    streamablehttp_transport.assert_called_once_with(
        url="https://mcp.example.com", headers={"X-Api-Key": "abc"}, auth=None
    )


# Auth construction


def test_auth_builds_client_credentials_provider(streamablehttp_transport):
    client = MCPClient(
        url="https://mcp.example.com",
        auth=MCPClientCredentials(client_id="id", client_secret="secret"),
    )

    client._transport_callable()

    provider = streamablehttp_transport.call_args.kwargs["auth"]
    assert isinstance(provider, ClientCredentialsOAuthProvider)
    assert provider.context.server_url == "https://mcp.example.com"
    assert provider._fixed_client_info.client_id == "id"
    assert provider._fixed_client_info.client_secret == "secret"
    assert provider._fixed_client_info.scope is None


def test_auth_scopes_joined_with_spaces(streamablehttp_transport):
    client = MCPClient(
        url="https://mcp.example.com",
        auth=MCPClientCredentials(client_id="id", client_secret="secret", scopes=["read", "write"]),
    )

    client._transport_callable()

    provider = streamablehttp_transport.call_args.kwargs["auth"]
    assert provider._fixed_client_info.scope == "read write"


def test_auth_provider_passed_through(streamablehttp_transport):
    custom_provider = httpx.BasicAuth("user", "pass")
    client = MCPClient(url="https://mcp.example.com", auth_provider=custom_provider)

    client._transport_callable()

    assert streamablehttp_transport.call_args.kwargs["auth"] is custom_provider


def test_auth_and_headers_passed_together(streamablehttp_transport):
    client = MCPClient(
        url="https://mcp.example.com",
        auth=MCPClientCredentials(client_id="id", client_secret="secret"),
        headers={"X-Trace": "123"},
    )

    client._transport_callable()

    kwargs = streamablehttp_transport.call_args.kwargs
    assert kwargs["headers"] == {"X-Trace": "123"}
    assert isinstance(kwargs["auth"], ClientCredentialsOAuthProvider)


# Constructor invariants


def test_both_transport_callable_and_url_raises():
    with pytest.raises(ValueError, match="not both"):
        MCPClient(lambda: None, url="https://mcp.example.com")


def test_neither_transport_callable_nor_url_raises():
    with pytest.raises(ValueError, match="must be provided"):
        MCPClient()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auth": MCPClientCredentials(client_id="x", client_secret="y")},
        {"auth_provider": httpx.BasicAuth("user", "pass")},
        {"headers": {"X-Foo": "bar"}},
    ],
)
def test_auth_or_headers_with_transport_callable_raises(kwargs):
    with pytest.raises(ValueError, match="require 'url'"):
        MCPClient(lambda: None, **kwargs)


def test_both_auth_and_auth_provider_raises():
    with pytest.raises(ValueError, match="either 'auth' or 'auth_provider', not both"):
        MCPClient(
            url="https://mcp.example.com",
            auth=MCPClientCredentials(client_id="x", client_secret="y"),
            auth_provider=httpx.BasicAuth("user", "pass"),
        )


# In-memory token storage


@pytest.mark.asyncio
async def test_in_memory_token_storage_round_trip():
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = _InMemoryTokenStorage()
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None

    tokens = OAuthToken(access_token="token", token_type="Bearer")
    client_info = OAuthClientInformationFull(client_id="id", redirect_uris=None)
    await storage.set_tokens(tokens)
    await storage.set_client_info(client_info)

    assert await storage.get_tokens() is tokens
    assert await storage.get_client_info() is client_info
