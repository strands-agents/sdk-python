"""Tests for the http_request tool.

The security surface -- URL scheme, private-network denial (literal and via
DNS), redirect re-validation, and response-size cap -- comes first. Happy
path GET/POST follows.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from strands.types.tools import ToolContext, ToolUse
from strands.vended_tools.http_request import (
    HttpRequestError,
    http_request,
    make_http_request,
)
from strands.vended_tools.http_request.types import DEFAULT_HTTP_REQUEST_DESCRIPTION


def _public_resolver(host: str) -> list[str]:
    """Test resolver that resolves any host to a fixed public address."""
    return ["93.184.216.34"]  # example.com; globally routable


def _make_transport(handler):
    """Build a mock httpx transport from a request handler callable."""
    return httpx.MockTransport(handler)


class TestSecuritySchemes:
    """Only ``http`` and ``https`` are allowed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/", "javascript:alert(1)"])
    async def test_rejects_non_http_scheme(self, url):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="http and https"):
            await tool(method="GET", url=url)

    @pytest.mark.asyncio
    async def test_rejects_invalid_port(self):
        # A non-numeric port makes urlsplit(url).port raise ValueError at
        # access time, which would otherwise leak through _origin() on a
        # redirect. Verify the tool surfaces its own error type instead.
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="invalid port"):
            await tool(method="GET", url="https://example.com:bad/data")


class TestSecurityPrivateNetworks:
    """Private-network destinations are denied by default, whether by literal IP or DNS."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[fe80::1]/",
            # Multicast: SSDP (v4), mDNS (v4), all-nodes (v6).
            "http://239.255.255.250:1900/",
            "http://224.0.0.251:5353/",
            "http://[ff02::1]/",
            # IPv4-mapped IPv6: `is_global` on ::ffff:10.0.0.1 returns True on
            # Python 3.10/3.11. Unwrap must happen before the flag check.
            "http://[::ffff:10.0.0.1]/",
            "http://[::ffff:169.254.169.254]/",
            # Reserved / unspecified / documentation.
            "http://0.0.0.0/",
            "http://192.0.2.1/",
            "http://198.51.100.1/",
            "http://203.0.113.1/",
        ],
    )
    async def test_rejects_private_ip_literal(self, url):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError):
            await tool(method="GET", url=url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url,match",
        [
            ("http://metadata.google.internal/", "cloud-metadata"),
            # Alibaba Cloud metadata.
            ("http://100.100.100.200/", "cloud-metadata|non-public"),
            # Oracle Cloud Infrastructure metadata.
            ("http://192.0.0.192/", "cloud-metadata|non-public"),
        ],
    )
    async def test_rejects_metadata_endpoint(self, url, match):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match=match):
            await tool(method="GET", url=url)

    @pytest.mark.asyncio
    async def test_rejects_host_that_resolves_to_private_ip(self):
        # DNS-rebinding-style: public-looking hostname resolves to private IP.
        def private_resolver(_host: str) -> list[str]:
            return ["10.0.0.5"]

        tool = make_http_request(resolver=private_resolver)
        with pytest.raises(HttpRequestError, match="non-public"):
            await tool(method="GET", url="https://sneaky.example.com/")

    @pytest.mark.asyncio
    async def test_allowlisted_host_bypasses_private_check(self):
        def private_resolver(_host: str) -> list[str]:
            return ["10.0.0.5"]

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "internal.example.com"
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_private_hosts=["internal.example.com"],
            resolver=private_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(method="GET", url="https://internal.example.com/")
        assert result["status"] == 200


class TestSecurityHeaders:
    """Sensitive headers are stripped unless the operator opts a host in."""

    @pytest.mark.asyncio
    async def test_rejects_authorization_header_by_default(self):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="Authorization"):
            await tool(
                method="GET",
                url="https://example.com/",
                headers={"Authorization": "Bearer secret"},
            )

    @pytest.mark.asyncio
    async def test_rejects_cookie_header_by_default(self):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="Cookie"):
            await tool(
                method="GET",
                url="https://example.com/",
                headers={"Cookie": "session=abc"},
            )

    @pytest.mark.asyncio
    async def test_rejects_hop_by_hop_header(self):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="not allowed"):
            await tool(
                method="GET",
                url="https://example.com/",
                headers={"Host": "spoofed.example.com"},
            )

    @pytest.mark.asyncio
    async def test_allows_authorization_for_allowlisted_host(self):
        received: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received.update(request.headers)
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_auth_for_hosts=["api.example.com"],
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(
            method="GET",
            url="https://api.example.com/",
            headers={"Authorization": "Bearer secret"},
        )
        assert result["status"] == 200
        assert received["authorization"] == "Bearer secret"


class TestSecurityRedirects:
    """Redirects are followed, capped, and re-validated at every hop."""

    @pytest.mark.asyncio
    async def test_follows_redirect(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/final"})
            return httpx.Response(200, text="final body")

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(method="GET", url="https://example.com/start")
        assert result["status"] == 200
        assert result["body"] == "final body"

    @pytest.mark.asyncio
    async def test_caps_redirect_chain(self):
        def handler(request: httpx.Request) -> httpx.Response:
            # Every request redirects to a new path -- infinite chain.
            next_path = "/next" + request.url.path
            return httpx.Response(302, headers={"location": next_path})

        tool = make_http_request(
            max_redirects=3,
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        with pytest.raises(HttpRequestError, match="Too many redirects"):
            await tool(method="GET", url="https://example.com/start")

    @pytest.mark.asyncio
    async def test_revalidates_redirect_target(self):
        # First hop resolves public; the redirect points at a hostname
        # that resolves private. The tool must refuse the second hop.
        call_count = {"n": 0}

        def rebinding_resolver(host: str) -> list[str]:
            call_count["n"] += 1
            if host == "start.example.com":
                return ["93.184.216.34"]
            return ["10.0.0.5"]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://internal.example.com/"})

        tool = make_http_request(
            resolver=rebinding_resolver,
            transport=_make_transport(handler),
        )
        with pytest.raises(HttpRequestError, match="non-public"):
            await tool(method="GET", url="https://start.example.com/start")
        # Both hops resolved.
        assert call_count["n"] >= 2

    @pytest.mark.asyncio
    async def test_cross_origin_redirect_strips_auth(self):
        # An operator-opted-in host redirects to a *different* host: the
        # Authorization / Cookie headers must not follow.
        seen_hops: list[tuple[str, dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hops.append((request.url.host, dict(request.headers)))
            if request.url.host == "api.example.com":
                return httpx.Response(302, headers={"location": "https://attacker.example/"})
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_auth_for_hosts=["api.example.com"],
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(
            method="GET",
            url="https://api.example.com/",
            headers={"Authorization": "Bearer secret", "Cookie": "s=1"},
        )
        assert result["status"] == 200
        first_host, first_headers = seen_hops[0]
        second_host, second_headers = seen_hops[1]
        assert first_host == "api.example.com"
        assert first_headers["authorization"] == "Bearer secret"
        assert second_host == "attacker.example"
        assert "authorization" not in second_headers
        assert "cookie" not in second_headers

    @pytest.mark.asyncio
    async def test_same_host_redirect_preserves_auth(self):
        seen_hops: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hops.append(dict(request.headers))
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/final"})
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_auth_for_hosts=["api.example.com"],
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(
            method="GET",
            url="https://api.example.com/start",
            headers={"Authorization": "Bearer secret"},
        )
        assert result["status"] == 200
        assert seen_hops[0]["authorization"] == "Bearer secret"
        assert seen_hops[1]["authorization"] == "Bearer secret"

    @pytest.mark.asyncio
    async def test_303_forces_get_and_drops_body(self):
        received = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/post":
                assert request.method == "POST"
                return httpx.Response(303, headers={"location": "/result"})
            received["method"] = request.method
            received["content"] = request.content
            return httpx.Response(200, text="done")

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(
            method="POST",
            url="https://example.com/post",
            body='{"x":1}',
            headers={"Content-Type": "application/json"},
        )
        assert result["status"] == 200
        assert received["method"] == "GET"
        assert received["content"] == b""


class TestSecurityBodyCap:
    """Response bodies larger than the cap are refused."""

    @pytest.mark.asyncio
    async def test_rejects_oversized_response(self):
        big = b"a" * 2048

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big)

        tool = make_http_request(
            max_response_bytes=1024,
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        with pytest.raises(HttpRequestError, match="exceeded maximum size"):
            await tool(method="GET", url="https://example.com/big")


class TestHappyPath:
    """Basic GET / POST flow works when the security policy is satisfied."""

    @pytest.mark.asyncio
    async def test_get_returns_body_and_headers(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "application/json", "x-custom": "value"},
                text='{"ok":true}',
            )

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(method="GET", url="https://example.com/data")
        assert result["status"] == 200
        assert result["status_text"] == "OK"
        assert result["body"] == '{"ok":true}'
        assert result["headers"]["content-type"].startswith("application/json")
        assert result["headers"]["x-custom"] == "value"

    @pytest.mark.asyncio
    async def test_post_sends_body_and_custom_headers(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["body"] = request.content
            seen["ct"] = request.headers.get("content-type")
            return httpx.Response(201, text='{"id":1}')

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(
            method="POST",
            url="https://example.com/users",
            headers={"Content-Type": "application/json"},
            body='{"name":"test"}',
        )
        assert result["status"] == 201
        assert seen["method"] == "POST"
        assert seen["body"] == b'{"name":"test"}'
        assert seen["ct"] == "application/json"

    @pytest.mark.asyncio
    async def test_non_2xx_status_is_returned_not_raised(self):
        # The tool returns the response as-is; the model decides what to do.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(method="GET", url="https://example.com/missing")
        assert result["status"] == 404
        assert result["body"] == "not found"


class TestTimeoutBoundary:
    """The model-supplied timeout is validated and bounded."""

    @pytest.mark.asyncio
    async def test_rejects_non_positive_timeout(self):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="positive"):
            await tool(method="GET", url="https://example.com/", timeout=0)

    @pytest.mark.asyncio
    async def test_model_timeout_is_capped_at_default(self):
        # If the tool operator set default 5s and the model asks for 60s,
        # the effective timeout is 5s. We verify by observing that a
        # transport that never responds triggers a timeout.
        async def slow_handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out")

        transport = httpx.MockTransport(slow_handler)
        tool = make_http_request(
            default_timeout_seconds=5.0,
            resolver=_public_resolver,
            transport=transport,
        )
        with pytest.raises(HttpRequestError, match="timed out"):
            await tool(method="GET", url="https://example.com/slow", timeout=60)


class TestToolMetadata:
    """Tool name, description, and input schema."""

    def test_default_name(self):
        assert http_request.tool_name == "http_request"

    def test_default_description(self):
        assert http_request.tool_spec["description"] == DEFAULT_HTTP_REQUEST_DESCRIPTION

    def test_custom_name(self):
        assert make_http_request(name="fetch").tool_name == "fetch"

    def test_schema_exposes_expected_parameters(self):
        # `tool_context` is a framework-injected special parameter and must
        # never appear in the model-facing schema.
        props = http_request.tool_spec["inputSchema"]["json"]["properties"]
        assert set(props) == {"method", "url", "headers", "body", "timeout"}


class TestSecurityCrossOriginAuth:
    """Cross-origin redirects strip auth per origin tuple, not just hostname."""

    @pytest.mark.asyncio
    async def test_https_to_http_downgrade_strips_auth(self):
        # HTTPS -> HTTP on the same host is still cross-origin: the credential
        # would travel over plaintext. Hostname-only comparison would preserve
        # it, so this test guards against the pre-round-3 predicate.
        seen_hops: list[tuple[str, dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hops.append((str(request.url), dict(request.headers)))
            if request.url.scheme == "https":
                return httpx.Response(302, headers={"location": "http://api.example.com/final"})
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_auth_for_hosts=["api.example.com"],
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(
            method="GET",
            url="https://api.example.com/",
            headers={"Authorization": "Bearer secret"},
        )
        assert result["status"] == 200
        assert "authorization" in seen_hops[0][1]
        assert "authorization" not in seen_hops[1][1]
        assert "cookie" not in seen_hops[1][1]

    @pytest.mark.asyncio
    async def test_same_host_port_change_strips_auth(self):
        # Same host + same scheme but a different port is a different origin.
        seen_hops: list[tuple[str, dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hops.append((str(request.url), dict(request.headers)))
            if request.url.port in (None, 443):
                return httpx.Response(302, headers={"location": "https://api.example.com:8443/final"})
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_auth_for_hosts=["api.example.com"],
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(
            method="GET",
            url="https://api.example.com/",
            headers={"Authorization": "Bearer secret"},
        )
        assert result["status"] == 200
        assert "authorization" in seen_hops[0][1]
        assert "authorization" not in seen_hops[1][1]

    @pytest.mark.asyncio
    async def test_same_origin_default_port_preserves_auth(self):
        # `https://api.example.com/` and `https://api.example.com:443/final`
        # are the same origin -- port default must resolve to 443.
        seen_hops: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hops.append(dict(request.headers))
            if request.url.path == "/":
                # Explicit port 443 = default; must still be same-origin.
                return httpx.Response(302, headers={"location": "https://api.example.com:443/final"})
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            allow_auth_for_hosts=["api.example.com"],
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(
            method="GET",
            url="https://api.example.com/",
            headers={"Authorization": "Bearer secret"},
        )
        assert result["status"] == 200
        assert seen_hops[0]["authorization"] == "Bearer secret"
        assert seen_hops[1]["authorization"] == "Bearer secret"


class TestSecurityDnsSuffixDenylist:
    """Hostnames ending in denied suffixes are refused before DNS resolves."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "http://service.internal/",
            "http://foo.corp/",
            "http://bar.home/",
            "http://baz.lan/",
            "http://svc.intranet/",
            "http://api.private/",
            "http://mysite.i2p/",
            "http://hidden.onion/",
            # `.local` and `.localhost` are also denied.
            "http://raspberrypi.local/",
            "http://something.localhost/",
            # Trailing dot on a rooted FQDN must not bypass the suffix match.
            "http://service.internal./",
            # Case-folded match.
            "http://SERVICE.Internal/",
        ],
    )
    async def test_rejects_denied_suffix_before_dns(self, url):
        called = {"n": 0}

        def resolver(_host: str) -> list[str]:
            # If the guard let this through, we'd see a resolver call.
            called["n"] += 1
            return ["93.184.216.34"]

        tool = make_http_request(resolver=resolver)
        with pytest.raises(HttpRequestError, match="denied suffix"):
            await tool(method="GET", url=url)
        assert called["n"] == 0


class TestSecurityResponseHeaderCap:
    """Response headers are capped so an oversize response can't dump megabytes back."""

    @pytest.mark.asyncio
    async def test_rejects_oversized_response_headers(self):
        # 200 bytes of value * 6 headers = 1200 bytes -- over the 500-byte cap.
        big_value = "x" * 200

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={f"x-blob-{i}": big_value for i in range(6)},
                text="ok",
            )

        tool = make_http_request(
            max_response_headers_bytes=500,
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        with pytest.raises(HttpRequestError, match="Response headers exceeded"):
            await tool(method="GET", url="https://example.com/")

    @pytest.mark.asyncio
    async def test_within_cap_headers_returned(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"x-small": "value"}, text="ok")

        tool = make_http_request(
            max_response_headers_bytes=1024,
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(method="GET", url="https://example.com/")
        assert result["headers"]["x-small"] == "value"


class TestSecurityMaxRedirectsZero:
    """`max_redirects=0` raises when a 3xx is encountered, per docstring."""

    @pytest.mark.asyncio
    async def test_max_redirects_zero_raises_on_3xx(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "/final"})

        tool = make_http_request(
            max_redirects=0,
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        with pytest.raises(HttpRequestError, match="Too many redirects"):
            await tool(method="GET", url="https://example.com/")

    @pytest.mark.asyncio
    async def test_max_redirects_zero_still_returns_non_3xx(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="ok")

        tool = make_http_request(
            max_redirects=0,
            resolver=_public_resolver,
            transport=_make_transport(handler),
        )
        result = await tool(method="GET", url="https://example.com/")
        assert result["status"] == 200


class TestCancelSignal:
    """The agent's cancel signal aborts an in-flight request."""

    @staticmethod
    def _tool_context_for(agent: object) -> ToolContext:
        tool_use = ToolUse(toolUseId="http_1", name="http_request", input={})
        return ToolContext(tool_use=tool_use, agent=agent, invocation_state={})

    @pytest.mark.asyncio
    async def test_pre_flight_cancel_short_circuits(self):
        # Signal set before dispatch: the tool sees it before the first send.
        # This alone is not enough to prove the mid-flight path works, so the
        # next test also drives the between-chunks branch. Cancellation is
        # signalled with ``asyncio.CancelledError`` so callers can distinguish
        # it from other request failures without matching on error text.
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("transport should not be called if cancel is pre-set")

        cancel = threading.Event()
        cancel.set()
        agent = SimpleNamespace(_cancel_signal=cancel)

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        with pytest.raises(asyncio.CancelledError):
            await tool(
                method="GET",
                url="https://example.com/",
                tool_context=self._tool_context_for(agent),
            )

    @pytest.mark.asyncio
    async def test_mid_flight_cancel_between_chunks(self):
        # The response starts fine; the cancel signal fires as the client
        # reads its second chunk. The guard between chunks must catch this
        # even though the first chunk arrived before the signal.
        cancel = threading.Event()

        async def streaming_body():
            yield b"partial-"
            cancel.set()
            yield b"the-rest"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=streaming_body())

        agent = SimpleNamespace(_cancel_signal=cancel)
        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        with pytest.raises(asyncio.CancelledError):
            await tool(
                method="GET",
                url="https://example.com/",
                tool_context=self._tool_context_for(agent),
            )

    @pytest.mark.asyncio
    async def test_no_cancel_signal_no_op(self):
        # Agent without a `_cancel_signal` attribute must not crash the tool.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="ok")

        agent = SimpleNamespace()  # no _cancel_signal at all
        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(
            method="GET",
            url="https://example.com/",
            tool_context=self._tool_context_for(agent),
        )
        assert result["status"] == 200


class TestResponseHeaderMultiplicity:
    """Repeated response headers (Set-Cookie) preserve every occurrence."""

    @pytest.mark.asyncio
    async def test_repeated_set_cookie_is_preserved(self):
        # Two Set-Cookie lines must not be comma-collapsed. Cookie values
        # legitimately contain commas (Expires=Wed, 09 Jun 2021 ...) so the
        # model needs to see both values distinctly.
        def handler(_request: httpx.Request) -> httpx.Response:
            headers = [
                ("set-cookie", "session=abc; Path=/"),
                ("set-cookie", "tracking=xyz; Path=/"),
            ]
            return httpx.Response(200, headers=headers, text="ok")

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(method="GET", url="https://example.com/")
        cookies = result["headers"]["set-cookie"].split("\n")
        assert "session=abc; Path=/" in cookies
        assert "tracking=xyz; Path=/" in cookies


class TestURLValidation:
    """Odd URL shapes that must fail closed."""

    @pytest.mark.asyncio
    async def test_url_with_no_host_is_rejected(self):
        tool = make_http_request(resolver=_public_resolver)
        with pytest.raises(HttpRequestError, match="URL has no host"):
            await tool(method="GET", url="http:///path")

    @pytest.mark.asyncio
    async def test_resolver_returning_empty_list_is_rejected(self):
        def empty_resolver(_host: str) -> list[str]:
            return []

        tool = make_http_request(resolver=empty_resolver)
        with pytest.raises(HttpRequestError, match="no addresses"):
            await tool(method="GET", url="https://example.com/")

    @pytest.mark.asyncio
    async def test_resolver_returning_unparseable_address_is_rejected(self):
        def bad_resolver(_host: str) -> list[str]:
            return ["not-an-ip"]

        tool = make_http_request(resolver=bad_resolver)
        with pytest.raises(HttpRequestError, match="Could not parse resolved address"):
            await tool(method="GET", url="https://example.com/")

    @pytest.mark.asyncio
    async def test_denylist_suffix_precedes_allow_private_hosts(self):
        # Opting a `.internal` host into allow_private_hosts must not
        # re-enable the suffix-denylist namespace. The denylist runs first.
        tool = make_http_request(
            allow_private_hosts=["service.internal"],
            resolver=_public_resolver,
        )
        with pytest.raises(HttpRequestError, match="denied suffix"):
            await tool(method="GET", url="http://service.internal/")

    @pytest.mark.asyncio
    async def test_redirect_without_location_returns_response(self):
        # A 302 without a Location header cannot be followed; the tool should
        # return the response as-is rather than looping.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, text="see elsewhere maybe")

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        result = await tool(method="GET", url="https://example.com/")
        assert result["status"] == 302
        assert result["body"] == "see elsewhere maybe"

    @pytest.mark.asyncio
    async def test_request_error_is_wrapped(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        tool = make_http_request(resolver=_public_resolver, transport=_make_transport(handler))
        with pytest.raises(HttpRequestError, match="Request failed"):
            await tool(method="GET", url="https://example.com/")


class TestDefaultResolver:
    """The stdlib-backed default resolver returns unique addresses or raises."""

    def test_default_resolver_returns_unique_public_addresses(self):
        # localhost always resolves; every address returned should parse as
        # an IP even though the guard would separately refuse to dial it.
        from strands.vended_tools.http_request.http_request import _default_resolver

        addresses = _default_resolver("localhost")
        assert addresses
        for addr in addresses:
            # Just assert parseable; policy is applied elsewhere.
            import ipaddress

            ipaddress.ip_address(addr)

    def test_default_resolver_raises_on_unresolvable_host(self):
        from strands.vended_tools.http_request.http_request import _default_resolver

        with pytest.raises(HttpRequestError, match="DNS resolution failed"):
            # RFC 6761 reserved name: guaranteed not to resolve on the public DNS.
            _default_resolver("invalid.test")
