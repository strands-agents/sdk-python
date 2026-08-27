"""Tests for the web_fetch tool.

These focus on the security surface -- URL validation, SSRF address checks,
redirect revalidation, size caps -- and on the HTML-to-markdown extractor.
The transport layer is exercised in-process by monkeypatching ``_fetch_once``
so tests do not hit the network.
"""

from __future__ import annotations

import importlib

import pytest

# The tool named ``web_fetch`` (re-exported by the package ``__init__``) shadows
# the submodule of the same name in the package namespace. Load the submodule
# explicitly so monkeypatch can rebind attributes on it -- and load the ``_ssrf``
# submodule the same way so it can be monkeypatched even though the shadowing
# breaks dotted-string lookups through the package.
web_fetch_module = importlib.import_module("strands.vended_tools.web_fetch.web_fetch")
ssrf_module = importlib.import_module("strands.vended_tools.web_fetch._ssrf")

from strands.vended_tools.web_fetch import make_web_fetch, web_fetch  # noqa: E402
from strands.vended_tools.web_fetch._extract import html_to_markdown  # noqa: E402
from strands.vended_tools.web_fetch._ssrf import (  # noqa: E402
    assert_host_is_allowed,
    resolve_and_validate_host,
    validate_url_scheme,
)
from strands.vended_tools.web_fetch.types import WEB_FETCH_DESCRIPTION  # noqa: E402


class TestUrlSchemeValidation:
    """Only http:// and https:// URLs may reach the network layer."""

    def test_accepts_http(self):
        validate_url_scheme("http://example.com/path")

    def test_accepts_https(self):
        validate_url_scheme("https://example.com/path")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/foo",
            "gopher://example.com/",
            "javascript:alert(1)",
            "data:text/html,<script>1</script>",
            "chrome-extension://foo/bar",
        ],
    )
    def test_rejects_non_http_schemes(self, url):
        with pytest.raises(ValueError, match="Only http"):
            validate_url_scheme(url)

    def test_rejects_missing_host(self):
        with pytest.raises(ValueError, match="no host"):
            validate_url_scheme("http:///no-host-here")


class TestSsrfAddressValidation:
    """resolve_and_validate_host rejects private, loopback, link-local, etc."""

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.0.0.53",
            "0.0.0.0",
            "10.0.0.5",
            "10.255.255.254",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",  # AWS/GCP metadata endpoint
            "169.254.170.2",  # ECS task metadata
            "100.64.0.1",  # CGNAT (100.64.0.0/10)
            "100.127.255.254",  # CGNAT (100.64.0.0/10)
            "::1",
            "fe80::1",  # link-local IPv6
            "fc00::1",  # unique-local IPv6
            "224.0.0.1",  # multicast
            "255.255.255.255",  # broadcast/reserved
            "2001:db8::1",  # documentation range (compressed form)
            "fec0::1",  # site-local (deprecated, filtered anyway)
        ],
    )
    def test_rejects_non_public_addresses(self, host):
        # Some of these (169.254.169.254, 100.100.100.200 etc.) are refused by
        # the pre-DNS named-metadata check before they ever hit the address
        # predicate. Either error is fine as long as it's a hard refusal.
        with pytest.raises(ValueError, match=r"not public|metadata|denylist"):
            resolve_and_validate_host(host)

    @pytest.mark.parametrize(
        "host",
        [
            "::ffff:127.0.0.1",  # dotted-quad form
            "::ffff:169.254.169.254",  # metadata
            "::ffff:10.0.0.1",
            "::ffff:100.64.0.1",  # CGNAT
        ],
    )
    def test_rejects_ipv4_mapped_ipv6_variants(self, host):
        with pytest.raises(ValueError, match="not public"):
            resolve_and_validate_host(host)

    @pytest.mark.parametrize(
        "host",
        [
            "foo.internal",
            "foo.internal.",  # trailing dot must be stripped before match
            "FOO.INTERNAL",  # case-insensitive
            "bar.local",
            "localhost",
            "example.corp",
            "example.home",
            "example.lan",
            "example.intranet",
            "example.private",
            "site.i2p",
            "site.onion",
        ],
    )
    def test_dns_suffix_denylist_refused_before_dns(self, host):
        # These hostnames never even reach a DNS query.
        with pytest.raises(ValueError, match=r"denylist|metadata"):
            assert_host_is_allowed(host)

    @pytest.mark.parametrize(
        "host",
        [
            "metadata",
            "metadata.google.internal",
            "169.254.169.254",
            "fd00:ec2::254",
            "100.100.100.200",
            "192.0.0.192",
        ],
    )
    def test_metadata_endpoints_refused_before_dns(self, host):
        with pytest.raises(ValueError, match=r"metadata|denylist"):
            assert_host_is_allowed(host)

    def test_rejects_ipv4_mapped_ipv6_loopback(self):
        # An attacker who can influence DNS could serve ::ffff:127.0.0.1
        # to bypass a naive IPv4-only private check.
        with pytest.raises(ValueError, match="not public"):
            resolve_and_validate_host("::ffff:127.0.0.1")

    def test_accepts_public_ipv4_literal(self):
        # 8.8.8.8 is a canonical public address; validating a literal does not
        # cause a real DNS query.
        addrs = resolve_and_validate_host("8.8.8.8")
        assert addrs == ["8.8.8.8"]

    def test_rejects_when_any_resolution_is_private(self, monkeypatch):
        """DNS rebinding defense: if any returned A record is non-public, refuse."""

        def fake_getaddrinfo(host, *a, **kw):
            return [
                # 8.8.8.8 is a genuinely public address, so if the validator
                # short-circuited on the first record this test would pass
                # spuriously. Using a real public IP first proves the loop
                # actually walks every record.
                (0, 0, 0, "", ("8.8.8.8", 0)),
                (0, 0, 0, "", ("127.0.0.1", 0)),
            ]

        monkeypatch.setattr(ssrf_module.socket, "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(ValueError, match="not public"):
            resolve_and_validate_host("example.invalid")

    def test_dns_failure_surfaces_as_value_error(self, monkeypatch):
        import socket as _socket

        def boom(*a, **kw):
            raise _socket.gaierror("no such host")

        monkeypatch.setattr(ssrf_module.socket, "getaddrinfo", boom)
        with pytest.raises(ValueError, match="Could not resolve host"):
            resolve_and_validate_host("nonexistent.invalid")

    def test_pinned_https_connect_uses_pinned_ip(self, monkeypatch):
        """The pinned connection must call socket.create_connection with the pinned IP."""
        captured: dict[str, tuple[str, int]] = {}

        class _FakeSock:
            def close(self) -> None:  # pragma: no cover - defensive
                pass

        def fake_create_connection(address, timeout=None):
            captured["address"] = address
            return _FakeSock()

        class _FakeCtx:
            def wrap_socket(self, sock, server_hostname=None):
                captured["sni"] = server_hostname
                return sock

        monkeypatch.setattr(web_fetch_module.socket, "create_connection", fake_create_connection)
        conn = web_fetch_module._PinnedHTTPSConnection(
            "example.com", "203.0.113.5", 443, timeout=5.0, context=_FakeCtx()
        )
        conn.connect()
        assert captured["address"] == ("203.0.113.5", 443)
        assert captured["sni"] == "example.com"


class TestHtmlToMarkdown:
    """Extraction strips noise and preserves structure."""

    def test_strips_script_and_style(self):
        html = """
        <html><head><title>Hi</title>
        <style>body{color:red}</style>
        </head><body>
        <p>Hello world.</p>
        <script>alert('xss')</script>
        <p>After script.</p>
        </body></html>
        """
        md = html_to_markdown(html)
        assert "# Hi" in md
        assert "alert" not in md
        assert "color:red" not in md
        assert "Hello world." in md
        assert "After script." in md

    def test_strips_data_uri_images(self):
        # A giant data-URI image blob would blow up model context; drop it.
        big_blob = "A" * 10000
        html = f'<p>text</p><img src="data:image/png;base64,{big_blob}" alt="alt text">'
        md = html_to_markdown(html)
        assert big_blob not in md
        assert "data:" not in md
        # Alt text is preserved as a substitute so information is not lost.
        assert "alt text" in md

    def test_preserves_regular_images(self):
        html = '<img src="https://example.com/pic.png" alt="pic">'
        md = html_to_markdown(html)
        assert "![pic](https://example.com/pic.png)" in md

    def test_javascript_href_is_dropped(self):
        html = '<a href="javascript:alert(1)">click</a>'
        md = html_to_markdown(html)
        assert "javascript:" not in md
        assert "click" in md

    def test_javascript_img_src_is_dropped(self):
        html = '<img src="javascript:alert(1)" alt="x">'
        md = html_to_markdown(html)
        assert "javascript:" not in md

    def test_preserves_headings_lists_and_links(self):
        html = """
        <h1>Title</h1>
        <p>Intro paragraph with a <a href="https://ex.com/x">link</a>.</p>
        <ul><li>one</li><li>two</li></ul>
        <ol><li>first</li><li>second</li></ol>
        """
        md = html_to_markdown(html)
        assert "# Title" in md
        assert "[link](https://ex.com/x)" in md
        assert "- one" in md
        assert "- two" in md
        assert "1. first" in md
        assert "2. second" in md

    def test_preserves_code_blocks(self):
        html = "<pre><code>def f():\n    return 1</code></pre>"
        md = html_to_markdown(html)
        assert "```" in md
        assert "def f():" in md
        assert "return 1" in md

    def test_ignores_event_handler_attributes(self):
        # onclick/onerror never appear as content: HTMLParser only reports
        # data(), and we render only href/src/alt attributes.
        html = '<div onclick="alert(1)"><p onmouseover="x()">safe</p></div>'
        md = html_to_markdown(html)
        assert "alert" not in md
        assert "onclick" not in md
        assert "safe" in md

    def test_blockquote_prefixes_nested_block_content(self):
        # <blockquote> wrapping <p> is the common shape. Every line inside the
        # blockquote must be prefixed with "> ", including the paragraph that a
        # naive open-tag-only implementation would emit unquoted.
        html = "<blockquote><p>quoted line one.</p><p>quoted line two.</p></blockquote>"
        md = html_to_markdown(html)
        bq_lines = [line for line in md.splitlines() if "quoted line" in line]
        assert len(bq_lines) == 2
        for line in bq_lines:
            assert line.startswith("> "), line

    def test_nested_blockquote_stacks_prefix(self):
        html = "<blockquote><blockquote><p>deep.</p></blockquote></blockquote>"
        md = html_to_markdown(html)
        assert "> > deep." in md

    def test_survives_malformed_html(self):
        html = "<p>ok <b>bold <em>and italic</p>"
        md = html_to_markdown(html)
        assert "ok" in md
        assert "bold" in md

    def test_void_tag_inside_dropped_element_does_not_swallow_following_content(self):
        # <input> is a void tag: it never fires an end tag. A previous bug
        # incremented _drop_depth on every start tag inside a dropped element,
        # so <form><input></form> would leave _drop_depth > 0 forever and
        # discard everything after the </form>.
        html = "<form><input></form><p>after</p>"
        md = html_to_markdown(html)
        assert "after" in md

    def test_void_end_tag_inside_dropped_ancestor_does_not_leak(self):
        # A stray </input> or </embed> inside a dropped ancestor must not
        # decrement _drop_depth (mirror of the start-tag guard). Otherwise the
        # dropped element's remaining body leaks into output.
        html = "<form>secret<input></input>LEAKED</form>tail"
        md = html_to_markdown(html)
        assert "secret" not in md
        assert "LEAKED" not in md
        assert "tail" in md

    def test_javascript_href_with_leading_whitespace_is_dropped(self):
        # Leading whitespace/tab before "javascript:" was a bypass in the
        # previous version -- lstrip before the prefix check.
        html = '<a href="\tjavascript:alert(1)">click</a>'
        md = html_to_markdown(html)
        assert "javascript:" not in md
        assert "alert" not in md
        assert "click" in md


class TestMakeWebFetchValidation:
    """Non-trivial factory behavior."""

    def test_default_description(self):
        assert web_fetch.tool_spec["description"] == WEB_FETCH_DESCRIPTION

    def test_default_name(self):
        assert web_fetch.tool_name == "web_fetch"


class TestWebFetchToolCall:
    """End-to-end tool behavior with the network layer stubbed out."""

    @pytest.mark.asyncio
    async def test_html_response_returns_markdown(self, monkeypatch):
        html_body = b"<html><head><title>T</title></head><body><h1>Hi</h1></body></html>"

        def stub(url, timeout, max_bytes):
            return 200, {"content-type": "text/html; charset=utf-8"}, html_body

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        result = await web_fetch(url="https://example.com/")
        assert result["status"] == 200
        assert result["content_type"].startswith("text/html")
        assert "# T" in result["markdown"]
        assert "# Hi" in result["markdown"]

    @pytest.mark.asyncio
    async def test_non_html_response_returns_body(self, monkeypatch):
        body = b"plain text response"

        def stub(url, timeout, max_bytes):
            return 200, {"content-type": "text/plain"}, body

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        tru_result = await web_fetch(url="https://example.com/robots.txt")
        exp_result = {
            "status": 200,
            "content_type": "text/plain",
            "markdown": "plain text response",
        }
        assert tru_result == exp_result

    @pytest.mark.asyncio
    async def test_redirect_is_followed_and_revalidated(self, monkeypatch):
        # Two hops: 302 -> 200. Each hop must go through _fetch_once (which
        # validates scheme + SSRF), proving revalidation is applied.
        seen = []

        def stub(url, timeout, max_bytes):
            seen.append(url)
            if url == "https://example.com/a":
                return 302, {"location": "https://example.com/b"}, b""
            return 200, {"content-type": "text/html"}, b"<p>ok</p>"

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        result = await web_fetch(url="https://example.com/a")
        assert seen == ["https://example.com/a", "https://example.com/b"]
        assert result["status"] == 200
        assert "ok" in result["markdown"]

    @pytest.mark.asyncio
    async def test_redirect_to_javascript_scheme_is_rejected(self, monkeypatch):
        def stub(url, timeout, max_bytes):
            return 302, {"location": "javascript:alert(1)"}, b""

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        with pytest.raises(ValueError, match="Only http"):
            await web_fetch(url="https://example.com/redir")

    @pytest.mark.asyncio
    async def test_redirect_cap_is_enforced(self, monkeypatch):
        counter = {"n": 0}

        def stub(url, timeout, max_bytes):
            counter["n"] += 1
            return 302, {"location": f"https://example.com/{counter['n']}"}, b""

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        fetch = make_web_fetch(max_redirects=2)
        with pytest.raises(ValueError, match="max_redirects"):
            await fetch(url="https://example.com/start")

    @pytest.mark.asyncio
    async def test_redirect_missing_location_errors(self, monkeypatch):
        def stub(url, timeout, max_bytes):
            return 302, {}, b""

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        with pytest.raises(ValueError, match="without a Location"):
            await web_fetch(url="https://example.com/x")

    @pytest.mark.asyncio
    async def test_private_ip_rejected_end_to_end(self):
        # 127.0.0.1 is an IP literal, so resolve_and_validate_host refuses
        # without any real network call.
        with pytest.raises(ValueError, match="not public"):
            await web_fetch(url="http://127.0.0.1/anything")

    @pytest.mark.asyncio
    async def test_metadata_endpoint_rejected(self):
        with pytest.raises(ValueError, match=r"not public|metadata"):
            await web_fetch(url="http://169.254.169.254/latest/meta-data/")

    @pytest.mark.asyncio
    async def test_non_http_scheme_rejected_end_to_end(self):
        with pytest.raises(ValueError, match="Only http"):
            await web_fetch(url="file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_zero_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout"):
            await web_fetch(url="https://example.com/", timeout=0)

    @pytest.mark.asyncio
    async def test_total_timeout_wraps_slow_transport(self, monkeypatch):
        # Simulate a transport that would run past the requested timeout: the
        # outer asyncio.wait_for must trip regardless.
        import time

        def slow(url, timeout, max_bytes):
            time.sleep(2.0)
            return 200, {"content-type": "text/html"}, b"<p>slow</p>"

        monkeypatch.setattr(web_fetch_module, "_fetch_once", slow)
        with pytest.raises(TimeoutError, match="total timeout"):
            await web_fetch(url="https://example.com/", timeout=1)


class TestSizeCap:
    """Oversized responses are rejected without buffering the excess."""

    def test_fetch_once_rejects_body_over_cap(self, monkeypatch):
        class _FakeResponse:
            status = 200

            def __init__(self, body: bytes) -> None:
                self._body = body

            def getheaders(self):
                return [("Content-Type", "text/html")]

            def read(self, n):
                return self._body[:n]

        class _FakeConn:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def request(self, *a, **kw):
                pass

            def getresponse(self):
                return _FakeResponse(self._body)

            def close(self):
                pass

        big = b"x" * 100

        def fake_pinned_https(host, pinned_ip, port, timeout, context):
            return _FakeConn(big)

        # Bypass SSRF for this narrow test -- we are exercising the size cap
        # only. Real end-to-end SSRF is covered elsewhere.
        monkeypatch.setattr(web_fetch_module, "_PinnedHTTPSConnection", fake_pinned_https)
        monkeypatch.setattr(web_fetch_module, "resolve_and_validate_host", lambda h: ["203.0.113.1"])

        with pytest.raises(ValueError, match="max_bytes"):
            web_fetch_module._fetch_once("https://example.com/", timeout=5.0, max_bytes=50)


class TestMultiAddressFallback:
    """If the first validated address fails to connect, the next is tried."""

    def test_second_address_used_when_first_refuses_connection(self, monkeypatch):
        # Simulate the common case of "AAAA first, no IPv6 route" -- the first
        # address raises a transport error, the second succeeds. Every address
        # in the list has already passed the SSRF check.
        tried: list[str] = []

        class _FakeResponse:
            status = 200

            def getheaders(self):
                return [("Content-Type", "text/plain")]

            def read(self, n):
                return b"ok"

        class _FakeConn:
            def __init__(self, host, pinned_ip, port, timeout, *a, **kw):
                self.pinned_ip = pinned_ip

            def request(self, *a, **kw):
                tried.append(self.pinned_ip)
                if self.pinned_ip == "2606:4700:4700::1111":
                    raise OSError("no route to host")

            def getresponse(self):
                return _FakeResponse()

            def close(self):
                pass

        monkeypatch.setattr(web_fetch_module, "_PinnedHTTPSConnection", _FakeConn)
        monkeypatch.setattr(
            web_fetch_module,
            "resolve_and_validate_host",
            lambda h: ["2606:4700:4700::1111", "1.1.1.1"],
        )

        status, _, body = web_fetch_module._fetch_once("https://example.com/", timeout=5.0, max_bytes=1024)
        assert status == 200
        assert body == b"ok"
        assert tried == ["2606:4700:4700::1111", "1.1.1.1"]

    def test_all_addresses_fail_surfaces_connection_error(self, monkeypatch):
        class _FakeConn:
            def __init__(self, host, pinned_ip, port, timeout, *a, **kw):
                pass

            def request(self, *a, **kw):
                raise OSError("no route to host")

            def close(self):
                pass

        monkeypatch.setattr(web_fetch_module, "_PinnedHTTPSConnection", _FakeConn)
        monkeypatch.setattr(
            web_fetch_module,
            "resolve_and_validate_host",
            lambda h: ["2606:4700:4700::1111", "1.1.1.1"],
        )

        with pytest.raises(ConnectionError, match="Could not connect to any validated address"):
            web_fetch_module._fetch_once("https://example.com/", timeout=5.0, max_bytes=1024)


class TestRequestHeaders:
    """Outgoing request headers must not leak URL userinfo or port state."""

    def _capture_request(self, monkeypatch, url: str) -> dict:
        captured: dict = {}

        class _FakeResponse:
            status = 200

            def getheaders(self):
                return [("Content-Type", "text/plain")]

            def read(self, n):
                return b""

        class _FakeConn:
            def request(self, method, path, headers):
                captured["headers"] = headers
                captured["path"] = path

            def getresponse(self):
                return _FakeResponse()

            def close(self):
                pass

        def fake_pinned_https(host, pinned_ip, port, timeout, context):
            return _FakeConn()

        def fake_pinned_http(host, pinned_ip, port, timeout):
            return _FakeConn()

        monkeypatch.setattr(web_fetch_module, "_PinnedHTTPSConnection", fake_pinned_https)
        monkeypatch.setattr(web_fetch_module, "_PinnedHTTPConnection", fake_pinned_http)
        monkeypatch.setattr(web_fetch_module, "resolve_and_validate_host", lambda h: ["203.0.113.1"])
        web_fetch_module._fetch_once(url, timeout=5.0, max_bytes=1024)
        return captured

    def test_host_header_omits_userinfo(self, monkeypatch):
        # A URL with basic-auth credentials must not leak them into the Host header.
        captured = self._capture_request(monkeypatch, "https://alice:pw@example.com/path")
        assert captured["headers"]["Host"] == "example.com"

    def test_host_header_includes_port(self, monkeypatch):
        captured = self._capture_request(monkeypatch, "https://example.com:8443/path")
        assert captured["headers"]["Host"] == "example.com:8443"

    def test_host_header_hides_userinfo_with_port(self, monkeypatch):
        captured = self._capture_request(monkeypatch, "https://alice:pw@example.com:8443/path")
        assert captured["headers"]["Host"] == "example.com:8443"

    def test_host_header_brackets_ipv6_with_port(self, monkeypatch):
        # RFC 3986 requires IPv6 literals in the Host header to be bracketed.
        # We stub SSRF because 2606:4700:4700::1111 is genuinely public and
        # we don't want to depend on DNS in a unit test.
        captured = self._capture_request(monkeypatch, "https://[2606:4700:4700::1111]:8443/path")
        assert captured["headers"]["Host"] == "[2606:4700:4700::1111]:8443"

    def test_host_header_brackets_ipv6_without_port(self, monkeypatch):
        captured = self._capture_request(monkeypatch, "https://[2606:4700:4700::1111]/path")
        assert captured["headers"]["Host"] == "[2606:4700:4700::1111]"


class TestRedirectBodyShortCircuit:
    """A 3xx response must not buffer the full body -- only the drain cap."""

    def test_redirect_body_is_not_buffered(self, monkeypatch):
        reads: list[int] = []

        class _FakeResponse:
            status = 302

            def getheaders(self):
                return [("Location", "https://example.com/next")]

            def read(self, n):
                reads.append(n)
                return b""

        class _FakeConn:
            def request(self, *a, **kw):
                pass

            def getresponse(self):
                return _FakeResponse()

            def close(self):
                pass

        def fake_pinned_https(host, pinned_ip, port, timeout, context):
            return _FakeConn()

        monkeypatch.setattr(web_fetch_module, "_PinnedHTTPSConnection", fake_pinned_https)
        monkeypatch.setattr(web_fetch_module, "resolve_and_validate_host", lambda h: ["203.0.113.1"])

        status, headers, body = web_fetch_module._fetch_once(
            "https://example.com/redir", timeout=5.0, max_bytes=10_000_000
        )
        assert status == 302
        assert headers["location"] == "https://example.com/next"
        assert body == b""
        # We should have asked for the small drain cap, not max_bytes + 1.
        assert reads == [web_fetch_module._REDIRECT_BODY_DRAIN_CAP]


class TestCharsetDecoding:
    """Content-Type charset -- quoted or unquoted -- is honored."""

    @pytest.mark.parametrize(
        "header,body,expected",
        [
            ("text/html; charset=utf-8", b"hello", "hello"),
            ('text/html; charset="iso-8859-1"', b"caf\xe9", "café"),
            ("text/html; charset='windows-1252'", b"\xa9 2026", "© 2026"),
        ],
    )
    def test_decode_body_honors_charset(self, header, body, expected):
        assert web_fetch_module._decode_body(body, header) == expected

    def test_decode_body_defaults_to_utf8_on_missing_charset(self):
        assert web_fetch_module._decode_body(b"plain", "text/plain") == "plain"

    def test_decode_body_falls_back_on_unknown_charset(self):
        # Some fictitious charset the runtime doesn't know -- fall back to utf-8
        # rather than raising.
        assert web_fetch_module._decode_body(b"plain", "text/plain; charset=xyz-fake-99") == "plain"
