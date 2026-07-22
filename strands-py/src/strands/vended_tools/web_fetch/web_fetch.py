"""Web fetch tool: fetch a URL and return clean markdown for a model to read.

Distinct from the http_request tool, which returns raw response bodies for API
calls. This tool is intentionally narrow: HTTP(S) GET, decoded body, HTML to
markdown. It is not a general-purpose scraper.

Only http and https URLs are accepted; every host is resolved and every
returned address is required to be publicly routable, and IPv4-mapped IPv6
addresses are unwrapped before the check. Redirects are re-validated against
the same rules. The tool connects to an already-validated IP address so a DNS
rebinder cannot substitute a private address between validation and connect.
Response size is capped and scripts, styles, and data URI images are stripped
from the extracted markdown. See :mod:`._ssrf` for the address-level defense.
"""

from __future__ import annotations

import asyncio
import http.client
import socket
import ssl
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from ...tools.decorator import tool
from ._extract import html_to_markdown
from ._ssrf import resolve_and_validate_host, validate_url_scheme
from .types import WEB_FETCH_DESCRIPTION, WebFetchOutput

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
_DEFAULT_MAX_REDIRECTS = 5
_USER_AGENT = "strands-agents-web-fetch/1.0"


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that connects to a caller-chosen IP but sends the original Host header.

    We validate the DNS answer up front and then pin the connect target so a
    later rebind cannot substitute a private IP. ``host`` is preserved for the
    Host header and (for HTTPS) SNI/certificate validation.
    """

    def __init__(self, host: str, pinned_ip: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS variant of :class:`_PinnedHTTPConnection`.

    Preserves ``host`` for SNI and certificate hostname verification while
    connecting to a pinned IP.
    """

    def __init__(
        self,
        host: str,
        pinned_ip: str,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        # Do NOT pass ``context`` up to ``HTTPSConnection.__init__``. On CPython
        # 3.11 the parent constructor probes ``context.verify_mode`` and
        # ``context.check_hostname`` before we get a chance to override
        # ``connect``. Some callers (and every reasonable test double) don't
        # want that surface, and we never use ``self._context`` -- ``connect``
        # below wraps with ``self._ssl_context`` directly.
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip
        self._ssl_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)
        # Wrap with SNI = self.host so certificate verification still uses the
        # public hostname (not the pinned IP literal).
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_REDIRECT_BODY_DRAIN_CAP = 4096


def _fetch_once(url: str, timeout: float, max_bytes: int) -> tuple[int, dict[str, str], bytes, str]:
    """Perform one HTTP(S) request with SSRF defenses and no redirect handling.

    Returns:
        A tuple of (status, headers, body bytes, effective URL). On a 3xx
        response the body is not buffered beyond a small drain cap: the caller
        only needs the ``Location`` header.
    """
    validate_url_scheme(url)
    parts = urlsplit(url)
    host = parts.hostname
    if host is None:
        raise ValueError(f"URL has no host: {url!r}")

    addresses = resolve_and_validate_host(host)

    port = parts.port
    if port is None:
        port = 443 if parts.scheme.lower() == "https" else 80

    # Build a Host header from hostname (+ optional port) so any userinfo in
    # the URL is not leaked into the header. IPv6 literals must be bracketed
    # per RFC 3986 -- otherwise "2001:db8::1:8443" is ambiguous.
    is_ipv6_literal = ":" in host
    if is_ipv6_literal:
        host_header = f"[{host}]" if parts.port is None else f"[{host}]:{parts.port}"
    else:
        host_header = host if parts.port is None else f"{host}:{parts.port}"

    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"

    # Try each validated address in turn. This matters on hosts where getaddrinfo
    # returns an AAAA record first but the runtime has no working IPv6, or vice
    # versa -- every address has already passed the SSRF check, so falling
    # through does not weaken the security posture.
    last_error: OSError | None = None
    for pinned_ip in addresses:
        conn: http.client.HTTPConnection
        if parts.scheme.lower() == "https":
            ctx = ssl.create_default_context()
            conn = _PinnedHTTPSConnection(host, pinned_ip, port, timeout, ctx)
        else:
            conn = _PinnedHTTPConnection(host, pinned_ip, port, timeout)

        try:
            conn.request(
                "GET",
                path,
                headers={
                    "Host": host_header,
                    "User-Agent": _USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = conn.getresponse()
            headers = {k.lower(): v for k, v in response.getheaders()}
            if response.status in _REDIRECT_STATUSES:
                # We only need the Location header on a redirect -- draining the
                # body to a small cap is enough to let the connection close
                # cleanly without buffering the discarded payload.
                response.read(_REDIRECT_BODY_DRAIN_CAP)
                return response.status, headers, b"", url
            # Read up to max_bytes + 1 so we can detect overflow.
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError(f"Response body exceeded max_bytes={max_bytes}. Refusing to buffer more.")
            return response.status, headers, body, url
        except OSError as e:
            # Connect-time or transport-level errors are worth retrying against
            # the next validated address. ValueError (size cap, etc.) is a
            # policy decision -- do not retry those.
            last_error = e
            continue
        finally:
            conn.close()

    # All addresses failed to connect. Surface the last transport error.
    raise ConnectionError(
        f"Could not connect to any validated address for host {host!r} (tried {len(addresses)}): {last_error}"
    ) from last_error


def _follow_redirects(
    url: str,
    timeout: float,
    max_bytes: int,
    max_redirects: int,
) -> tuple[int, dict[str, str], bytes, str]:
    """Fetch ``url``, following up to ``max_redirects`` redirects, revalidating each hop."""
    current = url
    seen: set[str] = set()
    for _ in range(max_redirects + 1):
        if current in seen:
            raise ValueError(f"Redirect loop detected involving {current!r}")
        seen.add(current)

        status, headers, body, effective = _fetch_once(current, timeout=timeout, max_bytes=max_bytes)
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                raise ValueError(f"Redirect {status} from {current!r} without a Location header")
            # Resolve relative Location against the request URL.
            current = urljoin(current, location)
            # Re-validate scheme immediately so we do not, for example, follow a
            # javascript: or file: redirect.
            validate_url_scheme(current)
            continue
        return status, headers, body, effective

    raise ValueError(f"Exceeded max_redirects={max_redirects} following {url!r}")


def _decode_body(body: bytes, content_type: str) -> str:
    """Decode a response body using the charset from Content-Type, defaulting to utf-8."""
    charset = "utf-8"
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            charset = part.split("=", 1)[1].strip().strip('"').strip("'") or "utf-8"
            break
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def make_web_fetch(
    *,
    name: str = "web_fetch",
    description: str = WEB_FETCH_DESCRIPTION,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
) -> DecoratedFunctionTool:
    """Create a web fetch tool.

    Args:
        name: Tool name. Defaults to ``"web_fetch"``.
        description: Tool description shown to the model.
        max_bytes: Maximum response body size, in bytes. Larger responses are
            rejected without buffering the entire body.
        max_redirects: Maximum number of HTTP redirects to follow. Each hop is
            revalidated against the same SSRF rules as the initial URL.

    Returns:
        A decorated tool that fetches a URL and returns the extracted markdown.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if max_redirects < 0:
        raise ValueError(f"max_redirects must be non-negative, got {max_redirects}")

    @tool(name=name, description=description)
    async def web_fetch_tool(url: str, timeout: int = _DEFAULT_TIMEOUT) -> WebFetchOutput:
        """Fetches an HTTP(S) URL and returns readable markdown.

        Only ``http://`` and ``https://`` URLs are accepted. Requests to
        private, loopback, link-local, multicast, or reserved addresses are
        refused. Response body size is capped and redirects are revalidated.

        Args:
            url: The URL to fetch. Must be ``http://`` or ``https://``.
            timeout: Total wall-clock timeout in seconds (default: 30). Enforced
                as an outer ``asyncio.wait_for`` so a trickling server cannot
                keep the request open past this cap.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        try:
            status, headers, body, effective_url = await asyncio.wait_for(
                asyncio.to_thread(
                    _follow_redirects,
                    url,
                    float(timeout),
                    max_bytes,
                    max_redirects,
                ),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"web_fetch exceeded total timeout of {timeout}s for {url!r}") from e

        content_type = headers.get("content-type", "")
        text = _decode_body(body, content_type)

        # If the response is not HTML, return the decoded body verbatim (up to
        # the size cap) so the model can still read plain-text or markdown
        # responses that were served with a non-HTML Content-Type.
        if "html" not in content_type.lower() and "xml" not in content_type.lower():
            return {
                "url": effective_url,
                "status": status,
                "content_type": content_type,
                "title": "",
                "markdown": text,
            }

        title, markdown = html_to_markdown(text)
        return {
            "url": effective_url,
            "status": status,
            "content_type": content_type,
            "title": title,
            "markdown": markdown,
        }

    return web_fetch_tool


web_fetch = make_web_fetch()
"""Default web fetch tool with conservative size and redirect limits."""
