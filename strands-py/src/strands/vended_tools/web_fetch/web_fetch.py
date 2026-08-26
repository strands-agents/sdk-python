"""Web fetch tool: fetch a URL and return clean markdown for a model to read.

Distinct from the http_request tool, which returns raw response bodies for API
calls. This tool is intentionally narrow: HTTP(S) GET, decoded body, HTML to
markdown. It is not a general-purpose scraper.

Requests are SSRF-guarded and the tool connects to an already-validated IP so a
redirect or DNS rebind cannot reach a private address.
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

_HEADERS = {
    "User-Agent": "strands-agents-web-fetch/1.0",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "identity",
    "Connection": "close",
}
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
_DEFAULT_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_REDIRECT_BODY_DRAIN_CAP = 4096


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that connects to a pre-validated IP, preventing DNS rebinding."""

    def __init__(self, host: str, pinned_ip: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS variant of :class:`_PinnedHTTPConnection`; keeps ``host`` for SNI and cert checks."""

    def __init__(
        self,
        host: str,
        pinned_ip: str,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip
        self._ssl_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)
        try:
            self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


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
    scheme = parts.scheme.lower()
    port = parts.port or (443 if scheme == "https" else 80)

    # IPv6 literals must be bracketed per RFC 3986 to avoid ambiguity with the port separator.
    if ":" in host:
        host_header = f"[{host}]" if parts.port is None else f"[{host}]:{parts.port}"
    else:
        host_header = host if parts.port is None else f"{host}:{parts.port}"
    path = f"{parts.path or '/'}?{parts.query}" if parts.query else (parts.path or "/")
    headers = {**_HEADERS, "Host": host_header}

    last_error: OSError | None = None
    for pinned_ip in addresses:
        conn: http.client.HTTPConnection
        if scheme == "https":
            conn = _PinnedHTTPSConnection(host, pinned_ip, port, timeout, ssl.create_default_context())
        else:
            conn = _PinnedHTTPConnection(host, pinned_ip, port, timeout)
        try:
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            resp_headers = {k.lower(): v for k, v in response.getheaders()}
            if response.status in _REDIRECT_STATUSES:
                response.read(_REDIRECT_BODY_DRAIN_CAP)
                return response.status, resp_headers, b"", url
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError(f"Response body exceeded max_bytes={max_bytes}. Refusing to buffer more.")
            return response.status, resp_headers, body, url
        except OSError as e:
            # Every address already passed the SSRF check, so trying the next on a connect error does not
            # weaken the guard; it just tolerates a dead A/AAAA record.
            last_error = e
            continue
        finally:
            conn.close()

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
            current = urljoin(current, location)
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

        # Non-HTML responses are returned verbatim so the model can read plain-text or markdown
        # served with a non-HTML Content-Type.
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
