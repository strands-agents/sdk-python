"""HTTP request tool for making raw HTTP calls to external APIs.

Provides :func:`make_http_request` (a factory that lets the tool operator
configure allowlists and limits) and :data:`http_request` (a default instance
with the safe defaults applied).

The tool is a thin shim over ``httpx.AsyncClient``. Its job is to guard the
network boundary the model can reach:

- reject non-http(s) schemes;
- reject hostnames that end in the SSRF-spec denylist suffixes
  (``.internal``, ``.local``, ``.corp``, ``.onion``, ...);
- resolve the target host to an IP and refuse non-public destinations
  (RFC1918, loopback, link-local, multicast, reserved, IPv4-mapped-private
  IPv6, cloud-metadata addresses) unless the operator has explicitly
  allowlisted the host at construction time;
- re-validate every redirect hop against the same policy;
- strip cross-origin auth (``Authorization`` / ``Cookie`` /
  ``Proxy-Authorization``) on every redirect that changes origin -- a scheme
  downgrade, a port change, or a host change all strip;
- cap the total request time, the redirect count, the response body size,
  and the total response-header size;
- reject model-supplied ``Authorization`` / ``Cookie`` /
  ``Proxy-Authorization`` headers unless the operator's config permits
  them for the target host;
- propagate the parent agent's cancel signal (``Agent._cancel_signal``) so an
  in-flight fetch aborts when the agent is cancelled. Cancellation is
  signalled with :class:`asyncio.CancelledError`, matching the
  :func:`asyncio.sleep`-style vended-tool convention.

DNS-rebinding note. The check-then-connect pattern here leaves a small TOCTOU
window: an attacker-controlled name server can return a public address to
``getaddrinfo`` and a private one to httpx's own resolve at connect time.
Pinning the resolved IP through httpx's transport is impractical without
reaching into httpcore internals; we accept this residual risk against
dynamic DNS and rely on the guard for static DNS. The rest of the controls
still hold: metadata addresses and non-http(s) schemes remain blocked, and
redirects are re-validated.

These controls are set by the tool operator at construction time and are
never controllable by the model.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import DEFAULT_HTTP_REQUEST_DESCRIPTION, HttpMethod, HttpRequestOutput

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool


Resolver = Callable[[str], list[str]]
"""Function that maps a hostname to a list of IP-address strings."""


_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_REDIRECTS = 5
_DEFAULT_MAX_RESPONSE_HEADERS_BYTES = 64 * 1024
"""Cap on the total size of response headers returned to the model."""

_DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}
"""Default ports used when computing an origin tuple for redirect auth checks."""

_DENYLIST_SUFFIXES = (
    ".internal",
    ".local",
    ".localhost",
    ".corp",
    ".home",
    ".lan",
    ".intranet",
    ".private",
    ".i2p",
    ".onion",
)
"""Hostname suffixes that must never resolve, per the shared SSRF spec (§4)."""

_SENSITIVE_HEADER_NAMES = frozenset({"authorization", "cookie", "proxy-authorization"})
"""Headers that the model may not set unless the operator opts the target host in."""

_HOP_BY_HOP_HEADER_NAMES = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)
"""Hop-by-hop and forbidden headers the model must not set."""

_CLOUD_METADATA_HOSTS = frozenset(
    {
        # AWS / Azure / OpenStack / DigitalOcean
        "169.254.169.254",
        # AWS IMDSv2 IPv6
        "fd00:ec2::254",
        # GCP metadata by DNS (bare label is checked before DNS)
        "metadata.google.internal",
        "metadata",
        # Alibaba Cloud
        "100.100.100.200",
        # Oracle Cloud Infrastructure
        "192.0.0.192",
    }
)
"""Well-known metadata endpoints that must never be reachable by default."""


class HttpRequestError(RuntimeError):
    """Raised when a request is rejected or fails."""


@dataclass(frozen=True)
class HttpRequestConfig:
    """Operator-controlled configuration for :func:`make_http_request`.

    Every field is set by the tool operator at construction time. Nothing here
    is under the model's control.

    Attributes:
        allow_private_hosts: Hostnames (case-insensitive) that are permitted to
            resolve to a private-network IP. An entry ``"internal.example.com"``
            allows exactly that host; there is no wildcard matching. Empty by
            default -- private destinations are denied by default.
        allow_auth_for_hosts: Hostnames for which the model may set
            ``Authorization`` / ``Cookie`` / ``Proxy-Authorization`` headers.
            Any request to a host not in this set is rejected outright if it
            carries one of those headers; on a cross-origin redirect away
            from an allowlisted host, the same headers are stripped so the
            credential never travels to the new origin.
        max_response_bytes: Hard cap on response body size in bytes.
        max_response_headers_bytes: Hard cap on total response-header size in
            bytes. Response headers are returned to the model, so this bounds
            what an oversized ``Set-Cookie`` (or similar) can push through.
        max_redirects: Hard cap on redirect chain length. A value of ``0``
            raises ``HttpRequestError`` if a 3xx is encountered.
        default_timeout_seconds: Timeout used when the model does not supply
            one (also acts as an upper bound on any timeout the model asks
            for).
    """

    allow_private_hosts: frozenset[str] = field(default_factory=frozenset)
    allow_auth_for_hosts: frozenset[str] = field(default_factory=frozenset)
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES
    max_response_headers_bytes: int = _DEFAULT_MAX_RESPONSE_HEADERS_BYTES
    max_redirects: int = _DEFAULT_MAX_REDIRECTS
    default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


def _default_resolver(host: str) -> list[str]:
    """Resolve a hostname to a list of unique IP-address strings."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        raise HttpRequestError(f"DNS resolution failed for {host}: {e}") from e
    seen: list[str] = []
    for info in infos:
        raw = info[4][0]
        # IPv4 tuples return a str; IPv6 sockaddr can include a scope id. Only str
        # addresses are interpretable here.
        if not isinstance(raw, str):
            continue
        if raw not in seen:
            seen.append(raw)
    return seen


def make_http_request(
    *,
    allow_private_hosts: Iterable[str] | None = None,
    allow_auth_for_hosts: Iterable[str] | None = None,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    max_response_headers_bytes: int = _DEFAULT_MAX_RESPONSE_HEADERS_BYTES,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
    default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    name: str = "http_request",
    description: str = DEFAULT_HTTP_REQUEST_DESCRIPTION,
    resolver: Resolver | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DecoratedFunctionTool:
    """Create an HTTP request tool configured for a given security posture.

    The safe default posture (no arguments) denies every private-network
    destination, denies model-supplied ``Authorization`` / ``Cookie``, caps
    the response body at 10 MiB, follows at most five redirects, and times
    out after 30 seconds.

    Args:
        allow_private_hosts: Hostnames whose DNS answers may point at private
            IPs (loopback / RFC1918 / link-local / etc.). Use only for hosts
            you fully trust the model to reach on the internal network.
        allow_auth_for_hosts: Hostnames for which the model is allowed to set
            ``Authorization`` / ``Cookie`` / ``Proxy-Authorization`` headers.
        max_response_bytes: Hard cap on response body size.
        max_response_headers_bytes: Hard cap on total response-header size
            (default 64 KiB).
        max_redirects: Hard cap on redirect chain length. ``0`` raises
            ``HttpRequestError`` if a 3xx is encountered instead of following.
        default_timeout_seconds: Default and upper-bound per-request timeout.
        name: Tool name shown to the model.
        description: Tool description shown to the model.
        resolver: Injection point for the DNS resolver. Callers can pass a
            custom resolver (e.g. to force a specific mode); tests pass a
            stub. Defaults to a resolver backed by ``socket.getaddrinfo``.
        transport: Injection point for the ``httpx`` transport. Callers can
            pass a mock transport in tests. Defaults to the standard
            connection-based transport.

    Returns:
        A decorated tool that makes HTTP requests within the configured policy.
    """
    config = HttpRequestConfig(
        allow_private_hosts=frozenset(h.lower() for h in (allow_private_hosts or ())),
        allow_auth_for_hosts=frozenset(h.lower() for h in (allow_auth_for_hosts or ())),
        max_response_bytes=max_response_bytes,
        max_response_headers_bytes=max_response_headers_bytes,
        max_redirects=max_redirects,
        default_timeout_seconds=default_timeout_seconds,
    )
    active_resolver: Resolver = resolver if resolver is not None else _default_resolver
    active_transport = transport

    @tool(name=name, description=description, context=True)
    async def http_request_tool(
        method: HttpMethod,
        url: str,
        headers: dict[str, str] | None = None,
        body: str | None = None,
        timeout: float | None = None,
        tool_context: ToolContext | None = None,
    ) -> HttpRequestOutput:
        """Make an HTTP request to a URL and return the response.

        Args:
            method: HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``,
                ``PATCH``, ``HEAD``, ``OPTIONS``).
            url: Absolute ``http://`` or ``https://`` URL to request.
            headers: Optional request headers.
            body: Optional request body as a string.
            timeout: Optional per-request timeout in seconds. Capped at the
                tool's configured default.
            tool_context: Framework-injected. Not model-visible. Carries the
                agent so the tool can read its cancel signal mid-flight.
        """
        effective_timeout = _resolve_timeout(timeout, config.default_timeout_seconds)
        sanitized_headers = _sanitize_headers(headers, url, config)
        cancel_signal = _extract_cancel_signal(tool_context)

        return await _perform_request(
            method=method,
            url=url,
            headers=sanitized_headers,
            body=body,
            timeout=effective_timeout,
            config=config,
            resolver=active_resolver,
            transport=active_transport,
            cancel_signal=cancel_signal,
        )

    return http_request_tool


http_request = make_http_request()
"""Default HTTP request tool with the safe posture applied."""


# ---- Internals ----


def _resolve_timeout(model_timeout: float | None, default: float) -> float:
    """Return the effective request timeout, bounded by the configured default."""
    if model_timeout is None:
        return default
    if model_timeout <= 0:
        raise HttpRequestError("timeout must be positive")
    return min(float(model_timeout), default)


def _format_number(value: float) -> str:
    """Format a numeric value without a trailing ``.0`` for whole numbers.

    Keeps error messages byte-identical to the TypeScript sibling, which
    stringifies integers-as-numbers without the fractional part.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sanitize_headers(
    headers: dict[str, str] | None,
    url: str,
    config: HttpRequestConfig,
) -> dict[str, str]:
    """Drop or reject headers the model must not set for this URL."""
    if not headers:
        return {}
    host = (urlsplit(url).hostname or "").lower()
    auth_allowed = host in config.allow_auth_for_hosts
    result: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADER_NAMES:
            raise HttpRequestError(f"Header not allowed: {key}")
        if lowered in _SENSITIVE_HEADER_NAMES and not auth_allowed:
            raise HttpRequestError(
                f"Header {key} is not allowed for {host or 'this URL'}. "
                "The tool operator has not opted this host into auth header passthrough."
            )
        result[key] = value
    return result


async def _validate_url(url: str, config: HttpRequestConfig, resolver: Resolver) -> str:
    """Validate a URL against the security policy and return the resolved host.

    - Only ``http`` and ``https`` schemes are allowed.
    - The host must be present.
    - Well-known cloud-metadata hosts are blocked outright.
    - Hostnames on the SSRF-spec suffix denylist are refused before DNS runs;
      the operator's ``allow_private_hosts`` does not override this because a
      ``.internal`` or ``.corp`` host that the operator wants reachable should
      have a real name outside the reserved namespace.
    - The host must not resolve to a private/reserved IP unless allowlisted.

    The resolver call is off-loaded to a worker thread with ``asyncio.to_thread``
    so a slow or failing name server does not stall the event loop.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise HttpRequestError(f"Only http and https URLs are allowed (got scheme {parts.scheme!r})")
    if not parts.hostname:
        raise HttpRequestError("URL has no host")
    # `parts.port` raises ValueError on non-numeric ports (e.g. "example.com:bad"),
    # which would otherwise propagate as an unhandled exception through _origin()
    # on a redirect. Fail closed with the tool's own error type.
    try:
        _ = parts.port
    except ValueError as error:
        raise HttpRequestError(f"URL has an invalid port: {url!r}") from error
    host = parts.hostname.lower()

    if host in _CLOUD_METADATA_HOSTS:
        raise HttpRequestError(f"Host {host} is a known cloud-metadata endpoint and is not allowed")

    # DNS suffix denylist per SSRF spec §4. Applied before ``allow_private_hosts``
    # so that opting in a single private IP cannot accidentally re-enable a
    # reserved namespace like ``.internal`` or ``.corp``. Strip a single
    # trailing dot (rooted FQDN) before matching so ``foo.internal.`` also
    # resolves to the denylist.
    denylist_host = host.rstrip(".")
    for suffix in _DENYLIST_SUFFIXES:
        if denylist_host == suffix.lstrip(".") or denylist_host.endswith(suffix):
            raise HttpRequestError(f"Refusing to resolve {host}: hostname ends with denied suffix {suffix!r}")

    if host in config.allow_private_hosts:
        return host

    # If the URL uses an IP literal, validate it directly -- do not consult the resolver.
    literal_ip = _parse_ip_literal(parts.hostname)
    if literal_ip is not None:
        if not _is_public_ip(literal_ip):
            raise HttpRequestError(f"Refusing to connect to non-public address {parts.hostname}")
        return host

    addresses = await asyncio.to_thread(resolver, host)
    if not addresses:
        raise HttpRequestError(f"DNS resolution returned no addresses for {host}")

    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as e:
            raise HttpRequestError(f"Could not parse resolved address {addr!r} for {host}") from e
        if not _is_public_ip(ip):
            raise HttpRequestError(f"Refusing to connect to {host}: resolves to non-public address {addr}")
    return host


def _parse_ip_literal(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return the IP address if ``raw`` is an IP literal, else ``None``.

    ``urlsplit`` strips the ``[...]`` around an IPv6 literal, so ``raw`` is
    already the bare address form here.
    """
    try:
        return ipaddress.ip_address(raw)
    except ValueError:
        return None


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP is safe to dial (globally routable, non-metadata).

    Follows the shared SSRF spec's predicate set. IPv4-mapped IPv6 addresses
    are unwrapped first so `is_global` on ``::ffff:10.0.0.1`` (which returns
    True on Python 3.10/3.11) does not create a bypass.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if str(ip) in _CLOUD_METADATA_HOSTS:
        return False
    # `is_global` alone returns True for multicast on all supported Python
    # versions, so we assert each predicate the spec calls out explicitly.
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified or ip.is_link_local:
        return False
    if getattr(ip, "is_site_local", False):
        return False
    return bool(ip.is_global)


def _origin(url: str) -> tuple[str, str, int | None]:
    """Return the ``(scheme, hostname, port)`` origin tuple for a URL.

    ``port`` falls back to the default for the scheme (80/443) so a URL that
    omits the port and one that spells it out compare equal. Scheme is
    case-folded; hostname is already case-insensitive per RFC 3986.
    """
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    port = parts.port if parts.port is not None else _DEFAULT_SCHEME_PORTS.get(scheme)
    return scheme, host, port


def _strip_cross_origin_auth(headers: dict[str, str], from_url: str, to_url: str) -> dict[str, str]:
    """Drop credentialing headers when a redirect changes origin.

    A 3xx to an attacker-controlled origin would otherwise forward whatever
    ``Authorization`` / ``Cookie`` / ``Proxy-Authorization`` header the
    operator opted the original host into. Only exact-origin redirects
    (same scheme, host, and port) preserve the headers -- a scheme downgrade
    (HTTPS -> HTTP), a port change, or a host change all strip.
    """
    if _origin(from_url) == _origin(to_url):
        return headers
    return {k: v for k, v in headers.items() if k.lower() not in _SENSITIVE_HEADER_NAMES}


async def _perform_request(
    *,
    method: HttpMethod,
    url: str,
    headers: dict[str, str],
    body: str | None,
    timeout: float,
    config: HttpRequestConfig,
    resolver: Resolver,
    transport: httpx.AsyncBaseTransport | None,
    cancel_signal: threading.Event | None = None,
) -> HttpRequestOutput:
    """Perform the HTTP request, honouring the redirect and body-size caps.

    Redirects are followed manually so every hop can be re-validated against
    the resolver policy and cross-origin auth can be stripped before the
    next request is built.
    """
    await _validate_url(url, config, resolver)

    client_kwargs: dict[str, object] = {
        "timeout": timeout,
        "follow_redirects": False,
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    current_method: str = method
    current_url: str = url
    current_body: str | None = body
    current_headers: dict[str, str] = dict(headers)

    async with httpx.AsyncClient(**client_kwargs) as client:  # type: ignore[arg-type]
        for _ in range(config.max_redirects + 1):
            _check_cancelled(cancel_signal)
            try:
                request = client.build_request(
                    current_method,
                    current_url,
                    headers=current_headers or None,
                    content=current_body,
                )
                response = await client.send(request, stream=True)
            except httpx.TimeoutException as e:
                raise HttpRequestError(f"Request timed out after {_format_number(timeout)} seconds") from e
            except httpx.RequestError as e:
                raise HttpRequestError(f"Request failed: {e}") from e

            try:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if location:
                        next_url = str(response.url.join(location))
                        await _validate_url(next_url, config, resolver)
                        current_headers = _strip_cross_origin_auth(current_headers, current_url, next_url)
                        # 303 semantics: force GET and drop body.
                        if response.status_code == 303 and current_method.upper() != "HEAD":
                            current_method = "GET"
                            current_body = None
                        current_url = next_url
                        await response.aclose()
                        continue

                headers_out = _capped_response_headers(response, config.max_response_headers_bytes)
                body_text = await _read_capped_body(response, config.max_response_bytes, cancel_signal=cancel_signal)
            finally:
                if not response.is_closed:
                    await response.aclose()

            return HttpRequestOutput(
                status=response.status_code,
                status_text=response.reason_phrase or "",
                headers=headers_out,
                body=body_text,
            )

    raise HttpRequestError(f"Too many redirects (limit {config.max_redirects})")


async def _read_capped_body(
    response: httpx.Response,
    max_bytes: int,
    *,
    cancel_signal: threading.Event | None = None,
) -> str:
    """Read the response body streaming, aborting if it exceeds ``max_bytes``."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        _check_cancelled(cancel_signal)
        total += len(chunk)
        if total > max_bytes:
            raise HttpRequestError(f"Response body exceeded maximum size of {max_bytes} bytes")
        chunks.append(chunk)
    raw = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _capped_response_headers(response: httpx.Response, max_bytes: int) -> dict[str, str]:
    r"""Return response headers as a lower-cased dict, capped at ``max_bytes``.

    Response headers are surfaced to the model, so an oversized ``Set-Cookie``
    or debug header could shove megabytes back through. The cap is on the
    summed size of the ``name: value`` pairs; we walk in order and refuse the
    whole response if the running total goes over.

    ``multi_items`` preserves every occurrence, so a response with two
    ``Set-Cookie`` lines does not get comma-collapsed into an unparseable
    single value (cookie values legitimately contain commas). Repeated
    headers are joined with a newline separator, which mirrors the wire form
    ``Set-Cookie: a=1\r\nSet-Cookie: b=2`` closely enough for a model to
    split on.
    """
    result: dict[str, str] = {}
    total = 0
    for key, value in response.headers.multi_items():
        # 2 bytes for ": " between name/value; a rough model of on-the-wire size.
        total += len(key) + len(value) + 2
        if total > max_bytes:
            raise HttpRequestError(f"Response headers exceeded maximum size of {max_bytes} bytes")
        lowered = key.lower()
        existing = result.get(lowered)
        result[lowered] = value if existing is None else f"{existing}\n{value}"
    return result


def _extract_cancel_signal(tool_context: ToolContext | None) -> threading.Event | None:
    """Return the agent's cancellation event when available.

    The event lives on ``Agent._cancel_signal``. It is a private attribute --
    we access it defensively so the tool works with any object shape (mocks,
    ``BidiAgent``, etc.) rather than crashing when the attribute is missing.
    """
    if tool_context is None:
        return None
    agent: Any = getattr(tool_context, "agent", None)
    signal: Any = getattr(agent, "_cancel_signal", None)
    return signal if isinstance(signal, threading.Event) else None


def _check_cancelled(cancel_signal: threading.Event | None) -> None:
    """Raise :class:`asyncio.CancelledError` if the agent's cancel signal has been set.

    Cancellation is signalled with the same exception the surrounding
    :func:`asyncio` task would raise, so callers can distinguish "cancelled"
    from other request failures without pattern-matching on error text.
    """
    if cancel_signal is not None and cancel_signal.is_set():
        raise asyncio.CancelledError("Request cancelled")
