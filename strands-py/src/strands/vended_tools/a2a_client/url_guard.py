"""URL validation for the a2a_client tool.

Blocks the SSRF surface a model can steer via a URL: scheme allowlist,
private/loopback/link-local/CGNAT/multicast/reserved IPs, cloud-metadata IPs,
and ``.internal`` / ``.local`` names. Validation happens both on the initial URL
and on every host we subsequently connect to (agent card URL, resolved-card
``url``), because a remote card can point at a different host.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Hostnames whose DNS suffix we refuse outright. Cheaper than DNS resolution
# and catches operators who put private services on `.internal` / `.local` /
# `.corp` / `.home` even when those don't resolve to a private IP for us.
_BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (
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

# Bare-hostname denylist checked before DNS resolution. GCP's metadata server
# answers on ``http://metadata/`` inside a VPC, and DNS may return any address
# for ``metadata`` depending on the resolver — better to short-circuit on the
# label alone.
_BLOCKED_BARE_HOSTNAMES: frozenset[str] = frozenset({"metadata"})

# Well-known cloud metadata addresses. Layered on top of the general
# categorical checks so a future refactor that weakens a predicate doesn't
# silently expose them.
_BLOCKED_METADATA_ADDRESSES: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "fd00:ec2::254",
        "100.100.100.200",
        "192.0.0.192",
    }
)


class UrlNotAllowedError(ValueError):
    """Raised when a URL fails validation before we make a network call."""


def validate_url(url: str, allowed_prefixes: tuple[str, ...] | None = None) -> str:
    """Validate a URL and return its resolved-safe host component for logging.

    Enforces:

    * http/https only,
    * a non-empty hostname,
    * hostname is not a bare metadata label (e.g. ``metadata``),
    * hostname not on the blocked-suffix list,
    * every DNS-resolved address is public — explicitly rejects private,
      loopback, link-local, CGNAT, multicast, reserved, and unspecified
      ranges (Python's ``is_global`` returns ``True`` for multicast on
      3.10–3.14, so the categorical checks are layered on top),
    * hostname not on the metadata address denylist,
    * URL starts with one of ``allowed_prefixes`` if the caller passed a list.

    Args:
        url: The URL to check.
        allowed_prefixes: Optional developer-supplied allowlist. When set,
            ``url`` must start with one of these prefixes.

    Returns:
        The parsed hostname (lowercased, trailing dot stripped).

    Raises:
        UrlNotAllowedError: If the URL fails any check.
    """
    if not isinstance(url, str):
        raise UrlNotAllowedError(f"url must be a string, got {type(url).__name__}")
    if allowed_prefixes and not any(url.startswith(prefix) for prefix in allowed_prefixes):
        raise UrlNotAllowedError(
            f"url {url!r} is not in the developer-configured allowlist (expected one of {list(allowed_prefixes)!r})"
        )

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise UrlNotAllowedError(f"url must use http or https, got scheme {scheme!r}")

    host = (parsed.hostname or "").lower()
    # Strip a trailing dot so `foo.internal.` is caught by the suffix check.
    if host.endswith("."):
        host = host[:-1]
    if not host:
        raise UrlNotAllowedError("url must include a hostname")

    if host in _BLOCKED_BARE_HOSTNAMES:
        # GCP's metadata server answers on `http://metadata/` inside a VPC;
        # reject the label before we ever call getaddrinfo.
        raise UrlNotAllowedError(f"hostname {host!r} is a cloud metadata label")

    for suffix in _BLOCKED_HOST_SUFFIXES:
        if host == suffix.lstrip(".") or host.endswith(suffix):
            raise UrlNotAllowedError(f"hostname {host!r} matches blocked suffix {suffix!r}")

    _assert_host_resolves_to_public_ips(host)
    return host


def _assert_host_resolves_to_public_ips(host: str) -> None:
    """Resolve ``host`` and reject if any address is not globally routable.

    If ``host`` is a bare IP literal we skip DNS and check it directly.

    Args:
        host: The hostname to resolve and validate.

    Raises:
        UrlNotAllowedError: If any resolved IP fails the public-only check,
            or if the hostname cannot be resolved.
    """
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        _assert_ip_is_public(literal)
        return

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlNotAllowedError(f"could not resolve hostname {host!r}: {exc}") from exc

    if not infos:
        raise UrlNotAllowedError(f"hostname {host!r} resolved to no addresses")

    seen: set[str] = set()
    for info in infos:
        raw_address = info[4][0]
        address = str(raw_address)
        if address in seen:
            continue
        seen.add(address)
        _assert_ip_is_public(ipaddress.ip_address(address))


def _assert_ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Raise if ``ip`` is any flavor of non-global-unicast address.

    Python's ``ipaddress.is_global`` returns ``True`` for both IPv4 and IPv6
    multicast on every supported CPython (3.10-3.14), so ``not is_global``
    alone is not sufficient — an attacker could point at ``239.255.255.250``
    (SSDP) or ``ff02::1`` and reach neighbours the host expected to be
    isolated from. We layer explicit ``is_multicast`` / ``is_reserved`` /
    ``is_unspecified`` / ``is_link_local`` checks on top before consulting
    ``is_global``.

    Args:
        ip: An ``IPv4Address`` or ``IPv6Address``.

    Raises:
        UrlNotAllowedError: If the IP is private, loopback, link-local, multicast,
            reserved, unspecified, CGNAT, or a known cloud-metadata address.
    """
    # IPv4-mapped IPv6 (`::ffff:a.b.c.d`) — unwrap to the embedded IPv4 so the
    # underlying v4 category (loopback, private, CGNAT, …) is what gets checked.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    if str(ip) in _BLOCKED_METADATA_ADDRESSES:
        raise UrlNotAllowedError(f"ip {ip!s} is a cloud metadata address")

    # Categorical checks that must run before `is_global` — see docstring.
    if ip.is_multicast:
        raise UrlNotAllowedError(f"ip {ip!s} is multicast")
    if ip.is_unspecified:
        raise UrlNotAllowedError(f"ip {ip!s} is unspecified")
    if ip.is_link_local:
        raise UrlNotAllowedError(f"ip {ip!s} is link-local")
    if ip.is_reserved:
        raise UrlNotAllowedError(f"ip {ip!s} is reserved")

    if not ip.is_global:
        # Emit a specific category label where possible so operators can tell
        # the classes of rejection apart in logs.
        category = _classify_non_global(ip)
        raise UrlNotAllowedError(f"ip {ip!s} is {category}")


def _classify_non_global(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Return a human label for a non-global IP for use in error messages."""
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_reserved:
        return "reserved"
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.IPv4Network("100.64.0.0/10"):
        return "private (CGNAT)"
    if ip.is_private:
        return "private"
    return "not globally routable"
