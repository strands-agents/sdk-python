"""SSRF defenses for the web fetch tool.

Only ``http`` and ``https`` schemes are allowed. Hostnames are resolved and every
returned address is checked against a deny list of private, loopback, link-local,
multicast, reserved, CGNAT, site-local, and unspecified ranges. Named cloud
metadata endpoints and a DNS suffix denylist are refused before DNS is queried
at all. Every hop of a redirect chain is re-validated so DNS rebinding cannot
swap a public address for a private one between the check and the actual
connect.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# DNS suffix denylist. Common private-network TLDs and anonymized-routing
# namespaces that must never leak a lookup to the public resolver. Compared
# case-folded with any trailing "." stripped. Kept in sync with the TypeScript
# implementation in ``strands-ts/src/vended-tools/web-fetch/ssrf.ts``.
_DENIED_DNS_SUFFIXES = (
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

# Explicit metadata hostnames. Belt-and-suspenders: every one of these resolves
# to an address the range checks below would already refuse, but naming them
# here means a future refactor of those predicates cannot silently expose them
# without also removing them from this list. Covers AWS/GCP IMDSv1
# (169.254.169.254), IPv6 EC2 (fd00:ec2::254), Alibaba
# (100.100.100.200 / 192.0.0.192), and GCP's DNS name.
_DENIED_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "fd00:ec2::254",
        "100.100.100.200",
        "192.0.0.192",
        "metadata.google.internal",
    }
)

# Site-local IPv6 (deprecated but still commonly deployed internally).
_IPV6_SITE_LOCAL = ipaddress.IPv6Network("fec0::/10")


def validate_url_scheme(url: str) -> None:
    """Raise ValueError if ``url`` is not an ``http://`` or ``https://`` URL."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Only http:// and https:// URLs are allowed. Got scheme={parts.scheme!r} for URL {url!r}.")
    if not parts.hostname:
        raise ValueError(f"URL has no host: {url!r}")


def _normalize_host_for_denylist(host: str) -> str:
    """Lower-case ``host`` and strip a single trailing ``.`` for suffix match."""
    lower = host.lower()
    if lower.endswith("."):
        lower = lower[:-1]
    return lower


def assert_host_is_allowed(host: str) -> None:
    """Refuse the DNS suffix denylist and named metadata endpoints.

    Runs before DNS resolution -- these hostnames must never generate a lookup
    at all. Raises :class:`ValueError` on refusal; returns silently on allow.
    """
    normalized = _normalize_host_for_denylist(host)
    if not normalized:
        return
    if normalized == "metadata":
        raise ValueError(f"Refusing to fetch host {host!r}: bare-label metadata endpoint is refused.")
    if normalized in _DENIED_METADATA_HOSTS:
        raise ValueError(f"Refusing to fetch host {host!r}: cloud metadata endpoint {normalized!r} is refused.")
    for suffix in _DENIED_DNS_SUFFIXES:
        if normalized == suffix[1:] or normalized.endswith(suffix):
            raise ValueError(f"Refusing to fetch host {host!r}: DNS suffix {suffix!r} is on the denylist.")


def _address_is_public(ip_text: str) -> bool:
    """Return True only for globally routable unicast addresses.

    Layers explicit rejections on top of ``ipaddress.is_global``:

    - ``is_global`` returns ``True`` for **multicast** on every CPython from
      3.10 through 3.14, so ``is_multicast`` must be checked explicitly.
    - Site-local IPv6 (``fec0::/10``) is not consistently caught by
      ``is_global`` either -- reject it explicitly.
    - ``is_reserved``, ``is_link_local``, and ``is_unspecified`` are layered
      too, as belt-and-suspenders against any future looseness in
      ``is_global``.

    IPv4-mapped IPv6 addresses (``::ffff:a.b.c.d``, including the fully expanded
    ``0:0:0:0:0:ffff:a.b.c.d`` form) are unwrapped first so the underlying IPv4
    category is what gets checked.
    """
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False

    # Unwrap IPv4-mapped IPv6 addresses so we compare against the mapped IPv4
    # address's real category.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    if ip.is_multicast:
        return False
    if ip.is_reserved:
        return False
    if ip.is_link_local:
        return False
    if ip.is_unspecified:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip in _IPV6_SITE_LOCAL:
        return False
    return bool(getattr(ip, "is_global", False))


def resolve_and_validate_host(host: str) -> list[str]:
    """Resolve ``host`` to IP addresses and require every one to be public.

    Returns the resolved addresses on success. Refuses the DNS suffix denylist
    and named metadata endpoints without querying DNS. If the host is an IP
    literal it is validated directly. If any resolved address is
    private/loopback/etc., raises :class:`ValueError`. Every address is
    re-checked, so a DNS-rebinding attempt that returns multiple A records is
    rejected.

    The caller is expected to connect using one of these already-validated
    addresses so the check-time and connect-time addresses agree.
    """
    # Refuse .internal / .onion / metadata.google.internal etc. before DNS.
    assert_host_is_allowed(host)

    # If host is a bracketed IPv6 literal, urlsplit strips the brackets already.
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host {host!r}: {e}") from e

    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        # sockaddr is (host, port) for AF_INET and (host, port, flow, scope) for AF_INET6.
        # ``sockaddr[0]`` is typed as ``str | int`` because ``socket.getaddrinfo`` can
        # return numeric addresses when NUMERICHOST is set; we always want the string.
        ip_text = str(sockaddr[0])
        if not _address_is_public(ip_text):
            raise ValueError(
                f"Refusing to fetch host {host!r}: resolved address {ip_text!r} is not public "
                "(private, loopback, link-local, site-local, multicast, CGNAT, or reserved)."
            )
        addresses.append(ip_text)

    if not addresses:
        raise ValueError(f"Could not resolve host {host!r}: no addresses returned")
    return addresses
