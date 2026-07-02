"""Skill data model and loading utilities for AgentSkills.io skills.

This module defines the Skill dataclass and provides classmethods for
discovering, parsing, and loading skills from the filesystem, raw content,
or HTTPS URLs. Skills are directories containing a SKILL.md file with YAML
frontmatter metadata and markdown instructions.
"""

from __future__ import annotations

import http.client
import ipaddress
import logging
import re
import socket
import ssl
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

logger = logging.getLogger(__name__)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_SKILL_NAME_LENGTH = 64

# Schemes a remote skill may be fetched over. Anything else (file, gopher, ftp,
# data, ...) is rejected so the fetch cannot be redirected onto a non-network or
# lower-trust transport.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Maximum number of bytes read from a remote SKILL.md response.
_MAX_FETCH_BYTES = 5 * 1024 * 1024

# A small set of non-public IPv6 addresses that ``ipaddress`` does not flag via
# its is_* properties, so they are matched explicitly.
_EXTRA_NON_PUBLIC_IPS = frozenset({ipaddress.ip_address("fd00:ec2::254")})

# The well-known NAT64 (64:ff9b::/96) prefix embeds an IPv4 address that
# ``ipaddress`` does not surface via ``ipv4_mapped`` or ``sixtofour``.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Extract any IPv4 address embedded in an IPv6 address.

    Covers IPv4-mapped / IPv4-compatible addresses (``::ffff:a.b.c.d`` and
    ``::a.b.c.d``), 6to4 (``2002:AABB:CCDD::/48``), and the well-known NAT64
    prefix (``64:ff9b::a.b.c.d``). These encodings can carry an IPv4 address
    such as ``169.254.169.254`` past the standard ``is_*`` checks, so the
    embedded IPv4 must be validated in its own right.

    Args:
        ip: The IPv6 address to inspect.

    Returns:
        The embedded IPv4 address, or None if the address does not embed one.
    """
    mapped = ip.ipv4_mapped
    if mapped is not None:
        return mapped

    packed = int(ip)

    # IPv4-compatible (deprecated) addresses: ``::a.b.c.d`` where only the low
    # 32 bits are set. ``::`` and ``::1`` are excluded as they are already
    # covered by the unspecified/loopback checks.
    if packed >> 32 == 0 and packed > 1:
        return ipaddress.IPv4Address(packed & 0xFFFFFFFF)

    if ip.sixtofour is not None:
        return ip.sixtofour

    if ip in _NAT64_PREFIX:
        return ipaddress.IPv4Address(packed & 0xFFFFFFFF)

    return None


def _is_disallowed_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if a remote skill must not be fetched from this IP address.

    Allows only public, routable addresses. Loopback, link-local, private,
    multicast, reserved, and unspecified ranges are rejected, along with a few
    extra non-public addresses not covered by the standard properties. For IPv6
    addresses that embed an IPv4 address (IPv4-mapped/compatible, 6to4, NAT64),
    the embedded IPv4 is checked as well so encodings like
    ``::ffff:169.254.169.254`` cannot slip past.

    Args:
        ip: The resolved IP address to check.

    Returns:
        True if the address is in a disallowed range, False otherwise.
    """
    if ip in _EXTRA_NON_PUBLIC_IPS:
        return True
    if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True

    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None and _is_disallowed_address(embedded):
            return True

    return False


def _resolve_and_validate_host(url: str) -> list[str]:
    """Resolve a URL's host and validate every resolved address.

    The host is resolved exactly once here and the resulting IP address(es) are
    both validated and returned so the caller can connect to a pre-validated
    address rather than letting the socket layer re-resolve the name (which
    would reopen a DNS-rebinding / TOCTOU window).

    Args:
        url: The URL whose host should be resolved and validated.

    Returns:
        The list of validated IP address strings the host resolves to.

    Raises:
        ValueError: If the host is missing, unresolvable, or any resolved
            address is in a disallowed range.
    """
    host = urlsplit(url).hostname
    if not host:
        raise ValueError(f"url=<{url}> | missing host")

    # A bracketed/bare IP literal is parsed by urlsplit without brackets, so it
    # can be checked directly; otherwise resolve the hostname to its addresses.
    try:
        literal = ipaddress.ip_address(host)
        candidates = [literal]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as e:
            raise ValueError(f"url=<{url}> | could not resolve host") from e
        candidates = [ipaddress.ip_address(info[4][0]) for info in infos]

    if not candidates:
        raise ValueError(f"url=<{url}> | could not resolve host")

    for ip in candidates:
        if _is_disallowed_address(ip):
            raise ValueError(f"url=<{url}> | host is not allowed")

    return [str(ip) for ip in candidates]


def _validate_fetch_url(url: str) -> list[str]:
    """Validate a fetch URL's scheme and host, returning validated IPs.

    Enforces the scheme allowlist (http/https only) and then resolves and
    validates the host. Used for the initial URL and re-run on every redirect
    hop.

    Args:
        url: The URL to validate.

    Returns:
        The list of validated IP address strings the host resolves to.

    Raises:
        ValueError: If the scheme is not allowed or the host is disallowed.
    """
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"url=<{url}> | scheme is not allowed")
    return _resolve_and_validate_host(url)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that dials a pre-validated IP but keeps the real host.

    ``urllib`` re-resolves the hostname independently at connect time, so the
    address validated in :func:`_resolve_and_validate_host` is never the one the
    socket actually connects to (a DNS-rebinding / TOCTOU hole). This subclass
    connects to a specific, already-validated IP address while leaving ``host``
    (and therefore the TLS SNI and certificate hostname check) set to the
    original hostname, so certificate verification stays intact.
    """

    def __init__(self, host: str, pinned_ip: str, ssl_context: ssl.SSLContext, **kwargs: Any) -> None:
        super().__init__(host, context=ssl_context, **kwargs)
        self._pinned_ip = pinned_ip
        self._ssl_context = ssl_context

    def connect(self) -> None:
        # Open the raw socket to the validated IP rather than re-resolving, then
        # wrap with TLS using the original hostname for SNI and cert validation.
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that dials a pre-validated IP but keeps the real host.

    The plaintext counterpart to :class:`_PinnedHTTPSConnection`; used only when
    a validated redirect target is served over http.
    """

    def __init__(self, host: str, pinned_ip: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


def _fetch_url_content(url: str, *, timeout: float = 30.0) -> str:
    """Fetch text content from a URL with SSRF protections.

    Enforces the scheme allowlist and host validation on the initial URL and on
    every redirect hop, and pins each connection to a pre-validated IP so the
    socket cannot be steered to a different address than the one that was
    checked.

    Args:
        url: The URL to fetch.
        timeout: Per-connection timeout in seconds.

    Returns:
        The decoded UTF-8 response body.

    Raises:
        ValueError: If the URL (or any redirect target) fails validation.
        RuntimeError: If the content cannot be fetched.
    """
    context = ssl.create_default_context()
    current_url = url
    initial_scheme = urlsplit(url).scheme.lower()
    # Follow a bounded number of redirects, re-validating each hop ourselves.
    for _ in range(10):
        # Block a downgrade from https to http (e.g. an https->http redirect to
        # an internal endpoint) on top of the scheme allowlist.
        if initial_scheme == "https" and urlsplit(current_url).scheme.lower() == "http":
            raise ValueError(f"url=<{current_url}> | insecure redirect from https to http is not allowed")

        validated_ips = _validate_fetch_url(current_url)
        parts = urlsplit(current_url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        pinned_ip = validated_ips[0]

        conn: http.client.HTTPConnection
        if parts.scheme == "https":
            conn = _PinnedHTTPSConnection(host, pinned_ip, context, port=port, timeout=timeout)
        else:
            conn = _PinnedHTTPConnection(host, pinned_ip, port=port, timeout=timeout)

        request_target = parts.path or "/"
        if parts.query:
            request_target = f"{request_target}?{parts.query}"

        try:
            # The connection host is the original hostname, so http.client emits
            # the correct Host header (and TLS SNI) for the pinned IP socket.
            conn.request("GET", request_target, headers={"User-Agent": "strands-agents-sdk"})
            response = conn.getresponse()

            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.read()
                if not location:
                    raise RuntimeError(f"url=<{current_url}> | redirect without Location header")
                # Resolve relative redirects against the current URL, then loop
                # so the scheme and host of the target are validated afresh.
                current_url = urllib.parse.urljoin(current_url, location)
                continue

            if response.status >= 400:
                raise RuntimeError(f"url=<{current_url}> | HTTP {response.status}: {response.reason}")

            body = response.read(_MAX_FETCH_BYTES + 1)
            if len(body) > _MAX_FETCH_BYTES:
                raise RuntimeError(f"url=<{current_url}> | skill content exceeds size limit")
            return body.decode("utf-8")
        except (OSError, http.client.HTTPException) as e:
            raise RuntimeError(f"url=<{current_url}> | failed to fetch skill: {e}") from e
        finally:
            conn.close()

    raise RuntimeError(f"url=<{url}> | too many redirects")


def _find_skill_md(skill_dir: Path) -> Path:
    """Find the SKILL.md file in a skill directory.

    Searches for SKILL.md (case-sensitive preferred) or skill.md as a fallback.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        Path to the SKILL.md file.

    Raises:
        FileNotFoundError: If no SKILL.md file is found in the directory.
    """
    for name in ("SKILL.md", "skill.md"):
        candidate = skill_dir / name
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"path=<{skill_dir}> | no SKILL.md found in skill directory")


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and body from SKILL.md content.

    Extracts the YAML frontmatter between ``---`` delimiters at line boundaries
    and returns parsed key-value pairs along with the remaining markdown body.

    Args:
        content: Full content of a SKILL.md file.

    Returns:
        Tuple of (frontmatter_dict, body_string).

    Raises:
        ValueError: If the frontmatter is malformed or missing required delimiters.
    """
    stripped = content.strip()
    if not stripped.startswith("---"):
        raise ValueError("SKILL.md must start with --- frontmatter delimiter")

    # Find the closing --- delimiter (first line after the opener that is only dashes)
    match = re.search(r"\n^---\s*$", stripped, re.MULTILINE)
    if match is None:
        raise ValueError("SKILL.md frontmatter missing closing --- delimiter")

    frontmatter_str = stripped[3 : match.start()].strip()
    body = stripped[match.end() :].strip()

    try:
        result = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError:
        # AgentSkills spec recommends handling malformed YAML (e.g. unquoted colons in values)
        # to improve cross-client compatibility. See: agentskills.io/client-implementation/adding-skills-support
        logger.warning("YAML parse failed, retrying with colon-quoting fallback")
        fixed = _fix_yaml_colons(frontmatter_str)
        result = yaml.safe_load(fixed)

    frontmatter: dict[str, Any] = result if isinstance(result, dict) else {}
    return frontmatter, body


def _fix_yaml_colons(yaml_str: str) -> str:
    """Attempt to fix common YAML issues like unquoted colons in values.

    Wraps values containing colons in double quotes to handle cases like:
    ``description: Use this skill when: the user asks about PDFs``

    Args:
        yaml_str: The raw YAML string to fix.

    Returns:
        The fixed YAML string.
    """
    lines: list[str] = []
    for line in yaml_str.splitlines():
        # Match key: value where value contains another colon
        match = re.match(r"^(\s*\w[\w-]*):\s+(.+)$", line)
        if match:
            key, value = match.group(1), match.group(2)
            # If value contains a colon and isn't already quoted
            if ":" in value and not (value.startswith('"') or value.startswith("'")):
                line = f'{key}: "{value}"'
        lines.append(line)
    return "\n".join(lines)


def _validate_skill_name(name: str, dir_path: Path | None = None, *, strict: bool = False) -> None:
    """Validate a skill name per the AgentSkills.io specification.

    In lenient mode (default), logs warnings for cosmetic issues but does not raise.
    In strict mode, raises ValueError for any validation failure.

    Rules checked:
    - 1-64 characters long
    - Lowercase alphanumeric characters and hyphens only
    - Cannot start or end with a hyphen
    - No consecutive hyphens
    - Must match parent directory name (if loaded from disk)

    Args:
        name: The skill name to validate.
        dir_path: Optional path to the skill directory for name matching.
        strict: If True, raise ValueError on any issue. If False (default), log warnings.

    Raises:
        ValueError: If the skill name is empty, or if strict=True and any rule is violated.
    """
    if not name:
        raise ValueError("Skill name cannot be empty")

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        msg = "name=<%s> | skill name exceeds %d character limit"
        if strict:
            raise ValueError(msg % (name, _MAX_SKILL_NAME_LENGTH))
        logger.warning(msg, name, _MAX_SKILL_NAME_LENGTH)

    if not _SKILL_NAME_PATTERN.match(name):
        msg = (
            "name=<%s> | skill name should be 1-64 lowercase alphanumeric characters or hyphens, "
            "should not start/end with hyphen"
        )
        if strict:
            raise ValueError(msg % name)
        logger.warning(msg, name)

    if "--" in name:
        msg = "name=<%s> | skill name contains consecutive hyphens"
        if strict:
            raise ValueError(msg % name)
        logger.warning(msg, name)

    if dir_path is not None and dir_path.name != name:
        msg = "name=<%s>, directory=<%s> | skill name does not match parent directory name"
        if strict:
            raise ValueError(msg % (name, dir_path.name))
        logger.warning(msg, name, dir_path.name)


def _build_skill_from_frontmatter(
    frontmatter: dict[str, Any],
    body: str,
) -> Skill:
    """Build a Skill instance from parsed frontmatter and body.

    Args:
        frontmatter: Parsed YAML frontmatter dict.
        body: Markdown body content.

    Returns:
        A populated Skill instance.
    """
    # Parse allowed-tools (space-delimited string or YAML list)
    allowed_tools_raw = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    allowed_tools: list[str] | None = None
    if isinstance(allowed_tools_raw, str) and allowed_tools_raw.strip():
        allowed_tools = allowed_tools_raw.strip().split()
    elif isinstance(allowed_tools_raw, list):
        allowed_tools = [str(item) for item in allowed_tools_raw if item]

    # Parse metadata (nested mapping)
    metadata_raw = frontmatter.get("metadata", {})
    metadata: dict[str, Any] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): v for k, v in metadata_raw.items()}

    skill_license = frontmatter.get("license")
    compatibility = frontmatter.get("compatibility")

    return Skill(
        name=frontmatter["name"],
        description=frontmatter["description"],
        instructions=body,
        allowed_tools=allowed_tools,
        metadata=metadata,
        license=str(skill_license) if skill_license else None,
        compatibility=str(compatibility) if compatibility else None,
    )


@dataclass
class Skill:
    r"""Represents an agent skill with metadata and instructions.

    A skill encapsulates a set of instructions and metadata that can be
    dynamically loaded by an agent at runtime. Skills support progressive
    disclosure: metadata is shown upfront in the system prompt, and full
    instructions are loaded on demand via a tool.

    Skills can be created directly or via convenience classmethods::

        # From a skill directory on disk
        skill = Skill.from_file("./skills/my-skill")

        # From raw SKILL.md content
        skill = Skill.from_content("---\nname: my-skill\n...")

        # Load all skills from a parent directory
        skills = Skill.from_directory("./skills/")

        # From an HTTPS URL
        skill = Skill.from_url("https://example.com/SKILL.md")

    Attributes:
        name: Unique identifier for the skill (1-64 chars, lowercase alphanumeric + hyphens).
        description: Human-readable description of what the skill does.
        instructions: Full markdown instructions from the SKILL.md body.
        path: Filesystem path to the skill directory, if loaded from disk.
        allowed_tools: List of tool names the skill is allowed to use. (Experimental: not yet enforced)
        metadata: Additional key-value metadata from the SKILL.md frontmatter.
        license: License identifier (e.g., "Apache-2.0").
        compatibility: Compatibility information string.
    """

    name: str
    description: str
    instructions: str = ""
    path: Path | None = None
    allowed_tools: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    compatibility: str | None = None

    @classmethod
    def from_file(cls, skill_path: str | Path, *, strict: bool = False) -> Skill:
        """Load a single skill from a directory containing SKILL.md.

        Resolves the filesystem path, reads the file content, and delegates
        to ``from_content`` for parsing. After loading, sets the skill's
        ``path`` and validates the skill name against the parent directory.

        Args:
            skill_path: Path to the skill directory or the SKILL.md file itself.
            strict: If True, raise on any validation issue. If False (default), warn and load anyway.

        Returns:
            A Skill instance populated from the SKILL.md file.

        Raises:
            FileNotFoundError: If the path does not exist or SKILL.md is not found.
            ValueError: If the skill metadata is invalid.
        """
        skill_path = Path(skill_path).resolve()

        if skill_path.is_file() and skill_path.name.lower() == "skill.md":
            skill_md_path = skill_path
            skill_dir = skill_path.parent
        elif skill_path.is_dir():
            skill_dir = skill_path
            skill_md_path = _find_skill_md(skill_dir)
        else:
            raise FileNotFoundError(
                f"path=<{skill_path}> | skill path does not exist or is not a valid skill directory"
            )

        logger.debug("path=<%s> | loading skill", skill_md_path)

        content = skill_md_path.read_text(encoding="utf-8")
        skill = cls.from_content(content, strict=strict)

        # Set path and check directory name match (from_content already validated the name format)
        skill.path = skill_dir
        if skill_dir.name != skill.name:
            msg = "name=<%s>, directory=<%s> | skill name does not match parent directory name"
            if strict:
                raise ValueError(msg % (skill.name, skill_dir.name))
            logger.warning(msg, skill.name, skill_dir.name)

        logger.debug("name=<%s>, path=<%s> | skill loaded successfully", skill.name, skill.path)
        return skill

    @classmethod
    def from_content(cls, content: str, *, strict: bool = False) -> Skill:
        """Parse SKILL.md content into a Skill instance.

        This is a convenience method for creating a Skill from raw SKILL.md
        content (YAML frontmatter + markdown body) without requiring a file on
        disk.

        Example::

            content = '''---
            name: my-skill
            description: Does something useful
            ---
            # Instructions
            Follow these steps...
            '''
            skill = Skill.from_content(content)

        Args:
            content: Raw SKILL.md content with YAML frontmatter and markdown body.
            strict: If True, raise on any validation issue. If False (default), warn and load anyway.

        Returns:
            A Skill instance populated from the parsed content.

        Raises:
            ValueError: If the content is missing required fields or has invalid frontmatter.
        """
        frontmatter, body = _parse_frontmatter(content)

        name = frontmatter.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("SKILL.md content must have a 'name' field in frontmatter")

        description = frontmatter.get("description")
        if not isinstance(description, str) or not description:
            raise ValueError("SKILL.md content must have a 'description' field in frontmatter")

        _validate_skill_name(name, strict=strict)

        return _build_skill_from_frontmatter(frontmatter, body)

    @classmethod
    def from_url(cls, url: str, *, strict: bool = False) -> Skill:
        """Load a skill by fetching its SKILL.md content from an HTTPS URL.

        Fetches the raw SKILL.md content over HTTPS and parses it using
        :meth:`from_content`.  The URL must point directly to the raw
        file content (not an HTML page).

        Example::

            skill = Skill.from_url(
                "https://raw.githubusercontent.com/org/repo/main/SKILL.md"
            )

        Args:
            url: An ``https://`` URL pointing directly to raw SKILL.md content.
            strict: If True, raise on any validation issue. If False (default),
                warn and load anyway.

        Returns:
            A Skill instance populated from the fetched SKILL.md content.

        Raises:
            ValueError: If ``url`` is not an ``https://`` URL, or if its host
                (or any redirect target's host) resolves to a loopback,
                link-local, or other non-public address.
            RuntimeError: If the SKILL.md content cannot be fetched.
        """
        if not url.startswith("https://"):
            raise ValueError(f"url=<{url}> | not a valid HTTPS URL")

        logger.info("url=<%s> | fetching skill content", url)

        content = _fetch_url_content(url)

        return cls.from_content(content, strict=strict)

    @classmethod
    def from_directory(cls, skills_dir: str | Path, *, strict: bool = False) -> list[Skill]:
        """Load all skills from a parent directory containing skill subdirectories.

        Each subdirectory containing a SKILL.md file is treated as a skill.
        Subdirectories without SKILL.md are silently skipped.

        Args:
            skills_dir: Path to the parent directory containing skill subdirectories.
            strict: If True, raise on any validation issue. If False (default), warn and load anyway.

        Returns:
            List of Skill instances loaded from the directory.

        Raises:
            FileNotFoundError: If the skills directory does not exist.
        """
        skills_dir = Path(skills_dir).resolve()

        if not skills_dir.is_dir():
            raise FileNotFoundError(f"path=<{skills_dir}> | skills directory does not exist")

        skills: list[Skill] = []

        for child in sorted(skills_dir.iterdir()):
            if not child.is_dir():
                continue

            try:
                _find_skill_md(child)
            except FileNotFoundError:
                logger.debug("path=<%s> | skipping directory without SKILL.md", child)
                continue

            try:
                skill = cls.from_file(child, strict=strict)
                skills.append(skill)
            except (ValueError, FileNotFoundError) as e:
                logger.warning("path=<%s> | skipping skill due to error: %s", child, e)

        logger.debug("path=<%s>, count=<%d> | loaded skills from directory", skills_dir, len(skills))
        return skills
