"""Utilities for loading skills from HTTPS URLs.

This module provides functions to detect URL-type skill sources, resolve
GitHub web URLs to raw content URLs, and fetch SKILL.md content over HTTPS.
No git dependency or local caching is required.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Matches GitHub /tree/<ref>/path and /blob/<ref>/path URLs.
# e.g. /owner/repo/tree/main/skills/my-skill -> groups: (owner/repo, main, skills/my-skill)
_GITHUB_TREE_PATTERN = re.compile(r"^/([^/]+/[^/]+)/(?:tree|blob)/([^/]+)(?:/(.+?))?/?$")


def is_url(source: str) -> bool:
    """Check whether a skill source string looks like an HTTPS URL.

    Only ``https://`` URLs are supported; plaintext ``http://`` is rejected
    for security (MITM risk).

    Args:
        source: The skill source string to check.

    Returns:
        True if the source is an ``https://`` URL.
    """
    return source.startswith("https://")


def resolve_to_raw_url(url: str) -> str:
    """Resolve a GitHub web URL to a raw content URL for SKILL.md.

    Supports several GitHub URL patterns and converts them to
    ``raw.githubusercontent.com`` URLs::

        # Repository root (assumes HEAD and SKILL.md at root)
        https://github.com/owner/repo
            -> https://raw.githubusercontent.com/owner/repo/HEAD/SKILL.md

        # Repository root with @ref
        https://github.com/owner/repo@v1.0
            -> https://raw.githubusercontent.com/owner/repo/v1.0/SKILL.md

        # Tree URL pointing to a directory
        https://github.com/owner/repo/tree/main/skills/my-skill
            -> https://raw.githubusercontent.com/owner/repo/main/skills/my-skill/SKILL.md

        # Blob URL pointing to SKILL.md directly
        https://github.com/owner/repo/blob/main/skills/my-skill/SKILL.md
            -> https://raw.githubusercontent.com/owner/repo/main/skills/my-skill/SKILL.md

    Non-GitHub URLs and ``raw.githubusercontent.com`` URLs are returned as-is.

    Args:
        url: An HTTPS URL, possibly a GitHub web URL.

    Returns:
        A URL that can be fetched directly to obtain SKILL.md content.
    """
    # Already a raw URL — return as-is
    if "raw.githubusercontent.com" in url:
        return url

    # Not a github.com URL — return as-is (user provides a direct link)
    if not url.startswith("https://github.com"):
        return url

    # Parse the path portion
    path_start = len("https://github.com")
    path = url[path_start:]

    # Handle /tree/<ref>/path and /blob/<ref>/path
    tree_match = _GITHUB_TREE_PATTERN.match(path)
    if tree_match:
        owner_repo = tree_match.group(1)
        ref = tree_match.group(2)
        subpath = tree_match.group(3) or ""

        # If the URL points to a SKILL.md file directly, use it as-is
        if subpath.lower().endswith("skill.md"):
            return f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{subpath}"

        # Otherwise, assume it's a directory and append SKILL.md
        if subpath:
            return f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{subpath}/SKILL.md"
        return f"https://raw.githubusercontent.com/{owner_repo}/{ref}/SKILL.md"

    # Handle plain repo URL: /owner/repo or /owner/repo@ref
    # Strip leading slash and any trailing slash
    clean_path = path.strip("/")

    # Check for @ref suffix
    if "@" in clean_path:
        at_idx = clean_path.rfind("@")
        owner_repo = clean_path[:at_idx]
        ref = clean_path[at_idx + 1 :]
    else:
        owner_repo = clean_path
        ref = "HEAD"

    # Only match owner/repo (exactly two path segments)
    if owner_repo.count("/") == 1:
        return f"https://raw.githubusercontent.com/{owner_repo}/{ref}/SKILL.md"

    # Unrecognised GitHub URL pattern — return as-is
    return url


def fetch_skill_content(url: str) -> str:
    """Fetch SKILL.md content from an HTTPS URL.

    Uses ``urllib.request`` (stdlib) so no additional dependencies are needed.

    Args:
        url: The HTTPS URL to fetch.

    Returns:
        The response body as a string.

    Raises:
        RuntimeError: If the fetch fails (network error, 404, etc.).
    """
    resolved = resolve_to_raw_url(url)
    logger.info("url=<%s> | fetching skill content from %s", url, resolved)

    try:
        req = urllib.request.Request(resolved, headers={"User-Agent": "strands-agents-sdk"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"url=<{resolved}> | HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"url=<{resolved}> | failed to fetch skill: {e.reason}") from e
