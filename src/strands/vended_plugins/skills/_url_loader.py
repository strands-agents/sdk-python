"""Utilities for loading skills from HTTPS URLs.

This module provides functions to detect URL-type skill sources and
fetch SKILL.md content over HTTPS.  No git dependency, local caching,
or URL resolution is required — callers provide a direct URL to the
raw SKILL.md content.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


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


def fetch_skill_content(url: str) -> str:
    """Fetch SKILL.md content from an HTTPS URL.

    Uses ``urllib.request`` (stdlib) so no additional dependencies are needed.

    Args:
        url: The HTTPS URL to fetch.  Must point directly to the raw
            SKILL.md content (for example,
            ``https://raw.githubusercontent.com/org/repo/main/SKILL.md``).

    Returns:
        The response body as a string.

    Raises:
        ValueError: If ``url`` is not an ``https://`` URL.
        RuntimeError: If the fetch fails (network error, 404, etc.).
    """
    if not url.startswith("https://"):
        raise ValueError(f"url=<{url}> | only https:// URLs are supported")

    logger.info("url=<%s> | fetching skill content", url)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "strands-agents-sdk"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            content: str = response.read().decode("utf-8")
            return content
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"url=<{url}> | HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"url=<{url}> | failed to fetch skill: {e.reason}") from e
