"""Utilities for loading skills from remote Git repository URLs.

This module provides functions to detect URL-type skill sources, parse
optional version references, clone repositories with shallow depth, and
manage a local cache of cloned skill repositories.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "strands" / "skills"

# Patterns that indicate a string is a URL rather than a local path
_URL_PREFIXES = ("https://", "http://", "git@", "ssh://")

# Regex to strip .git suffix from URLs before ref parsing
_GIT_SUFFIX = re.compile(r"\.git$")

# Matches GitHub /tree/<ref> or /tree/<ref>/<path> (also /blob/)
# e.g. /owner/repo/tree/main/skills/my-skill -> groups: (/owner/repo, main, skills/my-skill)
_GITHUB_TREE_PATTERN = re.compile(r"^(/[^/]+/[^/]+)/(?:tree|blob)/([^/]+)(?:/(.+?))?/?$")


def is_url(source: str) -> bool:
    """Check whether a skill source string looks like a remote URL.

    Args:
        source: The skill source string to check.

    Returns:
        True if the source appears to be a URL.
    """
    return any(source.startswith(prefix) for prefix in _URL_PREFIXES)


def parse_url_ref(url: str) -> tuple[str, str | None, str | None]:
    """Parse a skill URL into a clone URL, optional Git ref, and optional subpath.

    Supports an ``@ref`` suffix for specifying a branch, tag, or commit::

        https://github.com/org/skill-repo@v1.0.0  -> (https://github.com/org/skill-repo, v1.0.0, None)
        https://github.com/org/skill-repo         -> (https://github.com/org/skill-repo, None, None)

    Also supports GitHub web URLs with ``/tree/<ref>/path`` ::

        https://github.com/org/repo/tree/main/skills/my-skill
            -> (https://github.com/org/repo, main, skills/my-skill)

    Args:
        url: The skill URL, optionally with an ``@ref`` suffix or ``/tree/`` path.

    Returns:
        Tuple of (clone_url, ref_or_none, subpath_or_none).
    """
    if url.startswith(("https://", "http://", "ssh://")):
        # Find the path portion after the host
        scheme_end = url.index("//") + 2
        host_end = url.find("/", scheme_end)
        if host_end == -1:
            return url, None, None

        path_part = url[host_end:]

        # Handle GitHub /tree/<ref>/path and /blob/<ref>/path URLs
        tree_match = _GITHUB_TREE_PATTERN.match(path_part)
        if tree_match:
            owner_repo = tree_match.group(1)
            ref = tree_match.group(2)
            subpath = tree_match.group(3) or None
            clone_url = url[:host_end] + owner_repo
            return clone_url, ref, subpath

        # Strip .git suffix before looking for @ref so that
        # "repo.git@v1" is handled correctly
        clean_path = _GIT_SUFFIX.sub("", path_part)
        had_git_suffix = clean_path != path_part

        if "@" in clean_path:
            at_idx = clean_path.rfind("@")
            ref = clean_path[at_idx + 1 :]
            base_path = clean_path[:at_idx]
            if had_git_suffix:
                base_path += ".git"
            return url[:host_end] + base_path, ref, None

        return url, None, None

    if url.startswith("git@"):
        # SSH format: git@host:owner/repo.git@ref
        # The first @ is part of the SSH URL format.
        first_at = url.index("@")
        rest = url[first_at + 1 :]

        clean_rest = _GIT_SUFFIX.sub("", rest)
        had_git_suffix = clean_rest != rest

        if "@" in clean_rest:
            at_idx = clean_rest.rfind("@")
            ref = clean_rest[at_idx + 1 :]
            base_rest = clean_rest[:at_idx]
            if had_git_suffix:
                base_rest += ".git"
            return url[: first_at + 1] + base_rest, ref, None

        return url, None, None

    return url, None, None


def cache_key(url: str, ref: str | None) -> str:
    """Generate a deterministic cache directory name from a URL and ref.

    Args:
        url: The clone URL.
        ref: The optional Git ref.

    Returns:
        A short hex digest suitable for use as a directory name.
    """
    key_input = f"{url}@{ref}" if ref else url
    return hashlib.sha256(key_input.encode()).hexdigest()[:16]


def clone_skill_repo(
    url: str,
    *,
    ref: str | None = None,
    subpath: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Clone a skill repository to a local cache directory.

    Uses ``git clone --depth 1`` for efficiency. If a ``ref`` is provided it
    is passed as ``--branch`` (works for branches and tags). Repositories are
    cached by a hash of (url, ref) so repeated loads are instant.

    If ``subpath`` is provided, the returned path points to that subdirectory
    within the cloned repository (useful for mono-repos containing skills in
    nested directories).

    Args:
        url: The Git clone URL.
        ref: Optional branch or tag to check out.
        subpath: Optional path within the repo to return (e.g. ``skills/my-skill``).
        cache_dir: Override the default cache directory
            (``~/.cache/strands/skills/``).

    Returns:
        Path to the cloned repository root, or to ``subpath`` within it.

    Raises:
        RuntimeError: If the clone fails or ``git`` is not installed.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(url, ref)
    target = cache_dir / key

    if not target.exists():
        logger.info("url=<%s>, ref=<%s> | cloning skill repository", url, ref)

        cmd: list[str] = ["git", "clone", "--depth", "1"]
        if ref:
            cmd.extend(["--branch", ref])
        cmd.extend([url, str(target)])

        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            # Clean up any partial clone
            if target.exists():
                shutil.rmtree(target)
            raise RuntimeError(
                f"url=<{url}>, ref=<{ref}> | failed to clone skill repository: {e.stderr.strip()}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError("git is required to load skills from URLs but was not found on PATH") from e
    else:
        logger.debug("url=<%s>, ref=<%s> | using cached skill at %s", url, ref, target)

    result = target / subpath if subpath else target

    if subpath and not result.is_dir():
        raise RuntimeError(f"url=<{url}>, subpath=<{subpath}> | subdirectory does not exist in cloned repository")

    logger.debug("url=<%s>, ref=<%s> | resolved to %s", url, ref, result)
    return result
