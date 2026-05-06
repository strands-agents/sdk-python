"""Skill data model and loading utilities for AgentSkills.io skills.

This module defines the Skill dataclass and provides classmethods for
discovering, parsing, and loading skills from the filesystem, raw content,
HTTPS URLs, or Amazon S3. Skills are directories containing a SKILL.md file
with YAML frontmatter metadata and markdown instructions.
"""

from __future__ import annotations

import logging
import re
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import yaml

logger = logging.getLogger(__name__)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_SKILL_NAME_LENGTH = 64
_S3_MIRROR_MAX_WORKERS = 4
_S3_MIRROR_CACHE: dict[tuple[str, str | None], Path] = {}


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


def _mirror_skills_from_s3(
    bucket: str,
    prefix: str | None = None,
    *,
    s3_client: Any = None,
    local_dir: str | Path | None = None,
) -> Path | None:
    """Mirror skill directories from S3 to a local filesystem path.

    Discovers all SKILL.md files under the given prefix, then downloads each
    skill directory (SKILL.md + subdirectories like scripts/, references/,
    assets/) in parallel using a thread pool.

    Results are cached per (bucket, prefix) pair for the lifetime of the
    process. Subsequent calls with the same arguments return the previously
    mirrored directory without re-downloading.

    Args:
        bucket: S3 bucket name.
        prefix: Optional key prefix to scope the scan (e.g. "agents/orchestrator/").
            If None, scans the entire bucket.
        s3_client: Optional pre-configured boto3 S3 client. If None, a default
            client is created via ``boto3.client("s3")``.
        local_dir: Optional local directory to mirror skills into. If None, a
            temporary directory is created.

    Returns:
        Path to the local directory containing mirrored skill subdirectories,
        or None if no skills were found.
    """
    cache_key = (bucket, prefix)
    if cache_key in _S3_MIRROR_CACHE:
        logger.debug("bucket=<%s>, prefix=<%s> | s3 mirror cache hit", bucket, prefix or "")
        return _S3_MIRROR_CACHE[cache_key]

    if s3_client is None:
        s3_client = boto3.client("s3")

    # Normalize prefix
    normalized_prefix = ""
    if prefix:
        normalized_prefix = prefix.rstrip("/") + "/"

    # Discover all objects under the prefix
    objects = _s3_list_all_objects(s3_client, bucket, normalized_prefix)

    if not objects:
        logger.warning("bucket=<%s>, prefix=<%s> | no objects found", bucket, normalized_prefix)
        return None

    # Find skill directories (those containing SKILL.md)
    skill_dirs = _s3_find_skill_directories(objects, normalized_prefix)

    if not skill_dirs:
        logger.warning("bucket=<%s>, prefix=<%s> | no SKILL.md files found", bucket, normalized_prefix)
        return None

    logger.info(
        "bucket=<%s>, prefix=<%s>, count=<%d> | found skills: %s",
        bucket,
        normalized_prefix,
        len(skill_dirs),
        ", ".join(skill_dirs),
    )

    dest = _s3_resolve_local_dir(local_dir)
    download_tasks = _s3_build_download_tasks(objects, skill_dirs, normalized_prefix)
    _s3_download_parallel(s3_client, bucket, download_tasks, dest)

    _S3_MIRROR_CACHE[cache_key] = dest
    return dest


def _s3_resolve_local_dir(local_dir: str | Path | None) -> Path:
    """Resolve or create the local directory for S3 mirroring."""
    if local_dir is None:
        path = Path(tempfile.mkdtemp(prefix="strands-s3-skills-"))
    else:
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
    return path


def _s3_list_all_objects(s3_client: Any, bucket: str, prefix: str) -> list[str]:
    """List all object keys under a prefix, handling pagination."""
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return keys


def _s3_find_skill_directories(object_keys: list[str], prefix: str) -> list[str]:
    """Find skill directory names (relative to prefix) that contain SKILL.md.

    Returns a sorted list of skill directory names (e.g. ["code-review", "pdf-processing"]).
    """
    skill_dirs: list[str] = []

    for key in object_keys:
        relative = key[len(prefix) :]
        parts = relative.split("/")

        # We expect: <skill-name>/SKILL.md (or skill.md)
        if len(parts) == 2 and parts[1].lower() == "skill.md":
            skill_dirs.append(parts[0])

    return sorted(set(skill_dirs))


def _s3_build_download_tasks(
    object_keys: list[str],
    skill_dirs: list[str],
    prefix: str,
) -> list[tuple[str, str]]:
    """Build a list of (s3_key, relative_local_path) tuples to download.

    Only includes objects that belong to a discovered skill directory.
    """
    skill_dir_set = set(skill_dirs)
    tasks: list[tuple[str, str]] = []

    for key in object_keys:
        relative = key[len(prefix) :]
        parts = relative.split("/")

        if parts[0] in skill_dir_set and len(parts) >= 2:
            if not key.endswith("/"):
                tasks.append((key, relative))

    return tasks


def _s3_download_parallel(
    s3_client: Any,
    bucket: str,
    tasks: list[tuple[str, str]],
    dest: Path,
) -> None:
    """Download files from S3 in parallel using a thread pool."""
    if not tasks:
        return

    def _download_one(task: tuple[str, str]) -> str:
        s3_key, relative_path = task
        local_path = dest / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, str(local_path))
        return relative_path

    with ThreadPoolExecutor(max_workers=_S3_MIRROR_MAX_WORKERS) as executor:
        futures = {executor.submit(_download_one, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                path = future.result()
                logger.debug("s3_path=<%s> | downloaded", path)
            except Exception:
                s3_key, _ = futures[future]
                logger.warning("bucket=<%s>, key=<%s> | failed to download", bucket, s3_key)
                raise


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
            ValueError: If ``url`` is not an ``https://`` URL.
            RuntimeError: If the SKILL.md content cannot be fetched.
        """
        if not url.startswith("https://"):
            raise ValueError(f"url=<{url}> | not a valid HTTPS URL")

        logger.info("url=<%s> | fetching skill content", url)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strands-agents-sdk"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
                content: str = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"url=<{url}> | HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"url=<{url}> | failed to fetch skill: {e.reason}") from e

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

    @classmethod
    def from_s3(
        cls,
        bucket: str,
        prefix: str | None = None,
        *,
        s3_client: Any = None,
        local_dir: str | Path | None = None,
        strict: bool = False,
    ) -> list[Skill]:
        """Load skills from an Amazon S3 bucket.

        Scans ``bucket`` (under ``prefix`` if given) for directories containing
        a SKILL.md file, mirrors each skill directory (including subdirectories
        like scripts/, references/, assets/) to a local path, then delegates to
        :meth:`from_directory` for parsing.

        Results are cached per (bucket, prefix) pair for the lifetime of the
        process — subsequent calls with the same arguments return the cached
        list without re-downloading.

        Example::

            from strands.vended_plugins.skills import Skill

            # Load all skills under a prefix
            skills = Skill.from_s3("my-bucket", prefix="agents/director/")

            # Load from bucket root with a custom S3 client
            import boto3
            s3 = boto3.client("s3", region_name="eu-west-1")
            skills = Skill.from_s3("my-bucket", s3_client=s3)

        Args:
            bucket: S3 bucket name.
            prefix: Optional key prefix to scope the scan (e.g. "agents/director/").
                If None, scans the entire bucket.
            s3_client: Optional pre-configured boto3 S3 client. If None, a default
                client is created via ``boto3.client("s3")``.
            local_dir: Optional local directory to mirror skills into. If None, a
                temporary directory is created.
            strict: If True, raise on skill validation issues. If False (default),
                warn and load anyway. Passed through to :meth:`from_directory`.

        Returns:
            List of Skill instances loaded from S3.
        """
        local_path = _mirror_skills_from_s3(
            bucket=bucket,
            prefix=prefix,
            s3_client=s3_client,
            local_dir=local_dir,
        )

        if local_path is None:
            return []

        return cls.from_directory(local_path, strict=strict)
