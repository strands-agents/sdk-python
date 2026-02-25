"""Skill loading and parsing utilities for AgentSkills.io skills.

This module provides functions for discovering, parsing, and loading skills
from the filesystem. Skills are directories containing a SKILL.md file with
YAML frontmatter metadata and markdown instructions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .skill import Skill

logger = logging.getLogger(__name__)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_SKILL_NAME_LENGTH = 64


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


def _parse_yaml(yaml_text: str) -> dict[str, Any]:
    """Parse YAML text into a dictionary.

    Args:
        yaml_text: YAML-formatted text to parse.

    Returns:
        Dictionary of parsed key-value pairs.
    """
    result = yaml.safe_load(yaml_text)
    return result if isinstance(result, dict) else {}


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

    frontmatter = _parse_yaml(frontmatter_str)
    return frontmatter, body


def _validate_skill_name(name: str, dir_path: Path | None = None) -> None:
    """Validate a skill name per the AgentSkills.io specification.

    Rules:
    - 1-64 characters long
    - Lowercase alphanumeric characters and hyphens only
    - Cannot start or end with a hyphen
    - No consecutive hyphens
    - Must match parent directory name (if loaded from disk)

    Args:
        name: The skill name to validate.
        dir_path: Optional path to the skill directory for name matching.

    Raises:
        ValueError: If the skill name is invalid.
    """
    if not name:
        raise ValueError("Skill name cannot be empty")

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        raise ValueError(f"name=<{name}> | skill name exceeds {_MAX_SKILL_NAME_LENGTH} character limit")

    if not _SKILL_NAME_PATTERN.match(name):
        raise ValueError(
            f"name=<{name}> | skill name must be 1-64 lowercase alphanumeric characters or hyphens, "
            "cannot start/end with hyphen"
        )

    if "--" in name:
        raise ValueError(f"name=<{name}> | skill name cannot contain consecutive hyphens")

    if dir_path is not None and dir_path.name != name:
        raise ValueError(f"name=<{name}>, directory=<{dir_path.name}> | skill name must match parent directory name")


def load_skill(skill_path: str | Path) -> Skill:
    """Load a single skill from a directory containing SKILL.md.

    Args:
        skill_path: Path to the skill directory or the SKILL.md file itself.

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
        raise FileNotFoundError(f"path=<{skill_path}> | skill path does not exist or is not a valid skill directory")

    logger.debug("path=<%s> | loading skill", skill_md_path)

    content = skill_md_path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content)

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"path=<{skill_md_path}> | SKILL.md must have a 'name' field in frontmatter")

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError(f"path=<{skill_md_path}> | SKILL.md must have a 'description' field in frontmatter")

    _validate_skill_name(name, skill_dir)

    # Parse allowed-tools (space-delimited string or YAML list)
    allowed_tools_raw = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    allowed_tools: list[str] | None = None
    if isinstance(allowed_tools_raw, str) and allowed_tools_raw.strip():
        allowed_tools = allowed_tools_raw.strip().split()
    elif isinstance(allowed_tools_raw, list):
        allowed_tools = [str(item) for item in allowed_tools_raw if item]

    # Parse metadata (nested mapping)
    metadata_raw = frontmatter.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    skill_license = frontmatter.get("license")
    compatibility = frontmatter.get("compatibility")

    skill = Skill(
        name=name,
        description=description,
        instructions=body,
        path=skill_dir,
        allowed_tools=allowed_tools,
        metadata=metadata,
        license=str(skill_license) if skill_license else None,
        compatibility=str(compatibility) if compatibility else None,
    )

    logger.debug("name=<%s>, path=<%s> | skill loaded successfully", skill.name, skill.path)
    return skill


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """Load all skills from a parent directory containing skill subdirectories.

    Each subdirectory containing a SKILL.md file is treated as a skill.
    Subdirectories without SKILL.md are silently skipped.

    Args:
        skills_dir: Path to the parent directory containing skill subdirectories.

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
            skill = load_skill(child)
            skills.append(skill)
        except (ValueError, FileNotFoundError) as e:
            logger.warning("path=<%s> | skipping skill due to error: %s", child, e)

    logger.debug("path=<%s>, count=<%d> | loaded skills from directory", skills_dir, len(skills))
    return skills
