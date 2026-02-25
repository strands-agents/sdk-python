"""Skill data model for the AgentSkills.io integration.

This module defines the Skill dataclass, which represents a single AgentSkills.io
skill with its metadata and instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    """Represents an agent skill with metadata and instructions.

    A skill encapsulates a set of instructions and metadata that can be
    dynamically loaded by an agent at runtime. Skills support progressive
    disclosure: metadata is shown upfront in the system prompt, and full
    instructions are loaded on demand via a tool.

    Attributes:
        name: Unique identifier for the skill (1-64 chars, lowercase alphanumeric + hyphens).
        description: Human-readable description of what the skill does.
        instructions: Full markdown instructions from the SKILL.md body.
        path: Filesystem path to the skill directory, if loaded from disk.
        allowed_tools: List of tool names the skill is allowed to use.
        metadata: Additional key-value metadata from the SKILL.md frontmatter.
        license: License identifier (e.g., "Apache-2.0").
        compatibility: Compatibility information string.
    """

    name: str
    description: str
    instructions: str = ""
    path: Path | None = None
    allowed_tools: list[str] | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    license: str | None = None
    compatibility: str | None = None
