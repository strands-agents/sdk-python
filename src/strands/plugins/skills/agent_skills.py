"""AgentSkills plugin for integrating Agent Skills into Strands agents.

This module provides the AgentSkills class that extends the Plugin base class
to add Agent Skills support. The plugin registers a tool for activating
skills, and injects skill metadata into the system prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from xml.sax.saxutils import escape

from ...hooks.events import BeforeInvocationEvent
from ...plugins import Plugin, hook
from ...tools.decorator import tool
from ...types.tools import ToolContext
from .skill import Skill

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

_DEFAULT_STATE_KEY = "agent_skills"
_RESOURCE_DIRS = ("scripts", "references", "assets")
_DEFAULT_MAX_RESOURCE_FILES = 20

SkillSource: TypeAlias = str | Path | Skill
"""A single skill source: path string, Path object, or Skill instance."""

SkillSources: TypeAlias = SkillSource | list[SkillSource]
"""One or more skill sources."""


def _normalize_sources(sources: SkillSources) -> list[SkillSource]:
    """Normalize a single source or list of sources into a list."""
    if isinstance(sources, list):
        return sources
    return [sources]


class AgentSkills(Plugin):
    """Plugin that integrates Agent Skills into a Strands agent.

    The AgentSkills plugin extends the Plugin base class and provides:

    1. A ``skills`` tool that allows the agent to activate skills on demand
    2. System prompt injection of available skill metadata before each invocation
    3. Session persistence of active skill state via ``agent.state``

    Skills can be provided as filesystem paths (to individual skill directories or
    parent directories containing multiple skills) or as pre-built ``Skill`` instances.

    Example:
        ```python
        from strands import Agent
        from strands.plugins.skills import Skill, AgentSkills

        # Load from filesystem
        plugin = AgentSkills(skills=["./skills/pdf-processing", "./skills/"])

        # Or provide Skill instances directly
        skill = Skill(name="my-skill", description="A custom skill", instructions="Do the thing")
        plugin = AgentSkills(skills=[skill])

        agent = Agent(plugins=[plugin])
        ```
    """

    name = "agent_skills"

    def __init__(
        self,
        skills: SkillSources,
        state_key: str = _DEFAULT_STATE_KEY,
        max_resource_files: int = _DEFAULT_MAX_RESOURCE_FILES,
        strict: bool = False,
    ) -> None:
        """Initialize the AgentSkills plugin.

        Args:
            skills: One or more skill sources. Can be a single value or a list. Each element can be:

                - A ``str`` or ``Path`` to a skill directory (containing SKILL.md)
                - A ``str`` or ``Path`` to a parent directory (containing skill subdirectories)
                - A ``Skill`` dataclass instance
            state_key: Key used to store plugin state in ``agent.state``.
            max_resource_files: Maximum number of resource files to list in skill responses.
            strict: If True, raise on skill validation issues. If False (default), warn and load anyway.
        """
        self._strict = strict
        self._skills: dict[str, Skill] = self._resolve_skills(_normalize_sources(skills))
        self._state_key = state_key
        self._max_resource_files = max_resource_files
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Initialize the plugin with an agent instance.

        Decorated hooks and tools are auto-registered by the plugin registry.

        Args:
            agent: The agent instance to extend with skills support.
        """
        if not self._skills:
            logger.warning("no skills were loaded, the agent will have no skills available")
        logger.debug("skill_count=<%d> | skills plugin initialized", len(self._skills))

    @tool(context=True)
    def skills(self, skill_name: str, tool_context: ToolContext) -> str:  # noqa: D417
        """Activate a skill to load its full instructions.

        Use this tool to load the complete instructions for a skill listed in
        the available_skills section of your system prompt.

        Args:
            skill_name: Name of the skill to activate.
        """
        if not skill_name:
            available = ", ".join(self._skills)
            return f"Error: skill_name is required. Available skills: {available}"

        found = self._skills.get(skill_name)
        if found is None:
            available = ", ".join(self._skills)
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        logger.debug("skill_name=<%s> | skill activated", skill_name)
        self._track_activated_skill(tool_context.agent, skill_name)
        return self._format_skill_response(found)

    @hook
    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Inject skill metadata into the system prompt before each invocation.

        Removes the previously injected XML block (if any) via exact string
        replacement, then appends a fresh one. Uses agent state to track the
        injected XML per-agent, so a single plugin instance can be shared
        across multiple agents safely.

        Args:
            event: The before-invocation event containing the agent reference.
        """
        agent = event.agent

        current_prompt = agent.system_prompt or ""

        # Remove the previously injected XML block by exact match
        state_data = agent.state.get(self._state_key)
        last_injected_xml = state_data.get("last_injected_xml") if isinstance(state_data, dict) else None
        if last_injected_xml is not None:
            if last_injected_xml in current_prompt:
                current_prompt = current_prompt.replace(last_injected_xml, "")
            else:
                logger.warning("unable to find previously injected skills XML in system prompt, re-appending")

        skills_xml = self._generate_skills_xml()
        injection = f"\n\n{skills_xml}"
        new_prompt = f"{current_prompt}{injection}" if current_prompt else skills_xml

        new_injected_xml = injection if current_prompt else skills_xml
        self._set_state_field(agent, "last_injected_xml", new_injected_xml)
        agent.system_prompt = new_prompt

    def get_available_skills(self) -> list[Skill]:
        """Get the list of available skills.

        Returns:
            A copy of the current skills list.
        """
        return list(self._skills.values())

    def set_available_skills(self, skills: SkillSources) -> None:
        """Set the available skills, replacing any existing ones.

        Each element can be a ``Skill`` instance, a ``str`` or ``Path`` to a
        skill directory (containing SKILL.md), or a ``str`` or ``Path`` to a
        parent directory containing skill subdirectories.

        Note: this does not persist state or deactivate skills on any agent.
        Active skill state is managed per-agent and will be reconciled on the
        next tool call or invocation.

        Args:
            skills: One or more skill sources to resolve and set.
        """
        self._skills = self._resolve_skills(_normalize_sources(skills))


    def _format_skill_response(self, skill: Skill) -> str:
        """Format the tool response when a skill is activated.

        Includes the full instructions along with relevant metadata fields
        and a listing of available resource files (scripts, references, assets)
        for filesystem-based skills.

        Args:
            skill: The activated skill.

        Returns:
            Formatted string with skill instructions and metadata.
        """
        if not skill.instructions:
            return f"Skill '{skill.name}' activated (no instructions available)."

        parts: list[str] = [skill.instructions]

        metadata_lines: list[str] = []
        if skill.allowed_tools:
            metadata_lines.append(f"Allowed tools: {', '.join(skill.allowed_tools)}")
        if skill.compatibility:
            metadata_lines.append(f"Compatibility: {skill.compatibility}")
        if skill.path is not None:
            metadata_lines.append(f"Location: {skill.path / 'SKILL.md'}")

        if metadata_lines:
            parts.append("\n---\n" + "\n".join(metadata_lines))

        if skill.path is not None:
            resources = self._list_skill_resources(skill.path)
            if resources:
                parts.append("\nAvailable resources:\n" + "\n".join(f"  {r}" for r in resources))

        return "\n".join(parts)

    def _list_skill_resources(self, skill_path: Path) -> list[str]:
        """List resource files in a skill's optional directories.

        Scans the ``scripts/``, ``references/``, and ``assets/`` subdirectories
        for files, returning relative paths. Results are capped at
        ``max_resource_files`` to avoid context bloat.

        Args:
            skill_path: Path to the skill directory.

        Returns:
            List of relative file paths (e.g. ``scripts/extract.py``).
        """
        files: list[str] = []

        for dir_name in _RESOURCE_DIRS:
            resource_dir = skill_path / dir_name
            if not resource_dir.is_dir():
                continue

            for file_path in sorted(resource_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                files.append(file_path.relative_to(skill_path).as_posix())
                if len(files) >= self._max_resource_files:
                    files.append(f"... (truncated at {self._max_resource_files} files)")
                    return files

        return files

    def _generate_skills_xml(self) -> str:
        """Generate the XML block listing available skills for the system prompt.

        When no skills are loaded, returns a block indicating no skills are available.
        Otherwise includes a ``<location>`` element for skills loaded from the filesystem,
        following the AgentSkills.io integration spec.

        Returns:
            XML-formatted string with skill metadata.
        """
        if not self._skills:
            return "<available_skills>\nNo skills are currently available.\n</available_skills>"

        lines: list[str] = ["<available_skills>"]

        for skill in self._skills.values():
            lines.append("<skill>")
            lines.append(f"<name>{escape(skill.name)}</name>")
            lines.append(f"<description>{escape(skill.description)}</description>")
            if skill.path is not None:
                lines.append(f"<location>{escape(str(skill.path / 'SKILL.md'))}</location>")
            lines.append("</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def _resolve_skills(self, sources: list[SkillSource]) -> dict[str, Skill]:
        """Resolve a list of skill sources into Skill instances.

        Each source can be a Skill instance, a path to a skill directory,
        or a path to a parent directory containing multiple skills.

        Args:
            sources: List of skill sources to resolve.

        Returns:
            Dict mapping skill names to Skill instances.
        """
        resolved: dict[str, Skill] = {}

        for source in sources:
            if isinstance(source, Skill):
                if source.name in resolved:
                    logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", source.name)
                resolved[source.name] = source
            else:
                path = Path(source).resolve()
                if not path.exists():
                    logger.warning("path=<%s> | skill source path does not exist, skipping", path)
                    continue

                if path.is_dir():
                    # Check if this directory itself is a skill (has SKILL.md)
                    has_skill_md = (path / "SKILL.md").is_file() or (path / "skill.md").is_file()

                    if has_skill_md:
                        try:
                            skill = Skill.from_file(path, strict=self._strict)
                            if skill.name in resolved:
                                logger.warning(
                                    "name=<%s> | duplicate skill name, overwriting previous skill", skill.name
                                )
                            resolved[skill.name] = skill
                        except (ValueError, FileNotFoundError) as e:
                            logger.warning("path=<%s> | failed to load skill: %s", path, e)
                    else:
                        # Treat as parent directory containing skill subdirectories
                        for skill in Skill.from_directory(path, strict=self._strict):
                            if skill.name in resolved:
                                logger.warning(
                                    "name=<%s> | duplicate skill name, overwriting previous skill", skill.name
                                )
                            resolved[skill.name] = skill
                elif path.is_file() and path.name.lower() == "skill.md":
                    try:
                        skill = Skill.from_file(path, strict=self._strict)
                        if skill.name in resolved:
                            logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", skill.name)
                        resolved[skill.name] = skill
                    except (ValueError, FileNotFoundError) as e:
                        logger.warning("path=<%s> | failed to load skill: %s", path, e)

        logger.debug("source_count=<%d>, resolved_count=<%d> | skills resolved", len(sources), len(resolved))
        return resolved

    def _set_state_field(self, agent: Agent, key: str, value: Any) -> None:
        """Set a single field in the plugin's agent state dict.

        Args:
            agent: The agent whose state to update.
            key: The state field key.
            value: The value to set.

        Raises:
            TypeError: If the existing state value is not a dict.
        """
        state_data = agent.state.get(self._state_key)
        if state_data is not None and not isinstance(state_data, dict):
            raise TypeError(f"expected dict for state key '{self._state_key}', got {type(state_data).__name__}")
        if state_data is None:
            state_data = {}
        state_data[key] = value
        agent.state.set(self._state_key, state_data)

    def _track_activated_skill(self, agent: Agent, skill_name: str) -> None:
        """Record a skill activation in agent state.

        Maintains an ordered list of activated skill names (most recent last),
        without duplicates.

        Args:
            agent: The agent whose state to update.
            skill_name: Name of the activated skill.
        """
        state_data = agent.state.get(self._state_key)
        activated: list[str] = state_data.get("activated_skills", []) if isinstance(state_data, dict) else []
        if skill_name in activated:
            activated.remove(skill_name)
        activated.append(skill_name)
        self._set_state_field(agent, "activated_skills", activated)

    def get_activated_skills(self, agent: Agent) -> list[str]:
        """Get the list of skills activated by this agent.

        Returns skill names in activation order (most recent last).

        Args:
            agent: The agent to query.

        Returns:
            List of activated skill names.
        """
        state_data = agent.state.get(self._state_key)
        if isinstance(state_data, dict):
            return list(state_data.get("activated_skills", []))
        return []
