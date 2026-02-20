"""SkillsPlugin for integrating AgentSkills.io skills into Strands agents.

This module provides the SkillsPlugin class that extends the Plugin base class
to add AgentSkills.io skill support. The plugin registers a tool for activating
skills, and injects skill metadata into the system prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...hooks.events import BeforeInvocationEvent
from ...plugins import Plugin, hook
from ...tools.decorator import tool
from .loader import load_skill, load_skills
from .skill import Skill

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

_STATE_KEY = "skills_plugin"


class SkillsPlugin(Plugin):
    """Plugin that integrates AgentSkills.io skills into a Strands agent.

    The SkillsPlugin extends the Plugin base class and provides:

    1. A ``skills`` tool that allows the agent to activate skills on demand
    2. System prompt injection of available skill metadata before each invocation
    3. Single active skill management (activating a new skill replaces the previous one)
    4. Session persistence of active skill state via ``agent.state``

    Skills can be provided as filesystem paths (to individual skill directories or
    parent directories containing multiple skills) or as pre-built ``Skill`` instances.

    Example:
        ```python
        from strands import Agent
        from strands.plugins.skills import Skill, SkillsPlugin

        # Load from filesystem
        plugin = SkillsPlugin(skills=["./skills/pdf-processing", "./skills/"])

        # Or provide Skill instances directly
        skill = Skill(name="my-skill", description="A custom skill", instructions="Do the thing")
        plugin = SkillsPlugin(skills=[skill])

        agent = Agent(plugins=[plugin])
        ```
    """

    @property
    def name(self) -> str:
        """A stable string identifier for the plugin."""
        return "skills"

    def __init__(self, skills: list[str | Path | Skill]) -> None:
        """Initialize the SkillsPlugin.

        Args:
            skills: List of skill sources. Each element can be:

                - A ``str`` or ``Path`` to a skill directory (containing SKILL.md)
                - A ``str`` or ``Path`` to a parent directory (containing skill subdirectories)
                - A ``Skill`` dataclass instance
        """
        self._skills: list[Skill] = self._resolve_skills(skills)
        self._active_skill: Skill | None = None
        self._agent: Agent | None = None
        self._original_system_prompt: str | None = None
        super().__init__()

    def init_plugin(self, agent: Agent) -> None:
        """Initialize the plugin with an agent instance.

        Registers the skills tool and hooks with the agent, then restores
        any persisted state from a previous session.

        Args:
            agent: The agent instance to extend with skills support.
        """
        self._agent = agent
        super().init_plugin(agent)
        self._restore_state()
        logger.debug("skill_count=<%d> | skills plugin initialized", len(self._skills))

    @tool
    def skills(self, skill_name: str) -> str:
        """Activate a skill to load its full instructions.

        Use this tool to load the complete instructions for a skill listed in
        the available_skills section of your system prompt. Activating a new
        skill replaces the previously active one.

        Args:
            skill_name: Name of the skill to activate.
        """
        if not skill_name:
            return "Error: skill_name is required."

        found = self._find_skill(skill_name)
        if found is None:
            available = ", ".join(s.name for s in self._skills)
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        self._active_skill = found
        self._persist_state()

        logger.debug("skill_name=<%s> | skill activated", skill_name)
        return found.instructions or f"Skill '{skill_name}' activated (no instructions available)."

    @hook
    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Inject skill metadata into the system prompt before each invocation.

        Captures the original system prompt on first call, then rebuilds the
        prompt with the skills XML block on each invocation.

        Args:
            event: The before-invocation event containing the agent reference.
        """
        agent = event.agent

        # Capture the original system prompt on first invocation
        if self._original_system_prompt is None:
            self._original_system_prompt = agent._system_prompt or ""

        if not self._skills:
            return

        skills_xml = self._generate_skills_xml()
        new_prompt = f"{self._original_system_prompt}\n\n{skills_xml}" if self._original_system_prompt else skills_xml

        agent._system_prompt = new_prompt
        agent._system_prompt_content = [{"text": new_prompt}]

    @property
    def available_skills(self) -> list[Skill]:
        """Get the list of available skills.

        Returns:
            A copy of the current skills list.
        """
        return list(self._skills)

    @available_skills.setter
    def available_skills(self, value: list[str | Path | Skill]) -> None:
        """Set the available skills, resolving paths as needed.

        Deactivates any currently active skill when skills are changed.

        Args:
            value: List of skill sources to resolve.
        """
        self._skills = self._resolve_skills(value)
        self._active_skill = None
        self._persist_state()

    @property
    def active_skill(self) -> Skill | None:
        """Get the currently active skill.

        Returns:
            The active Skill instance, or None if no skill is active.
        """
        return self._active_skill

    def _generate_skills_xml(self) -> str:
        """Generate the XML block listing available skills for the system prompt.

        Returns:
            XML-formatted string with skill metadata.
        """
        lines: list[str] = ["<available_skills>"]

        for skill in self._skills:
            lines.append("<skill>")
            lines.append(f"<name>{skill.name}</name>")
            lines.append(f"<description>{skill.description}</description>")
            lines.append("</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def _find_skill(self, skill_name: str) -> Skill | None:
        """Find a skill by name in the available skills list.

        Args:
            skill_name: The name of the skill to find.

        Returns:
            The matching Skill instance, or None if not found.
        """
        for skill in self._skills:
            if skill.name == skill_name:
                return skill
        return None

    def _resolve_skills(self, sources: list[str | Path | Skill]) -> list[Skill]:
        """Resolve a list of skill sources into Skill instances.

        Each source can be a Skill instance, a path to a skill directory,
        or a path to a parent directory containing multiple skills.

        Args:
            sources: List of skill sources to resolve.

        Returns:
            List of resolved Skill instances.
        """
        resolved: list[Skill] = []

        for source in sources:
            if isinstance(source, Skill):
                resolved.append(source)
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
                            resolved.append(load_skill(path))
                        except (ValueError, FileNotFoundError) as e:
                            logger.warning("path=<%s> | failed to load skill: %s", path, e)
                    else:
                        # Treat as parent directory containing skill subdirectories
                        resolved.extend(load_skills(path))
                elif path.is_file() and path.name.lower() == "skill.md":
                    try:
                        resolved.append(load_skill(path))
                    except (ValueError, FileNotFoundError) as e:
                        logger.warning("path=<%s> | failed to load skill: %s", path, e)

        logger.debug("source_count=<%d>, resolved_count=<%d> | skills resolved", len(sources), len(resolved))
        return resolved

    def _persist_state(self) -> None:
        """Persist the active skill name to agent state for session recovery."""
        if self._agent is None:
            return

        state_data: dict[str, Any] = {
            "active_skill_name": self._active_skill.name if self._active_skill else None,
        }
        self._agent.state.set(_STATE_KEY, state_data)

    def _restore_state(self) -> None:
        """Restore the active skill from agent state if available."""
        if self._agent is None:
            return

        state_data = self._agent.state.get(_STATE_KEY)
        if not isinstance(state_data, dict):
            return

        active_name = state_data.get("active_skill_name")
        if isinstance(active_name, str):
            self._active_skill = self._find_skill(active_name)
            if self._active_skill:
                logger.debug("skill_name=<%s> | restored active skill from state", active_name)
