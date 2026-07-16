"""AgentSkills plugin for integrating Agent Skills into Strands agents.

This module provides the AgentSkills class that extends the Plugin base class
to add Agent Skills support. The plugin injects skill metadata into the system
prompt and exposes tools for activating skills. Two ``injection_mode`` values
control how a skill's full instructions reach the model: ``"tool"`` returns them
as a tool result (progressive disclosure), while ``"system_prompt"`` delivers the
full skills block (metadata plus the instructions of activated skills) into the
per-call system prompt through an internal :class:`ContextInjector`, and exposes a
``skills`` tool with ``activate`` / ``deactivate`` actions to toggle them. The
``system_prompt`` delivery is ephemeral: the model sees the block on every call
while ``agent.system_prompt`` itself is never modified.

Filesystem skill sources are loaded through the agent's sandbox (host or
container) at ``init_agent`` time, not at construction, so each agent sees the
skills present on its own filesystem. Skill instances and ``https://`` URLs are
sandbox-independent and resolve eagerly at construction.

:meth:`Skill.from_url` is synchronous, so URLs resolve at construction and no
readiness barrier is needed. The observable effect is benign: URL skills are
available immediately after construction, while filesystem skills appear once
an agent loads them through its sandbox.
"""

from __future__ import annotations

import logging
import re
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from xml.sax.saxutils import escape

from ...hooks.events import BeforeInvocationEvent
from ...injection.types import InjectionContext
from ...plugins import Plugin, hook
from ...tools.decorator import tool
from ...types.content import SystemContentBlock
from ...types.tools import ToolContext
from ..context_injector import ContextInjector
from .skill import Skill

if TYPE_CHECKING:
    from ...agent.agent import Agent
    from ...sandbox import FileInfo, Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_STATE_KEY = "agent_skills"
_RESOURCE_DIRS = ("scripts", "references", "assets")
_DEFAULT_MAX_RESOURCE_FILES = 20
_MAX_RESOURCE_DEPTH = 3

SkillSource: TypeAlias = str | Path | Skill
"""A single skill source: path string, Path object, or Skill instance."""

SkillSources: TypeAlias = SkillSource | list[SkillSource]
"""One or more skill sources."""

SkillInjectionMode: TypeAlias = Literal["tool", "system_prompt"]
"""How activated skill instructions reach the model.

- ``"tool"``: skill metadata is injected into the system prompt and full instructions are
  returned on demand as the result of the ``skills`` tool (progressive disclosure).
- ``"system_prompt"``: the skills block — metadata plus the full instructions of *activated*
  skills — is delivered into the per-call system prompt through an internal
  :class:`ContextInjector` (ephemeral: ``agent.system_prompt`` is never modified). The agent
  toggles skills with the ``skills`` tool's ``activate`` / ``deactivate`` actions (or the
  matching programmatic methods) instead of receiving instructions as a tool result.
"""

_VALID_INJECTION_MODES: tuple[SkillInjectionMode, ...] = ("tool", "system_prompt")

_CLOSING_DELIMITER_RE = re.compile(r"</\s*(instructions|available_skills)\s*>", re.IGNORECASE)
"""Matches the closing delimiters of the elements wrapping injected instruction bodies,
including whitespace and case variants (``</instructions >``, ``</INSTRUCTIONS>``) that a
lenient parser would also treat as valid closers."""


def _neutralize_delimiters(text: str) -> str:
    """Neutralize closing delimiters inside a skill instruction body.

    Instruction bodies are injected verbatim -- matching ``tool`` mode, which returns them raw
    as a tool result -- so code samples containing ``<``, ``>``, or ``&`` reach the model
    unmodified. Only the closing sequences that could break out of the enclosing elements are
    defanged, including whitespace/case variants that a lenient parser would also accept as
    valid closers.

    Args:
        text: The raw instruction body.

    Returns:
        The body with ``</instructions>`` and ``</available_skills>`` sequences neutralized.
    """
    return _CLOSING_DELIMITER_RE.sub(lambda match: f"<\\/{match.group(1)}>", text)


def _normalize_sources(sources: SkillSources) -> list[SkillSource]:
    """Normalize a single source or list of sources into a list."""
    if isinstance(sources, list):
        return sources
    return [sources]


def _find_skill_md_name(entries: list[FileInfo]) -> str | None:
    """Find the SKILL.md filename among directory entries.

    Prefers ``SKILL.md`` over ``skill.md`` (matching :meth:`Skill.from_file`
    precedence). Returns ``None`` if neither is present.

    Args:
        entries: Directory entries from a sandbox ``list_files`` call.

    Returns:
        The SKILL.md filename, or ``None`` if not found.
    """
    for name in ("SKILL.md", "skill.md"):
        if any(not entry.is_dir and entry.name == name for entry in entries):
            return name
    return None


class AgentSkills(Plugin):
    """Plugin that integrates Agent Skills into a Strands agent.

    The AgentSkills plugin extends the Plugin base class and provides:

    1. Tools that allow the agent to activate skills on demand
    2. System prompt injection of available skill metadata before each invocation
    3. Session persistence of active skill state via ``agent.state``

    The plugin supports two ``injection_mode`` values that control how a skill's full
    instructions reach the model once activated:

    - ``"tool"`` (default): the agent calls the ``skills`` tool with a skill name and
      receives the full instructions as the tool result (progressive disclosure). This
      keeps the system prompt small and is ideal when many skills are available.
    - ``"system_prompt"``: the agent calls the ``skills`` tool with an ``activate`` or
      ``deactivate`` action to toggle skills on and off. The skills block — metadata plus
      the full instructions of active skills — is delivered into the per-call system
      prompt through an internal :class:`ContextInjector`, so the guidance persists
      across turns without re-reading a tool result. The delivery is ephemeral:
      ``agent.system_prompt`` is never modified, and the block is re-rendered from the
      current activation state on every model call. This suits long-running workflows
      where a skill should stay "loaded" for the remainder of the conversation.

    Skills can be provided as filesystem paths (to individual skill directories or
    parent directories containing multiple skills), ``https://`` URLs pointing to
    raw SKILL.md content, or as pre-built ``Skill`` instances.

    Filesystem paths are read through the agent's sandbox at ``init_agent`` time,
    so each agent loads the skills present on its own filesystem (host or
    container). Skill instances and URLs are sandbox-independent and resolve at
    construction. As a result, ``get_available_skills`` returns filesystem skills
    only when passed the agent they were loaded for.

    Example:
        ```python
        from strands import Agent
        from strands.vended_plugins.skills import Skill, AgentSkills

        # Load from filesystem
        plugin = AgentSkills(skills=["./skills/pdf-processing", "./skills/"])

        # Or provide Skill instances directly
        skill = Skill(name="my-skill", description="A custom skill", instructions="Do the thing")
        plugin = AgentSkills(skills=[skill])

        # Inject activated skill instructions into the system prompt instead of
        # returning them as a tool result
        plugin = AgentSkills(skills=["./skills/"], injection_mode="system_prompt")

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
        *,
        injection_mode: SkillInjectionMode = "tool",
    ) -> None:
        """Initialize the AgentSkills plugin.

        Args:
            skills: One or more skill sources. Can be a single value or a list. Each element can be:

                - A ``str`` or ``Path`` to a skill directory (containing SKILL.md)
                - A ``str`` or ``Path`` to a parent directory (containing skill subdirectories)
                - A ``Skill`` dataclass instance
                - An ``https://`` URL pointing directly to raw SKILL.md content
            state_key: Key used to store plugin state in ``agent.state``.
            max_resource_files: Maximum number of resource files to list in skill responses.
            strict: If True, raise on skill validation issues. If False (default), warn and load anyway.
            injection_mode: How activated skill instructions reach the model. ``"tool"`` (default)
                returns instructions as the result of the ``skills`` tool. ``"system_prompt"`` injects
                the instructions of activated skills directly into the system prompt; the ``skills``
                tool instead takes ``activate`` / ``deactivate`` actions to toggle them.

        Raises:
            ValueError: If ``injection_mode`` is not one of ``"tool"`` or ``"system_prompt"``.
        """
        if injection_mode not in _VALID_INJECTION_MODES:
            raise ValueError(f"injection_mode=<{injection_mode}> | must be one of {_VALID_INJECTION_MODES}")
        self._strict = strict
        self._state_key = state_key
        self._max_resource_files = max_resource_files
        self._injection_mode: SkillInjectionMode = injection_mode
        # Skill instances and URLs resolve now (both sandbox-independent and synchronous in
        # Python). Filesystem paths are deferred to init_agent, where the agent's sandbox is
        # available, so a path may resolve differently per agent (host vs. container).
        self._skills, self._skill_paths = self._resolve_skills(_normalize_sources(skills))
        # Per-agent full skill set (base skills + path-loaded skills from that agent's sandbox).
        # Per-agent map (WeakKeyDictionary) so a single plugin
        # instance can serve multiple agents without leaking references once an agent is collected.
        self._agent_skills: weakref.WeakKeyDictionary[Agent, dict[str, Skill]] = weakref.WeakKeyDictionary()
        super().__init__()
        # Both modes surface a single tool named ``skills``; only the variant matching the
        # active injection mode is exposed. ``tool`` mode uses the activate-and-return variant
        # (the ``skills`` method); ``system_prompt`` mode uses the action-based toggle variant
        # (the ``manage_skills`` method, published under the ``skills`` tool name). The variants
        # share a tool name, so filtering is by the defining method's name.
        wanted_method = self._tool_method_for_mode()
        self._tools = [t for t in self._tools if t.__name__ == wanted_method]
        # In system_prompt mode the skills block reaches the model through the injection
        # primitive rather than by mutating agent.system_prompt: an internal ContextInjector
        # appends the block to the per-call system prompt on every model call
        # (trigger="everyTurn", so activation via a tool call shows up on the very next
        # turn). Rendering happens at call time from the current activation state, so
        # toggles, session-restored state, and per-agent isolation all fall out of the
        # ephemeral delivery — there is nothing durable to remove, sweep, or desync.
        self._instruction_injector = (
            ContextInjector(
                self._render_skills_block,
                name=f"{self.name}:context-injector",
                trigger="everyTurn",
                location="systemPrompt",
            )
            if injection_mode == "system_prompt"
            else None
        )

    async def init_agent(self, agent: Agent) -> None:
        """Initialize the plugin with an agent instance.

        Loads any deferred filesystem skill paths through the agent's sandbox,
        building the agent's full skill set. Decorated hooks and tools are
        auto-registered by the plugin registry.

        Args:
            agent: The agent instance to extend with skills support.
        """
        await self._load_skill_paths(agent)
        if self._instruction_injector is not None:
            self._instruction_injector.init_agent(agent)
        skills = self._agent_skills.get(agent, self._skills)
        if not skills:
            logger.warning("no skills were loaded, the agent will have no skills available")
        logger.debug("skill_count=<%d> | skills plugin initialized", len(skills))

    @tool(context=True)
    async def skills(self, skill_name: str, tool_context: ToolContext) -> str:
        """Activate a skill to load its full instructions.

        Use this tool to load the complete instructions for a skill listed in
        the available_skills section of your system prompt.

        Args:
            skill_name: Name of the skill to activate, as listed in the ``<name>`` element of
                the available_skills section of your system prompt.
            tool_context: Injected by the framework. Not user-facing.
        """
        agent = tool_context.agent
        skills = self._skills_for(agent)

        if not skill_name:
            available = ", ".join(skills)
            return f"Error: skill_name is required. Available skills: {available}"

        found = skills.get(skill_name)
        if found is None:
            available = ", ".join(skills)
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        logger.debug("skill_name=<%s> | skill activated", skill_name)
        self._track_activated_skill(agent, skill_name)
        return await self._format_skill_response(found, agent.sandbox)

    @tool(name="skills", context=True)
    async def manage_skills(
        self,
        action: Literal["activate", "deactivate"],
        skill_name: str,
        tool_context: ToolContext,
    ) -> str:
        """Activate or deactivate a skill.

        Activating a skill loads its complete instructions into your system prompt, where
        they stay for the rest of the conversation until you deactivate it. Deactivate a
        skill that is no longer relevant to remove its instructions and free up context.

        Args:
            action: ``"activate"`` to load a skill's instructions into your system prompt,
                ``"deactivate"`` to remove a previously activated skill's instructions.
            skill_name: Name of the skill. For ``"activate"``, use the ``<name>`` element from
                the available_skills section of your system prompt; for ``"deactivate"``, it must
                match a currently active skill.
            tool_context: Injected by the framework. Not user-facing.
        """
        agent = tool_context.agent

        if action == "activate":
            skills = self._skills_for(agent)

            if not skill_name:
                available = ", ".join(skills)
                return f"Error: skill_name is required. Available skills: {available}"

            found = skills.get(skill_name)
            if found is None:
                available = ", ".join(skills)
                return f"Skill '{skill_name}' not found. Available skills: {available}"

            self.activate_skill_for(agent, skill_name)
            logger.debug("skill_name=<%s> | skill activated (system_prompt mode)", skill_name)

            if not found.instructions:
                return f"Skill '{skill_name}' activated (no instructions available)."

            parts = [
                f"Skill '{skill_name}' activated. Its instructions will appear in your system prompt "
                "from your next turn onward."
            ]
            if found.path is not None:
                resources = await self._list_skill_resources(agent.sandbox, str(found.path))
                if resources:
                    parts.append("Available resources:\n" + "\n".join(f"  {r}" for r in resources))
            return "\n".join(parts)

        if action == "deactivate":
            if not skill_name:
                active = ", ".join(self.get_activated_skills(agent))
                return f"Error: skill_name is required. Active skills: {active}"

            if skill_name not in self.get_activated_skills(agent):
                active = ", ".join(self.get_activated_skills(agent))
                return f"Skill '{skill_name}' is not active. Active skills: {active}"

            self.deactivate_skill_for(agent, skill_name)
            logger.debug("skill_name=<%s> | skill deactivated (system_prompt mode)", skill_name)
            return f"Skill '{skill_name}' deactivated. Its instructions were removed from your system prompt."

        return f"Error: unknown action '{action}'. Valid actions: activate, deactivate"

    @hook
    async def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Prepare skills before each invocation.

        On first invocation for an agent (or after ``set_available_skills`` reset
        the per-agent cache), loads that agent's deferred filesystem skill paths
        through its sandbox. In ``tool`` mode it then injects the metadata block into
        the agent's system prompt; in ``system_prompt`` mode delivery is handled
        ephemerally by the internal :class:`ContextInjector` on each model call.

        Args:
            event: The before-invocation event containing the agent reference.
        """
        agent = event.agent

        # Lazily load filesystem skills if this agent has not been initialized yet
        # Keeps skills correct
        # after set_available_skills, which clears the per-agent cache.
        if agent not in self._agent_skills:
            await self._load_skill_paths(agent)

        # tool mode injects the metadata listing durably into agent.system_prompt (the
        # pre-existing behaviour). system_prompt mode skips this entirely: the internal
        # ContextInjector delivers the block ephemerally on each model call instead.
        if self._injection_mode == "tool":
            self._inject_skills(agent)

    def _inject_skills(self, agent: Agent) -> None:
        """Inject the skills metadata block into the agent's system prompt (``tool`` mode).

        Removes the previously injected block (if any) and appends a fresh one, using the
        exact text recorded in agent state to find it. Agent state tracks the injected text
        per-agent, so a single plugin instance can be shared across multiple agents safely.

        When the agent has a structured system prompt (list of SystemContentBlock),
        the injection is done at the block level so that cache points and other
        structured blocks are preserved. When the agent has no system prompt at all,
        the skills block becomes the entire prompt.

        Args:
            agent: The agent whose system prompt to update.
        """
        state_data = agent.state.get(self._state_key)
        last_injected_xml = state_data.get("last_injected_xml") if isinstance(state_data, dict) else None

        skills_xml = self._generate_skills_xml(agent)
        content = agent.system_prompt_content

        if content is not None:
            # Content-block path: preserve cache points and other structured blocks
            blocks: list[SystemContentBlock] = list(content)
            if last_injected_xml is not None:
                injected_block: SystemContentBlock = {"text": last_injected_xml}
                if injected_block in blocks:
                    blocks.remove(injected_block)
                else:
                    logger.warning("unable to find previously injected skills XML in system prompt, re-appending")
            blocks.append({"text": skills_xml})
            self._set_state_field(agent, "last_injected_xml", skills_xml)
            agent.system_prompt = blocks
        else:
            # split_system_prompt() returns (None, None) only for a None system prompt, so
            # reaching this branch means the agent has no system prompt at all: there is
            # nothing to remove, and the skills block becomes the entire prompt.
            self._set_state_field(agent, "last_injected_xml", skills_xml)
            agent.system_prompt = skills_xml

    def _render_skills_block(self, context: InjectionContext) -> str:
        """Render the skills block for the internal ContextInjector (``system_prompt`` mode).

        Called before every model call; re-renders the block from the agent's current
        activation state so ``skills`` tool toggles from the previous turn are reflected on
        the next one.

        Args:
            context: The injection context for the current model call.

        Returns:
            The XML skills block to append to the per-call system prompt.
        """
        return self._generate_skills_xml(context.agent)

    def get_available_skills(self, agent: Agent | None = None) -> list[Skill]:
        """Get the list of available skills.

        Args:
            agent: When provided, returns that agent's full skill set (base skills
                plus filesystem skills loaded from its sandbox). When omitted,
                returns only the sandbox-independent base skills (Skill instances
                and URLs); filesystem skills are excluded because they are loaded
                per-agent at ``init_agent`` time.

        Returns:
            A copy of the resolved skills list.
        """
        skills = self._skills_for(agent) if agent is not None else self._skills
        return list(skills.values())

    def set_available_skills(self, skills: SkillSources) -> None:
        """Set the available skills, replacing any existing ones.

        Each element can be a ``Skill`` instance, a ``str`` or ``Path`` to a
        skill directory (containing SKILL.md), a ``str`` or ``Path`` to a
        parent directory containing skill subdirectories, or an ``https://``
        URL pointing directly to raw SKILL.md content.

        Filesystem paths are re-loaded per-agent on the next invocation. Note:
        this does not persist state or deactivate skills on any agent. Active
        skill state is managed per-agent and is not pruned: an activated name
        that no longer matches an available skill simply stops being rendered.

        Args:
            skills: One or more skill sources to resolve and set.
        """
        self._skills, self._skill_paths = self._resolve_skills(_normalize_sources(skills))
        # Drop per-agent caches so deferred paths reload against each agent's sandbox.
        self._agent_skills = weakref.WeakKeyDictionary()

    def _skills_for(self, agent: Agent | None) -> dict[str, Skill]:
        """Return the skill set for an agent, falling back to base skills.

        An agent appears in the per-agent map once :meth:`init_agent` (or the
        before-invocation hook) has loaded its filesystem paths. Before that (or
        for agents that only use Skill instances and URLs), the base skills are
        returned.

        Args:
            agent: The agent whose skill set to retrieve, or ``None``.

        Returns:
            The agent's full skill set, or the base skills.
        """
        if agent is None:
            return self._skills
        return self._agent_skills.get(agent, self._skills)

    async def _load_skill_paths(self, agent: Agent) -> None:
        """Load deferred filesystem skill paths through the agent's sandbox.

        Mirrors :meth:`Skill.from_file` / :meth:`Skill.from_directory`: a path may
        be a SKILL.md file, a skill directory, or a parent directory of skill
        subdirectories. Per-path failures are logged and skipped so one bad skill
        does not abort its siblings. The resulting full skill set is stored in the
        per-agent map.

        Args:
            agent: The agent whose sandbox is used to read skill files.
        """
        skills = dict(self._skills)
        if not self._skill_paths:
            self._agent_skills[agent] = skills
            return

        # Falls back to the default NotASandboxLocalEnvironment when the agent has no explicit sandbox.
        sandbox = agent.sandbox

        async def load_skill(skill_dir: str, md_path: str) -> None:
            # A failure (e.g. malformed SKILL.md) is logged and skipped so it does not abort
            # sibling skills, matching Skill.from_directory's per-skill resilience.
            try:
                skill = Skill.from_content(await sandbox.read_text(md_path), strict=self._strict)
                # Set the sandbox path as-is (not host-resolved): the file may live in a container.
                # Then replicate Skill.from_file's directory-name check, which from_content does not
                # perform (Python's from_content takes no path parameter).
                skill.path = Path(skill_dir)
                if skill.path.name != skill.name:
                    msg = "name=<%s>, directory=<%s> | skill name does not match parent directory name"
                    if self._strict:
                        raise ValueError(msg % (skill.name, skill.path.name))
                    logger.warning(msg, skill.name, skill.path.name)
                if skill.name in skills:
                    logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", skill.name)
                skills[skill.name] = skill
            except Exception as e:
                logger.warning("path=<%s> | failed to load skill: %s", skill_dir, e)

        for skill_path in self._skill_paths:
            skill_path_str = str(skill_path)
            try:
                entries = await sandbox.list_files(skill_path_str)
            except Exception:
                # Not a directory: accept a direct path to a SKILL.md file, as Skill.from_file does.
                if skill_path_str.lower().endswith("skill.md"):
                    slash_index = skill_path_str.rfind("/")
                    await load_skill("." if slash_index == -1 else skill_path_str[:slash_index], skill_path_str)
                else:
                    logger.warning("path=<%s> | skill source does not exist or is not a valid path", skill_path_str)
                continue

            md_name = _find_skill_md_name(entries)
            if md_name:
                await load_skill(skill_path_str, f"{skill_path_str}/{md_name}")
                continue

            # Parent directory: load each subdirectory that contains a skill.
            for entry in sorted((e for e in entries if e.is_dir), key=lambda e: e.name):
                child_dir = f"{skill_path_str}/{entry.name}"
                try:
                    child_entries = await sandbox.list_files(child_dir)
                except Exception as e:
                    logger.warning("path=<%s> | failed to load skill from sandbox: %s", child_dir, e)
                    continue
                child_md = _find_skill_md_name(child_entries)
                if child_md:
                    await load_skill(child_dir, f"{child_dir}/{child_md}")

        self._agent_skills[agent] = skills

    async def _format_skill_response(self, skill: Skill, sandbox: Sandbox) -> str:
        """Format the tool response when a skill is activated.

        Includes the full instructions along with relevant metadata fields
        and a listing of available resource files (scripts, references, assets)
        read through the sandbox for filesystem-based skills.

        Args:
            skill: The activated skill.
            sandbox: The agent's sandbox, used to list resource files.

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
            resources = await self._list_skill_resources(sandbox, str(skill.path))
            if resources:
                parts.append("\nAvailable resources:\n" + "\n".join(f"  {r}" for r in resources))

        return "\n".join(parts)

    async def _list_skill_resources(self, sandbox: Sandbox, skill_path: str) -> list[str]:
        """List resource files in a skill's optional directories through the sandbox.

        Scans the ``scripts/``, ``references/``, and ``assets/`` subdirectories
        for files, returning relative paths. Results are capped at
        ``max_resource_files`` to avoid context bloat.

        Args:
            sandbox: The agent's sandbox, used to list directory contents.
            skill_path: Path to the skill directory (a sandbox path).

        Returns:
            List of relative file paths (e.g. ``scripts/extract.py``).
        """
        files: list[str] = []

        # List a directory recursively through the sandbox, returning paths relative to its root.
        # Replaces Path.rglob, which has no sandbox equivalent.
        async def list_files_recursive(directory: str, depth: int = 0) -> list[str]:
            if depth >= _MAX_RESOURCE_DEPTH:
                return []
            result: list[str] = []
            for entry in await sandbox.list_files(directory):
                if entry.is_dir:
                    nested = await list_files_recursive(f"{directory}/{entry.name}", depth + 1)
                    result.extend(f"{entry.name}/{p}" for p in nested)
                else:
                    result.append(entry.name)
            return result

        for dir_name in _RESOURCE_DIRS:
            resource_dir = f"{skill_path}/{dir_name}"
            try:
                entries = await list_files_recursive(resource_dir)
            except Exception:
                # Missing directory (or unreadable): skip, as the optional dirs need not exist.
                continue

            for entry in sorted(entries):
                files.append(f"{dir_name}/{entry}")
                if len(files) >= self._max_resource_files:
                    files.append(f"... (truncated at {self._max_resource_files} files)")
                    return files

        return files

    def _generate_skills_xml(self, agent: Agent | None = None) -> str:
        """Generate the XML block listing available skills for the system prompt.

        When no skills are loaded, returns a block indicating no skills are available.
        Otherwise includes a ``<location>`` element for skills loaded from the filesystem,
        following the AgentSkills.io integration spec.

        In ``system_prompt`` mode, the block opens with a ``<usage>`` hint that tells the
        model how to toggle skills with the ``skills`` tool, and skills activated by the
        agent additionally carry an
        ``active="true"`` attribute and an ``<instructions>`` element with their full
        body, so the model can follow them directly from the system prompt. Instruction
        bodies are injected verbatim (matching what ``tool`` mode returns) with only the
        closing delimiters neutralized. In ``tool`` mode the output is the plain metadata
        listing, byte-identical to previous releases.

        Args:
            agent: When provided, lists that agent's full skill set (and, in
                ``system_prompt`` mode, renders its activation state); otherwise lists
                only the base skills.

        Returns:
            XML-formatted string with skill metadata.
        """
        skills = self._skills_for(agent)
        if not skills:
            return "<available_skills>\nNo skills are currently available.\n</available_skills>"

        embed_instructions = self._injection_mode == "system_prompt" and agent is not None
        active = set(self.get_activated_skills(agent)) if embed_instructions else set()

        lines: list[str] = ["<available_skills>"]
        if embed_instructions:
            lines.append(
                '<usage>Activate a skill with the skills tool (action="activate") to load its full '
                'instructions into this section; action="deactivate" removes them. Skills marked '
                'active="true" are in effect - follow their instructions.</usage>'
            )

        for skill in skills.values():
            is_active = skill.name in active
            lines.append('<skill active="true">' if is_active else "<skill>")
            lines.append(f"<name>{escape(skill.name)}</name>")
            lines.append(f"<description>{escape(skill.description)}</description>")
            if skill.path is not None:
                lines.append(f"<location>{escape(str(skill.path / 'SKILL.md'))}</location>")
            if is_active and skill.instructions:
                lines.append(f"<instructions>\n{_neutralize_delimiters(skill.instructions)}\n</instructions>")
            lines.append("</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def _tool_method_for_mode(self) -> str:
        """Return the name of the method backing the ``skills`` tool for the active mode.

        Returns:
            ``"skills"`` (activate-and-return) for ``tool`` mode, or ``"manage_skills"``
            (action-based activate/deactivate) for ``system_prompt`` mode. Both are
            published under the ``skills`` tool name.
        """
        if self._injection_mode == "system_prompt":
            return "manage_skills"
        return "skills"

    def activate_skill_for(self, agent: Agent, skill_name: str) -> None:
        """Activate a skill for an agent, persisting it in the activation list.

        Programmatic counterpart to the ``skills`` tool's ``activate`` action. In ``system_prompt`` mode
        the skill's instructions are injected into the system prompt on the next model call.
        In ``"tool"`` mode the activation is recorded but has no effect on the system prompt
        (instructions are only delivered as tool results there). The change is recorded in
        ``agent.state`` and persists across runs via the session manager.

        Args:
            agent: The agent to activate the skill for.
            skill_name: Name of the skill to activate.
        """
        self._track_activated_skill(agent, skill_name)

    def deactivate_skill_for(self, agent: Agent, skill_name: str) -> None:
        """Deactivate a skill for an agent, removing it from the activation list.

        Programmatic counterpart to the ``skills`` tool's ``deactivate`` action. Deactivating a skill that
        is not active is a no-op.

        Args:
            agent: The agent to deactivate the skill for.
            skill_name: Name of the skill to deactivate.
        """
        state_data = agent.state.get(self._state_key)
        activated: list[str] = state_data.get("activated_skills", []) if isinstance(state_data, dict) else []
        if skill_name in activated:
            activated = [name for name in activated if name != skill_name]
            self._set_state_field(agent, "activated_skills", activated)

    def _resolve_skills(self, sources: list[SkillSource]) -> tuple[dict[str, Skill], list[SkillSource]]:
        """Resolve sandbox-independent sources and collect deferred filesystem paths.

        Skill instances and ``https://`` URLs resolve immediately (both are
        synchronous and filesystem-independent). Filesystem paths (``str`` or
        ``Path``) are collected and returned unresolved, to be loaded per-agent
        through the sandbox in :meth:`_load_skill_paths`.

        Args:
            sources: List of skill sources to resolve.

        Returns:
            A tuple of (base skills mapping name to Skill, deferred filesystem paths).
        """
        resolved: dict[str, Skill] = {}
        skill_paths: list[SkillSource] = []

        for source in sources:
            if isinstance(source, Skill):
                if source.name in resolved:
                    logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", source.name)
                resolved[source.name] = source
            elif isinstance(source, str) and source.startswith("https://"):
                try:
                    skill = Skill.from_url(source, strict=self._strict)
                    if skill.name in resolved:
                        logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", skill.name)
                    resolved[skill.name] = skill
                except (RuntimeError, ValueError) as e:
                    logger.warning("url=<%s> | failed to load skill from URL: %s", source, e)
            else:
                # Filesystem path: defer to init_agent, where the agent's sandbox is available.
                skill_paths.append(source)

        logger.debug(
            "source_count=<%d>, resolved_count=<%d>, deferred_path_count=<%d> | skills resolved",
            len(sources),
            len(resolved),
            len(skill_paths),
        )
        return resolved, skill_paths

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
