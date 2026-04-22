"""Tests for sandbox-based skill loading in the AgentSkills plugin.

Tests cover:
- Skill.from_sandbox() — loading a single skill from sandbox
- Skill.from_sandbox_directory() — loading multiple skills from sandbox
- AgentSkills with "sandbox:/path" sources — deferred loading in init_agent
- Mixed local + sandbox sources
- Error handling (missing files, invalid content, sandbox failures)
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from strands.hooks.registry import HookRegistry
from strands.sandbox.base import FileInfo, Sandbox
from strands.types.tools import ToolContext
from strands.vended_plugins.skills.agent_skills import AgentSkills
from strands.vended_plugins.skills.skill import Skill

# --- Helpers ---

SKILL_CONTENT = """---
name: sandbox-skill
description: A skill loaded from sandbox
allowed-tools: shell editor
---
# Sandbox Skill Instructions

Follow these steps to do the thing.
"""

SKILL_B_CONTENT = """---
name: another-skill
description: Another sandbox skill
---
# Another Skill

More instructions.
"""

INVALID_SKILL_CONTENT = """---
description: Missing name field
---
# Broken
"""


def _make_mock_sandbox(files: dict[str, str | bytes] | None = None, dirs: dict[str, list[FileInfo]] | None = None):
    """Create a mock Sandbox with configurable file system.

    Args:
        files: Mapping of path -> content (str for text, bytes for binary).
        dirs: Mapping of path -> list of FileInfo entries.
    """
    files = files or {}
    dirs = dirs or {}

    sandbox = AsyncMock(spec=Sandbox)

    async def mock_read_file(path, **kwargs):
        if path in files:
            content = files[path]
            return content.encode("utf-8") if isinstance(content, str) else content
        raise FileNotFoundError(f"No such file: {path}")

    async def mock_read_text(path, encoding="utf-8", **kwargs):
        if path in files:
            content = files[path]
            return content if isinstance(content, str) else content.decode(encoding)
        raise FileNotFoundError(f"No such file: {path}")

    async def mock_list_files(path, **kwargs):
        if path in dirs:
            return dirs[path]
        raise FileNotFoundError(f"No such directory: {path}")

    sandbox.read_file = AsyncMock(side_effect=mock_read_file)
    sandbox.read_text = AsyncMock(side_effect=mock_read_text)
    sandbox.list_files = AsyncMock(side_effect=mock_list_files)

    return sandbox


def _mock_agent(sandbox=None):
    """Create a mock agent with sandbox support."""
    agent = MagicMock()
    agent._system_prompt = "You are an agent."

    type(agent).system_prompt = property(
        lambda self: self._system_prompt,
        lambda self, value: setattr(self, "_system_prompt", value),
    )

    agent.hooks = HookRegistry()
    agent.add_hook = MagicMock(
        side_effect=lambda callback, event_type=None: agent.hooks.add_callback(event_type, callback)
    )
    agent.tool_registry = MagicMock()
    agent.tool_registry.process_tools = MagicMock(return_value=["skills"])

    state_store: dict[str, object] = {}
    agent.state = MagicMock()
    agent.state.get = MagicMock(side_effect=lambda key: state_store.get(key))
    agent.state.set = MagicMock(side_effect=lambda key, value: state_store.__setitem__(key, value))

    if sandbox is not None:
        agent.sandbox = sandbox
    else:
        agent.sandbox = _make_mock_sandbox()

    return agent


# --- Tests for Skill.from_sandbox ---


class TestSkillFromSandbox:
    """Tests for Skill.from_sandbox classmethod."""

    @pytest.mark.asyncio
    async def test_load_skill_from_sandbox_directory(self):
        """Test loading a skill from a directory containing SKILL.md."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT}
        )

        skill = await Skill.from_sandbox(sandbox, "/home/skills/my-skill")

        assert skill.name == "sandbox-skill"
        assert skill.description == "A skill loaded from sandbox"
        assert "Follow these steps" in skill.instructions
        assert skill.allowed_tools == ["shell", "editor"]

    @pytest.mark.asyncio
    async def test_load_skill_from_direct_skill_md_path(self):
        """Test loading a skill by pointing directly to SKILL.md."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT}
        )

        skill = await Skill.from_sandbox(sandbox, "/home/skills/my-skill/SKILL.md")

        assert skill.name == "sandbox-skill"

    @pytest.mark.asyncio
    async def test_load_skill_lowercase_skill_md(self):
        """Test loading skill from skill.md (lowercase)."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/skill.md": SKILL_CONTENT}
        )

        skill = await Skill.from_sandbox(sandbox, "/home/skills/my-skill")

        assert skill.name == "sandbox-skill"

    @pytest.mark.asyncio
    async def test_prefers_uppercase_skill_md(self):
        """Test that SKILL.md is preferred over skill.md."""
        sandbox = _make_mock_sandbox(
            files={
                "/home/skills/my-skill/SKILL.md": SKILL_CONTENT,
                "/home/skills/my-skill/skill.md": SKILL_B_CONTENT,
            }
        )

        skill = await Skill.from_sandbox(sandbox, "/home/skills/my-skill")

        assert skill.name == "sandbox-skill"  # From SKILL.md, not skill.md

    @pytest.mark.asyncio
    async def test_raises_when_no_skill_md(self):
        """Test FileNotFoundError when directory has no SKILL.md."""
        sandbox = _make_mock_sandbox()

        with pytest.raises(FileNotFoundError, match="no SKILL.md found"):
            await Skill.from_sandbox(sandbox, "/home/skills/empty-dir")

    @pytest.mark.asyncio
    async def test_raises_on_invalid_content(self):
        """Test ValueError when SKILL.md has invalid content."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/bad-skill/SKILL.md": INVALID_SKILL_CONTENT}
        )

        with pytest.raises(ValueError, match="name"):
            await Skill.from_sandbox(sandbox, "/home/skills/bad-skill")

    @pytest.mark.asyncio
    async def test_path_trailing_slash(self):
        """Test that trailing slashes in path are handled correctly."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT}
        )

        skill = await Skill.from_sandbox(sandbox, "/home/skills/my-skill/")

        assert skill.name == "sandbox-skill"

    @pytest.mark.asyncio
    async def test_strict_mode(self):
        """Test strict validation mode."""
        content = """---
name: Bad_Name
description: Has invalid name
---
# Instructions
"""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/bad/SKILL.md": content}
        )

        with pytest.raises(ValueError, match="skill name"):
            await Skill.from_sandbox(sandbox, "/home/skills/bad", strict=True)


class TestSkillFromSandboxDirectory:
    """Tests for Skill.from_sandbox_directory classmethod."""

    @pytest.mark.asyncio
    async def test_load_multiple_skills(self):
        """Test loading multiple skills from a parent directory."""
        sandbox = _make_mock_sandbox(
            files={
                "/home/skills/skill-a/SKILL.md": SKILL_CONTENT,
                "/home/skills/skill-b/SKILL.md": SKILL_B_CONTENT,
            },
            dirs={
                "/home/skills": [
                    FileInfo(name="skill-a", is_dir=True),
                    FileInfo(name="skill-b", is_dir=True),
                ]
            },
        )

        skills = await Skill.from_sandbox_directory(sandbox, "/home/skills")

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert "sandbox-skill" in names
        assert "another-skill" in names

    @pytest.mark.asyncio
    async def test_skips_non_directory_entries(self):
        """Test that files in the parent directory are skipped."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/skill-a/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [
                    FileInfo(name="skill-a", is_dir=True),
                    FileInfo(name="README.md", is_dir=False),
                ]
            },
        )

        skills = await Skill.from_sandbox_directory(sandbox, "/home/skills")

        assert len(skills) == 1
        assert skills[0].name == "sandbox-skill"

    @pytest.mark.asyncio
    async def test_skips_directories_without_skill_md(self):
        """Test that directories without SKILL.md are silently skipped."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/skill-a/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [
                    FileInfo(name="skill-a", is_dir=True),
                    FileInfo(name="empty-dir", is_dir=True),
                ]
            },
        )

        skills = await Skill.from_sandbox_directory(sandbox, "/home/skills")

        assert len(skills) == 1

    @pytest.mark.asyncio
    async def test_empty_directory(self):
        """Test loading from an empty directory."""
        sandbox = _make_mock_sandbox(
            dirs={"/home/skills": []}
        )

        skills = await Skill.from_sandbox_directory(sandbox, "/home/skills")

        assert skills == []

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self):
        """Test loading from a directory that doesn't exist."""
        sandbox = _make_mock_sandbox()

        skills = await Skill.from_sandbox_directory(sandbox, "/nonexistent")

        assert skills == []

    @pytest.mark.asyncio
    async def test_skips_invalid_skills_with_warning(self, caplog):
        """Test that invalid skills are skipped with a debug log."""
        sandbox = _make_mock_sandbox(
            files={
                "/home/skills/good-skill/SKILL.md": SKILL_CONTENT,
                "/home/skills/bad-skill/SKILL.md": INVALID_SKILL_CONTENT,
            },
            dirs={
                "/home/skills": [
                    FileInfo(name="bad-skill", is_dir=True),
                    FileInfo(name="good-skill", is_dir=True),
                ]
            },
        )

        with caplog.at_level(logging.DEBUG):
            skills = await Skill.from_sandbox_directory(sandbox, "/home/skills")

        assert len(skills) == 1
        assert skills[0].name == "sandbox-skill"

    @pytest.mark.asyncio
    async def test_sorted_by_directory_name(self):
        """Test that skills are loaded in sorted directory name order."""
        sandbox = _make_mock_sandbox(
            files={
                "/home/skills/z-skill/SKILL.md": SKILL_B_CONTENT,
                "/home/skills/a-skill/SKILL.md": SKILL_CONTENT,
            },
            dirs={
                "/home/skills": [
                    FileInfo(name="z-skill", is_dir=True),
                    FileInfo(name="a-skill", is_dir=True),
                ]
            },
        )

        skills = await Skill.from_sandbox_directory(sandbox, "/home/skills")

        assert len(skills) == 2
        # Loaded in sorted order (a-skill first, z-skill second)
        assert skills[0].name == "sandbox-skill"  # from a-skill/
        assert skills[1].name == "another-skill"  # from z-skill/


# --- Tests for AgentSkills with "sandbox:" sources ---


class TestAgentSkillsSandboxSources:
    """Tests for AgentSkills plugin with sandbox: URI sources."""

    @pytest.mark.asyncio
    async def test_sandbox_source_parsed_from_constructor(self):
        """Test that 'sandbox:' prefixed sources are stored as pending."""
        plugin = AgentSkills(skills=["sandbox:/home/skills"])

        assert len(plugin._sandbox_sources) == 1
        assert plugin._sandbox_sources[0] == "/home/skills"
        assert len(plugin._skills) == 0  # Not loaded yet

    @pytest.mark.asyncio
    async def test_sandbox_skills_loaded_in_init_agent(self):
        """Test that sandbox skills are loaded during init_agent."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [
                    FileInfo(name="my-skill", is_dir=True),
                ]
            },
        )
        agent = _mock_agent(sandbox=sandbox)
        plugin = AgentSkills(skills=["sandbox:/home/skills"])

        assert len(plugin._skills) == 0

        await plugin.init_agent(agent)

        assert len(plugin._skills) == 1
        assert "sandbox-skill" in plugin._skills

    @pytest.mark.asyncio
    async def test_mixed_local_and_sandbox_sources(self, tmp_path):
        """Test mixing local filesystem and sandbox sources."""
        # Create a local skill
        local_skill_dir = tmp_path / "local-skill"
        local_skill_dir.mkdir()
        (local_skill_dir / "SKILL.md").write_text(
            "---\nname: local-skill\ndescription: From filesystem\n---\n# Local\n"
        )

        # Create sandbox with a different skill
        sandbox = _make_mock_sandbox(
            files={"/home/skills/remote-skill/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [
                    FileInfo(name="remote-skill", is_dir=True),
                ]
            },
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=[str(local_skill_dir), "sandbox:/home/skills"])

        # Local skill resolved immediately
        assert "local-skill" in plugin._skills
        assert "sandbox-skill" not in plugin._skills

        # After init_agent, sandbox skill is also available
        await plugin.init_agent(agent)

        assert "local-skill" in plugin._skills
        assert "sandbox-skill" in plugin._skills

    @pytest.mark.asyncio
    async def test_sandbox_single_skill_directory(self):
        """Test sandbox source pointing to a single skill directory (not parent)."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT}
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/home/skills/my-skill"])
        await plugin.init_agent(agent)

        assert "sandbox-skill" in plugin._skills

    @pytest.mark.asyncio
    async def test_sandbox_source_not_found_warns(self, caplog):
        """Test that missing sandbox paths warn and don't crash."""
        sandbox = _make_mock_sandbox()
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/nonexistent"])

        with caplog.at_level(logging.WARNING):
            await plugin.init_agent(agent)

        assert len(plugin._skills) == 0

    @pytest.mark.asyncio
    async def test_sandbox_source_duplicate_overwrites(self):
        """Test that duplicate skill names from sandbox overwrite earlier ones."""
        sandbox = _make_mock_sandbox(
            files={"/sandbox/skills/dupe/SKILL.md": SKILL_CONTENT}
        )
        agent = _mock_agent(sandbox=sandbox)

        # Pre-load a skill with the same name
        existing = Skill(name="sandbox-skill", description="Original", instructions="Old")
        plugin = AgentSkills(skills=[existing, "sandbox:/sandbox/skills/dupe"])
        await plugin.init_agent(agent)

        # Sandbox version should overwrite
        assert plugin._skills["sandbox-skill"].description == "A skill loaded from sandbox"

    @pytest.mark.asyncio
    async def test_multiple_sandbox_sources(self):
        """Test multiple sandbox: sources."""
        sandbox = _make_mock_sandbox(
            files={
                "/skills-a/s1/SKILL.md": SKILL_CONTENT,
                "/skills-b/s2/SKILL.md": SKILL_B_CONTENT,
            },
            dirs={
                "/skills-a": [FileInfo(name="s1", is_dir=True)],
                "/skills-b": [FileInfo(name="s2", is_dir=True)],
            },
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/skills-a", "sandbox:/skills-b"])
        await plugin.init_agent(agent)

        assert len(plugin._skills) == 2
        assert "sandbox-skill" in plugin._skills
        assert "another-skill" in plugin._skills

    @pytest.mark.asyncio
    async def test_set_available_skills_rejects_sandbox(self):
        """Test that set_available_skills raises on sandbox sources."""
        plugin = AgentSkills(skills=[])

        with pytest.raises(ValueError, match="Sandbox sources"):
            plugin.set_available_skills(["sandbox:/home/skills"])

    @pytest.mark.asyncio
    async def test_sandbox_skills_appear_in_system_prompt(self):
        """Test that sandbox-loaded skills appear in the system prompt XML."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [FileInfo(name="my-skill", is_dir=True)],
            },
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/home/skills"])
        await plugin.init_agent(agent)

        # Simulate before_invocation hook
        xml = plugin._generate_skills_xml()

        assert "sandbox-skill" in xml
        assert "A skill loaded from sandbox" in xml

    @pytest.mark.asyncio
    async def test_sandbox_skills_activatable_via_tool(self):
        """Test that sandbox-loaded skills can be activated via the skills tool."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [FileInfo(name="my-skill", is_dir=True)],
            },
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/home/skills"])
        await plugin.init_agent(agent)

        # Create tool context and activate the skill
        tool_use = {"toolUseId": "test-id", "name": "skills", "input": {}}
        tool_context = ToolContext(tool_use=tool_use, agent=agent, invocation_state={"agent": agent})

        result = plugin.skills(skill_name="sandbox-skill", tool_context=tool_context)

        assert "Follow these steps" in result

    @pytest.mark.asyncio
    async def test_no_sandbox_sources_stays_sync(self):
        """Test that plugin with no sandbox sources works fine with sync init."""
        skill = Skill(name="local", description="Local skill", instructions="Do it")
        plugin = AgentSkills(skills=[skill])
        agent = _mock_agent()

        # init_agent should work even though it's async
        await plugin.init_agent(agent)

        assert "local" in plugin._skills
        assert len(plugin._sandbox_sources) == 0

    @pytest.mark.asyncio
    async def test_sandbox_source_string_format(self):
        """Test various sandbox: source string formats."""
        plugin = AgentSkills(
            skills=[
                "sandbox:/home/skills",
                "sandbox:/absolute/path/to/skills",
                "sandbox:/tmp/skills/",
            ]
        )

        assert len(plugin._sandbox_sources) == 3
        assert plugin._sandbox_sources[0] == "/home/skills"
        assert plugin._sandbox_sources[1] == "/absolute/path/to/skills"
        assert plugin._sandbox_sources[2] == "/tmp/skills/"


class TestAgentSkillsSandboxPluginRegistry:
    """Test that sandbox skills work through the plugin registry (async init_agent)."""

    @pytest.mark.asyncio
    async def test_plugin_registry_handles_async_init(self):
        """Test that _PluginRegistry correctly handles async init_agent from AgentSkills."""
        from strands.plugins.registry import _PluginRegistry

        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [FileInfo(name="my-skill", is_dir=True)],
            },
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/home/skills"])

        # The registry should handle the async init_agent
        registry = _PluginRegistry(agent)
        registry.add_and_init(plugin)

        # After registry init, sandbox skills should be loaded
        assert "sandbox-skill" in plugin._skills

    @pytest.mark.asyncio
    async def test_get_available_skills_after_sandbox_load(self):
        """Test get_available_skills returns sandbox-loaded skills."""
        sandbox = _make_mock_sandbox(
            files={"/home/skills/my-skill/SKILL.md": SKILL_CONTENT},
            dirs={
                "/home/skills": [FileInfo(name="my-skill", is_dir=True)],
            },
        )
        agent = _mock_agent(sandbox=sandbox)

        plugin = AgentSkills(skills=["sandbox:/home/skills"])
        await plugin.init_agent(agent)

        available = plugin.get_available_skills()
        assert len(available) == 1
        assert available[0].name == "sandbox-skill"
