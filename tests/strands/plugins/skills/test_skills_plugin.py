"""Tests for the SkillsPlugin."""

from pathlib import Path
from unittest.mock import MagicMock

from strands.hooks.events import AfterInvocationEvent, BeforeInvocationEvent
from strands.hooks.registry import HookRegistry
from strands.plugins.skills.skill import Skill
from strands.plugins.skills.skills_plugin import SkillsPlugin, _make_skills_tool


def _make_skill(name: str = "test-skill", description: str = "A test skill", instructions: str = "Do the thing."):
    """Helper to create a Skill instance."""
    return Skill(name=name, description=description, instructions=instructions)


def _make_skill_dir(parent: Path, name: str, description: str = "A test skill") -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n# Instructions for {name}\n"
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


def _mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent._system_prompt = "You are an agent."
    agent._system_prompt_content = [{"text": "You are an agent."}]
    agent.hooks = HookRegistry()
    agent.tool_registry = MagicMock()
    agent.tool_registry.process_tools = MagicMock(return_value=["skills"])
    agent.state = MagicMock()
    agent.state.get = MagicMock(return_value=None)
    agent.state.set = MagicMock()
    return agent


class TestSkillsPluginInit:
    """Tests for SkillsPlugin initialization."""

    def test_init_with_skill_instances(self):
        """Test initialization with Skill instances."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])

        assert len(plugin.skills) == 1
        assert plugin.skills[0].name == "test-skill"

    def test_init_with_filesystem_paths(self, tmp_path):
        """Test initialization with filesystem paths."""
        _make_skill_dir(tmp_path, "fs-skill")
        plugin = SkillsPlugin(skills=[str(tmp_path / "fs-skill")])

        assert len(plugin.skills) == 1
        assert plugin.skills[0].name == "fs-skill"

    def test_init_with_parent_directory(self, tmp_path):
        """Test initialization with a parent directory containing skills."""
        _make_skill_dir(tmp_path, "skill-a")
        _make_skill_dir(tmp_path, "skill-b")
        plugin = SkillsPlugin(skills=[tmp_path])

        assert len(plugin.skills) == 2

    def test_init_with_mixed_sources(self, tmp_path):
        """Test initialization with mixed skill sources."""
        _make_skill_dir(tmp_path, "fs-skill")
        direct_skill = _make_skill(name="direct-skill", description="Direct")
        plugin = SkillsPlugin(skills=[str(tmp_path / "fs-skill"), direct_skill])

        assert len(plugin.skills) == 2
        names = {s.name for s in plugin.skills}
        assert names == {"fs-skill", "direct-skill"}

    def test_init_skips_nonexistent_paths(self, tmp_path):
        """Test that nonexistent paths are skipped gracefully."""
        plugin = SkillsPlugin(skills=[str(tmp_path / "nonexistent")])
        assert len(plugin.skills) == 0

    def test_init_empty_skills(self):
        """Test initialization with empty skills list."""
        plugin = SkillsPlugin(skills=[])
        assert plugin.skills == []
        assert plugin.active_skill is None

    def test_name_attribute(self):
        """Test that the plugin has the correct name."""
        plugin = SkillsPlugin(skills=[])
        assert plugin.name == "skills"


class TestSkillsPluginInitPlugin:
    """Tests for the init_plugin method."""

    def test_registers_tool(self):
        """Test that init_plugin registers the skills tool."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()

        plugin.init_plugin(agent)

        agent.tool_registry.process_tools.assert_called_once()
        args = agent.tool_registry.process_tools.call_args[0][0]
        assert len(args) == 1

    def test_registers_hooks(self):
        """Test that init_plugin registers hook callbacks."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()

        plugin.init_plugin(agent)

        # Verify hooks were registered by checking the registry has callbacks
        assert agent.hooks.has_callbacks()

    def test_stores_agent_reference(self):
        """Test that init_plugin stores the agent reference."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()

        plugin.init_plugin(agent)

        assert plugin._agent is agent

    def test_restores_state(self):
        """Test that init_plugin restores active skill from state."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])
        agent = _mock_agent()
        agent.state.get = MagicMock(return_value={"active_skill_name": "test-skill"})

        plugin.init_plugin(agent)

        assert plugin.active_skill is not None
        assert plugin.active_skill.name == "test-skill"


class TestSkillsPluginProperties:
    """Tests for SkillsPlugin properties."""

    def test_skills_getter_returns_copy(self):
        """Test that the skills getter returns a copy of the list."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])

        skills_list = plugin.skills
        skills_list.append(_make_skill(name="another-skill", description="Another"))

        assert len(plugin.skills) == 1

    def test_skills_setter(self):
        """Test setting skills via the property setter."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        plugin._agent = _mock_agent()

        new_skill = _make_skill(name="new-skill", description="New")
        plugin.skills = [new_skill]

        assert len(plugin.skills) == 1
        assert plugin.skills[0].name == "new-skill"

    def test_skills_setter_deactivates_current(self):
        """Test that setting skills deactivates the current active skill."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        plugin._agent = _mock_agent()
        plugin._active_skill = _make_skill()

        plugin.skills = [_make_skill(name="new-skill", description="New")]

        assert plugin.active_skill is None

    def test_active_skill_initially_none(self):
        """Test that active_skill is None initially."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        assert plugin.active_skill is None


class TestSkillsTool:
    """Tests for the skills tool function."""

    def test_activate_skill(self):
        """Test activating a skill returns its instructions."""
        skill = _make_skill(instructions="Full instructions here.")
        plugin = SkillsPlugin(skills=[skill])
        plugin._agent = _mock_agent()

        skills_tool = _make_skills_tool(plugin)
        result = skills_tool(action="activate", skill_name="test-skill")

        assert result == "Full instructions here."
        assert plugin.active_skill is not None
        assert plugin.active_skill.name == "test-skill"

    def test_activate_nonexistent_skill(self):
        """Test activating a nonexistent skill returns error message."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])
        plugin._agent = _mock_agent()

        skills_tool = _make_skills_tool(plugin)
        result = skills_tool(action="activate", skill_name="nonexistent")

        assert "not found" in result
        assert "test-skill" in result

    def test_activate_replaces_previous(self):
        """Test that activating a new skill replaces the previous one."""
        skill1 = _make_skill(name="skill-a", description="A", instructions="A instructions")
        skill2 = _make_skill(name="skill-b", description="B", instructions="B instructions")
        plugin = SkillsPlugin(skills=[skill1, skill2])
        plugin._agent = _mock_agent()

        skills_tool = _make_skills_tool(plugin)
        skills_tool(action="activate", skill_name="skill-a")
        assert plugin.active_skill.name == "skill-a"

        skills_tool(action="activate", skill_name="skill-b")
        assert plugin.active_skill.name == "skill-b"

    def test_activate_without_name(self):
        """Test activating without a skill name returns error."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        plugin._agent = _mock_agent()

        skills_tool = _make_skills_tool(plugin)
        result = skills_tool(action="activate", skill_name="")

        assert "required" in result.lower()

    def test_deactivate_skill(self):
        """Test deactivating a skill."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])
        plugin._agent = _mock_agent()
        plugin._active_skill = skill

        skills_tool = _make_skills_tool(plugin)
        result = skills_tool(action="deactivate", skill_name="test-skill")

        assert "deactivated" in result.lower()
        assert plugin.active_skill is None

    def test_unknown_action(self):
        """Test unknown action returns error message."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        plugin._agent = _mock_agent()

        skills_tool = _make_skills_tool(plugin)
        result = skills_tool(action="unknown")

        assert "Unknown action" in result

    def test_activate_persists_state(self):
        """Test that activating a skill persists state."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        plugin._agent = agent

        skills_tool = _make_skills_tool(plugin)
        skills_tool(action="activate", skill_name="test-skill")

        agent.state.set.assert_called()


class TestSystemPromptInjection:
    """Tests for system prompt injection via hooks."""

    def test_before_invocation_appends_skills_xml(self):
        """Test that before_invocation appends skills XML to system prompt."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])
        agent = _mock_agent()

        event = BeforeInvocationEvent(agent=agent)
        plugin._on_before_invocation(event)

        assert "<available_skills>" in agent._system_prompt
        assert "<name>test-skill</name>" in agent._system_prompt
        assert "<description>A test skill</description>" in agent._system_prompt

    def test_before_invocation_preserves_existing_prompt(self):
        """Test that existing system prompt content is preserved."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        agent._system_prompt = "Original prompt."
        agent._system_prompt_content = [{"text": "Original prompt."}]

        event = BeforeInvocationEvent(agent=agent)
        plugin._on_before_invocation(event)

        assert agent._system_prompt.startswith("Original prompt.")
        assert "<available_skills>" in agent._system_prompt

    def test_after_invocation_restores_prompt(self):
        """Test that after_invocation restores the original system prompt."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        original_prompt = "Original prompt."
        original_content = [{"text": "Original prompt."}]
        agent._system_prompt = original_prompt
        agent._system_prompt_content = original_content

        # Simulate before/after cycle
        before_event = BeforeInvocationEvent(agent=agent)
        plugin._on_before_invocation(before_event)
        assert agent._system_prompt != original_prompt

        after_event = AfterInvocationEvent(agent=agent)
        plugin._on_after_invocation(after_event)
        assert agent._system_prompt == original_prompt
        assert agent._system_prompt_content == original_content

    def test_no_skills_skips_injection(self):
        """Test that injection is skipped when no skills are available."""
        plugin = SkillsPlugin(skills=[])
        agent = _mock_agent()
        original_prompt = "Original prompt."
        agent._system_prompt = original_prompt
        agent._system_prompt_content = [{"text": original_prompt}]

        event = BeforeInvocationEvent(agent=agent)
        plugin._on_before_invocation(event)

        assert agent._system_prompt == original_prompt

    def test_none_system_prompt_handled(self):
        """Test handling when system prompt is None."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        agent._system_prompt = None
        agent._system_prompt_content = None

        event = BeforeInvocationEvent(agent=agent)
        plugin._on_before_invocation(event)

        assert "<available_skills>" in agent._system_prompt


class TestSkillsXmlGeneration:
    """Tests for _generate_skills_xml."""

    def test_single_skill(self):
        """Test XML generation with a single skill."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        xml = plugin._generate_skills_xml()

        assert "<available_skills>" in xml
        assert "</available_skills>" in xml
        assert "<name>test-skill</name>" in xml
        assert "<description>A test skill</description>" in xml

    def test_multiple_skills(self):
        """Test XML generation with multiple skills."""
        skills = [
            _make_skill(name="skill-a", description="Skill A"),
            _make_skill(name="skill-b", description="Skill B"),
        ]
        plugin = SkillsPlugin(skills=skills)
        xml = plugin._generate_skills_xml()

        assert "<name>skill-a</name>" in xml
        assert "<name>skill-b</name>" in xml

    def test_empty_skills(self):
        """Test XML generation with no skills."""
        plugin = SkillsPlugin(skills=[])
        xml = plugin._generate_skills_xml()

        assert "<available_skills>" in xml
        assert "</available_skills>" in xml


class TestHookRegistration:
    """Tests for hook registration."""

    def test_register_hooks(self):
        """Test that register_hooks adds callbacks to the registry."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        registry = HookRegistry()

        plugin.register_hooks(registry)

        assert registry.has_callbacks()


class TestSessionPersistence:
    """Tests for session state persistence."""

    def test_persist_state_with_active_skill(self):
        """Test persisting active skill name."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        plugin._agent = agent
        plugin._active_skill = _make_skill()

        plugin._persist_state()

        agent.state.set.assert_called_once_with("skills_plugin", {"active_skill_name": "test-skill"})

    def test_persist_state_without_active_skill(self):
        """Test persisting None when no skill is active."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        plugin._agent = agent

        plugin._persist_state()

        agent.state.set.assert_called_once_with("skills_plugin", {"active_skill_name": None})

    def test_restore_state_activates_skill(self):
        """Test restoring active skill from state."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])
        agent = _mock_agent()
        agent.state.get = MagicMock(return_value={"active_skill_name": "test-skill"})
        plugin._agent = agent

        plugin._restore_state()

        assert plugin.active_skill is not None
        assert plugin.active_skill.name == "test-skill"

    def test_restore_state_no_data(self):
        """Test restore when no state data exists."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        agent.state.get = MagicMock(return_value=None)
        plugin._agent = agent

        plugin._restore_state()

        assert plugin.active_skill is None

    def test_restore_state_skill_not_found(self):
        """Test restore when saved skill is no longer available."""
        plugin = SkillsPlugin(skills=[_make_skill()])
        agent = _mock_agent()
        agent.state.get = MagicMock(return_value={"active_skill_name": "removed-skill"})
        plugin._agent = agent

        plugin._restore_state()

        assert plugin.active_skill is None

    def test_persist_state_without_agent(self):
        """Test that persist_state is a no-op without agent."""
        plugin = SkillsPlugin(skills=[_make_skill()])

        # Should not raise
        plugin._persist_state()


class TestResolveSkills:
    """Tests for _resolve_skills."""

    def test_resolve_skill_instances(self):
        """Test resolving Skill instances (pass-through)."""
        skill = _make_skill()
        plugin = SkillsPlugin(skills=[skill])

        assert len(plugin._skills) == 1
        assert plugin._skills[0] is skill

    def test_resolve_skill_directory_path(self, tmp_path):
        """Test resolving a path to a skill directory."""
        _make_skill_dir(tmp_path, "path-skill")
        plugin = SkillsPlugin(skills=[tmp_path / "path-skill"])

        assert len(plugin._skills) == 1
        assert plugin._skills[0].name == "path-skill"

    def test_resolve_parent_directory_path(self, tmp_path):
        """Test resolving a path to a parent directory."""
        _make_skill_dir(tmp_path, "child-a")
        _make_skill_dir(tmp_path, "child-b")
        plugin = SkillsPlugin(skills=[tmp_path])

        assert len(plugin._skills) == 2

    def test_resolve_skill_md_file_path(self, tmp_path):
        """Test resolving a path to a SKILL.md file."""
        skill_dir = _make_skill_dir(tmp_path, "file-skill")
        plugin = SkillsPlugin(skills=[skill_dir / "SKILL.md"])

        assert len(plugin._skills) == 1
        assert plugin._skills[0].name == "file-skill"

    def test_resolve_nonexistent_path(self, tmp_path):
        """Test that nonexistent paths are skipped."""
        plugin = SkillsPlugin(skills=[str(tmp_path / "ghost")])
        assert len(plugin._skills) == 0


class TestImports:
    """Tests for module imports."""

    def test_import_from_plugins(self):
        """Test importing SkillsPlugin from strands.plugins."""
        from strands.plugins import SkillsPlugin as SP

        assert SP is SkillsPlugin

    def test_import_skill_from_strands(self):
        """Test importing Skill from top-level strands package."""
        from strands import Skill as S

        assert S is Skill

    def test_import_from_skills_package(self):
        """Test importing from strands.plugins.skills package."""
        from strands.plugins.skills import Skill, SkillsPlugin, load_skill, load_skills

        assert Skill is not None
        assert SkillsPlugin is not None
        assert load_skill is not None
        assert load_skills is not None

    def test_skills_plugin_is_plugin_subclass(self):
        """Test that SkillsPlugin is a subclass of the Plugin ABC."""
        from strands.plugins import Plugin

        assert issubclass(SkillsPlugin, Plugin)

    def test_skills_plugin_isinstance_check(self):
        """Test that SkillsPlugin instances pass isinstance check against Plugin."""
        from strands.plugins import Plugin

        plugin = SkillsPlugin(skills=[])
        assert isinstance(plugin, Plugin)
