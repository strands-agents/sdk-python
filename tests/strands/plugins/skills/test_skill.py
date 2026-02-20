"""Tests for the Skill dataclass."""

from pathlib import Path

import pytest

from strands.plugins.skills.skill import Skill


class TestSkillDataclass:
    """Tests for the Skill dataclass creation and properties."""

    def test_skill_minimal(self):
        """Test creating a Skill with only required fields."""
        skill = Skill(name="test-skill", description="A test skill")

        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.instructions == ""
        assert skill.path is None
        assert skill.allowed_tools is None
        assert skill.metadata == {}
        assert skill.license is None
        assert skill.compatibility is None

    def test_skill_full(self):
        """Test creating a Skill with all fields."""
        skill = Skill(
            name="full-skill",
            description="A fully specified skill",
            instructions="# Full Instructions\nDo the thing.",
            path=Path("/tmp/skills/full-skill"),
            allowed_tools=["tool1", "tool2"],
            metadata={"author": "test-org"},
            license="Apache-2.0",
            compatibility="strands>=1.0",
        )

        assert skill.name == "full-skill"
        assert skill.description == "A fully specified skill"
        assert skill.instructions == "# Full Instructions\nDo the thing."
        assert skill.path == Path("/tmp/skills/full-skill")
        assert skill.allowed_tools == ["tool1", "tool2"]
        assert skill.metadata == {"author": "test-org"}
        assert skill.license == "Apache-2.0"
        assert skill.compatibility == "strands>=1.0"

    def test_skill_metadata_default_is_not_shared(self):
        """Test that default metadata dict is not shared between instances."""
        skill1 = Skill(name="skill-1", description="First")
        skill2 = Skill(name="skill-2", description="Second")

        skill1.metadata["key"] = "value"
        assert "key" not in skill2.metadata

    def test_skill_from_path(self, tmp_path):
        """Test loading a Skill from a path using from_path classmethod."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill\n---\n# Instructions\nDo stuff.\n"
        )

        skill = Skill.from_path(skill_dir)

        assert skill.name == "my-skill"
        assert skill.description == "Test skill"
        assert "Do stuff." in skill.instructions

    def test_skill_from_path_not_found(self, tmp_path):
        """Test that from_path raises FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            Skill.from_path(tmp_path / "nonexistent")
