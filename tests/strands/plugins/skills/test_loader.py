"""Tests for the skill loader module."""

from pathlib import Path

import pytest

from strands.plugins.skills.loader import (
    _find_skill_md,
    _parse_frontmatter,
    _validate_skill_name,
    load_skill,
    load_skills,
)


class TestFindSkillMd:
    """Tests for _find_skill_md."""

    def test_finds_uppercase_skill_md(self, tmp_path):
        """Test finding SKILL.md (uppercase)."""
        (tmp_path / "SKILL.md").write_text("test")
        result = _find_skill_md(tmp_path)
        assert result.name == "SKILL.md"

    def test_finds_lowercase_skill_md(self, tmp_path):
        """Test finding skill.md (lowercase)."""
        (tmp_path / "skill.md").write_text("test")
        result = _find_skill_md(tmp_path)
        assert result.name.lower() == "skill.md"

    def test_prefers_uppercase(self, tmp_path):
        """Test that SKILL.md is preferred over skill.md."""
        (tmp_path / "SKILL.md").write_text("uppercase")
        (tmp_path / "skill.md").write_text("lowercase")
        result = _find_skill_md(tmp_path)
        assert result.name == "SKILL.md"

    def test_raises_when_not_found(self, tmp_path):
        """Test FileNotFoundError when no SKILL.md exists."""
        with pytest.raises(FileNotFoundError, match="no SKILL.md found"):
            _find_skill_md(tmp_path)


class TestParseFrontmatter:
    """Tests for _parse_frontmatter."""

    def test_valid_frontmatter(self):
        """Test parsing valid frontmatter."""
        content = "---\nname: test-skill\ndescription: A test\n---\n# Instructions\nDo things."
        frontmatter, body = _parse_frontmatter(content)
        assert frontmatter["name"] == "test-skill"
        assert frontmatter["description"] == "A test"
        assert "# Instructions" in body
        assert "Do things." in body

    def test_missing_opening_delimiter(self):
        """Test error when opening --- is missing."""
        with pytest.raises(ValueError, match="must start with ---"):
            _parse_frontmatter("name: test\n---\n")

    def test_missing_closing_delimiter(self):
        """Test error when closing --- is missing."""
        with pytest.raises(ValueError, match="missing closing ---"):
            _parse_frontmatter("---\nname: test\n")

    def test_empty_body(self):
        """Test frontmatter with empty body."""
        content = "---\nname: test-skill\ndescription: test\n---\n"
        frontmatter, body = _parse_frontmatter(content)
        assert frontmatter["name"] == "test-skill"
        assert body == ""

    def test_frontmatter_with_metadata(self):
        """Test frontmatter with nested metadata."""
        content = "---\nname: test-skill\ndescription: test\nmetadata:\n  author: acme\n---\nBody here."
        frontmatter, body = _parse_frontmatter(content)
        assert frontmatter["name"] == "test-skill"
        assert isinstance(frontmatter["metadata"], dict)
        assert frontmatter["metadata"]["author"] == "acme"
        assert body == "Body here."

    def test_frontmatter_with_dashes_in_yaml_value(self):
        """Test that --- inside a YAML value does not break parsing."""
        content = "---\nname: test-skill\ndescription: has --- inside\n---\nBody here."
        frontmatter, body = _parse_frontmatter(content)
        assert frontmatter["name"] == "test-skill"
        assert frontmatter["description"] == "has --- inside"
        assert body == "Body here."


class TestValidateSkillName:
    """Tests for _validate_skill_name."""

    def test_valid_names(self):
        """Test that valid names pass validation."""
        valid_names = ["a", "test", "my-skill", "skill-123", "a1b2c3"]
        for name in valid_names:
            _validate_skill_name(name)  # Should not raise

    def test_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_skill_name("")

    def test_too_long_name(self):
        """Test that names exceeding 64 chars raise ValueError."""
        with pytest.raises(ValueError, match="exceeds 64 character limit"):
            _validate_skill_name("a" * 65)

    def test_uppercase_rejected(self):
        """Test that uppercase characters are rejected."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            _validate_skill_name("MySkill")

    def test_starts_with_hyphen(self):
        """Test that names starting with hyphen are rejected."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            _validate_skill_name("-skill")

    def test_ends_with_hyphen(self):
        """Test that names ending with hyphen are rejected."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            _validate_skill_name("skill-")

    def test_consecutive_hyphens(self):
        """Test that consecutive hyphens are rejected."""
        with pytest.raises(ValueError, match="consecutive hyphens"):
            _validate_skill_name("my--skill")

    def test_special_characters(self):
        """Test that special characters are rejected."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            _validate_skill_name("my_skill")

    def test_directory_name_mismatch(self, tmp_path):
        """Test that skill name must match directory name."""
        skill_dir = tmp_path / "wrong-name"
        skill_dir.mkdir()
        with pytest.raises(ValueError, match="must match parent directory name"):
            _validate_skill_name("my-skill", skill_dir)

    def test_directory_name_match(self, tmp_path):
        """Test that matching directory name passes."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        _validate_skill_name("my-skill", skill_dir)  # Should not raise


def _make_skill_dir(parent: Path, name: str, description: str = "A test skill", body: str = "Instructions.") -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


class TestLoadSkill:
    """Tests for load_skill."""

    def test_load_from_directory(self, tmp_path):
        """Test loading a skill from a directory path."""
        skill_dir = _make_skill_dir(tmp_path, "my-skill", "My description", "# Hello\nWorld.")
        skill = load_skill(skill_dir)

        assert skill.name == "my-skill"
        assert skill.description == "My description"
        assert "# Hello" in skill.instructions
        assert "World." in skill.instructions
        assert skill.path == skill_dir.resolve()

    def test_load_from_skill_md_file(self, tmp_path):
        """Test loading a skill by pointing directly to SKILL.md."""
        skill_dir = _make_skill_dir(tmp_path, "direct-skill")
        skill = load_skill(skill_dir / "SKILL.md")

        assert skill.name == "direct-skill"

    def test_load_with_allowed_tools(self, tmp_path):
        """Test loading a skill with allowed-tools field as space-delimited string."""
        skill_dir = tmp_path / "tool-skill"
        skill_dir.mkdir()
        content = "---\nname: tool-skill\ndescription: test\nallowed-tools: read write execute\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = load_skill(skill_dir)
        assert skill.allowed_tools == ["read", "write", "execute"]

    def test_load_with_allowed_tools_yaml_list(self, tmp_path):
        """Test loading a skill with allowed-tools as a YAML list."""
        skill_dir = tmp_path / "list-skill"
        skill_dir.mkdir()
        content = "---\nname: list-skill\ndescription: test\nallowed-tools:\n  - read\n  - write\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = load_skill(skill_dir)
        assert skill.allowed_tools == ["read", "write"]

    def test_load_with_metadata(self, tmp_path):
        """Test loading a skill with nested metadata."""
        skill_dir = tmp_path / "meta-skill"
        skill_dir.mkdir()
        content = "---\nname: meta-skill\ndescription: test\nmetadata:\n  author: acme\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = load_skill(skill_dir)
        assert skill.metadata == {"author": "acme"}

    def test_load_with_license_and_compatibility(self, tmp_path):
        """Test loading a skill with license and compatibility fields."""
        skill_dir = tmp_path / "licensed-skill"
        skill_dir.mkdir()
        content = "---\nname: licensed-skill\ndescription: test\nlicense: MIT\ncompatibility: v1\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = load_skill(skill_dir)
        assert skill.license == "MIT"
        assert skill.compatibility == "v1"

    def test_load_missing_name(self, tmp_path):
        """Test error when SKILL.md is missing name field."""
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: test\n---\nBody.")

        with pytest.raises(ValueError, match="must have a 'name' field"):
            load_skill(skill_dir)

    def test_load_missing_description(self, tmp_path):
        """Test error when SKILL.md is missing description field."""
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\nBody.")

        with pytest.raises(ValueError, match="must have a 'description' field"):
            load_skill(skill_dir)

    def test_load_nonexistent_path(self, tmp_path):
        """Test FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_skill(tmp_path / "nonexistent")

    def test_load_name_directory_mismatch(self, tmp_path):
        """Test error when skill name doesn't match directory name."""
        skill_dir = tmp_path / "wrong-dir"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: right-name\ndescription: test\n---\nBody.")

        with pytest.raises(ValueError, match="must match parent directory name"):
            load_skill(skill_dir)


class TestLoadSkills:
    """Tests for load_skills."""

    def test_load_multiple_skills(self, tmp_path):
        """Test loading multiple skills from a parent directory."""
        _make_skill_dir(tmp_path, "skill-a", "Skill A")
        _make_skill_dir(tmp_path, "skill-b", "Skill B")

        skills = load_skills(tmp_path)

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-a", "skill-b"}

    def test_skips_directories_without_skill_md(self, tmp_path):
        """Test that directories without SKILL.md are silently skipped."""
        _make_skill_dir(tmp_path, "valid-skill")
        (tmp_path / "no-skill-here").mkdir()

        skills = load_skills(tmp_path)

        assert len(skills) == 1
        assert skills[0].name == "valid-skill"

    def test_skips_files_in_parent(self, tmp_path):
        """Test that files in the parent directory are ignored."""
        _make_skill_dir(tmp_path, "real-skill")
        (tmp_path / "readme.txt").write_text("not a skill")

        skills = load_skills(tmp_path)

        assert len(skills) == 1

    def test_empty_directory(self, tmp_path):
        """Test loading from an empty directory."""
        skills = load_skills(tmp_path)
        assert skills == []

    def test_nonexistent_directory(self, tmp_path):
        """Test FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            load_skills(tmp_path / "nonexistent")

    def test_skips_invalid_skills(self, tmp_path):
        """Test that invalid skills are skipped with a warning."""
        _make_skill_dir(tmp_path, "good-skill")

        # Create an invalid skill (name mismatch)
        bad_dir = tmp_path / "bad-dir"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("---\nname: wrong-name\ndescription: test\n---\nBody.")

        skills = load_skills(tmp_path)

        assert len(skills) == 1
        assert skills[0].name == "good-skill"
