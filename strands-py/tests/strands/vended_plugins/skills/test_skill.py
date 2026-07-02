"""Tests for the Skill dataclass and loading utilities."""

import logging
from pathlib import Path

import pytest

from strands.vended_plugins.skills.skill import (
    Skill,
    _find_skill_md,
    _fix_yaml_colons,
    _parse_frontmatter,
    _validate_skill_name,
)


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
    """Tests for _validate_skill_name (lenient validation)."""

    def test_valid_names(self):
        """Test that valid names pass validation without warnings."""
        valid_names = ["a", "test", "my-skill", "skill-123", "a1b2c3"]
        for name in valid_names:
            _validate_skill_name(name)  # Should not raise

    def test_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_skill_name("")

    def test_too_long_name_warns(self, caplog):
        """Test that names exceeding 64 chars warn but do not raise."""
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("a" * 65)
        assert "exceeds" in caplog.text

    def test_uppercase_warns(self, caplog):
        """Test that uppercase characters warn but do not raise."""
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("MySkill")
        assert "lowercase alphanumeric" in caplog.text

    def test_starts_with_hyphen_warns(self, caplog):
        """Test that names starting with hyphen warn but do not raise."""
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("-skill")
        assert "lowercase alphanumeric" in caplog.text

    def test_ends_with_hyphen_warns(self, caplog):
        """Test that names ending with hyphen warn but do not raise."""
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("skill-")
        assert "lowercase alphanumeric" in caplog.text

    def test_consecutive_hyphens_warns(self, caplog):
        """Test that consecutive hyphens warn but do not raise."""
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("my--skill")
        assert "consecutive hyphens" in caplog.text

    def test_special_characters_warns(self, caplog):
        """Test that special characters warn but do not raise."""
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("my_skill")
        assert "lowercase alphanumeric" in caplog.text

    def test_directory_name_mismatch_warns(self, tmp_path, caplog):
        """Test that skill name not matching directory name warns but does not raise."""
        skill_dir = tmp_path / "wrong-name"
        skill_dir.mkdir()
        with caplog.at_level(logging.WARNING):
            _validate_skill_name("my-skill", skill_dir)
        assert "does not match parent directory name" in caplog.text

    def test_directory_name_match(self, tmp_path):
        """Test that matching directory name passes."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        _validate_skill_name("my-skill", skill_dir)  # Should not raise or warn


class TestValidateSkillNameStrict:
    """Tests for _validate_skill_name with strict=True."""

    def test_strict_valid_name(self):
        """Test that valid names pass strict validation."""
        _validate_skill_name("my-skill", strict=True)  # Should not raise

    def test_strict_empty_name(self):
        """Test that empty name raises in strict mode."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_skill_name("", strict=True)

    def test_strict_too_long_name(self):
        """Test that names exceeding 64 chars raise in strict mode."""
        with pytest.raises(ValueError, match="exceeds 64 character limit"):
            _validate_skill_name("a" * 65, strict=True)

    def test_strict_uppercase_rejected(self):
        """Test that uppercase characters raise in strict mode."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            _validate_skill_name("MySkill", strict=True)

    def test_strict_starts_with_hyphen(self):
        """Test that names starting with hyphen raise in strict mode."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            _validate_skill_name("-skill", strict=True)

    def test_strict_consecutive_hyphens(self):
        """Test that consecutive hyphens raise in strict mode."""
        with pytest.raises(ValueError, match="consecutive hyphens"):
            _validate_skill_name("my--skill", strict=True)

    def test_strict_directory_mismatch(self, tmp_path):
        """Test that directory name mismatch raises in strict mode."""
        skill_dir = tmp_path / "wrong-name"
        skill_dir.mkdir()
        with pytest.raises(ValueError, match="does not match parent directory name"):
            _validate_skill_name("my-skill", skill_dir, strict=True)


class TestFixYamlColons:
    """Tests for _fix_yaml_colons."""

    def test_fixes_unquoted_colon_in_value(self):
        """Test that an unquoted colon in a value gets quoted."""
        raw = "description: Use this skill when: the user asks about PDFs"
        fixed = _fix_yaml_colons(raw)
        assert fixed == 'description: "Use this skill when: the user asks about PDFs"'

    def test_leaves_already_double_quoted_value(self):
        """Test that already double-quoted values are not re-quoted."""
        raw = 'description: "already: quoted"'
        assert _fix_yaml_colons(raw) == raw

    def test_leaves_already_single_quoted_value(self):
        """Test that already single-quoted values are not re-quoted."""
        raw = "description: 'already: quoted'"
        assert _fix_yaml_colons(raw) == raw

    def test_leaves_value_without_colon(self):
        """Test that values without colons are unchanged."""
        raw = "name: my-skill"
        assert _fix_yaml_colons(raw) == raw

    def test_multiline_mixed(self):
        """Test fixing only the lines that need it in a multi-line string."""
        raw = "name: my-skill\ndescription: Use when: needed\nversion: 1.0"
        fixed = _fix_yaml_colons(raw)
        assert fixed == 'name: my-skill\ndescription: "Use when: needed"\nversion: 1.0'

    def test_empty_string(self):
        """Test that an empty string is returned unchanged."""
        assert _fix_yaml_colons("") == ""

    def test_preserves_indented_lines_without_colons(self):
        """Test that indented lines without key-value patterns are preserved."""
        raw = "  - item one\n  - item two"
        assert _fix_yaml_colons(raw) == raw


class TestParseFrontmatterYamlFallback:
    """Tests for YAML colon-quoting fallback in _parse_frontmatter."""

    def test_fallback_on_unquoted_colon(self):
        """Test that frontmatter with unquoted colons in values is parsed via fallback."""
        content = "---\nname: my-skill\ndescription: Use when: the user asks\n---\nBody."
        frontmatter, body = _parse_frontmatter(content)
        assert frontmatter["name"] == "my-skill"
        assert "Use when" in frontmatter["description"]
        assert body == "Body."

    def test_fallback_preserves_valid_yaml(self):
        """Test that valid YAML is parsed normally without triggering fallback."""
        content = "---\nname: my-skill\ndescription: A simple description\n---\nBody."
        frontmatter, body = _parse_frontmatter(content)
        assert frontmatter["name"] == "my-skill"
        assert frontmatter["description"] == "A simple description"


def _make_skill_dir(parent: Path, name: str, description: str = "A test skill", body: str = "Instructions.") -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


class TestSkillFromFile:
    """Tests for Skill.from_file."""

    def test_load_from_directory(self, tmp_path):
        """Test loading a skill from a directory path."""
        skill_dir = _make_skill_dir(tmp_path, "my-skill", "My description", "# Hello\nWorld.")
        skill = Skill.from_file(skill_dir)

        assert skill.name == "my-skill"
        assert skill.description == "My description"
        assert "# Hello" in skill.instructions
        assert "World." in skill.instructions
        assert skill.path == skill_dir.resolve()

    def test_load_from_skill_md_file(self, tmp_path):
        """Test loading a skill by pointing directly to SKILL.md."""
        skill_dir = _make_skill_dir(tmp_path, "direct-skill")
        skill = Skill.from_file(skill_dir / "SKILL.md")

        assert skill.name == "direct-skill"

    def test_load_with_allowed_tools(self, tmp_path):
        """Test loading a skill with allowed-tools field as space-delimited string."""
        skill_dir = tmp_path / "tool-skill"
        skill_dir.mkdir()
        content = "---\nname: tool-skill\ndescription: test\nallowed-tools: read write execute\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = Skill.from_file(skill_dir)
        assert skill.allowed_tools == ["read", "write", "execute"]

    def test_load_with_allowed_tools_yaml_list(self, tmp_path):
        """Test loading a skill with allowed-tools as a YAML list."""
        skill_dir = tmp_path / "list-skill"
        skill_dir.mkdir()
        content = "---\nname: list-skill\ndescription: test\nallowed-tools:\n  - read\n  - write\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = Skill.from_file(skill_dir)
        assert skill.allowed_tools == ["read", "write"]

    def test_load_with_metadata(self, tmp_path):
        """Test loading a skill with nested metadata."""
        skill_dir = tmp_path / "meta-skill"
        skill_dir.mkdir()
        content = "---\nname: meta-skill\ndescription: test\nmetadata:\n  author: acme\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = Skill.from_file(skill_dir)
        assert skill.metadata == {"author": "acme"}

    def test_load_with_license_and_compatibility(self, tmp_path):
        """Test loading a skill with license and compatibility fields."""
        skill_dir = tmp_path / "licensed-skill"
        skill_dir.mkdir()
        content = "---\nname: licensed-skill\ndescription: test\nlicense: MIT\ncompatibility: v1\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content)

        skill = Skill.from_file(skill_dir)
        assert skill.license == "MIT"
        assert skill.compatibility == "v1"

    def test_load_missing_name(self, tmp_path):
        """Test error when SKILL.md is missing name field."""
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: test\n---\nBody.")

        with pytest.raises(ValueError, match="must have a 'name' field"):
            Skill.from_file(skill_dir)

    def test_load_missing_description(self, tmp_path):
        """Test error when SKILL.md is missing description field."""
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\nBody.")

        with pytest.raises(ValueError, match="must have a 'description' field"):
            Skill.from_file(skill_dir)

    def test_load_nonexistent_path(self, tmp_path):
        """Test FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            Skill.from_file(tmp_path / "nonexistent")

    def test_load_name_directory_mismatch_warns(self, tmp_path, caplog):
        """Test that skill name not matching directory name warns but still loads."""
        skill_dir = tmp_path / "wrong-dir"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: right-name\ndescription: test\n---\nBody.")

        with caplog.at_level(logging.WARNING):
            skill = Skill.from_file(skill_dir)

        assert skill.name == "right-name"
        assert "does not match parent directory name" in caplog.text

    def test_strict_rejects_name_mismatch(self, tmp_path):
        """Test that strict mode raises on name/directory mismatch."""
        skill_dir = tmp_path / "wrong-dir"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: right-name\ndescription: test\n---\nBody.")

        with pytest.raises(ValueError, match="does not match parent directory name"):
            Skill.from_file(skill_dir, strict=True)

    def test_strict_accepts_valid_skill(self, tmp_path):
        """Test that strict mode loads a valid skill without error."""
        _make_skill_dir(tmp_path, "valid-skill")
        skill = Skill.from_file(tmp_path / "valid-skill", strict=True)
        assert skill.name == "valid-skill"


class TestSkillFromDirectory:
    """Tests for Skill.from_directory."""

    def test_load_multiple_skills(self, tmp_path):
        """Test loading multiple skills from a parent directory."""
        _make_skill_dir(tmp_path, "skill-a", "Skill A")
        _make_skill_dir(tmp_path, "skill-b", "Skill B")

        skills = Skill.from_directory(tmp_path)

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-a", "skill-b"}

    def test_skips_directories_without_skill_md(self, tmp_path):
        """Test that directories without SKILL.md are silently skipped."""
        _make_skill_dir(tmp_path, "valid-skill")
        (tmp_path / "no-skill-here").mkdir()

        skills = Skill.from_directory(tmp_path)

        assert len(skills) == 1
        assert skills[0].name == "valid-skill"

    def test_skips_files_in_parent(self, tmp_path):
        """Test that files in the parent directory are ignored."""
        _make_skill_dir(tmp_path, "real-skill")
        (tmp_path / "readme.txt").write_text("not a skill")

        skills = Skill.from_directory(tmp_path)

        assert len(skills) == 1

    def test_empty_directory(self, tmp_path):
        """Test loading from an empty directory."""
        skills = Skill.from_directory(tmp_path)
        assert skills == []

    def test_nonexistent_directory(self, tmp_path):
        """Test FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            Skill.from_directory(tmp_path / "nonexistent")

    def test_loads_mismatched_name_with_warning(self, tmp_path, caplog):
        """Test that skills with name/directory mismatch are loaded with a warning."""
        _make_skill_dir(tmp_path, "good-skill")

        bad_dir = tmp_path / "bad-dir"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("---\nname: wrong-name\ndescription: test\n---\nBody.")

        with caplog.at_level(logging.WARNING):
            skills = Skill.from_directory(tmp_path)

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"good-skill", "wrong-name"}
        assert "does not match parent directory name" in caplog.text


class TestSkillFromContent:
    def test_basic_content(self):
        """Test parsing basic SKILL.md content."""
        content = "---\nname: my-skill\ndescription: A useful skill\n---\n# Instructions\nDo the thing."
        skill = Skill.from_content(content)

        assert skill.name == "my-skill"
        assert skill.description == "A useful skill"
        assert "Do the thing." in skill.instructions
        assert skill.path is None

    def test_with_allowed_tools(self):
        """Test parsing content with allowed-tools field."""
        content = "---\nname: my-skill\ndescription: A skill\nallowed-tools: Bash Read\n---\nInstructions."
        skill = Skill.from_content(content)

        assert skill.allowed_tools == ["Bash", "Read"]

    def test_with_metadata(self):
        """Test parsing content with metadata field."""
        content = "---\nname: my-skill\ndescription: A skill\nmetadata:\n  key: value\n---\nInstructions."
        skill = Skill.from_content(content)

        assert skill.metadata == {"key": "value"}

    def test_with_license_and_compatibility(self):
        """Test parsing content with license and compatibility fields."""
        content = (
            "---\nname: my-skill\ndescription: A skill\n"
            "license: Apache-2.0\ncompatibility: Requires docker\n---\nInstructions."
        )
        skill = Skill.from_content(content)

        assert skill.license == "Apache-2.0"
        assert skill.compatibility == "Requires docker"

    def test_missing_name_raises(self):
        """Test that missing name raises ValueError."""
        content = "---\ndescription: A skill\n---\nInstructions."
        with pytest.raises(ValueError, match="name"):
            Skill.from_content(content)

    def test_missing_description_raises(self):
        """Test that missing description raises ValueError."""
        content = "---\nname: my-skill\n---\nInstructions."
        with pytest.raises(ValueError, match="description"):
            Skill.from_content(content)

    def test_missing_frontmatter_raises(self):
        """Test that content without frontmatter raises ValueError."""
        content = "# Just markdown\nNo frontmatter here."
        with pytest.raises(ValueError, match="frontmatter"):
            Skill.from_content(content)

    def test_empty_body(self):
        """Test parsing content with empty body."""
        content = "---\nname: my-skill\ndescription: A skill\n---\n"
        skill = Skill.from_content(content)

        assert skill.name == "my-skill"
        assert skill.instructions == ""

    def test_strict_mode(self):
        """Test Skill.from_content with strict=True raises on validation issues."""
        content = "---\nname: BAD_NAME\ndescription: Bad\n---\nBody."
        with pytest.raises(ValueError):
            Skill.from_content(content, strict=True)


class TestSkillFromUrl:
    """Tests for Skill.from_url."""

    _SKILL_MODULE = "strands.vended_plugins.skills.skill"
    _SAMPLE_CONTENT = "---\nname: my-skill\ndescription: A remote skill\n---\nRemote instructions.\n"

    @pytest.fixture(autouse=True)
    def _resolve_to_public(self):
        """Resolve all hostnames to a public address so tests do not hit DNS.

        Tests that exercise disallowed-host behavior override this with their
        own resolution.
        """
        from unittest.mock import patch

        with patch(
            f"{self._SKILL_MODULE}.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("93.184.216.34", 0))],
        ):
            yield

    def test_from_url_returns_skill(self):
        """Test loading a skill from a URL returns a single Skill."""
        from unittest.mock import patch

        with patch(f"{self._SKILL_MODULE}._fetch_url_content", return_value=self._SAMPLE_CONTENT):
            skill = Skill.from_url("https://raw.githubusercontent.com/org/repo/main/SKILL.md")

        assert isinstance(skill, Skill)
        assert skill.name == "my-skill"
        assert skill.description == "A remote skill"
        assert "Remote instructions." in skill.instructions
        assert skill.path is None

    def test_from_url_invalid_url_raises(self):
        """Test that a non-HTTPS URL raises ValueError."""
        with pytest.raises(ValueError, match="not a valid HTTPS URL"):
            Skill.from_url("./local-path")

    def test_from_url_http_rejected(self):
        """Test that http:// URLs are rejected."""
        with pytest.raises(ValueError, match="not a valid HTTPS URL"):
            Skill.from_url("http://example.com/SKILL.md")

    def test_from_url_http_error_raises(self):
        """Test that HTTP errors propagate as RuntimeError."""
        from unittest.mock import patch

        with patch(
            f"{self._SKILL_MODULE}._fetch_url_content",
            side_effect=RuntimeError("url=<https://example.com/SKILL.md> | HTTP 404: Not Found"),
        ):
            with pytest.raises(RuntimeError, match="HTTP 404"):
                Skill.from_url("https://example.com/SKILL.md")

    def test_from_url_network_error_raises(self):
        """Test that network errors propagate as RuntimeError."""
        from unittest.mock import patch

        with patch(
            f"{self._SKILL_MODULE}._fetch_url_content",
            side_effect=RuntimeError("url=<https://example.com/SKILL.md> | failed to fetch skill: refused"),
        ):
            with pytest.raises(RuntimeError, match="failed to fetch"):
                Skill.from_url("https://example.com/SKILL.md")

    def test_from_url_strict_mode(self):
        """Test that strict mode is forwarded to from_content."""
        from unittest.mock import patch

        bad_content = "---\nname: BAD_NAME\ndescription: Bad\n---\nBody."

        with patch(f"{self._SKILL_MODULE}._fetch_url_content", return_value=bad_content):
            with pytest.raises(ValueError):
                Skill.from_url("https://example.com/SKILL.md", strict=True)

    def test_from_url_invalid_content_raises(self):
        """Test that non-SKILL.md content (e.g. HTML page) raises ValueError."""
        from unittest.mock import patch

        html_content = "<html><body>Not a SKILL.md</body></html>"

        with patch(f"{self._SKILL_MODULE}._fetch_url_content", return_value=html_content):
            with pytest.raises(ValueError, match="frontmatter"):
                Skill.from_url("https://example.com/SKILL.md")

    @pytest.mark.parametrize(
        "url",
        [
            "https://169.254.169.254/SKILL.md",
            "https://[fd00:ec2::254]/SKILL.md",
            "https://127.0.0.1/SKILL.md",
            "https://[::1]/SKILL.md",
            "https://10.0.0.5/SKILL.md",
        ],
    )
    def test_from_url_literal_disallowed_host_rejected(self, url):
        """Test that non-public IP literals are rejected before any connection."""
        from unittest.mock import patch

        with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection") as mock_conn:
            with pytest.raises(ValueError, match="not allowed"):
                Skill.from_url(url)
            mock_conn.assert_not_called()

    def test_from_url_resolved_disallowed_host_rejected(self):
        """Test that a hostname resolving to a non-public address is rejected before any connection."""
        from unittest.mock import patch

        with patch(
            f"{self._SKILL_MODULE}.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("169.254.169.254", 0))],
        ):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection") as mock_conn:
                with pytest.raises(ValueError, match="not allowed"):
                    Skill.from_url("https://internal.example.com/SKILL.md")
                mock_conn.assert_not_called()

    def test_from_url_unresolvable_host_rejected(self):
        """Test that a host that cannot be resolved is rejected before any connection."""
        import socket
        from unittest.mock import patch

        with patch(f"{self._SKILL_MODULE}.socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection") as mock_conn:
                with pytest.raises(ValueError, match="could not resolve host"):
                    Skill.from_url("https://does-not-exist.invalid/SKILL.md")
                mock_conn.assert_not_called()

    def test_from_url_public_host_allowed(self):
        """Test that a host resolving to a public address is allowed to fetch."""
        from unittest.mock import patch

        with patch(f"{self._SKILL_MODULE}._fetch_url_content", return_value=self._SAMPLE_CONTENT):
            skill = Skill.from_url("https://example.com/SKILL.md")

        assert skill.name == "my-skill"


def _make_ip(addr: str):
    """Parse an IP literal into an ipaddress object for host-check tests."""
    import ipaddress

    return ipaddress.ip_address(addr)


class TestIsDisallowedAddress:
    """Tests for _is_disallowed_address, including IPv6-embedded IPv4 encodings."""

    @pytest.mark.parametrize(
        "addr",
        [
            "169.254.169.254",  # IMDS link-local
            "127.0.0.1",  # loopback
            "10.0.0.5",  # private
            "::1",  # IPv6 loopback
            "fd00:ec2::254",  # extra non-public
            "::ffff:169.254.169.254",  # IPv4-mapped IMDS
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "64:ff9b::a9fe:a9fe",  # NAT64 IMDS
            "2002:a9fe:a9fe::",  # 6to4 IMDS
        ],
    )
    def test_disallowed(self, addr):
        """Non-public addresses (including IPv6-embedded IPv4) are rejected."""
        from strands.vended_plugins.skills.skill import _is_disallowed_address

        assert _is_disallowed_address(_make_ip(addr)) is True

    @pytest.mark.parametrize(
        "addr",
        [
            "8.8.8.8",  # public IPv4
            "93.184.216.34",  # public IPv4
            "2606:4700:4700::1111",  # public IPv6
            "::ffff:8.8.8.8",  # IPv4-mapped public IPv4
        ],
    )
    def test_allowed(self, addr):
        """Public addresses are allowed."""
        from strands.vended_plugins.skills.skill import _is_disallowed_address

        assert _is_disallowed_address(_make_ip(addr)) is False


class _FakeResponse:
    """Minimal stand-in for http.client.HTTPResponse."""

    def __init__(self, status, body="", headers=None, reason="OK"):
        self.status = status
        self.reason = reason
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}

    def read(self, amt=None):
        if amt is None:
            return self._body
        return self._body[:amt]


class _FakeConnectionFactory:
    """Builds fake pinned-connection classes that record connect IPs.

    Each queued response is returned in order. The factory records the pinned
    IP each connection was constructed with, so tests can assert the socket was
    dialed at the pre-validated address rather than a re-resolved one.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.pinned_ips = []
        self.hosts = []
        factory = self

        class _FakeConn:
            def __init__(self, host, pinned_ip, *args, **kwargs):
                factory.pinned_ips.append(pinned_ip)
                factory.hosts.append(host)
                self._response = factory._responses.pop(0)

            def request(self, method, target, headers=None):
                self._headers = headers

            def getresponse(self):
                return self._response

            def close(self):
                pass

        self.conn_cls = _FakeConn


class TestFetchUrlIntegration:
    """Integration-style tests for the SSRF-hardened fetch path.

    These exercise the connection/redirect layer where the DNS-rebinding,
    redirect, and IPv6-encoding bypasses lived, rather than the host-check
    helper in isolation.
    """

    _SKILL_MODULE = "strands.vended_plugins.skills.skill"
    _SAMPLE_CONTENT = "---\nname: my-skill\ndescription: A remote skill\n---\nRemote instructions.\n"

    def test_pins_validated_ip_for_connection(self):
        """The connection is dialed at the pre-validated IP, not a re-resolved name."""
        from unittest.mock import patch

        factory = _FakeConnectionFactory([_FakeResponse(200, self._SAMPLE_CONTENT)])
        with patch(
            f"{self._SKILL_MODULE}.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("93.184.216.34", 0))],
        ):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                skill = Skill.from_url("https://example.com/SKILL.md")

        assert skill.name == "my-skill"
        # The socket must have been dialed at the validated address.
        assert factory.pinned_ips == ["93.184.216.34"]
        assert factory.hosts == ["example.com"]

    def test_rebinding_shape_rejected(self):
        """A host that resolves to an IMDS IP is rejected even though the check ran separately.

        The validation resolves once and pins that IP, so a name resolving to a
        disallowed address can never reach the connection layer.
        """
        from unittest.mock import patch

        factory = _FakeConnectionFactory([_FakeResponse(200, self._SAMPLE_CONTENT)])
        with patch(
            f"{self._SKILL_MODULE}.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("169.254.169.254", 0))],
        ):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                with pytest.raises(ValueError, match="not allowed"):
                    Skill.from_url("https://rebind.example.com/SKILL.md")

        assert factory.pinned_ips == []  # never connected

    def test_redirect_to_imds_blocked(self):
        """A 302 redirect to http://169.254.169.254/ is blocked on the redirect hop."""
        from unittest.mock import patch

        resolves = {
            "public.example.com": "93.184.216.34",
            "169.254.169.254": "169.254.169.254",
        }

        def fake_getaddrinfo(host, *args, **kwargs):
            return [(None, None, None, "", (resolves[host], 0))]

        # First hop: public host returns a 302 to the IMDS endpoint.
        factory = _FakeConnectionFactory(
            [_FakeResponse(302, "", headers={"Location": "http://169.254.169.254/latest/meta-data/"})]
        )
        with patch(f"{self._SKILL_MODULE}.socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                with patch(f"{self._SKILL_MODULE}._PinnedHTTPConnection", factory.conn_cls):
                    with pytest.raises(ValueError, match="not allowed|https to http"):
                        Skill.from_url("https://public.example.com/SKILL.md")

    def test_https_redirect_to_imds_revalidated(self):
        """An https->https 302 to the IMDS endpoint is blocked by host re-validation on the hop."""
        from unittest.mock import patch

        resolves = {
            "public.example.com": "93.184.216.34",
            "169.254.169.254": "169.254.169.254",
        }

        def fake_getaddrinfo(host, *args, **kwargs):
            return [(None, None, None, "", (resolves[host], 0))]

        factory = _FakeConnectionFactory(
            [_FakeResponse(302, "", headers={"Location": "https://169.254.169.254/latest/meta-data/"})]
        )
        with patch(f"{self._SKILL_MODULE}.socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                with pytest.raises(ValueError, match="not allowed"):
                    Skill.from_url("https://public.example.com/SKILL.md")

        # Only the first (public) hop connected; the redirect target was rejected.
        assert factory.pinned_ips == ["93.184.216.34"]

    def test_redirect_to_public_https_followed(self):
        """A redirect to another public https host is followed and re-validated."""
        from unittest.mock import patch

        resolves = {
            "first.example.com": "93.184.216.34",
            "second.example.com": "93.184.216.35",
        }

        def fake_getaddrinfo(host, *args, **kwargs):
            return [(None, None, None, "", (resolves[host], 0))]

        factory = _FakeConnectionFactory(
            [
                _FakeResponse(302, "", headers={"Location": "https://second.example.com/SKILL.md"}),
                _FakeResponse(200, self._SAMPLE_CONTENT),
            ]
        )
        with patch(f"{self._SKILL_MODULE}.socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                skill = Skill.from_url("https://first.example.com/SKILL.md")

        assert skill.name == "my-skill"
        assert factory.pinned_ips == ["93.184.216.34", "93.184.216.35"]

    @pytest.mark.parametrize(
        "resolved_ip",
        [
            "::ffff:169.254.169.254",  # IPv4-mapped IMDS
            "64:ff9b::a9fe:a9fe",  # NAT64 IMDS
            "2002:a9fe:a9fe::",  # 6to4 IMDS
        ],
    )
    def test_ipv6_encoded_imds_rejected(self, resolved_ip):
        """Hosts resolving to IPv6-encoded IMDS addresses are rejected before connecting."""
        from unittest.mock import patch

        factory = _FakeConnectionFactory([_FakeResponse(200, self._SAMPLE_CONTENT)])
        with patch(
            f"{self._SKILL_MODULE}.socket.getaddrinfo",
            return_value=[(None, None, None, "", (resolved_ip, 0))],
        ):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                with pytest.raises(ValueError, match="not allowed"):
                    Skill.from_url("https://sneaky.example.com/SKILL.md")

        assert factory.pinned_ips == []

    def test_too_many_redirects_error(self):
        """A redirect loop is capped and surfaced as an error."""
        from unittest.mock import patch

        factory = _FakeConnectionFactory(
            [_FakeResponse(302, "", headers={"Location": "https://public.example.com/next"}) for _ in range(12)]
        )
        with patch(
            f"{self._SKILL_MODULE}.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("93.184.216.34", 0))],
        ):
            with patch(f"{self._SKILL_MODULE}._PinnedHTTPSConnection", factory.conn_cls):
                with pytest.raises(RuntimeError, match="too many redirects"):
                    Skill.from_url("https://public.example.com/SKILL.md")


class TestSkillClassmethods:
    """Tests for Skill classmethod existence."""

    def test_skill_classmethods_exist(self):
        """Test that Skill has from_file, from_content, from_directory, and from_url classmethods."""
        assert callable(getattr(Skill, "from_file", None))
        assert callable(getattr(Skill, "from_content", None))
        assert callable(getattr(Skill, "from_directory", None))
        assert callable(getattr(Skill, "from_url", None))
