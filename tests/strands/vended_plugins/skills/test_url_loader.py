"""Tests for the _url_loader module."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from strands.vended_plugins.skills._url_loader import (
    fetch_skill_content,
    is_url,
    resolve_to_raw_url,
)


class TestIsUrl:
    """Tests for is_url."""

    def test_https_url(self):
        assert is_url("https://github.com/org/skill-repo") is True

    def test_https_raw_url(self):
        assert is_url("https://raw.githubusercontent.com/org/repo/main/SKILL.md") is True

    def test_http_rejected(self):
        """Plaintext http:// is rejected for security."""
        assert is_url("http://github.com/org/skill-repo") is False

    def test_ssh_rejected(self):
        """SSH URLs are not supported in HTTPS-only mode."""
        assert is_url("ssh://git@github.com/org/skill-repo") is False

    def test_git_at_rejected(self):
        """git@ URLs are not supported in HTTPS-only mode."""
        assert is_url("git@github.com:org/skill-repo.git") is False

    def test_local_relative_path(self):
        assert is_url("./skills/my-skill") is False

    def test_local_absolute_path(self):
        assert is_url("/home/user/skills/my-skill") is False

    def test_plain_directory_name(self):
        assert is_url("my-skill") is False

    def test_empty_string(self):
        assert is_url("") is False


class TestResolveToRawUrl:
    """Tests for resolve_to_raw_url."""

    def test_raw_url_passthrough(self):
        url = "https://raw.githubusercontent.com/org/repo/main/SKILL.md"
        assert resolve_to_raw_url(url) == url

    def test_non_github_passthrough(self):
        url = "https://example.com/skills/SKILL.md"
        assert resolve_to_raw_url(url) == url

    def test_repo_root(self):
        assert resolve_to_raw_url("https://github.com/org/repo") == (
            "https://raw.githubusercontent.com/org/repo/HEAD/SKILL.md"
        )

    def test_repo_root_trailing_slash(self):
        assert resolve_to_raw_url("https://github.com/org/repo/") == (
            "https://raw.githubusercontent.com/org/repo/HEAD/SKILL.md"
        )

    def test_repo_root_with_ref(self):
        assert resolve_to_raw_url("https://github.com/org/repo@v1.0.0") == (
            "https://raw.githubusercontent.com/org/repo/v1.0.0/SKILL.md"
        )

    def test_repo_root_with_branch_ref(self):
        assert resolve_to_raw_url("https://github.com/org/repo@main") == (
            "https://raw.githubusercontent.com/org/repo/main/SKILL.md"
        )

    def test_tree_url_directory(self):
        assert resolve_to_raw_url("https://github.com/org/repo/tree/main/skills/my-skill") == (
            "https://raw.githubusercontent.com/org/repo/main/skills/my-skill/SKILL.md"
        )

    def test_tree_url_branch_only(self):
        assert resolve_to_raw_url("https://github.com/org/repo/tree/main") == (
            "https://raw.githubusercontent.com/org/repo/main/SKILL.md"
        )

    def test_tree_url_trailing_slash(self):
        assert resolve_to_raw_url("https://github.com/org/repo/tree/main/skills/my-skill/") == (
            "https://raw.githubusercontent.com/org/repo/main/skills/my-skill/SKILL.md"
        )

    def test_tree_url_with_tag(self):
        assert resolve_to_raw_url("https://github.com/org/repo/tree/v2.0/skills/brainstorming") == (
            "https://raw.githubusercontent.com/org/repo/v2.0/skills/brainstorming/SKILL.md"
        )

    def test_blob_url_to_skill_md(self):
        assert resolve_to_raw_url("https://github.com/org/repo/blob/main/skills/my-skill/SKILL.md") == (
            "https://raw.githubusercontent.com/org/repo/main/skills/my-skill/SKILL.md"
        )

    def test_blob_url_to_lowercase_skill_md(self):
        assert resolve_to_raw_url("https://github.com/org/repo/blob/main/skills/my-skill/skill.md") == (
            "https://raw.githubusercontent.com/org/repo/main/skills/my-skill/skill.md"
        )


class TestFetchSkillContent:
    """Tests for fetch_skill_content."""

    _LOADER = "strands.vended_plugins.skills._url_loader"

    def test_fetch_success(self):
        """Test successful content fetch."""
        skill_content = "---\nname: test-skill\ndescription: A test\n---\n# Instructions\n"

        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(f"{self._LOADER}.urllib.request.urlopen", return_value=mock_response):
            result = fetch_skill_content("https://raw.githubusercontent.com/org/repo/main/SKILL.md")

        assert result == skill_content

    def test_fetch_resolves_github_url(self):
        """Test that GitHub web URLs are resolved before fetching."""
        skill_content = "---\nname: test\ndescription: test\n---\n"

        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(f"{self._LOADER}.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            fetch_skill_content("https://github.com/org/repo/tree/main/skills/my-skill")

        # Verify the resolved raw URL was used
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert "raw.githubusercontent.com" in request_obj.full_url

    def test_fetch_http_error(self):
        """Test that HTTP errors raise RuntimeError."""
        with patch(
            f"{self._LOADER}.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://example.com", code=404, msg="Not Found", hdrs=None, fp=None
            ),
        ):
            with pytest.raises(RuntimeError, match="HTTP 404"):
                fetch_skill_content("https://example.com/SKILL.md")

    def test_fetch_url_error(self):
        """Test that network errors raise RuntimeError."""
        with patch(
            f"{self._LOADER}.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="failed to fetch"):
                fetch_skill_content("https://example.com/SKILL.md")
