"""Tests for the _url_loader module."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from strands.vended_plugins.skills._url_loader import (
    fetch_skill_content,
    is_url,
)


class TestIsUrl:
    """Tests for is_url."""

    def test_https_url(self):
        assert is_url("https://example.com/SKILL.md") is True

    def test_https_raw_github_url(self):
        assert is_url("https://raw.githubusercontent.com/org/repo/main/SKILL.md") is True

    def test_http_rejected(self):
        """Plaintext http:// is rejected for security."""
        assert is_url("http://example.com/SKILL.md") is False

    def test_ssh_rejected(self):
        assert is_url("ssh://git@github.com/org/repo") is False

    def test_git_at_rejected(self):
        assert is_url("git@github.com:org/repo.git") is False

    def test_local_relative_path(self):
        assert is_url("./skills/my-skill") is False

    def test_local_absolute_path(self):
        assert is_url("/home/user/skills/my-skill") is False

    def test_plain_directory_name(self):
        assert is_url("my-skill") is False

    def test_empty_string(self):
        assert is_url("") is False


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

    def test_fetch_uses_url_directly(self):
        """Test that the URL is used as-is with no resolution."""
        url = "https://raw.githubusercontent.com/org/repo/main/skills/my-skill/SKILL.md"

        mock_response = MagicMock()
        mock_response.read.return_value = b"---\nname: t\ndescription: t\n---\n"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(f"{self._LOADER}.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            fetch_skill_content(url)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.full_url == url

    def test_fetch_sets_user_agent(self):
        """Test that requests include a User-Agent header."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"---\nname: t\ndescription: t\n---\n"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(f"{self._LOADER}.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            fetch_skill_content("https://example.com/SKILL.md")

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("User-agent") == "strands-agents-sdk"

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

    def test_fetch_rejects_non_https(self):
        """Test that non-https URLs are rejected."""
        with pytest.raises(ValueError, match="only https://"):
            fetch_skill_content("http://example.com/SKILL.md")
