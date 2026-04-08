"""Tests for the _url_loader module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from strands.vended_plugins.skills._url_loader import (
    cache_key,
    clone_skill_repo,
    is_url,
    parse_url_ref,
)


class TestIsUrl:
    """Tests for is_url."""

    def test_https_github_url(self):
        assert is_url("https://github.com/org/skill-repo") is True

    def test_http_url(self):
        assert is_url("http://github.com/org/skill-repo") is True

    def test_ssh_url(self):
        assert is_url("ssh://git@github.com/org/skill-repo") is True

    def test_git_at_url(self):
        assert is_url("git@github.com:org/skill-repo.git") is True

    def test_local_relative_path(self):
        assert is_url("./skills/my-skill") is False

    def test_local_absolute_path(self):
        assert is_url("/home/user/skills/my-skill") is False

    def test_plain_directory_name(self):
        assert is_url("my-skill") is False

    def test_empty_string(self):
        assert is_url("") is False

    def test_https_with_ref(self):
        assert is_url("https://github.com/org/skill@v1.0.0") is True


class TestParseUrlRef:
    """Tests for parse_url_ref."""

    def test_https_no_ref(self):
        url, ref = parse_url_ref("https://github.com/org/skill-repo")
        assert url == "https://github.com/org/skill-repo"
        assert ref is None

    def test_https_with_tag_ref(self):
        url, ref = parse_url_ref("https://github.com/org/skill-repo@v1.0.0")
        assert url == "https://github.com/org/skill-repo"
        assert ref == "v1.0.0"

    def test_https_with_branch_ref(self):
        url, ref = parse_url_ref("https://github.com/org/skill-repo@main")
        assert url == "https://github.com/org/skill-repo"
        assert ref == "main"

    def test_https_git_suffix_no_ref(self):
        url, ref = parse_url_ref("https://github.com/org/skill-repo.git")
        assert url == "https://github.com/org/skill-repo.git"
        assert ref is None

    def test_https_git_suffix_with_ref(self):
        url, ref = parse_url_ref("https://github.com/org/skill-repo.git@v2.0")
        assert url == "https://github.com/org/skill-repo.git"
        assert ref == "v2.0"

    def test_ssh_no_ref(self):
        url, ref = parse_url_ref("git@github.com:org/skill-repo.git")
        assert url == "git@github.com:org/skill-repo.git"
        assert ref is None

    def test_ssh_with_ref(self):
        url, ref = parse_url_ref("git@github.com:org/skill-repo.git@v1.0")
        assert url == "git@github.com:org/skill-repo.git"
        assert ref == "v1.0"

    def test_ssh_no_git_suffix_with_ref(self):
        url, ref = parse_url_ref("git@github.com:org/skill-repo@develop")
        assert url == "git@github.com:org/skill-repo"
        assert ref == "develop"

    def test_ssh_protocol_no_ref(self):
        url, ref = parse_url_ref("ssh://git@github.com/org/skill-repo")
        assert url == "ssh://git@github.com/org/skill-repo"
        assert ref is None

    def test_ssh_protocol_with_ref(self):
        url, ref = parse_url_ref("ssh://git@github.com/org/skill-repo@v3")
        assert url == "ssh://git@github.com/org/skill-repo"
        assert ref == "v3"

    def test_http_with_ref(self):
        url, ref = parse_url_ref("http://gitlab.com/org/skill@feature-branch")
        assert url == "http://gitlab.com/org/skill"
        assert ref == "feature-branch"

    def test_url_host_only(self):
        url, ref = parse_url_ref("https://example.com")
        assert url == "https://example.com"
        assert ref is None

    def test_non_url_passthrough(self):
        url, ref = parse_url_ref("/local/path")
        assert url == "/local/path"
        assert ref is None


class TestCacheKey:
    """Tests for cache_key."""

    def test_deterministic(self):
        key1 = cache_key("https://github.com/org/repo", None)
        key2 = cache_key("https://github.com/org/repo", None)
        assert key1 == key2

    def test_different_url_different_key(self):
        key1 = cache_key("https://github.com/org/repo-a", None)
        key2 = cache_key("https://github.com/org/repo-b", None)
        assert key1 != key2

    def test_different_ref_different_key(self):
        key1 = cache_key("https://github.com/org/repo", None)
        key2 = cache_key("https://github.com/org/repo", "v1.0")
        assert key1 != key2

    def test_length(self):
        key = cache_key("https://github.com/org/repo", "main")
        assert len(key) == 16

    def test_hex_characters_only(self):
        key = cache_key("https://github.com/org/repo", "main")
        assert all(c in "0123456789abcdef" for c in key)


class TestCloneSkillRepo:
    """Tests for clone_skill_repo."""

    def test_clone_success(self, tmp_path):
        """Test successful clone by mocking subprocess.run."""
        cache = tmp_path / "cache"

        def fake_clone(cmd, **kwargs):
            # Simulate a successful clone by creating the target directory
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            result = clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert result.exists()
        assert (result / "SKILL.md").exists()

    def test_clone_with_ref(self, tmp_path):
        """Test that ref is passed as --branch to git clone."""
        cache = tmp_path / "cache"
        captured_cmd = []

        def fake_clone(cmd, **kwargs):
            captured_cmd.extend(cmd)
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            clone_skill_repo("https://github.com/org/skill", ref="v1.0", cache_dir=cache)

        assert "--branch" in captured_cmd
        assert "v1.0" in captured_cmd

    def test_clone_without_ref(self, tmp_path):
        """Test that --branch is not passed when ref is None."""
        cache = tmp_path / "cache"
        captured_cmd = []

        def fake_clone(cmd, **kwargs):
            captured_cmd.extend(cmd)
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert "--branch" not in captured_cmd

    def test_uses_cache_on_second_call(self, tmp_path):
        """Test that the cache is used on the second call."""
        cache = tmp_path / "cache"
        call_count = 0

        def fake_clone(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            result1 = clone_skill_repo("https://github.com/org/skill", cache_dir=cache)
            result2 = clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert call_count == 1
        assert result1 == result2

    def test_clone_failure_raises_runtime_error(self, tmp_path):
        """Test that a failed clone raises RuntimeError."""
        cache = tmp_path / "cache"

        with patch(
            "strands.vended_plugins.skills._url_loader.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr="fatal: repo not found"),
        ):
            with pytest.raises(RuntimeError, match="failed to clone"):
                clone_skill_repo("https://github.com/org/nonexistent", cache_dir=cache)

    def test_clone_failure_cleans_up_partial(self, tmp_path):
        """Test that a failed clone removes any partial directory."""
        cache = tmp_path / "cache"

        def failing_clone(cmd, **kwargs):
            # Create partial clone then fail
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            raise subprocess.CalledProcessError(128, "git", stderr="fatal: error")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=failing_clone):
            with pytest.raises(RuntimeError):
                clone_skill_repo("https://github.com/org/broken", cache_dir=cache)

        # Verify no leftover directory
        assert len(list(cache.iterdir())) == 0

    def test_git_not_found_raises_runtime_error(self, tmp_path):
        """Test that missing git binary raises RuntimeError."""
        cache = tmp_path / "cache"

        with patch(
            "strands.vended_plugins.skills._url_loader.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            with pytest.raises(RuntimeError, match="git is required"):
                clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

    def test_creates_cache_dir_if_missing(self, tmp_path):
        """Test that the cache directory is created if it doesn't exist."""
        cache = tmp_path / "deep" / "nested" / "cache"

        def fake_clone(cmd, **kwargs):
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert cache.exists()

    def test_shallow_clone_depth_one(self, tmp_path):
        """Test that --depth 1 is always passed."""
        cache = tmp_path / "cache"
        captured_cmd = []

        def fake_clone(cmd, **kwargs):
            captured_cmd.extend(cmd)
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert "--depth" in captured_cmd
        depth_idx = captured_cmd.index("--depth")
        assert captured_cmd[depth_idx + 1] == "1"
