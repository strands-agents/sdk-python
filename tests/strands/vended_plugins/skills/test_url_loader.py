"""Tests for the _url_loader module."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from strands.vended_plugins.skills._url_loader import (
    _default_cache_dir,
    cache_key,
    clone_skill_repo,
    is_url,
    parse_url_ref,
)


class TestIsUrl:
    """Tests for is_url."""

    def test_https_github_url(self):
        assert is_url("https://github.com/org/skill-repo") is True

    def test_http_url_rejected(self):
        """Plaintext http:// is not supported for security reasons."""
        assert is_url("http://github.com/org/skill-repo") is False

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
        url, ref, subpath = parse_url_ref("https://github.com/org/skill-repo")
        assert url == "https://github.com/org/skill-repo"
        assert ref is None
        assert subpath is None

    def test_https_with_tag_ref(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/skill-repo@v1.0.0")
        assert url == "https://github.com/org/skill-repo"
        assert ref == "v1.0.0"
        assert subpath is None

    def test_https_with_branch_ref(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/skill-repo@main")
        assert url == "https://github.com/org/skill-repo"
        assert ref == "main"
        assert subpath is None

    def test_https_git_suffix_no_ref(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/skill-repo.git")
        assert url == "https://github.com/org/skill-repo.git"
        assert ref is None
        assert subpath is None

    def test_https_git_suffix_with_ref(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/skill-repo.git@v2.0")
        assert url == "https://github.com/org/skill-repo.git"
        assert ref == "v2.0"
        assert subpath is None

    def test_ssh_no_ref(self):
        url, ref, subpath = parse_url_ref("git@github.com:org/skill-repo.git")
        assert url == "git@github.com:org/skill-repo.git"
        assert ref is None
        assert subpath is None

    def test_ssh_with_ref(self):
        url, ref, subpath = parse_url_ref("git@github.com:org/skill-repo.git@v1.0")
        assert url == "git@github.com:org/skill-repo.git"
        assert ref == "v1.0"
        assert subpath is None

    def test_ssh_no_git_suffix_with_ref(self):
        url, ref, subpath = parse_url_ref("git@github.com:org/skill-repo@develop")
        assert url == "git@github.com:org/skill-repo"
        assert ref == "develop"
        assert subpath is None

    def test_ssh_protocol_no_ref(self):
        url, ref, subpath = parse_url_ref("ssh://git@github.com/org/skill-repo")
        assert url == "ssh://git@github.com/org/skill-repo"
        assert ref is None
        assert subpath is None

    def test_ssh_protocol_with_ref(self):
        url, ref, subpath = parse_url_ref("ssh://git@github.com/org/skill-repo@v3")
        assert url == "ssh://git@github.com/org/skill-repo"
        assert ref == "v3"
        assert subpath is None

    def test_http_with_ref(self):
        """http:// URLs are still parsed (parse_url_ref doesn't enforce security)."""
        url, ref, subpath = parse_url_ref("http://gitlab.com/org/skill@feature-branch")
        assert url == "http://gitlab.com/org/skill"
        assert ref == "feature-branch"
        assert subpath is None

    def test_url_host_only(self):
        url, ref, subpath = parse_url_ref("https://example.com")
        assert url == "https://example.com"
        assert ref is None
        assert subpath is None

    def test_non_url_passthrough(self):
        url, ref, subpath = parse_url_ref("/local/path")
        assert url == "/local/path"
        assert ref is None
        assert subpath is None


class TestParseUrlRefGitHubTree:
    """Tests for parse_url_ref with GitHub /tree/ and /blob/ URLs."""

    def test_tree_with_subpath(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/repo/tree/main/skills/my-skill")
        assert url == "https://github.com/org/repo"
        assert ref == "main"
        assert subpath == "skills/my-skill"

    def test_tree_branch_only(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/repo/tree/main")
        assert url == "https://github.com/org/repo"
        assert ref == "main"
        assert subpath is None

    def test_tree_with_tag(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/repo/tree/v1.0.0/skills/brainstorming")
        assert url == "https://github.com/org/repo"
        assert ref == "v1.0.0"
        assert subpath == "skills/brainstorming"

    def test_tree_deep_subpath(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/repo/tree/develop/a/b/c/d")
        assert url == "https://github.com/org/repo"
        assert ref == "develop"
        assert subpath == "a/b/c/d"

    def test_blob_url(self):
        """Test that /blob/ URLs are handled like /tree/."""
        url, ref, subpath = parse_url_ref("https://github.com/org/repo/blob/main/skills/my-skill")
        assert url == "https://github.com/org/repo"
        assert ref == "main"
        assert subpath == "skills/my-skill"

    def test_tree_trailing_slash(self):
        url, ref, subpath = parse_url_ref("https://github.com/org/repo/tree/main/skills/my-skill/")
        assert url == "https://github.com/org/repo"
        assert ref == "main"
        assert subpath == "skills/my-skill"


class TestDefaultCacheDir:
    """Tests for _default_cache_dir."""

    def test_respects_xdg_cache_home(self, tmp_path):
        """Test that XDG_CACHE_HOME is respected."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path / "xdg")}):
            result = _default_cache_dir()
        assert result == tmp_path / "xdg" / "strands" / "skills"

    def test_falls_back_to_home_cache(self):
        """Test that without XDG_CACHE_HOME, falls back to ~/.cache."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove XDG_CACHE_HOME if set
            env = os.environ.copy()
            env.pop("XDG_CACHE_HOME", None)
            with patch.dict(os.environ, env, clear=True):
                result = _default_cache_dir()
        assert result == Path.home() / ".cache" / "strands" / "skills"


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


def _fake_clone_factory():
    """Return a fake clone function that creates the target directory via atomic rename."""

    def fake_clone(cmd, **kwargs):
        target_dir = Path(cmd[-1])
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_clone


class TestCloneSkillRepo:
    """Tests for clone_skill_repo."""

    def test_clone_success(self, tmp_path):
        """Test successful clone by mocking subprocess.run."""
        cache = tmp_path / "cache"

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=_fake_clone_factory()):
            result = clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert result.exists()

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

    def test_clone_failure_cleans_up_temp_dir(self, tmp_path):
        """Test that a failed clone removes the temp directory."""
        cache = tmp_path / "cache"
        cache.mkdir()

        with patch(
            "strands.vended_plugins.skills._url_loader.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr="fatal: error"),
        ):
            with pytest.raises(RuntimeError):
                clone_skill_repo("https://github.com/org/broken", cache_dir=cache)

        # Only the cache dir itself should remain, no temp or target dirs
        remaining = [p for p in cache.iterdir() if not p.name.startswith(".")]
        assert len(remaining) == 0

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

    def test_subpath_returns_subdirectory(self, tmp_path):
        """Test that subpath parameter returns the subdirectory within the clone."""
        cache = tmp_path / "cache"

        def fake_clone(cmd, **kwargs):
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            skill_dir = target_dir / "skills" / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            result = clone_skill_repo("https://github.com/org/repo", subpath="skills/my-skill", cache_dir=cache)

        assert result.name == "my-skill"
        assert (result / "SKILL.md").exists()

    def test_subpath_nonexistent_raises(self, tmp_path):
        """Test that a nonexistent subpath raises RuntimeError."""
        cache = tmp_path / "cache"

        def fake_clone(cmd, **kwargs):
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            with pytest.raises(RuntimeError, match="subdirectory does not exist"):
                clone_skill_repo("https://github.com/org/repo", subpath="nonexistent/path", cache_dir=cache)

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

    def test_force_refresh_reclones(self, tmp_path):
        """Test that force_refresh=True deletes cache and re-clones."""
        cache = tmp_path / "cache"
        call_count = 0

        def fake_clone(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "marker.txt").write_text(f"clone-{call_count}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            result1 = clone_skill_repo("https://github.com/org/skill", cache_dir=cache)
            assert (result1 / "marker.txt").read_text() == "clone-1"

            result2 = clone_skill_repo("https://github.com/org/skill", cache_dir=cache, force_refresh=True)
            assert (result2 / "marker.txt").read_text() == "clone-2"

        assert call_count == 2
        assert result1 == result2

    def test_force_refresh_noop_when_no_cache(self, tmp_path):
        """Test that force_refresh=True works even when there's no existing cache."""
        cache = tmp_path / "cache"

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=_fake_clone_factory()):
            result = clone_skill_repo("https://github.com/org/skill", cache_dir=cache, force_refresh=True)

        assert result.exists()

    def test_race_condition_other_process_wins(self, tmp_path):
        """Test that when another process clones first (rename fails), we use their clone."""
        cache = tmp_path / "cache"
        key_hash = cache_key("https://github.com/org/skill", None)
        target = cache / key_hash

        def fake_clone(cmd, **kwargs):
            target_dir = Path(cmd[-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            # Simulate another process completing the clone first
            target.mkdir(parents=True, exist_ok=True)
            (target / "SKILL.md").write_text("---\nname: winner\ndescription: test\n---\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("strands.vended_plugins.skills._url_loader.subprocess.run", side_effect=fake_clone):
            result = clone_skill_repo("https://github.com/org/skill", cache_dir=cache)

        assert result == target
        assert result.exists()
