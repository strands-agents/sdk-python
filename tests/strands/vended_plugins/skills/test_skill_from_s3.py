"""Tests for Skill.from_s3 using moto to mock S3."""

import boto3
import pytest
from moto import mock_aws

from strands.vended_plugins.skills.skill import (
    _S3_MIRROR_CACHE,
    Skill,
    _mirror_skills_from_s3,
    _s3_build_download_tasks,
    _s3_find_skill_directories,
    _s3_list_all_objects,
)

BUCKET = "test-skills-bucket"
PREFIX = "my-agent/"

SKILL_MD_CONTENT = """\
---
name: {name}
description: {description}
---
# {name}

These are the instructions for {name}.
"""


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the S3 mirror cache before each test."""
    _S3_MIRROR_CACHE.clear()
    yield
    _S3_MIRROR_CACHE.clear()


def _put_skill(s3_client, skill_name: str, *, prefix: str = PREFIX, with_resources: bool = False):
    """Upload a skill directory to the mocked S3 bucket."""
    content = SKILL_MD_CONTENT.format(
        name=skill_name,
        description=f"Description for {skill_name}",
    )
    s3_client.put_object(
        Bucket=BUCKET,
        Key=f"{prefix}{skill_name}/SKILL.md",
        Body=content.encode(),
    )

    if with_resources:
        s3_client.put_object(
            Bucket=BUCKET,
            Key=f"{prefix}{skill_name}/scripts/run.py",
            Body=b"#!/usr/bin/env python3\nprint('hello')\n",
        )
        s3_client.put_object(
            Bucket=BUCKET,
            Key=f"{prefix}{skill_name}/references/guide.md",
            Body=b"# Guide\nSome reference content.\n",
        )
        s3_client.put_object(
            Bucket=BUCKET,
            Key=f"{prefix}{skill_name}/assets/config.json",
            Body=b'{"key": "value"}\n',
        )


class TestSkillFromS3:
    """Tests for Skill.from_s3 classmethod."""

    @mock_aws
    def test_loads_single_skill(self, tmp_path):
        """Single skill with no resources loads correctly."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        _put_skill(s3_client, "code-review")

        skills = Skill.from_s3(
            BUCKET,
            prefix=PREFIX,
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
        )

        assert len(skills) == 1
        assert skills[0].name == "code-review"
        assert skills[0].description == "Description for code-review"
        assert "instructions for code-review" in skills[0].instructions

    @mock_aws
    def test_loads_multiple_skills(self, tmp_path):
        """Multiple skills are discovered and loaded."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        _put_skill(s3_client, "pdf-processing")
        _put_skill(s3_client, "code-review")
        _put_skill(s3_client, "summarize")

        skills = Skill.from_s3(
            BUCKET,
            prefix=PREFIX,
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
        )

        assert len(skills) == 3
        names = {s.name for s in skills}
        assert names == {"pdf-processing", "code-review", "summarize"}

    @mock_aws
    def test_mirrors_resource_files(self, tmp_path):
        """Resource directories (scripts/, references/, assets/) are mirrored."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        _put_skill(s3_client, "data-analysis", with_resources=True)

        skills = Skill.from_s3(
            BUCKET,
            prefix=PREFIX,
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
        )

        assert len(skills) == 1
        skill_path = skills[0].path
        assert skill_path is not None
        assert (skill_path / "SKILL.md").exists()
        assert (skill_path / "scripts" / "run.py").exists()
        assert (skill_path / "references" / "guide.md").exists()
        assert (skill_path / "assets" / "config.json").exists()

    @mock_aws
    def test_empty_prefix_returns_empty(self, tmp_path):
        """Empty bucket prefix returns an empty list without error."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)

        skills = Skill.from_s3(
            BUCKET,
            prefix="nonexistent/",
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
        )

        assert skills == []

    @mock_aws
    def test_no_prefix_scans_entire_bucket(self, tmp_path):
        """When prefix is None, scans the entire bucket."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)

        content = SKILL_MD_CONTENT.format(name="root-skill", description="A root-level skill")
        s3_client.put_object(
            Bucket=BUCKET,
            Key="root-skill/SKILL.md",
            Body=content.encode(),
        )

        skills = Skill.from_s3(
            BUCKET,
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
        )

        assert len(skills) == 1
        assert skills[0].name == "root-skill"

    @mock_aws
    def test_caching(self, tmp_path):
        """Second call with same (bucket, prefix) returns cached result without re-downloading."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)

        content = SKILL_MD_CONTENT.format(name="cached-skill", description="Cached")
        s3_client.put_object(
            Bucket=BUCKET,
            Key="cache-test/cached-skill/SKILL.md",
            Body=content.encode(),
        )

        local = tmp_path / "skills"
        skills1 = Skill.from_s3(
            BUCKET,
            prefix="cache-test/",
            s3_client=s3_client,
            local_dir=local,
        )

        # Delete from S3 — second call should still return cached result
        s3_client.delete_object(Bucket=BUCKET, Key="cache-test/cached-skill/SKILL.md")
        skills2 = Skill.from_s3(
            BUCKET,
            prefix="cache-test/",
            s3_client=s3_client,
            local_dir=local,
        )

        assert len(skills1) == len(skills2)
        assert skills1[0].name == skills2[0].name

    @mock_aws
    def test_strict_mode_propagates(self, tmp_path):
        """Strict mode is passed through to from_directory, skipping invalid skills."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)

        # Create a skill with a name that doesn't match its directory
        s3_client.put_object(
            Bucket=BUCKET,
            Key=f"{PREFIX}wrong-dir/SKILL.md",
            Body=b"---\nname: right-name\ndescription: test\n---\nBody.",
        )

        # In strict mode, from_file raises but from_directory catches and skips
        skills = Skill.from_s3(
            BUCKET,
            prefix=PREFIX,
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
            strict=True,
        )

        # The mismatched skill is skipped, resulting in an empty list
        assert skills == []

    @mock_aws
    def test_prefix_trailing_slash_normalized(self, tmp_path):
        """Prefix with or without trailing slash produces the same result."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        _put_skill(s3_client, "my-skill", prefix="agents/")

        skills_with_slash = Skill.from_s3(
            BUCKET,
            prefix="agents/",
            s3_client=s3_client,
            local_dir=tmp_path / "s1",
        )

        _S3_MIRROR_CACHE.clear()

        skills_without_slash = Skill.from_s3(
            BUCKET,
            prefix="agents",
            s3_client=s3_client,
            local_dir=tmp_path / "s2",
        )

        assert len(skills_with_slash) == len(skills_without_slash) == 1
        assert skills_with_slash[0].name == skills_without_slash[0].name

    @mock_aws
    def test_no_skill_md_returns_empty(self, tmp_path):
        """Bucket with files but no SKILL.md returns empty list."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        s3_client.put_object(
            Bucket=BUCKET,
            Key=f"{PREFIX}some-dir/readme.txt",
            Body=b"not a skill",
        )

        skills = Skill.from_s3(
            BUCKET,
            prefix=PREFIX,
            s3_client=s3_client,
            local_dir=tmp_path / "skills",
        )

        assert skills == []


class TestMirrorSkillsFromS3:
    """Tests for the _mirror_skills_from_s3 helper."""

    @mock_aws
    def test_creates_temp_dir_when_local_dir_is_none(self):
        """When local_dir is None, a temp directory is created."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        _put_skill(s3_client, "temp-skill")

        result = _mirror_skills_from_s3(BUCKET, prefix=PREFIX, s3_client=s3_client)

        assert result.exists()
        assert result.is_dir()
        assert "strands-s3-skills-" in str(result)

    @mock_aws
    def test_creates_local_dir_if_not_exists(self, tmp_path):
        """When local_dir doesn't exist, it is created."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        _put_skill(s3_client, "new-dir-skill")

        target = tmp_path / "nested" / "dir"
        result = _mirror_skills_from_s3(BUCKET, prefix=PREFIX, s3_client=s3_client, local_dir=target)

        assert result == target
        assert result.exists()


class TestS3HelperFunctions:
    """Tests for internal S3 helper functions."""

    def test_find_skill_directories(self):
        """Test _s3_find_skill_directories discovers correct directories."""
        keys = [
            "prefix/skill-a/SKILL.md",
            "prefix/skill-a/scripts/run.py",
            "prefix/skill-b/SKILL.md",
            "prefix/other/readme.txt",
        ]
        result = _s3_find_skill_directories(keys, "prefix/")
        assert result == ["skill-a", "skill-b"]

    def test_find_skill_directories_empty(self):
        """Test _s3_find_skill_directories with no SKILL.md files."""
        keys = ["prefix/dir/readme.txt", "prefix/dir/other.py"]
        result = _s3_find_skill_directories(keys, "prefix/")
        assert result == []

    def test_build_download_tasks(self):
        """Test _s3_build_download_tasks filters to skill directories only."""
        keys = [
            "prefix/skill-a/SKILL.md",
            "prefix/skill-a/scripts/run.py",
            "prefix/other/readme.txt",
            "prefix/skill-a/",  # directory marker, should be skipped
        ]
        tasks = _s3_build_download_tasks(keys, ["skill-a"], "prefix/")

        assert len(tasks) == 2
        assert ("prefix/skill-a/SKILL.md", "skill-a/SKILL.md") in tasks
        assert ("prefix/skill-a/scripts/run.py", "skill-a/scripts/run.py") in tasks

    @mock_aws
    def test_list_all_objects_pagination(self):
        """Test _s3_list_all_objects handles paginated results."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)

        # Upload enough objects to potentially trigger pagination
        for i in range(5):
            s3_client.put_object(
                Bucket=BUCKET,
                Key=f"prefix/file-{i}.txt",
                Body=b"content",
            )

        keys = _s3_list_all_objects(s3_client, BUCKET, "prefix/")
        assert len(keys) == 5
        assert all(k.startswith("prefix/") for k in keys)
