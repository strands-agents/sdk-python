"""Tests for deployment packaging utilities."""

import zipfile
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from strands.experimental.deploy._exceptions import DeployPackagingException
from strands.experimental.deploy._packaging import (
    _should_exclude,
    _strip_deploy_call,
    create_code_zip,
    generate_agentcore_entrypoint,
)


class TestShouldExclude:
    def test_excludes_strands_dir(self):
        assert _should_exclude("/project/.strands_deploy/state.json", "/project")

    def test_excludes_git_dir(self):
        assert _should_exclude("/project/.git/config", "/project")

    def test_excludes_pycache(self):
        assert _should_exclude("/project/__pycache__/module.pyc", "/project")

    def test_excludes_venv(self):
        assert _should_exclude("/project/.venv/lib/python3.12", "/project")
        assert _should_exclude("/project/venv/lib/python3.12", "/project")

    def test_excludes_egg_info(self):
        assert _should_exclude("/project/package.egg-info/PKG-INFO", "/project")

    def test_includes_normal_files(self):
        assert not _should_exclude("/project/main.py", "/project")
        assert not _should_exclude("/project/src/agent.py", "/project")


class TestCreateCodeZip:
    def test_zips_directory_contents(self, tmp_path):
        # Create test files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "calc.py").write_text("def add(a, b): return a + b")

        zip_bytes = create_code_zip("# entrypoint", base_dir=str(tmp_path))

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "main.py" in names
            assert "tools/calc.py" in names
            assert "_strands_entrypoint.py" in names

    def test_excludes_hidden_dirs(self, tmp_path):
        (tmp_path / "main.py").write_text("code")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / ".strands_deploy").mkdir()
        (tmp_path / ".strands_deploy" / "state.json").write_text("{}")

        zip_bytes = create_code_zip("# entrypoint", base_dir=str(tmp_path))

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert not any(".git" in n for n in names)
            assert not any(".strands_deploy" in n for n in names)

    def test_excludes_pyc_files(self, tmp_path):
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "main.pyc").write_text("bytecode")

        zip_bytes = create_code_zip("# entrypoint", base_dir=str(tmp_path))

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "main.py" in names
            assert "main.pyc" not in names

    def test_includes_generated_entrypoint(self, tmp_path):
        (tmp_path / "app.py").write_text("agent = Agent()")
        entrypoint_code = "from strands import Agent\nagent = Agent()"

        zip_bytes = create_code_zip(entrypoint_code, base_dir=str(tmp_path))

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            content = zf.read("_strands_entrypoint.py").decode()
            assert "from strands import Agent" in content


class TestStripDeployCall:
    def test_removes_deploy_call(self):
        source = "from strands import Agent\nagent = Agent()\ndeploy(agent, name='test')\n"
        result = _strip_deploy_call(source)
        assert "deploy(" not in result
        assert "agent = Agent()" in result

    def test_removes_module_deploy_call(self):
        source = "import strands\nagent = Agent()\nstrands.deploy(agent)\n"
        result = _strip_deploy_call(source)
        assert "deploy(" not in result

    def test_removes_if_name_main(self):
        source = "agent = Agent()\nif __name__ == '__main__':\n    deploy(agent)\n"
        result = _strip_deploy_call(source)
        assert "__name__" not in result
        assert "__main__" not in result


    def test_preserves_non_deploy_code(self):
        source = "from strands import Agent\nfrom my_tools import search\nagent = Agent(tools=[search])\n"
        result = _strip_deploy_call(source)
        assert "from strands import Agent" in result
        assert "from my_tools import search" in result
        assert "Agent(tools=[search])" in result


class TestGenerateAgentcoreEntrypoint:
    def test_raises_when_caller_source_not_found(self):
        """When caller source can't be found (e.g., REPL), raises an error."""
        from unittest.mock import patch

        agent = MagicMock()
        agent.name = "my-agent"

        with patch("strands.experimental.deploy._packaging._find_caller_info", return_value=None):
            with pytest.raises(DeployPackagingException, match="Could not find the source file"):
                generate_agentcore_entrypoint(agent)
