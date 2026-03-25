"""Tests for deployment packaging utilities."""

import zipfile
from io import BytesIO
from unittest.mock import MagicMock

from strands.deploy._packaging import (
    _should_exclude,
    create_code_zip,
    extract_agent_config,
    generate_agentcore_entrypoint,
)


class TestShouldExclude:
    def test_excludes_strands_dir(self):
        assert _should_exclude("/project/.strands/state.json", "/project")

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
        (tmp_path / ".strands").mkdir()
        (tmp_path / ".strands" / "state.json").write_text("{}")

        zip_bytes = create_code_zip("# entrypoint", base_dir=str(tmp_path))

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert not any(".git" in n for n in names)
            assert not any(".strands" in n for n in names)

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


class TestExtractAgentConfig:
    def test_extracts_name_and_system_prompt(self):
        agent = MagicMock()
        agent.name = "test-agent"
        agent.system_prompt = "You are a helpful assistant."
        agent.model = MagicMock()
        agent.model.config = {"model_id": "us.anthropic.claude-sonnet-4-20250514"}

        config = extract_agent_config(agent)

        assert config["name"] == "test-agent"
        assert config["system_prompt"] == "You are a helpful assistant."
        assert config["model_id"] == "us.anthropic.claude-sonnet-4-20250514"

    def test_handles_missing_model_config(self):
        agent = MagicMock()
        agent.name = "test"
        agent.system_prompt = None
        agent.model = MagicMock(spec=[])  # No config attr

        config = extract_agent_config(agent)
        assert config["name"] == "test"
        assert "model_id" not in config


class TestGenerateAgentcoreEntrypoint:
    def test_generates_valid_python(self):
        agent = MagicMock()
        agent.name = "my-agent"
        agent.system_prompt = "Be helpful."
        agent.model = MagicMock()
        agent.model.config = {"model_id": "us.anthropic.claude-sonnet-4-20250514"}

        code = generate_agentcore_entrypoint(agent)

        # Should be valid Python (compile check)
        compile(code, "<test>", "exec")

        assert "BedrockAgentCoreApp" in code
        assert "us.anthropic.claude-sonnet-4-20250514" in code
        assert "Be helpful." in code
        assert "@app.entrypoint" in code

    def test_handles_none_system_prompt(self):
        agent = MagicMock()
        agent.name = "test"
        agent.system_prompt = None
        agent.model = MagicMock(spec=[])

        code = generate_agentcore_entrypoint(agent)
        compile(code, "<test>", "exec")
        assert "system_prompt=None" in code
