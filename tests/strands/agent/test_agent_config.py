"""Tests for agent configuration functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strands.agent import Agent
from strands.agent.config import AgentConfig


class TestBackwardCompatibility:
    """Test that existing Agent usage continues to work."""

    def test_existing_agent_usage_still_works(self):
        """Test that Agent can be created without config parameter."""
        with patch("strands.agent.agent.BedrockModel") as mock_bedrock:
            # This should work exactly as before - no config parameter
            agent = Agent(model="us.anthropic.claude-3-haiku-20240307-v1:0", system_prompt="You are helpful")

            mock_bedrock.assert_called_with(model_id="us.anthropic.claude-3-haiku-20240307-v1:0")
            assert agent.system_prompt == "You are helpful"
            assert agent.name == "Strands Agents"  # Default name


class TestAgentConfig:
    """Test AgentConfig class functionality."""

    def test_load_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "tools": ["calculator", "shell"],
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "prompt": "You are a helpful assistant",
        }

        config = AgentConfig(config_dict)

        assert config.tools == ["calculator", "shell"]
        assert config.model == "us.anthropic.claude-sonnet-4-20250514-v1:0"
        assert config.system_prompt == "You are a helpful assistant"

    def test_load_from_file(self):
        """Test loading config from JSON file."""
        config_dict = {
            "tools": ["./tools/shell_tool.py"],
            "model": "us.anthropic.claude-3-haiku-20240307-v1:0",
            "prompt": "You are a coding assistant",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            config = AgentConfig(temp_path)

            assert config.tools == ["./tools/shell_tool.py"]
            assert config.model == "us.anthropic.claude-3-haiku-20240307-v1:0"
            assert config.system_prompt == "You are a coding assistant"
        finally:
            Path(temp_path).unlink()

    def test_missing_file_error(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError, match="Agent config file not found"):
            AgentConfig("/nonexistent/path/config.json")

    def test_invalid_json_error(self):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError, match="Invalid JSON in config file"):
                AgentConfig(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_invalid_config_source_type(self):
        """Test error handling for invalid config source type."""
        with pytest.raises(ValueError, match="config_source must be a file path string or dictionary"):
            AgentConfig(123)

    def test_missing_fields(self):
        """Test handling of missing configuration fields."""
        config = AgentConfig({})

        assert config.tools is None
        assert config.model is None
        assert config.system_prompt is None


class TestAgentWithConfig:
    """Test Agent class with configuration support."""

    def test_agent_with_config_dict(self):
        """Test Agent initialization with config dictionary."""
        # Mock the strands_tools import
        mock_tool = MagicMock()
        mock_tool.name = "file_read"
        mock_tool.spec = {"name": "file_read", "description": "Mock file read tool"}

        config = {
            "tools": ["file_read"],
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "prompt": "You are helpful",
        }

        with patch("strands.agent.agent.BedrockModel") as mock_bedrock, patch("importlib.import_module") as mock_import:
            # Mock the strands_tools.file_read module
            mock_module = MagicMock()
            mock_module.file_read = mock_tool

            def side_effect(module_name):
                if module_name == "strands_tools.file_read":
                    return mock_module
                raise ImportError(f"No module named '{module_name}'")

            mock_import.side_effect = side_effect

            agent = Agent(config=config)

            mock_bedrock.assert_called_with(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
            assert agent.system_prompt == "You are helpful"
            assert len(agent.tool_registry.get_all_tool_specs()) == 1

    def test_agent_with_config_file(self):
        """Test Agent initialization with config file."""
        # Mock the strands_tools import
        mock_tool = MagicMock()
        mock_tool.name = "shell"
        mock_tool.spec = {"name": "shell", "description": "Mock shell tool"}

        config_dict = {
            "tools": ["shell"],
            "model": "us.anthropic.claude-3-haiku-20240307-v1:0",
            "prompt": "You are a coding assistant",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            with (
                patch("strands.agent.agent.BedrockModel") as mock_bedrock,
                patch("importlib.import_module") as mock_import,
            ):
                # Mock the strands_tools.shell module
                mock_module = MagicMock()
                mock_module.shell = mock_tool
                mock_import.return_value = mock_module

                agent = Agent(config=temp_path)

                mock_bedrock.assert_called_with(model_id="us.anthropic.claude-3-haiku-20240307-v1:0")
                assert agent.system_prompt == "You are a coding assistant"
                assert len(agent.tool_registry.get_all_tool_specs()) == 1
        finally:
            Path(temp_path).unlink()

    def test_constructor_params_override_config(self):
        """Test that constructor parameters override config values."""
        # Mock the strands_tools import
        mock_tool = MagicMock()
        mock_tool.name = "file_read"
        mock_tool.spec = {"name": "file_read", "description": "Mock file read tool"}

        config = {
            "tools": ["file_read"],
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "prompt": "Config prompt",
        }

        with patch("strands.agent.agent.BedrockModel") as mock_bedrock, patch("importlib.import_module") as mock_import:
            # Mock the strands_tools.file_read module
            mock_module = MagicMock()
            mock_module.file_read = mock_tool
            mock_import.return_value = mock_module

            agent = Agent(
                config=config, model="us.anthropic.claude-3-haiku-20240307-v1:0", system_prompt="Constructor prompt"
            )

            mock_bedrock.assert_called_with(model_id="us.anthropic.claude-3-haiku-20240307-v1:0")
            assert agent.system_prompt == "Constructor prompt"
            assert len(agent.tool_registry.get_all_tool_specs()) == 1

    def test_config_values_used_when_constructor_params_none(self):
        """Test that config values are used when constructor parameters are None."""
        # Mock the strands_tools import
        mock_tool = MagicMock()
        mock_tool.name = "file_write"
        mock_tool.spec = {"name": "file_write", "description": "Mock file write tool"}

        config = {
            "tools": ["file_write"],
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "prompt": "Config prompt",
        }

        with patch("strands.agent.agent.BedrockModel") as mock_bedrock, patch("importlib.import_module") as mock_import:
            # Mock the strands_tools.file_write module
            mock_module = MagicMock()
            mock_module.file_write = mock_tool
            mock_import.return_value = mock_module

            agent = Agent(config=config, model=None, system_prompt=None)

            mock_bedrock.assert_called_with(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
            assert agent.system_prompt == "Config prompt"
            assert len(agent.tool_registry.get_all_tool_specs()) == 1

    def test_agent_without_config(self):
        """Test that Agent works normally without config parameter."""
        with patch("strands.agent.agent.BedrockModel") as mock_bedrock:
            agent = Agent(model="us.anthropic.claude-3-haiku-20240307-v1:0", system_prompt="Test prompt")

            mock_bedrock.assert_called_with(model_id="us.anthropic.claude-3-haiku-20240307-v1:0")
            assert agent.system_prompt == "Test prompt"

    def test_config_error_handling(self):
        """Test error handling for invalid config."""
        with pytest.raises(ValueError, match="Failed to load agent configuration"):
            Agent(config="/nonexistent/config.json")

    def test_partial_config(self):
        """Test Agent with partial config (only some fields specified)."""
        config = {"model": "us.anthropic.claude-sonnet-4-20250514-v1:0"}

        with patch("strands.agent.agent.BedrockModel") as mock_bedrock:
            agent = Agent(config=config, system_prompt="Constructor prompt")

            mock_bedrock.assert_called_with(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
            assert agent.system_prompt == "Constructor prompt"
