"""Tests for experimental config_to_agent function."""

import json
import os
import platform
import tempfile
from unittest.mock import patch

import pytest

from strands.experimental import config_to_agent


class TestConfigToAgent:
    """Test experimental config_to_agent functionality."""

    def test_config_to_agent_with_dict(self):
        """Test config_to_agent can be created with dict config."""
        config = {"model": "test-model"}
        agent = config_to_agent(config)
        assert agent.model.config["model_id"] == "test-model"

    def test_config_to_agent_with_system_prompt(self):
        """Test config_to_agent handles system prompt correctly."""
        config = {"model": "test-model", "prompt": "Test prompt"}
        agent = config_to_agent(config)
        assert agent.system_prompt == "Test prompt"

    def test_config_to_agent_with_tools_list(self):
        """Test config_to_agent handles tools list without failing."""
        # Use a simple test that doesn't require actual tool loading
        config = {"model": "test-model", "tools": []}
        agent = config_to_agent(config)
        assert agent.model.config["model_id"] == "test-model"

    def test_config_to_agent_with_kwargs_override(self):
        """Test that kwargs can override config values."""
        config = {"model": "test-model", "prompt": "Config prompt"}
        agent = config_to_agent(config, system_prompt="Override prompt")
        assert agent.system_prompt == "Override prompt"

    def test_config_to_agent_file_prefix_required(self):
        """Test that file paths without file:// prefix work."""
        import tempfile
        import json

        config_data = {"model": "test-model"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            agent = config_to_agent(temp_path)
            assert agent.model.config["model_id"] == "test-model"
        finally:
            os.unlink(temp_path)

    def test_config_to_agent_file_prefix_valid(self):
        """Test that file:// prefix is properly handled."""
        config_data = {"model": "test-model", "prompt": "Test prompt"}
        
        if platform.system() == "Windows":
            # Use mkstemp approach on Windows for better permission handling
            fd, temp_path = tempfile.mkstemp(suffix=".json")
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(config_data, f)
                
                agent = config_to_agent(f"file://{temp_path}")
                assert agent.model.config["model_id"] == "test-model"
                assert agent.system_prompt == "Test prompt"
            finally:
                os.unlink(temp_path)
        else:
            # Use NamedTemporaryFile on non-Windows platforms
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
                json.dump(config_data, f)
                f.flush()  # Ensure data is written to disk

                agent = config_to_agent(f"file://{f.name}")
                assert agent.model.config["model_id"] == "test-model"
                assert agent.system_prompt == "Test prompt"

    def test_config_to_agent_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            config_to_agent("/nonexistent/path/config.json")

    def test_config_to_agent_invalid_json(self):
        """Test that JSONDecodeError is raised for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                config_to_agent(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_to_agent_invalid_config_type(self):
        """Test that ValueError is raised for invalid config types."""
        with pytest.raises(ValueError, match="Config must be a file path string or dictionary"):
            config_to_agent(123)

    def test_config_to_agent_with_name(self):
        """Test config_to_agent handles agent name."""
        config = {"model": "test-model", "name": "TestAgent"}
        agent = config_to_agent(config)
        assert agent.name == "TestAgent"

    def test_config_to_agent_with_agent_id(self):
        """Test config_to_agent handles agent_id."""
        config = {"model": "test-model", "agent_id": "test-agent-123"}
        agent = config_to_agent(config)
        assert agent.agent_id == "test-agent-123"

    def test_config_to_agent_ignores_none_values(self):
        """Test that None values in config are ignored."""
        config = {"model": "test-model", "prompt": None, "name": None}
        agent = config_to_agent(config)
        assert agent.model.config["model_id"] == "test-model"
        # Agent should use its defaults for None values
