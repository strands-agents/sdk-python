"""Tests for experimental AgentConfig."""

import re
from unittest.mock import patch

import pytest

from strands import tool
from strands.experimental.agent_config import AgentConfig
from strands.tools.registry import ToolRegistry
from strands.types.tools import AgentTool


class TestAgentConfig:
    """Test experimental AgentConfig functionality."""

    class MockTool(AgentTool):
        def __init__(self, name):
            self._name = name

        @property
        def tool_name(self):
            return self._name

        @property
        def tool_type(self):
            return "mock"

        @property
        def tool_spec(self):
            return {"name": self._name, "type": "mock"}

        @property
        def _is_dynamic(self):
            return False

        def stream(self, input_data, context):
            return iter([])

    def test_agent_config_creation(self):
        """Test AgentConfig can be created with dict config."""
        # Provide empty ToolRegistry since strands_tools not available in tests
        config = AgentConfig({"model": "test-model"}, tool_registry=ToolRegistry())
        assert config.model == "test-model"

    def test_agent_config_with_tools(self):
        """Test AgentConfig with basic configuration."""

        config = AgentConfig({"model": "test-model", "prompt": "Test prompt"}, tool_registry=ToolRegistry())

        assert config.model == "test-model"
        assert config.system_prompt == "Test prompt"

    def test_agent_config_file_prefix_required(self):
        """Test that file paths must have file:// prefix."""

        with pytest.raises(ValueError, match="File paths must be prefixed with 'file://'"):
            AgentConfig("/path/to/config.json")

    def test_agent_config_file_prefix_valid(self):
        """Test that file:// prefix is properly handled."""
        import json
        import tempfile

        # Create a temporary config file
        config_data = {"model": "test-model", "prompt": "Test prompt"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
            json.dump(config_data, f)
            f.flush()  # Ensure data is written to disk

            config = AgentConfig(f"file://{f.name}", tool_registry=ToolRegistry())
            assert config.model == "test-model"
            assert config.system_prompt == "Test prompt"

    @patch("strands.agent.agent.Agent")
    def test_to_agent_calls_agent_constructor(self, mock_agent):
        """Test that to_agent calls Agent constructor with correct parameters."""

        config = AgentConfig({"model": "test-model", "prompt": "Test prompt"}, tool_registry=ToolRegistry())

        config.to_agent()

        mock_agent.assert_called_once_with(model="test-model", tools=[], system_prompt="Test prompt")

    def test_agent_config_has_tool_registry(self):
        """Test AgentConfig creates ToolRegistry and tracks configured tools."""

        config = AgentConfig({"model": "test-model"}, tool_registry=ToolRegistry())
        assert hasattr(config, "tool_registry")
        assert hasattr(config, "configured_tools")
        assert config.configured_tools == []  # No tools configured

    @patch("strands.agent.agent.Agent")
    def test_to_agent_with_empty_tool_registry(self, mock_agent):
        """Test that to_agent uses empty ToolRegistry by default."""

        config = AgentConfig({"model": "test-model"}, tool_registry=ToolRegistry())
        config.to_agent()

        # Should be called with empty tools list
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == []

    def test_agent_config_with_tool_registry_constructor(self):
        """Test AgentConfig with ToolRegistry parameter in constructor."""
        # Create mock tools
        tool1 = self.MockTool("calculator")
        tool2 = self.MockTool("web_search")

        # Create ToolRegistry with tools
        tool_registry = ToolRegistry()
        tool_registry.process_tools([tool1, tool2])

        # Create config with tool selection
        config = AgentConfig(
            {"model": "test-model", "prompt": "Test prompt", "tools": ["calculator"]}, tool_registry=tool_registry
        )

        # Should have selected only calculator
        assert len(config.configured_tools) == 1
        assert config.configured_tools[0].tool_name == "calculator"

    def test_agent_config_tool_validation_error(self):
        """Test that invalid tool names raise validation error."""
        tool1 = self.MockTool("calculator")
        tool_registry = ToolRegistry()
        tool_registry.process_tools([tool1])

        # Should raise error for unknown tool
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tool(s) '{'unknown_tool'}' not found in ToolRegistry. Available tools: dict_keys(['calculator'])"
            ),
        ):
            AgentConfig({"model": "test-model", "tools": ["unknown_tool"]}, tool_registry=tool_registry)

    @patch("importlib.import_module")
    def test_agent_config_import_error(self, mock_import):
        """Test that import error for strands_tools is handled correctly."""
        mock_import.side_effect = ImportError("No module named 'strands_tools'")

        with pytest.raises(ImportError, match="strands_tools is not available and no ToolRegistry was specified"):
            AgentConfig({"model": "test-model"})

    def test_agent_config_skip_missing_tools(self):
        """Test that missing strands_tools can be skipped with flag."""
        # Should not raise error when flag is False and no ToolRegistry provided
        config = AgentConfig({"model": "test-model"}, raise_exception_on_missing_tool=False)
        assert config.model == "test-model"
        assert config.configured_tools == []  # No tools loaded since strands_tools missing

    def test_agent_config_skip_missing_tools_with_selection(self):
        """Test that missing tools in ToolRegistry can be skipped with flag."""

        existing_tool = self.MockTool("existing_tool")
        custom_tool_registry = ToolRegistry()
        custom_tool_registry.process_tools([existing_tool])

        # Should skip missing tool when flag is False
        config = AgentConfig(
            {
                "model": "test-model",
                "tools": ["existing_tool", "missing_tool"],  # One exists, one doesn't
            },
            tool_registry=custom_tool_registry,
            raise_exception_on_missing_tool=False,
        )

        # Should only have the existing tool
        assert len(config.configured_tools) == 1
        assert config.configured_tools[0].tool_name == "existing_tool"

    def test_agent_config_missing_tool_validation_with_flag_true(self):
        """Test that missing tools still raise error when flag is True."""

        existing_tool = self.MockTool("existing_tool")
        custom_tool_registry = ToolRegistry()
        custom_tool_registry.process_tools([existing_tool])

        # Should raise error for missing tool when flag is True (default)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tool(s) '{'missing_tool'}' not found in ToolRegistry. Available tools: dict_keys(['existing_tool'])"
            ),
        ):
            AgentConfig(
                {"model": "test-model", "tools": ["missing_tool"]},
                tool_registry=custom_tool_registry,
                raise_exception_on_missing_tool=True,
            )

    @patch("strands.experimental.agent_config.AgentConfig._create_default_tool_registry")
    def test_agent_config_tools_without_tool_registry_error(self, mock_create_registry):
        """Test that config can load tools from default ToolRegistry when strands_tools is available."""
        # Mock the default tool registry to return a registry with file_read tool
        mock_registry = ToolRegistry()
        
        @tool
        def file_read(path: str) -> str:
            return f"content of {path}"
        
        mock_registry.process_tools([file_read])
        mock_create_registry.return_value = mock_registry

        config = AgentConfig({"model": "test-model", "tools": ["file_read"]})
        assert len(config.configured_tools) == 1
        assert config.configured_tools[0].tool_name == "file_read"

    @patch("strands.experimental.agent_config.AgentConfig._create_default_tool_registry")
    def test_agent_config_loads_from_default_tools_without_tool_registry(self, mock_create_registry):
        """Test that config can load tools from default strands_tools without explicit tool registry."""
        # Mock the default tool registry to return a registry with file_read tool
        mock_registry = ToolRegistry()
        
        @tool
        def file_read(path: str) -> str:
            return f"content of {path}"
        
        mock_registry.process_tools([file_read])
        mock_create_registry.return_value = mock_registry

        config = AgentConfig({"model": "test-model", "tools": ["file_read"]})
        # Verify the tool was loaded from the default tool registry
        assert len(config.configured_tools) == 1
        assert config.configured_tools[0].tool_name == "file_read"
