"""Tests for experimental AgentConfig."""

import pytest
from unittest.mock import Mock, patch

from strands.experimental.agent_config import AgentConfig
from strands.experimental.tool_box import ToolBox
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
        
        def stream(self, input_data, context):
            return iter([])
    
    def test_agent_config_creation(self):
        """Test AgentConfig can be created with dict config."""
        # Provide empty ToolBox since strands_tools not available in tests
        config = AgentConfig({"model": "test-model"}, tool_box=ToolBox())
        assert config.model == "test-model"
    
    def test_agent_config_with_tools(self):
        """Test AgentConfig with basic configuration."""
        
        config = AgentConfig({
            "model": "test-model",
            "prompt": "Test prompt"
        }, tool_box=ToolBox())
        
        assert config.model == "test-model"
        assert config.system_prompt == "Test prompt"
    
    def test_agent_config_file_prefix_required(self):
        """Test that file paths must have file:// prefix."""
        
        with pytest.raises(ValueError, match="File paths must be prefixed with 'file://'"):
            AgentConfig("/path/to/config.json")
    
    def test_agent_config_file_prefix_valid(self):
        """Test that file:// prefix is properly handled."""
        import tempfile
        import json
        
        # Create a temporary config file
        config_data = {"model": "test-model", "prompt": "Test prompt"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as f:
            json.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            
            config = AgentConfig(f"file://{f.name}", tool_box=ToolBox())
            assert config.model == "test-model"
            assert config.system_prompt == "Test prompt"
    
    def test_to_agent_method_exists(self):
        """Test that to_agent method exists and is callable."""
        
        config = AgentConfig({"model": "test-model"}, tool_box=ToolBox())
        assert hasattr(config, 'to_agent')
        assert callable(config.to_agent)
    
    @patch('strands.agent.agent.Agent')
    def test_to_agent_calls_agent_constructor(self, mock_agent):
        """Test that to_agent calls Agent constructor with correct parameters."""
        
        config = AgentConfig({
            "model": "test-model",
            "prompt": "Test prompt"
        }, tool_box=ToolBox())
        
        config.to_agent()
        
        mock_agent.assert_called_once_with(
            model="test-model",
            tools=[],  # ToolBox converts to empty list
            system_prompt="Test prompt"
        )
    
    def test_agent_config_has_toolbox(self):
        """Test AgentConfig creates ToolBox and tracks configured tools."""
        
        config = AgentConfig({"model": "test-model"}, tool_box=ToolBox())
        assert hasattr(config, 'toolbox')
        assert hasattr(config, 'configured_tools')
        assert config.configured_tools == []  # No tools configured
    
    @patch('strands.agent.agent.Agent')
    def test_to_agent_with_empty_toolbox(self, mock_agent):
        """Test that to_agent uses empty ToolBox by default."""
        
        config = AgentConfig({"model": "test-model"}, tool_box=ToolBox())
        config.to_agent()
        
        # Should be called with empty tools list
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args[1]
        assert 'tools' in call_args
        assert call_args['tools'] == []
    
    @patch('strands.agent.agent.Agent')
    def test_to_agent_with_toolbox_parameter(self, mock_agent):
        """Test that to_agent uses configured ToolBox."""
        
        config = AgentConfig({"model": "test-model"}, tool_box=ToolBox())
        
        config.to_agent()
        
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args[1]
        assert 'tools' in call_args
    
    def test_agent_config_with_toolbox_constructor(self):
        """Test AgentConfig with ToolBox parameter in constructor."""
        # Create mock tools
        tool1 = self.MockTool("calculator")
        tool2 = self.MockTool("web_search")
        
        # Create ToolBox with tools
        toolbox = ToolBox([tool1, tool2])
        
        # Create config with tool selection
        config = AgentConfig({
            "model": "test-model",
            "prompt": "Test prompt",
            "tools": ["calculator"]
        }, tool_box=toolbox)
        
        # Should have selected only calculator
        assert len(config.configured_tools) == 1
        assert config.configured_tools[0].tool_name == "calculator"
    
    def test_agent_config_tool_validation_error(self):
        """Test that invalid tool names raise validation error."""
        tool1 = self.MockTool("calculator")
        toolbox = ToolBox([tool1])
        
        # Should raise error for unknown tool
        with pytest.raises(ValueError, match="Tool 'unknown_tool' not found in ToolBox"):
            AgentConfig({
                "model": "test-model",
                "tools": ["unknown_tool"]
            }, tool_box=toolbox)
    
    def test_agent_config_tools_without_toolbox_error(self):
        """Test that specifying tools without ToolBox raises error."""
        with pytest.raises(ValueError, match="Tool names specified in config but no ToolBox provided"):
            AgentConfig({
                "model": "test-model",
                "tools": ["calculator"]
            })
    
    def test_agent_config_no_strands_tools_error(self):
        """Test that missing strands_tools without ToolBox raises ImportError."""
        with pytest.raises(ImportError, match="strands_tools is not available and no ToolBox was specified"):
            AgentConfig({"model": "test-model"})
    
    def test_agent_config_skip_missing_tools(self):
        """Test that missing strands_tools can be skipped with flag."""
        # Should not raise error when flag is False and no ToolBox provided
        config = AgentConfig({"model": "test-model"}, raise_exception_on_missing_tool=False)
        assert config.model == "test-model"
        assert config.configured_tools == []  # No tools loaded since strands_tools missing
    
    def test_agent_config_skip_missing_tools_with_selection(self):
        """Test that missing tools in ToolBox can be skipped with flag."""
        # Create custom ToolBox with one tool
        from strands.types.tools import AgentTool
        
        existing_tool = self.MockTool("existing_tool")
        custom_toolbox = ToolBox([existing_tool])
        
        # Should skip missing tool when flag is False
        config = AgentConfig({
            "model": "test-model",
            "tools": ["existing_tool", "missing_tool"]  # One exists, one doesn't
        }, tool_box=custom_toolbox, raise_exception_on_missing_tool=False)
        
        # Should only have the existing tool
        assert len(config.configured_tools) == 1
        assert config.configured_tools[0].tool_name == "existing_tool"
    
    def test_agent_config_missing_tool_validation_with_flag_true(self):
        """Test that missing tools still raise error when flag is True."""
        from strands.types.tools import AgentTool
        
        existing_tool = self.MockTool("existing_tool")
        custom_toolbox = ToolBox([existing_tool])
        
        # Should raise error for missing tool when flag is True (default)
        with pytest.raises(ValueError, match="Tool 'missing_tool' not found in ToolBox"):
            AgentConfig({
                "model": "test-model",
                "tools": ["missing_tool"]
            }, tool_box=custom_toolbox, raise_exception_on_missing_tool=True)
