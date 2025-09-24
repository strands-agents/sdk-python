"""Tests for experimental ToolBox system."""

import pytest

from strands.types.tools import AgentTool
from strands.experimental.tool_box import ToolBox


class MockAgentTool(AgentTool):
    """Mock AgentTool for testing without full dependencies."""
    
    def __init__(self, name: str, tool_type: str = "mock", **metadata):
        self._name = name
        self._tool_type = tool_type
        self.metadata = metadata
    
    @property
    def tool_name(self) -> str:
        return self._name
    
    @property
    def tool_spec(self):
        return {"name": self._name, "description": "Mock tool", "inputSchema": {}}
    
    @property
    def tool_type(self) -> str:
        return self._tool_type
    
    def stream(self, tool_use, invocation_state, **kwargs):
        yield {"result": f"Mock result from {self._name}"}


class TestToolBox:
    """Test ToolBox functionality with existing AgentTool infrastructure."""
    
    def test_tool_pool_creation(self):
        """Test ToolBox can be created empty."""
        
        pool = ToolBox()
        assert pool.list_tool_names() == []
    
    def test_tool_pool_with_initial_tools(self):
        """Test ToolBox creation with initial AgentTool instances."""
        
        tool1 = MockAgentTool("tool1", "python")
        tool2 = MockAgentTool("tool2", "javascript")
        
        pool = ToolBox([tool1, tool2])
        assert set(pool.list_tool_names()) == {"tool1", "tool2"}
    
    def test_add_and_get_tool(self):
        """Test adding and retrieving AgentTool instances."""
        
        pool = ToolBox()
        tool = MockAgentTool("test_tool", "python")
        
        pool.add_tool(tool)
        assert "test_tool" in pool.list_tool_names()
        assert pool.get_tool("test_tool") == tool
        assert pool.get_tool("nonexistent") is None
    
    def test_list_tools(self):
        """Test listing tools as AgentTool instances."""
        
        tool1 = MockAgentTool("tool1", "python")
        tool2 = MockAgentTool("tool2", "javascript")
        
        pool = ToolBox([tool1, tool2])
        agent_tools = pool.list_tools()
        
        assert len(agent_tools) == 2
        assert all(hasattr(t, 'tool_name') for t in agent_tools)
        assert set(t.tool_name for t in agent_tools) == {"tool1", "tool2"}
    
    def test_add_tools_from_module(self):
        """Test adding tools from a module."""
        
        # Create mock module with tool function
        class MockModule:
            @staticmethod
            def mock_tool():
                return "result"
        
        # Add tool spec to simulate @tool decorator
        MockModule.mock_tool._strands_tool_spec = {
            "name": "mock_tool",
            "description": "Mock tool",
            "inputSchema": {}
        }
        
        pool = ToolBox()
        pool.add_tools_from_module(MockModule)
        
        assert "mock_tool" in pool.list_tool_names()
    
    def test_from_module_class_method(self):
        """Test creating ToolBox from module."""
        
        # Create mock module with tool function
        class MockModule:
            @staticmethod
            def tool1():
                return "result1"
            
            @staticmethod
            def tool2():
                return "result2"
            
            @staticmethod
            def not_a_tool():
                return "not a tool"
        
        # Add tool specs to simulate @tool decorator
        MockModule.tool1._strands_tool_spec = {"name": "tool1", "description": "Tool 1", "inputSchema": {}}
        MockModule.tool2._strands_tool_spec = {"name": "tool2", "description": "Tool 2", "inputSchema": {}}
        # not_a_tool doesn't have _strands_tool_spec
        
        pool = ToolBox.from_module(MockModule)
        
        assert set(pool.list_tool_names()) == {"tool1", "tool2"}
        assert "not_a_tool" not in pool.list_tool_names()
