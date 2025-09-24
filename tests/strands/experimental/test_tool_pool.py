# ABOUTME: Tests for experimental ToolPool using existing AgentTool infrastructure
# ABOUTME: Validates tool management, filtering, and integration with existing tools
"""Tests for experimental ToolPool system."""

import pytest

from strands.types.tools import AgentTool
from strands.experimental.tool_pool import ToolPool


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


class TestToolPool:
    """Test ToolPool functionality with existing AgentTool infrastructure."""
    
    def test_tool_pool_creation(self):
        """Test ToolPool can be created empty."""
        
        pool = ToolPool()
        assert pool.list_tool_names() == []
    
    def test_tool_pool_with_initial_tools(self):
        """Test ToolPool creation with initial AgentTool instances."""
        
        tool1 = MockAgentTool("tool1", "python")
        tool2 = MockAgentTool("tool2", "javascript")
        
        pool = ToolPool([tool1, tool2])
        assert set(pool.list_tool_names()) == {"tool1", "tool2"}
    
    def test_tool_pool_with_tool_functions(self):
        """Test ToolPool creation with @tool decorated functions."""
        
        # Create mock tool functions
        def tool1():
            return "result1"
        
        def tool2():
            return "result2"
        
        # Add tool specs to simulate @tool decorator
        tool1._strands_tool_spec = {"name": "tool1", "description": "Tool 1", "inputSchema": {}}
        tool2._strands_tool_spec = {"name": "tool2", "description": "Tool 2", "inputSchema": {}}
        
        pool = ToolPool([tool1, tool2])
        assert set(pool.list_tool_names()) == {"tool1", "tool2"}
    
    def test_tool_pool_with_mixed_tools(self):
        """Test ToolPool creation with mixed AgentTool instances and functions."""
        
        # AgentTool instance
        agent_tool = MockAgentTool("agent_tool", "python")
        
        # @tool function
        def func_tool():
            return "result"
        func_tool._strands_tool_spec = {"name": "func_tool", "description": "Function tool", "inputSchema": {}}
        
        pool = ToolPool([agent_tool, func_tool])
        assert set(pool.list_tool_names()) == {"agent_tool", "func_tool"}
    
    def test_add_and_get_tool(self):
        """Test adding and retrieving AgentTool instances."""
        
        pool = ToolPool()
        tool = MockAgentTool("test_tool", "python")
        
        pool.add_tool(tool)
        assert "test_tool" in pool.list_tool_names()
        assert pool.get_tool("test_tool") == tool
        assert pool.get_tool("nonexistent") is None
    
    def test_get_tools(self):
        """Test getting tools as AgentTool instances."""
        
        tool1 = MockAgentTool("tool1", "python")
        tool2 = MockAgentTool("tool2", "javascript")
        
        pool = ToolPool([tool1, tool2])
        agent_tools = pool.get_tools()
        
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
        
        pool = ToolPool()
        pool.add_tools_from_module(MockModule)
        
        assert "mock_tool" in pool.list_tool_names()
    
    def test_from_module_class_method(self):
        """Test creating ToolPool from module."""
        
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
        
        pool = ToolPool.from_module(MockModule)
        
        assert set(pool.list_tool_names()) == {"tool1", "tool2"}
        assert "not_a_tool" not in pool.list_tool_names()
