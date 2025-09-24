# ABOUTME: Tool pool system using existing AgentTool base class for tool selection and management
# ABOUTME: Integrates with existing tool infrastructure and @tool decorator pattern
"""Experimental tool pool system for structured tool management."""

from collections.abc import Callable

from ..types.tools import AgentTool
from ..tools.tools import PythonAgentTool
from ..tools.decorator import DecoratedFunctionTool


class ToolPool:
    """Pool of available tools for agent selection using existing tool infrastructure."""
    
    def __init__(self, tools: "list[AgentTool | Callable] | None" = None):
        """Initialize tool pool.
        
        Args:
            tools: List of AgentTool instances or @tool decorated functions
        """
        self._tools: dict[str, AgentTool] = {}
        if tools:
            for tool in tools:
                if isinstance(tool, AgentTool):
                    self.add_tool(tool)
                elif callable(tool):
                    self.add_tool_function(tool)
                else:
                    raise ValueError(f"Tool must be AgentTool instance or callable, got {type(tool)}")
    
    def add_tool(self, tool: AgentTool) -> None:
        """Add existing AgentTool instance to the pool.
        
        Args:
            tool: AgentTool instance to add
        """
        self._tools[tool.tool_name] = tool
    
    def add_tool_function(self, tool_func: Callable) -> None:
        """Add @tool decorated function to the pool.
        
        Args:
            tool_func: Function decorated with @tool
        """
        if hasattr(tool_func, '_strands_tool_spec'):
            # This is a decorated function tool
            tool_spec = tool_func._strands_tool_spec
            tool_name = tool_spec.get('name', tool_func.__name__)
            decorated_tool = DecoratedFunctionTool(
                tool_name=tool_name,
                tool_spec=tool_spec,
                tool_func=tool_func,
                metadata={}
            )
            self.add_tool(decorated_tool)
        else:
            raise ValueError(f"Function {tool_func.__name__} is not decorated with @tool")
    
    def add_tools_from_module(self, module: any) -> None:
        """Add all @tool decorated functions from a Python module.
        
        Args:
            module: Python module containing @tool decorated functions
        """
        import inspect
        
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, '_strands_tool_spec'):
                self.add_tool_function(obj)
    
    @classmethod
    def from_module(cls, module: any) -> "ToolPool":
        """Create ToolPool from all @tool functions in a module.
        
        Args:
            module: Python module containing @tool decorated functions
            
        Returns:
            ToolPool with all tools from the module
        """
        pool = cls()
        pool.add_tools_from_module(module)
        return pool
    
    def get_tool(self, name: str) -> AgentTool | None:
        """Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            AgentTool if found, None otherwise
        """
        return self._tools.get(name)
    
    def list_tool_names(self) -> list[str]:
        """List available tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tools(self) -> list[AgentTool]:
        """Get all tools as AgentTool instances.
        
        Returns:
            List of AgentTool instances
        """
        return list(self._tools.values())
