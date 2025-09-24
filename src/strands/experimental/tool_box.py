"""Experimental tool box system for structured tool management."""

from ..types.tools import AgentTool
from ..tools.tools import PythonAgentTool
from ..tools.decorator import DecoratedFunctionTool


class ToolBox:
    """Box of available tools for agent selection using existing tool infrastructure."""
    
    def __init__(self, tools: "list[AgentTool] | None" = None):
        """Initialize tool box.
        
        Args:
            tools: List of AgentTool instances
        """
        self._tools: dict[str, AgentTool] = {}
        if tools:
            for tool in tools:
                self.add_tool(tool)
    
    def add_tool(self, tool: AgentTool) -> None:
        """Add existing AgentTool instance to the pool.
        
        Args:
            tool: AgentTool instance to add
        """
        self._tools[tool.tool_name] = tool
    
    def add_tools_from_module(self, module: any) -> None:
        """Add all @tool decorated functions from a Python module.
        
        Args:
            module: Python module containing @tool decorated functions
        """
        import inspect
        
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, '_strands_tool_spec'):
                # Create DecoratedFunctionTool directly
                tool_spec = obj._strands_tool_spec
                tool_name = tool_spec.get('name', obj.__name__)
                decorated_tool = DecoratedFunctionTool(
                    tool_name=tool_name,
                    tool_spec=tool_spec,
                    tool_func=obj,
                    metadata={}
                )
                self.add_tool(decorated_tool)
    
    @classmethod
    def from_module(cls, module: any) -> "ToolBox":
        """Create ToolBox from all @tool functions in a module.
        
        Args:
            module: Python module containing @tool decorated functions
            
        Returns:
            ToolBox with all tools from the module
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
    
    def list_tools(self) -> list[AgentTool]:
        """List all tools as AgentTool instances.
        
        Returns:
            List of AgentTool instances in the box
        """
        return list(self._tools.values())
