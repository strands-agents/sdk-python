"""MCP Agent Tool module for adapting Model Context Protocol tools to the agent framework.

This module provides the MCPAgentTool class which serves as an adapter between
MCP (Model Context Protocol) tools and the agent framework's tool interface.
It allows MCP tools to be seamlessly integrated and used within the agent ecosystem.
"""

import logging
from typing import TYPE_CHECKING, Any

from mcp.types import Tool as MCPTool

from ...types.tools import AgentTool, ToolResult, ToolSpec, ToolUse

if TYPE_CHECKING:
    from .mcp_client import MCPClient

logger = logging.getLogger(__name__)


class MCPAgentTool(AgentTool):
    """Adapter class that wraps an MCP tool and exposes it as an AgentTool.

    This class bridges the gap between the MCP protocol's tool representation
    and the agent framework's tool interface, allowing MCP tools to be used
    seamlessly within the agent framework.
    """

    def __init__(self, mcp_tool: MCPTool, mcp_client: "MCPClient") -> None:
        """Initialize a new MCPAgentTool instance.

        Args:
            mcp_tool: The MCP tool to adapt
            mcp_client: The MCP server connection to use for tool invocation
        """
        super().__init__()
        logger.debug("tool_name=<%s> | creating mcp agent tool", mcp_tool.name)
        self.mcp_tool = mcp_tool
        self.mcp_client = mcp_client

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            str: The name of the MCP tool
        """
        return self.mcp_tool.name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the specification of the tool.

        This method converts the MCP tool specification to the agent framework's
        ToolSpec format, including the input schema and description.

        Returns:
            ToolSpec: The tool specification in the agent framework format
        """
        description: str = self.mcp_tool.description or f"Tool which performs {self.mcp_tool.name}"
        return {
            "inputSchema": {"json": self.mcp_tool.inputSchema},
            "name": self.mcp_tool.name,
            "description": description,
        }

    @property
    def tool_type(self) -> str:
        """Get the type of the tool.

        Returns:
            str: The type of the tool, always "python" for MCP tools
        """
        return "python"

    def invoke(self, tool: ToolUse, *args: Any, **kwargs: dict[str, Any]) -> ToolResult:
        """Invoke the MCP tool.

        This method delegates the tool invocation to the MCP server connection,
        passing the tool use ID, tool name, and input arguments.
        """
        logger.debug("invoking MCP tool '%s' with tool_use_id=%s", self.tool_name, tool["toolUseId"])
        return self.mcp_client.call_tool_sync(
            tool_use_id=tool["toolUseId"], name=self.tool_name, arguments=tool["input"]
        )
