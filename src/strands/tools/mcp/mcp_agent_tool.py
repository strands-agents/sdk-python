"""MCP Agent Tool module for adapting Model Context Protocol tools to the agent framework.

This module provides the MCPAgentTool class which serves as an adapter between
MCP (Model Context Protocol) tools and the agent framework's tool interface.
It allows MCP tools to be seamlessly integrated and used within the agent ecosystem.
"""

import logging
from typing import TYPE_CHECKING, Any

from mcp.types import Tool as MCPTool
from typing_extensions import override

from ...types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

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

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream the MCP tool.

        This method delegates the tool stream to the MCP server connection, passing the tool use ID, tool name, and
        input arguments.

        Args:
            tool_use: The tool use request containing tool ID and parameters.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        logger.debug("tool_name=<%s>, tool_use_id=<%s> | streaming", self.tool_name, tool_use["toolUseId"])

        result = await self.mcp_client.call_tool_async(
            tool_use_id=tool_use["toolUseId"],
            name=self.tool_name,
            arguments=tool_use["input"],
        )
        yield result
