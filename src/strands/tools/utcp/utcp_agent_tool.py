"""UTCP Agent Tool module for adapting Universal Tool Calling Protocol tools to the agent framework.

This module provides the UTCPAgentTool class which serves as an adapter between
UTCP (Universal Tool Calling Protocol) tools and the agent framework's tool interface.
It allows UTCP tools to be seamlessly integrated and used within the agent ecosystem.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, cast

from typing_extensions import override
from utcp.data.tool import Tool as UTCPTool

from ...types._events import ToolResultEvent
from ...types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

if TYPE_CHECKING:
    from .utcp_client import UTCPClient

logger = logging.getLogger(__name__)


class UTCPAgentTool(AgentTool):
    """Adapter class that wraps a UTCP tool and exposes it as an AgentTool.

    This class bridges the gap between the UTCP protocol's tool representation
    and the agent framework's tool interface, allowing UTCP tools to be used
    seamlessly within the agent framework.
    """

    def __init__(self, utcp_tool: UTCPTool, utcp_client: "UTCPClient") -> None:
        """Initialize a new UTCPAgentTool instance.

        Args:
            utcp_tool: The UTCP tool to adapt
            utcp_client: The UTCP client connection to use for tool invocation
        """
        super().__init__()
        logger.debug("tool_name=<%s> | creating UTCP agent tool", utcp_tool.name)
        self.utcp_tool = utcp_tool
        self.utcp_client = utcp_client

    @classmethod
    def from_utcp_tool(cls, utcp_tool: UTCPTool, utcp_client: "UTCPClient") -> "UTCPAgentTool":
        """Create a UTCPAgentTool from a UTCP Tool and client.

        Args:
            utcp_tool: The UTCP tool to adapt
            utcp_client: The UTCP client connection to use for tool invocation

        Returns:
            UTCPAgentTool instance
        """
        return cls(utcp_tool, utcp_client)

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            str: The name of the UTCP tool, sanitized for Bedrock compatibility
        """
        # TEMPORARY FIX: Replace dots with underscores to make tool names Bedrock-compatible
        # Bedrock requires tool names to match pattern [a-zA-Z0-9_-]+
        return self.utcp_tool.name.replace(".", "_")

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the specification of the tool.

        This method converts the UTCP tool specification to the agent framework's
        ToolSpec format, including the input schema and description.

        Returns:
            ToolSpec: The tool specification in the agent framework format
        """
        description: str = self.utcp_tool.description or f"Tool which performs {self.utcp_tool.name}"

        # Convert UTCP ToolInputOutputSchema to JSON schema format
        input_schema = {
            "type": self.utcp_tool.inputs.type,
            "properties": self.utcp_tool.inputs.properties,
        }

        # Add optional fields if they exist
        if self.utcp_tool.inputs.required:
            input_schema["required"] = self.utcp_tool.inputs.required
        if self.utcp_tool.inputs.description:
            input_schema["description"] = self.utcp_tool.inputs.description
        if self.utcp_tool.inputs.title:
            input_schema["title"] = str(self.utcp_tool.inputs.title)
        if self.utcp_tool.inputs.items:
            input_schema["items"] = cast(Dict[str, Any], self.utcp_tool.inputs.items)
        if self.utcp_tool.inputs.enum:
            input_schema["enum"] = cast(List[str], self.utcp_tool.inputs.enum)
        if self.utcp_tool.inputs.minimum is not None:
            input_schema["minimum"] = str(int(self.utcp_tool.inputs.minimum))
        if self.utcp_tool.inputs.maximum is not None:
            input_schema["maximum"] = str(int(self.utcp_tool.inputs.maximum))
        if self.utcp_tool.inputs.format:
            input_schema["format"] = self.utcp_tool.inputs.format

        return {
            "inputSchema": {"json": input_schema},
            "name": self.tool_name,  # Use sanitized tool name for Bedrock compatibility
            "description": description,
        }

    @property
    def tool_type(self) -> str:
        """Get the type of the tool.

        Returns:
            str: The type of the tool, always "python" for UTCP tools
        """
        return "python"

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream the UTCP tool.

        This method delegates the tool execution to the UTCP client connection, passing the tool use ID, tool name, and
        input arguments.

        Args:
            tool_use: The tool use request containing tool ID and parameters.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        logger.debug("tool_name=<%s>, tool_use_id=<%s> | streaming", self.tool_name, tool_use["toolUseId"])

        result = await self.utcp_client.call_tool_async(
            tool_use_id=tool_use["toolUseId"],
            tool_name=self.utcp_tool.name,  # Use original UTCP tool name for execution
            arguments=tool_use["input"] or {},
        )
        yield ToolResultEvent(cast(ToolResult, result))
