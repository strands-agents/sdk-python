"""This module provides handlers for managing tool invocations."""

import logging
from typing import Any, Optional

from ..tools.registry import ToolRegistry
from ..types.content import Messages
from ..types.models import Model
from ..types.tools import ToolConfig, ToolHandler, ToolUse

logger = logging.getLogger(__name__)


class AgentToolHandler(ToolHandler):
    """Handler for processing tool invocations in agent.

    This class implements the ToolHandler interface and provides functionality for looking up tools in a registry and
    invoking them with the appropriate parameters.
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """Initialize handler.

        Args:
            tool_registry: Registry of available tools.
        """
        self.tool_registry = tool_registry
        # 打印完整的tool registry 信息

    def process(
        self,
        tool: ToolUse,
        *,
        model: Model,
        system_prompt: Optional[str],
        messages: Messages,
        tool_config: ToolConfig,
        callback_handler: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Process a tool invocation.

        Looks up the tool in the registry and invokes it with the provided parameters.

        Args:
            tool: The tool object to process, containing name and parameters.
            model: The model being used for the agent.
            system_prompt: The system prompt for the agent.
            messages: The conversation history.
            tool_config: Configuration for the tool.
            callback_handler: Callback for processing events as they happen.
            kwargs: Additional keyword arguments passed to the tool.

        Returns:
            The result of the tool invocation, or an error response if the tool fails or is not found.
        """
        logger.debug("tool=<%s> | invoking", tool)
        tool_use_id = tool["toolUseId"]
        tool_name = tool["name"]
        tool_input = tool.get("input", {})

        # Get the tool info first to extract metadata
        tool_info = self.tool_registry.registry.get(tool_name)
        tool_func = tool_info if tool_info is not None else self.tool_registry.registry.get(tool_name)

        # Display tool call with parameters (if callback handler supports it)
        agent = kwargs.get("agent")
        agent_name = agent.name if agent and hasattr(agent, "name") and agent.name else "my_agent"
        callback_handler.tool_formatter.display_tool_call(tool_name, tool_input, agent_name)
        try:
            # Check if tool exists
            if not tool_func:
                logger.error(
                    "tool_name=<%s>, available_tools=<%s> | tool not found in registry",
                    tool_name,
                    list(self.tool_registry.registry.keys()),
                )
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Unknown tool: {tool_name}"}],
                }
            # Add standard arguments to kwargs for Python tools
            kwargs.update(
                {
                    "model": model,
                    "system_prompt": system_prompt,
                    "messages": messages,
                    "tool_config": tool_config,
                    "callback_handler": callback_handler,
                }
            )

            return tool_func.invoke(tool, **kwargs)

        except Exception as e:
            logger.exception("tool_name=<%s> | failed to process tool", tool_name)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }

    def _format_tool_name_with_context(self, tool_name: str, agent_name: str = "my_agent") -> str:
        """Format tool name with context information (agent name, MCP server, etc.).

        Args:
            tool_name: Original tool name
            agent_name: Name of the agent calling the tool

        Returns:
            Formatted tool name with context
        """
        # Check if this is an MCP tool and format accordingly
        if self._is_mcp_tool(tool_name):
            server_name, local_tool_name = self._parse_mcp_tool_name(tool_name)
            if server_name and local_tool_name and server_name != local_tool_name:
                return f"{agent_name} {server_name}({local_tool_name})"

        # For non-MCP tools, return original name
        return tool_name

    def _is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool is likely an MCP tool based on naming patterns.

        Args:
            tool_name: Tool name to check

        Returns:
            True if likely an MCP tool
        """
        return "/" in tool_name or any(
            tool_name.startswith(prefix)
            for prefix in [
                "playwright_",
                "filesystem_",
                "git_",
                "browser_",
                "aws_",
                "github_",
                "sqlite_",
                "postgres_",
                "docker_",
                "kubernetes_",
                "redis_",
                "mongodb_",
                "mysql_",
            ]
        )

    def _parse_mcp_tool_name(self, tool_name: str) -> tuple[str, str]:
        """Parse MCP tool name to extract server name and local tool name.

        Args:
            tool_name: Full MCP tool name

        Returns:
            Tuple of (server_name, local_tool_name)
        """
        # Handle xxx/yyy format
        if "/" in tool_name:
            parts = tool_name.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

        # Handle underscore format like "playwright_navigate"
        if "_" in tool_name:
            parts = tool_name.split("_", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

        # If no clear separation, return the tool name as both
        return tool_name, tool_name
