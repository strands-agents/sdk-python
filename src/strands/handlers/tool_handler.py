"""This module provides handlers for managing tool invocations."""

import logging
from typing import TYPE_CHECKING, Any

from ..types.tools import ToolGenerator, ToolUse

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


class AgentToolHandler:
    """Handler for processing tool invocations in agent.

    This class implements the ToolHandler interface and provides functionality for looking up tools in a registry and
    invoking them with the appropriate parameters.
    """

    def __init__(self, agent: "Agent") -> None:
        """Initialize handler.

        Args:
            agent: The agent for which this handler has been created.
        """
        self.agent = agent

    def run_tool(self, tool: ToolUse, kwargs: dict[str, Any]) -> ToolGenerator:
        """Process a tool invocation.

        Looks up the tool in the registry and invokes it with the provided parameters.

        Args:
            tool: The tool object to process, containing name and parameters.
            kwargs: Additional keyword arguments passed to the tool.

        Yields:
            Events of the tool invocation.

        Returns:
            The final tool result or an error response if the tool fails or is not found.
        """
        logger.debug("tool=<%s> | invoking", tool)
        tool_use_id = tool["toolUseId"]
        tool_name = tool["name"]

        # Get the tool info
        tool_info = self.agent.tool_registry.dynamic_tools.get(tool_name)
        tool_func = tool_info if tool_info is not None else self.agent.tool_registry.registry.get(tool_name)

        try:
            # Check if tool exists
            if not tool_func:
                logger.error(
                    "tool_name=<%s>, available_tools=<%s> | tool not found in registry",
                    tool_name,
                    list(self.agent.tool_registry.registry.keys()),
                )
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Unknown tool: {tool_name}"}],
                }
            # Add standard arguments to kwargs for Python tools
            kwargs.update(
                {
                    "model": self.agent.model,
                    "system_prompt": self.agent.system_prompt,
                    "messages": self.agent.messages,
                    "tool_config": self.agent.tool_config,
                }
            )

            result = tool_func.invoke(tool, **kwargs)
            yield {"result": result}  # Placeholder until tool_func becomes a generator from which we can yield from
            return result

        except Exception as e:
            logger.exception("tool_name=<%s> | failed to process tool", tool_name)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }
