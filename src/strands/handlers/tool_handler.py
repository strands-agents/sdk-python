"""This module provides handlers for managing tool streams."""

import logging
from typing import Any, Optional

from ..tools.registry import ToolRegistry
from ..types.content import Messages
from ..types.models import Model
from ..types.tools import ToolConfig, ToolGenerator, ToolHandler, ToolUse

logger = logging.getLogger(__name__)


class AgentToolHandler(ToolHandler):
    """Handler for processing tool streams.

    This class implements the ToolHandler interface and provides functionality for looking up tools in a registry and
    streaming them with the appropriate parameters.
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """Initialize handler.

        Args:
            tool_registry: Registry of available tools.
        """
        self.tool_registry = tool_registry

    def process(
        self,
        tool_use: ToolUse,
        *,
        model: Model,
        system_prompt: Optional[str],
        messages: Messages,
        tool_config: ToolConfig,
        kwargs: dict[str, Any],
    ) -> ToolGenerator:
        """Process a tool stream.

        Looks up the tool in the registry and streams it with the provided parameters.

        Args:
            tool_use: The tool object to process, containing name and parameters.
            model: The model being used for the agent.
            system_prompt: The system prompt for the agent.
            messages: The conversation history.
            tool_config: Configuration for the tool.
            kwargs: Additional keyword arguments passed to the tool.

        Yields:
            Events of the tool stream.

        Returns:
            The final tool result or an error response if the tool fails or is not found.
        """
        logger.debug("tool_use=<%s> | streaming", tool_use)
        tool_use_id = tool_use["toolUseId"]
        tool_name = tool_use["name"]

        # Get the tool info
        tool_info = self.tool_registry.dynamic_tools.get(tool_name)
        tool_func = tool_info if tool_info is not None else self.tool_registry.registry.get(tool_name)

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
                }
            )

            return (yield from tool_func.stream(tool_use, **kwargs))

        except Exception as e:
            logger.exception("tool_name=<%s> | failed to process tool", tool_name)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }
