"""This module provides handlers for managing tool invocations."""

import logging
from typing import Any, List, Optional

from ..tools.registry import ToolRegistry
from ..types.models import Model
from ..types.tools import ToolConfig, ToolHandler, ToolResult, ToolUse

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

    def preprocess(
        self,
        tool: ToolUse,
        tool_config: ToolConfig,
        **kwargs: Any,
    ) -> Optional[ToolResult]:
        """Preprocess a tool before invocation (not implemented).

        Args:
            tool: The tool use object to preprocess.
            tool_config: Configuration for the tool.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of preprocessing, if any.
        """
        pass

    def process(
        self,
        tool: Any,
        *,
        model: Model,
        system_prompt: Optional[str],
        messages: List[Any],
        tool_config: Any,
        callback_handler: Any,
        **kwargs: Any,
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
            **kwargs: Additional keyword arguments passed to the tool.

        Returns:
            The result of the tool invocation, or an error response if the tool fails or is not found.
        """
        logger.debug("tool=<%s> | invoking", tool)
        tool_use_id = tool["toolUseId"]
        tool_name = tool["name"]

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
