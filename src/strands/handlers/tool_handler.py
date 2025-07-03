"""This module provides handlers for managing tool invocations."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..experimental.hooks import AfterToolInvocation, BeforeToolInvocation
from ..tools.registry import ToolRegistry
from ..types.content import Messages
from ..types.models import Model
from ..types.tools import ToolConfig, ToolGenerator, ToolHandler, ToolResult, ToolUse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..agent import Agent


class AgentToolHandler(ToolHandler):
    """Handler for processing tool invocations in agent.

    This class implements the ToolHandler interface and provides functionality for looking up tools in a registry and
    invoking them with the appropriate parameters.
    """

    def __init__(self, agent: "Agent", tool_registry: ToolRegistry) -> None:
        """Initialize handler.

        Args:
            agent: The agent associated with this tool handler.
            tool_registry: Registry of available tools.
        """
        self.tool_registry = tool_registry
        self.agent = agent

    def process(
        self,
        tool: ToolUse,
        *,
        model: Model,
        system_prompt: Optional[str],
        messages: Messages,
        tool_config: ToolConfig,
        kwargs: dict[str, Any],
    ) -> ToolGenerator:
        """Process a tool invocation.

        Looks up the tool in the registry and invokes it with the provided parameters.

        Args:
            tool: The tool object to process, containing name and parameters.
            model: The model being used for the agent.
            system_prompt: The system prompt for the agent.
            messages: The conversation history.
            tool_config: Configuration for the tool.
            kwargs: Additional keyword arguments passed to the tool.

        Yields:
            Events of the tool invocation.

        Returns:
            The final tool result or an error response if the tool fails or is not found.
        """
        logger.debug("tool=<%s> | invoking", tool)
        tool_name = tool["name"]

        # Get the tool info
        tool_info = self.tool_registry.dynamic_tools.get(tool_name)
        tool_func = tool_info if tool_info is not None else self.tool_registry.registry.get(tool_name)

        # Add standard arguments to kwargs for Python tools
        kwargs.update(
            {
                "model": model,
                "system_prompt": system_prompt,
                "messages": messages,
                "tool_config": tool_config,
            }
        )

        before_event = BeforeToolInvocation(
            agent=self.agent,
            selected_tool=tool_func,
            tool_use=tool,
            kwargs=kwargs,
        )
        self.agent._hooks.invoke_callbacks(before_event)

        try:
            selected_tool = before_event.selected_tool
            tool_use = before_event.tool_use

            # Check if tool exists
            if not selected_tool:
                logger.error(
                    "tool_name=<%s>, available_tools=<%s> | tool not found in registry",
                    tool_name,
                    list(self.tool_registry.registry.keys()),
                )
                return {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Unknown tool: {tool_name}"}],
                }

            result = selected_tool.invoke(tool_use, **kwargs)
            after_event = AfterToolInvocation(
                agent=self.agent,
                selected_tool=selected_tool,
                tool_use=tool_use,
                kwargs=kwargs,
                result=result,
            )
            self.agent._hooks.invoke_callbacks(after_event)
            yield {
                "result": after_event.result
            }  # Placeholder until tool_func becomes a generator from which we can yield from
            return after_event.result

        except Exception as e:
            logger.exception("tool_name=<%s> | failed to process tool", tool_name)
            error_result: ToolResult = {
                "toolUseId": str(tool_use.get("toolUseId")),
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }
            after_event = AfterToolInvocation(
                agent=self.agent,
                selected_tool=selected_tool,
                tool_use=tool_use,
                kwargs=kwargs,
                result=error_result,
                exception=e,
            )
            self.agent._hooks.invoke_callbacks(after_event)
            return after_event.result
