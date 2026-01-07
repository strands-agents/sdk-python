"""LangChain tool wrapper for Strands Agents.

This module provides a Strands AgentTool that wraps LangChain BaseTool instances,
enabling seamless use of LangChain tools with Strands Agents.

All LangChain tools inherit from BaseTool, so this wrapper works with any LangChain tool:
tools created with the @tool decorator, StructuredTool instances, or custom BaseTool subclasses.

See: https://python.langchain.com/docs/concepts/tools/

Example:
    ```python
    from langchain_core.tools import tool as langchain_tool
    from strands import Agent
    from strands.experimental.tools import LangChainTool

    @langchain_tool
    def calculator(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    agent = Agent(tools=[LangChainTool(calculator)])
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool as LangChainBaseTool
from typing_extensions import override

from strands.types._events import ToolResultEvent
from strands.types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

logger = logging.getLogger(__name__)


class LangChainTool(AgentTool):
    """A Strands AgentTool that wraps a LangChain BaseTool.

    This class allows LangChain tools to be used directly with Strands Agents
    by wrapping them in the AgentTool interface.

    Example:
        ```python
        from langchain_core.tools import tool as langchain_tool

        @langchain_tool
        def calculator(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        # Wrap as Strands tool
        strands_calculator = LangChainTool(calculator)

        # Use with Strands Agent
        agent = Agent(tools=[strands_calculator])
        ```
    """

    _langchain_tool: LangChainBaseTool
    _tool_name: str
    _tool_spec: ToolSpec

    def __init__(
        self,
        tool: LangChainBaseTool,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize with a LangChain BaseTool.

        Args:
            tool: A LangChain BaseTool instance.
            name: Optional override for the tool name.
            description: Optional override for the tool description.
        """
        super().__init__()

        self._langchain_tool = tool
        self._tool_name = name or tool.name

        tool_description = description or tool.description or f"Tool: {self._tool_name}"

        # Build tool spec
        input_schema = self._build_input_schema(tool)
        self._tool_spec: ToolSpec = {
            "name": self._tool_name,
            "description": tool_description,
            "inputSchema": {"json": input_schema},
        }

    @staticmethod
    def _build_input_schema(tool: LangChainBaseTool) -> dict[str, object]:
        """Build JSON schema from a LangChain tool's args_schema.

        Args:
            tool: A LangChain BaseTool instance.

        Returns:
            A JSON schema dict suitable for Strands' inputSchema format.
        """
        args_schema = tool.args_schema

        if args_schema is None:
            return {
                "type": "object",
                "properties": {},
                "required": [],
            }

        # args_schema is a Pydantic model class or a dict
        # https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.BaseTool.html#langchain_core.tools.base.BaseTool.args_schema
        if isinstance(args_schema, dict):
            schema = args_schema.copy()
        elif hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
        else:
            return {
                "type": "object",
                "properties": {},
                "required": [],
            }

        # Remove fields that aren't needed for tool input schemas:
        # - title: Pydantic adds the class name, not useful for tool schemas
        # - additionalProperties: validation constraint, not needed by model providers
        schema.pop("title", None)
        schema.pop("additionalProperties", None)

        # Ensure required fields exist
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("required", [])

        return schema

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            The tool name.
        """
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification.

        Returns:
            The Strands-compatible tool specification.
        """
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Get the type of the tool.

        Returns:
            'langchain' to identify this as a wrapped LangChain tool.
        """
        return "langchain"

    @property
    def wrapped_tool(self) -> LangChainBaseTool:
        """Access the underlying LangChain tool.

        Returns:
            The original LangChain BaseTool instance.
        """
        return self._langchain_tool

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, object], **kwargs: object) -> ToolGenerator:
        """Execute the LangChain tool and stream the result.

        Args:
            tool_use: The tool use request containing input parameters.
            invocation_state: Context for the tool invocation.
            **kwargs: Additional keyword arguments.

        Yields:
            ToolResultEvent containing the tool execution result.
        """
        tool_use_id = tool_use.get("toolUseId", "unknown")
        tool_input = tool_use.get("input", {})

        result = await self._langchain_tool.ainvoke(tool_input)
        content = self._convert_result_to_content(result)

        yield ToolResultEvent(
            {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": content,
            }
        )

    def _convert_result_to_content(self, result: Any) -> list[dict[str, Any]]:
        """Convert a LangChain tool result to Strands content format.

        LangChain tools can return various content types defined in TOOL_MESSAGE_BLOCK_TYPES:
        https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/base.py

        Currently only string results are supported. Support for other types (text blocks,
        image, json, document, etc.) will be added in future versions.

        Args:
            result: The result from a LangChain tool invocation.

        Returns:
            A list of content blocks in Strands format.

        Raises:
            ValueError: If the result type is not supported.
        """
        # TODO: Expand support for other LangChain content types (text blocks, image, json, etc.)
        if isinstance(result, str):
            return [{"text": result}]

        raise ValueError(
            f"Unsupported LangChain result type: {type(result).__name__}. Only string results are currently supported."
        )
