"""Universal Tool Calling Protocol (UTCP) client wrapper for Strands integration.

This module provides the UTCPClient class which wraps the native UTCP client to provide
a simplified interface for the Strands agent framework.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, cast

from utcp.data.tool import Tool as UTCPTool
from utcp.data.utcp_client_config import UtcpClientConfig
from utcp.utcp_client import UtcpClient
from utcp_http.http_call_template import HttpCallTemplate

from ...types import PaginatedList
from ...types.exceptions import UTCPClientInitializationError
from ...types.tools import ToolResult
from .utcp_agent_tool import UTCPAgentTool

logger = logging.getLogger(__name__)


class UTCPClient:
    """Wrapper for UTCP client that provides Strands-compatible interface."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize a new UTCP client wrapper.

        Args:
            config: Configuration dictionary containing:
                   - 'manual_call_templates': List of call template configurations
        """
        self._config = config
        self._utcp_client: Optional[UtcpClient] = None
        logger.debug("initializing UTCPClient wrapper with config: %s", config)

    async def __aenter__(self) -> "UTCPClient":
        """Async context manager entry point which initializes the UTCP client."""
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit point that cleans up resources."""
        await self.stop()

    async def start(self) -> "UTCPClient":
        """Initialize and start the UTCP client.

        Returns:
            Self for method chaining

        Raises:
            UTCPClientInitializationError: If client initialization fails
        """
        try:
            # Convert dictionary configurations to proper call template objects
            call_templates = []
            for template_config in self._config.get("manual_call_templates", []):
                if template_config.get("call_template_type") == "http":
                    call_template = HttpCallTemplate(
                        name=template_config["name"],
                        call_template_type="http",
                        url=template_config["url"],
                        http_method=template_config.get("http_method", "GET"),
                        content_type=template_config.get("content_type", "application/json"),
                    )
                    call_templates.append(call_template)
                else:
                    logger.warning("Unsupported call template type: %s", template_config.get("call_template_type"))

            # Create UtcpClientConfig with proper call template objects
            from utcp.data.call_template import CallTemplate

            utcp_config = UtcpClientConfig(manual_call_templates=cast(List[CallTemplate], call_templates))

            # Create UTCP client
            self._utcp_client = await UtcpClient.create(config=utcp_config)
            logger.info("UTCP client initialized successfully")
            return self

        except Exception as e:
            logger.error("Failed to initialize UTCP client: %s", e)
            raise UTCPClientInitializationError(f"UTCP client initialization failed: {e}") from e

    async def stop(self) -> None:
        """Stop and cleanup the UTCP client."""
        if self._utcp_client:
            self._utcp_client = None
            logger.info("UTCP client stopped")

    def list_tools_sync(self, pagination_token: Optional[str] = None) -> PaginatedList[UTCPAgentTool]:
        """Synchronously retrieve the list of available tools from UTCP providers.

        Args:
            pagination_token: Optional pagination token (not used in current implementation)

        Returns:
            PaginatedList of UTCPAgentTool instances
        """
        if not self._utcp_client:
            raise UTCPClientInitializationError("UTCP client is not initialized. Call start() first.")

        try:
            # Use asyncio.run to call the async method
            utcp_tools = asyncio.run(self._get_tools_async())
        except Exception as e:
            logger.error("Failed to get tools from UTCP client: %s", e)
            utcp_tools = []

        logger.debug("received %d tools from UTCP client", len(utcp_tools))

        # Convert to UTCPAgentTool instances
        agent_tools = []
        for utcp_tool in utcp_tools:
            try:
                agent_tool = UTCPAgentTool.from_utcp_tool(utcp_tool, self)
                agent_tools.append(agent_tool)
            except Exception as e:
                logger.warning("Failed to convert UTCP tool %s to agent tool: %s", utcp_tool.name, e)

        return PaginatedList(data=agent_tools, token=None)

    async def _get_tools_async(self) -> List[UTCPTool]:
        """Asynchronously get tools from UTCP client.

        Returns:
            List of UTCP Tool objects
        """
        if not self._utcp_client:
            raise UTCPClientInitializationError("UTCP client is not initialized. Call start() first.")

        # Use the search_tools API with empty query to get all tools
        try:
            tools = await self._utcp_client.search_tools(query="", limit=1000)
            return tools
        except Exception as e:
            logger.error("Failed to search tools: %s", e)
            return []

    async def list_tools(self, pagination_token: Optional[str] = None) -> PaginatedList[UTCPAgentTool]:
        """Asynchronously retrieve the list of available tools from UTCP providers.

        Args:
            pagination_token: Optional pagination token (not used in current implementation)

        Returns:
            PaginatedList of UTCPAgentTool instances
        """
        try:
            utcp_tools = await self._get_tools_async()
        except Exception as e:
            logger.error("Failed to get tools from UTCP client: %s", e)
            utcp_tools = []

        logger.debug("received %d tools from UTCP client", len(utcp_tools))

        # Convert to UTCPAgentTool instances
        agent_tools = []
        for utcp_tool in utcp_tools:
            try:
                agent_tool = UTCPAgentTool.from_utcp_tool(utcp_tool, self)
                agent_tools.append(agent_tool)
            except Exception as e:
                logger.warning("Failed to convert UTCP tool %s to agent tool: %s", utcp_tool.name, e)

        return PaginatedList(data=agent_tools, token=None)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            UTCPClientInitializationError: If client not initialized
            ValueError: If tool not found or execution fails
        """
        if not self._utcp_client:
            raise UTCPClientInitializationError("UTCP client is not initialized. Call start() first.")

        try:
            logger.debug("calling tool %s with arguments: %s", tool_name, arguments)
            result = await self._utcp_client.call_tool(tool_name=tool_name, tool_args=arguments)
            logger.debug("tool %s returned: %s", tool_name, result)
            return result
        except Exception as e:
            logger.error("Failed to call tool %s: %s", tool_name, e)
            raise ValueError(f"Tool execution failed: {e}") from e

    def create_tool_result(self, tool_use_id: str, result: Any) -> ToolResult:
        """Create a ToolResult from a raw UTCP tool execution result.

        Args:
            tool_use_id: The tool use identifier
            result: The raw result from UTCP tool execution

        Returns:
            ToolResult instance
        """
        try:
            # Convert result to string if it's not already
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                content = json.dumps(result, indent=2)
            else:
                content = str(result)

            return ToolResult(toolUseId=tool_use_id, content=[{"text": content}], status="success")
        except Exception as e:
            logger.error("Failed to create tool result: %s", e)
            return ToolResult(
                toolUseId=tool_use_id, content=[{"text": f"Error processing result: {e}"}], status="error"
            )

    async def call_tool_async(self, tool_use_id: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool asynchronously and return formatted result.

        Args:
            tool_use_id: The tool use identifier
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Formatted tool result dictionary
        """
        if not self._utcp_client:
            raise UTCPClientInitializationError("UTCP client is not initialized. Call start() first.")

        try:
            result = await self.call_tool(tool_name, arguments)
            return self._handle_tool_result(tool_use_id, result)
        except Exception as e:
            return self._handle_tool_execution_error(tool_use_id, e)

    def call_tool_sync(self, tool_use_id: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool synchronously and return formatted result.

        Args:
            tool_use_id: The tool use identifier
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Formatted tool result dictionary
        """
        if not self._utcp_client:
            raise UTCPClientInitializationError("UTCP client is not initialized. Call start() first.")

        return asyncio.run(self.call_tool_async(tool_use_id, tool_name, arguments))

    async def search_tools(self, query: str, max_results: Optional[int] = None) -> List[UTCPAgentTool]:
        """Search for tools matching the given query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of matching UTCPAgentTool instances
        """
        if not self._utcp_client:
            raise UTCPClientInitializationError("UTCP client is not initialized. Call start() first.")

        try:
            limit = max_results or 100
            utcp_tools = await self._utcp_client.search_tools(query=query, limit=limit)

            # Convert to UTCPAgentTool instances
            agent_tools = []
            for utcp_tool in utcp_tools:
                try:
                    agent_tool = UTCPAgentTool.from_utcp_tool(utcp_tool, self)
                    agent_tools.append(agent_tool)
                except Exception as e:
                    logger.warning("Failed to convert UTCP tool %s to agent tool: %s", utcp_tool.name, e)

            return agent_tools
        except Exception as e:
            logger.error("Failed to search tools: %s", e)
            return []

    def _handle_tool_result(self, tool_use_id: str, result: Any) -> Dict[str, Any]:
        """Handle and format tool execution result.

        Args:
            tool_use_id: The tool use identifier
            result: Raw tool execution result

        Returns:
            Formatted result dictionary
        """
        try:
            if isinstance(result, str):
                content = result
            elif isinstance(result, (dict, list)):
                content = json.dumps(result, indent=2)
            else:
                content = str(result)

            return {"status": "success", "toolUseId": tool_use_id, "content": [{"text": content}]}
        except Exception as e:
            logger.error("Failed to process tool result: %s", e)
            return self._handle_tool_execution_error(tool_use_id, e)

    def _handle_tool_execution_error(self, tool_use_id: str, exception: Exception) -> Dict[str, Any]:
        """Handle tool execution errors and format error response.

        Args:
            tool_use_id: The tool use identifier
            exception: The exception that occurred

        Returns:
            Formatted error response dictionary
        """
        error_message = f"UTCP tool execution failed: {str(exception)}"
        return {"status": "error", "toolUseId": tool_use_id, "content": [{"text": error_message}]}
