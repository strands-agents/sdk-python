"""MCP Tool Provider implementation."""

import logging
from typing import Any, Callable, Optional, Pattern, Sequence, Union

from typing_extensions import TypedDict

from ....tools.mcp.mcp_agent_tool import MCPAgentTool
from ....tools.mcp.mcp_client import MCPClient
from ....types.exceptions import ToolProviderException
from ....types.tools import AgentTool
from ..tool_provider import ToolProvider

logger = logging.getLogger(__name__)

_ToolFilterCallback = Callable[[AgentTool], bool]
_ToolFilterPattern = Union[str, Pattern[str], _ToolFilterCallback]


class ToolFilters(TypedDict, total=False):
    """Filters for controlling which MCP tools are loaded and available.

    Tools are filtered in this order:
    1. If 'allowed' is specified, only tools matching these patterns are included
    2. Tools matching 'rejected' patterns are then excluded
    """

    allowed: list[_ToolFilterPattern]
    rejected: list[_ToolFilterPattern]


class MCPToolProvider(ToolProvider):
    """Tool provider for MCP clients with managed lifecycle."""

    def __init__(
        self,
        *,
        client: MCPClient,
        tool_filters: Optional[ToolFilters] = None,
        prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with an MCP client.

        Args:
            client: The MCP client to manage.
            tool_filters: Optional filters to apply to tools.
            prefix: Optional prefix for tool names.
            **kwargs: Additional arguments for future compatibility.
        """
        logger.debug("tool_filters=<%s>, prefix=<%s> | initializing MCPToolProvider", tool_filters, prefix)
        self._client = client
        self._tool_filters = tool_filters
        self._prefix = prefix
        self._tools: Optional[list[MCPAgentTool]] = None  # None = not loaded yet, [] = loaded but empty
        self._started = False

    async def load_tools(self, **kwargs: Any) -> Sequence[AgentTool]:
        """Load and return tools from the MCP client.

        Returns:
            List of tools from the MCP server.
        """
        logger.debug("started=<%s>, cached_tools=<%s> | loading tools", self._started, self._tools is not None)

        if not self._started:
            try:
                logger.debug("starting MCP client")
                self._client.start()
                self._started = True
                logger.debug("MCP client started successfully")
            except Exception as e:
                logger.error("error=<%s> | failed to start MCP client", e)
                raise ToolProviderException(f"Failed to start MCP client: {e}") from e

        if self._tools is None:
            logger.debug("loading tools from MCP server")
            self._tools = []
            pagination_token = None
            page_count = 0

            while True:
                logger.debug("page=<%d>, token=<%s> | fetching tools page", page_count, pagination_token)
                paginated_tools = self._client.list_tools_sync(pagination_token)

                # Process each tool as we get it
                for tool in paginated_tools:
                    # Apply filters
                    if self._should_include_tool(tool):
                        # Apply prefix if needed
                        processed_tool = self._apply_prefix(tool)
                        self._tools.append(processed_tool)

                logger.debug(
                    "page=<%d>, page_tools=<%d>, total_filtered=<%d> | processed page",
                    page_count,
                    len(paginated_tools),
                    len(self._tools),
                )

                pagination_token = paginated_tools.pagination_token
                page_count += 1

                if pagination_token is None:
                    break

            logger.debug("final_tools=<%d> | loading complete", len(self._tools))

        return self._tools

    def _should_include_tool(self, tool: MCPAgentTool) -> bool:
        """Check if a tool should be included based on allowed/rejected filters."""
        if not self._tool_filters:
            return True

        # Apply allowed filter
        if "allowed" in self._tool_filters:
            if not self._matches_patterns(tool, self._tool_filters["allowed"]):
                return False

        # Apply rejected filter
        if "rejected" in self._tool_filters:
            if self._matches_patterns(tool, self._tool_filters["rejected"]):
                return False

        return True

    def _apply_prefix(self, tool: MCPAgentTool) -> MCPAgentTool:
        """Apply prefix to a single tool if needed."""
        if not self._prefix:
            return tool

        # Create new tool with prefixed agent name but preserve original MCP name
        old_name = tool.tool_name
        new_agent_name = f"{self._prefix}_{tool.mcp_tool.name}"
        new_tool = MCPAgentTool(tool.mcp_tool, tool.mcp_client, agent_facing_tool_name=new_agent_name)
        logger.debug("tool_rename=<%s->%s> | renamed tool", old_name, new_agent_name)
        return new_tool

    def _matches_patterns(self, tool: MCPAgentTool, patterns: list[_ToolFilterPattern]) -> bool:
        """Check if tool matches any of the given patterns."""
        for pattern in patterns:
            if callable(pattern):
                if pattern(tool):
                    return True
            elif hasattr(pattern, "match") and hasattr(pattern, "pattern"):
                if pattern.match(tool.tool_name):
                    return True
            elif isinstance(pattern, str):
                if pattern == tool.tool_name:
                    return True
        return False

    async def cleanup(self, **kwargs: Any) -> None:
        """Clean up the MCP client connection."""
        if not self._started:
            return

        logger.debug("cleaning up MCP client")
        try:
            logger.debug("stopping MCP client")
            self._client.stop(None, None, None)
            logger.debug("MCP client stopped successfully")
        except Exception as e:
            logger.error("error=<%s> | failed to cleanup MCP client", e)
            raise ToolProviderException(f"Failed to cleanup MCP client: {e}") from e

        # Only reset state if cleanup succeeded
        self._started = False
        self._tools = None
        logger.debug("MCP client cleanup complete")
