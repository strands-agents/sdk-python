"""Model Context Protocol (MCP) server connection management module.

This module provides the MCPClient class which handles connections to MCP servers.
It manages the lifecycle of MCP connections, including initialization, tool discovery,
tool invocation, and proper cleanup of resources. The connection runs in a background
thread to avoid blocking the main application thread while maintaining communication
with the MCP service.
"""

import asyncio
import base64
import logging
import threading
import uuid
from asyncio import AbstractEventLoop
from concurrent import futures
from datetime import timedelta
from types import TracebackType
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar, Union

from mcp import ClientSession, ListToolsResult
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import GetPromptResult, ListPromptsResult
from mcp.types import ImageContent as MCPImageContent
from mcp.types import TextContent as MCPTextContent

from ...types import PaginatedList
from ...types.exceptions import MCPClientInitializationError
from ...types.media import ImageFormat
from ...types.tools import ToolResultContent, ToolResultStatus
from .mcp_agent_tool import MCPAgentTool
from .mcp_instrumentation import mcp_instrumentation
from .mcp_types import MCPToolResult, MCPTransport

logger = logging.getLogger(__name__)

T = TypeVar("T")

MIME_TO_FORMAT: Dict[str, ImageFormat] = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE = (
    "the client session is not running. Ensure the agent is used within "
    "the MCP client context manager. For more information see: "
    "https://strandsagents.com/latest/user-guide/concepts/tools/mcp-tools/#mcpclientinitializationerror"
)


class MCPClient:
    """Represents a connection to a Model Context Protocol (MCP) server.

    This class implements a context manager pattern for efficient connection management,
    allowing reuse of the same connection for multiple tool calls to reduce latency.
    It handles the creation, initialization, and cleanup of MCP connections.

    The connection runs in a background thread to avoid blocking the main application thread
    while maintaining communication with the MCP service. When structured content is available
    from MCP tools, it will be returned as the last item in the content array of the ToolResult.
    """

    def __init__(self, transport_callable: Callable[[], MCPTransport], *, startup_timeout: int = 30):
        """Initialize a new MCP Server connection.

        Args:
            transport_callable: A callable that returns an MCPTransport (read_stream, write_stream) tuple
            startup_timeout: Timeout after which MCP server initialization should be cancelled
                Defaults to 30.
        """
        self._startup_timeout = startup_timeout

        mcp_instrumentation()
        self._session_id = uuid.uuid4()
        self._log_debug_with_thread("initializing MCPClient connection")
        # Main thread blocks until future completesock
        self._init_future: futures.Future[None] = futures.Future()
        # Do not want to block other threads while close event is false
        self._close_event = asyncio.Event()
        self._transport_callable = transport_callable

        self._background_thread: threading.Thread | None = None
        self._background_thread_session: ClientSession
        self._background_thread_event_loop: AbstractEventLoop

    def __enter__(self) -> "MCPClient":
        """Context manager entry point which initializes the MCP server connection."""
        return self.start()

    def __exit__(self, exc_type: BaseException, exc_val: BaseException, exc_tb: TracebackType) -> None:
        """Context manager exit point that cleans up resources."""
        self.stop(exc_type, exc_val, exc_tb)

    def start(self) -> "MCPClient":
        """Starts the background thread and waits for initialization.

        This method starts the background thread that manages the MCP connection
        and blocks until the connection is ready or times out.

        Returns:
            self: The MCPClient instance

        Raises:
            Exception: If the MCP connection fails to initialize within the timeout period
        """
        if self._is_session_active():
            raise MCPClientInitializationError("the client session is currently running")

        self._log_debug_with_thread("entering MCPClient context")
        self._background_thread = threading.Thread(target=self._background_task, args=[], daemon=True)
        self._background_thread.start()
        self._log_debug_with_thread("background thread started, waiting for ready event")
        try:
            # Blocking main thread until session is initialized in other thread or if the thread stops
            self._init_future.result(timeout=self._startup_timeout)
            self._log_debug_with_thread("the client initialization was successful")
        except futures.TimeoutError as e:
            raise MCPClientInitializationError("background thread did not start in 30 seconds") from e
        except Exception as e:
            logger.exception("client failed to initialize")
            raise MCPClientInitializationError("the client initialization failed") from e
        return self

    def stop(
        self, exc_type: Optional[BaseException], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """Signals the background thread to stop and waits for it to complete, ensuring proper cleanup of all resources.

        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        self._log_debug_with_thread("exiting MCPClient context")

        async def _set_close_event() -> None:
            self._close_event.set()

        self._invoke_on_background_thread(_set_close_event()).result()
        self._log_debug_with_thread("waiting for background thread to join")
        if self._background_thread is not None:
            self._background_thread.join()
        self._log_debug_with_thread("background thread joined, MCPClient context exited")

        # Reset fields to allow instance reuse
        self._init_future = futures.Future()
        self._close_event = asyncio.Event()
        self._background_thread = None
        self._session_id = uuid.uuid4()

    def list_tools_sync(self, pagination_token: Optional[str] = None) -> PaginatedList[MCPAgentTool]:
        """Synchronously retrieves the list of available tools from the MCP server.

        This method calls the asynchronous list_tools method on the MCP session
        and adapts the returned tools to the AgentTool interface.

        Returns:
            List[AgentTool]: A list of available tools adapted to the AgentTool interface
        """
        self._log_debug_with_thread("listing MCP tools synchronously")
        if not self._is_session_active():
            raise MCPClientInitializationError(CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE)

        async def _list_tools_async() -> ListToolsResult:
            return await self._background_thread_session.list_tools(cursor=pagination_token)

        list_tools_response: ListToolsResult = self._invoke_on_background_thread(_list_tools_async()).result()
        self._log_debug_with_thread("received %d tools from MCP server", len(list_tools_response.tools))

        mcp_tools = [MCPAgentTool(tool, self) for tool in list_tools_response.tools]
        self._log_debug_with_thread("successfully adapted %d MCP tools", len(mcp_tools))
        return PaginatedList[MCPAgentTool](mcp_tools, token=list_tools_response.nextCursor)

    def list_prompts_sync(self, pagination_token: Optional[str] = None) -> ListPromptsResult:
        """Synchronously retrieves the list of available prompts from the MCP server.

        This method calls the asynchronous list_prompts method on the MCP session
        and returns the raw ListPromptsResult with pagination support.

        Args:
            pagination_token: Optional token for pagination

        Returns:
            ListPromptsResult: The raw MCP response containing prompts and pagination info
        """
        self._log_debug_with_thread("listing MCP prompts synchronously")
        if not self._is_session_active():
            raise MCPClientInitializationError(CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE)

        async def _list_prompts_async() -> ListPromptsResult:
            return await self._background_thread_session.list_prompts(cursor=pagination_token)

        list_prompts_result: ListPromptsResult = self._invoke_on_background_thread(_list_prompts_async()).result()
        self._log_debug_with_thread("received %d prompts from MCP server", len(list_prompts_result.prompts))
        for prompt in list_prompts_result.prompts:
            self._log_debug_with_thread(prompt.name)

        return list_prompts_result

    def get_prompt_sync(self, prompt_id: str, args: dict[str, Any]) -> GetPromptResult:
        """Synchronously retrieves a prompt from the MCP server.

        Args:
            prompt_id: The ID of the prompt to retrieve
            args: Optional arguments to pass to the prompt

        Returns:
            GetPromptResult: The prompt response from the MCP server
        """
        self._log_debug_with_thread("getting MCP prompt synchronously")
        if not self._is_session_active():
            raise MCPClientInitializationError(CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE)

        async def _get_prompt_async() -> GetPromptResult:
            return await self._background_thread_session.get_prompt(prompt_id, arguments=args)

        get_prompt_result: GetPromptResult = self._invoke_on_background_thread(_get_prompt_async()).result()
        self._log_debug_with_thread("received prompt from MCP server")

        return get_prompt_result

    def call_tool_sync(
        self,
        tool_use_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
    ) -> MCPToolResult:
        """Synchronously calls a tool on the MCP server.

        This method calls the asynchronous call_tool method on the MCP session
        and converts the result to the ToolResult format. If the MCP tool returns
        structured content, it will be included as the last item in the content array
        of the returned ToolResult.

        Args:
            tool_use_id: Unique identifier for this tool use
            name: Name of the tool to call
            arguments: Optional arguments to pass to the tool
            read_timeout_seconds: Optional timeout for the tool call

        Returns:
            MCPToolResult: The result of the tool call
        """
        self._log_debug_with_thread("calling MCP tool '%s' synchronously with tool_use_id=%s", name, tool_use_id)
        if not self._is_session_active():
            raise MCPClientInitializationError(CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE)

        async def _call_tool_async() -> MCPCallToolResult:
            return await self._background_thread_session.call_tool(name, arguments, read_timeout_seconds)

        try:
            call_tool_result: MCPCallToolResult = self._invoke_on_background_thread(_call_tool_async()).result()
            return self._handle_tool_result(tool_use_id, call_tool_result)
        except Exception as e:
            logger.exception("tool execution failed")
            return self._handle_tool_execution_error(tool_use_id, e)

    async def call_tool_async(
        self,
        tool_use_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
    ) -> MCPToolResult:
        """Asynchronously calls a tool on the MCP server.

        This method calls the asynchronous call_tool method on the MCP session
        and converts the result to the MCPToolResult format.

        Args:
            tool_use_id: Unique identifier for this tool use
            name: Name of the tool to call
            arguments: Optional arguments to pass to the tool
            read_timeout_seconds: Optional timeout for the tool call

        Returns:
            MCPToolResult: The result of the tool call
        """
        self._log_debug_with_thread("calling MCP tool '%s' asynchronously with tool_use_id=%s", name, tool_use_id)
        if not self._is_session_active():
            raise MCPClientInitializationError(CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE)

        async def _call_tool_async() -> MCPCallToolResult:
            return await self._background_thread_session.call_tool(name, arguments, read_timeout_seconds)

        try:
            future = self._invoke_on_background_thread(_call_tool_async())
            call_tool_result: MCPCallToolResult = await asyncio.wrap_future(future)
            return self._handle_tool_result(tool_use_id, call_tool_result)
        except Exception as e:
            logger.exception("tool execution failed")
            return self._handle_tool_execution_error(tool_use_id, e)

    def _handle_tool_execution_error(self, tool_use_id: str, exception: Exception) -> MCPToolResult:
        """Create error ToolResult with consistent logging."""
        return MCPToolResult(
            status="error",
            toolUseId=tool_use_id,
            content=[{"text": f"Tool execution failed: {str(exception)}"}],
        )

    def _handle_tool_result(self, tool_use_id: str, call_tool_result: MCPCallToolResult) -> MCPToolResult:
        """Maps MCP tool result to the agent's MCPToolResult format.

        This method processes the content from the MCP tool call result and converts it to the format
        expected by the framework.

        Args:
            tool_use_id: Unique identifier for this tool use
            call_tool_result: The result from the MCP tool call

        Returns:
            MCPToolResult: The converted tool result
        """
        self._log_debug_with_thread("received tool result with %d content items", len(call_tool_result.content))

        mapped_content = [
            mapped_content
            for content in call_tool_result.content
            if (mapped_content := self._map_mcp_content_to_tool_result_content(content)) is not None
        ]

        status: ToolResultStatus = "error" if call_tool_result.isError else "success"
        self._log_debug_with_thread("tool execution completed with status: %s", status)
        result = MCPToolResult(
            status=status,
            toolUseId=tool_use_id,
            content=mapped_content,
        )
        if call_tool_result.structuredContent:
            result["structuredContent"] = call_tool_result.structuredContent

        return result

    async def _async_background_thread(self) -> None:
        """Asynchronous method that runs in the background thread to manage the MCP connection.

        This method establishes the transport connection, creates and initializes the MCP session,
        signals readiness to the main thread, and waits for a close signal.
        """
        self._log_debug_with_thread("starting async background thread for MCP connection")
        try:
            async with self._transport_callable() as (read_stream, write_stream, *_):
                self._log_debug_with_thread("transport connection established")
                async with ClientSession(read_stream, write_stream) as session:
                    self._log_debug_with_thread("initializing MCP session")
                    await session.initialize()

                    self._log_debug_with_thread("session initialized successfully")
                    # Store the session for use while we await the close event
                    self._background_thread_session = session
                    # Signal that the session has been created and is ready for use
                    self._init_future.set_result(None)

                    self._log_debug_with_thread("waiting for close signal")
                    # Keep background thread running until signaled to close.
                    # Thread is not blocked as this is an asyncio.Event not a threading.Event
                    await self._close_event.wait()
                    self._log_debug_with_thread("close signal received")
        except Exception as e:
            # If we encounter an exception and the future is still running,
            # it means it was encountered during the initialization phase.
            if not self._init_future.done():
                self._init_future.set_exception(e)
            else:
                self._log_debug_with_thread(
                    "encountered exception on background thread after initialization %s", str(e)
                )

    def _background_task(self) -> None:
        """Sets up and runs the event loop in the background thread.

        This method creates a new event loop for the background thread,
        sets it as the current event loop, and runs the async_background_thread
        coroutine until completion. In this case "until completion" means until the _close_event is set.
        This allows for a long-running event loop.
        """
        self._log_debug_with_thread("setting up background task event loop")
        self._background_thread_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._background_thread_event_loop)
        self._background_thread_event_loop.run_until_complete(self._async_background_thread())

    def _map_mcp_content_to_tool_result_content(
        self,
        content: MCPTextContent | MCPImageContent | Any,
    ) -> Union[ToolResultContent, None]:
        """Maps MCP content types to tool result content types.

        This method converts MCP-specific content types to the generic
        ToolResultContent format used by the agent framework.

        Args:
            content: The MCP content to convert

        Returns:
            ToolResultContent or None: The converted content, or None if the content type is not supported
        """
        if isinstance(content, MCPTextContent):
            self._log_debug_with_thread("mapping MCP text content")
            return {"text": content.text}
        elif isinstance(content, MCPImageContent):
            self._log_debug_with_thread("mapping MCP image content with mime type: %s", content.mimeType)
            return {
                "image": {
                    "format": MIME_TO_FORMAT[content.mimeType],
                    "source": {"bytes": base64.b64decode(content.data)},
                }
            }
        else:
            self._log_debug_with_thread("unhandled content type: %s - dropping content", content.__class__.__name__)
            return None

    def _log_debug_with_thread(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logger helper to help differentiate logs coming from MCPClient background thread."""
        formatted_msg = msg % args if args else msg
        logger.debug(
            "[Thread: %s, Session: %s] %s", threading.current_thread().name, self._session_id, formatted_msg, **kwargs
        )

    def _invoke_on_background_thread(self, coro: Coroutine[Any, Any, T]) -> futures.Future[T]:
        if self._background_thread_session is None or self._background_thread_event_loop is None:
            raise MCPClientInitializationError("the client session was not initialized")
        return asyncio.run_coroutine_threadsafe(coro=coro, loop=self._background_thread_event_loop)

    def _is_session_active(self) -> bool:
        return self._background_thread is not None and self._background_thread.is_alive()
