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
from typing import Any, Callable, Coroutine, Dict, Optional, Pattern, Sequence, TypeVar, Union, cast

import anyio
from mcp import ClientSession, ListToolsResult
from mcp.client.session import ElicitationFnT
from mcp.types import BlobResourceContents, GetPromptResult, ListPromptsResult, TextResourceContents
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import EmbeddedResource as MCPEmbeddedResource
from mcp.types import ImageContent as MCPImageContent
from mcp.types import TextContent as MCPTextContent
from typing_extensions import Protocol, TypedDict

from ...experimental.tools import ToolProvider
from ...types import PaginatedList
from ...types.exceptions import MCPClientInitializationError, ToolProviderException
from ...types.media import ImageFormat
from ...types.tools import AgentTool, ToolResultContent, ToolResultStatus
from .mcp_agent_tool import MCPAgentTool
from .mcp_instrumentation import mcp_instrumentation
from .mcp_types import MCPToolResult, MCPTransport

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _ToolFilterCallback(Protocol):
    def __call__(self, tool: AgentTool, **kwargs: Any) -> bool: ...


_ToolMatcher = str | Pattern[str] | _ToolFilterCallback


class ToolFilters(TypedDict, total=False):
    """Filters for controlling which MCP tools are loaded and available.

    Tools are filtered in this order:
    1. If 'allowed' is specified, only tools matching these patterns are included
    2. Tools matching 'rejected' patterns are then excluded
    """

    allowed: list[_ToolMatcher]
    rejected: list[_ToolMatcher]


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

# Non-fatal error patterns that should not cause connection collapse
_NON_FATAL_ERROR_PATTERNS = [
    # Occurs when client receives response with unrecognized ID
    # Can occur after a client-side timeout
    # See: https://github.com/modelcontextprotocol/python-sdk/blob/c51936f61f35a15f0b1f8fb6887963e5baee1506/src/mcp/shared/session.py#L421
    "unknown request id",
]


class MCPClient(ToolProvider):
    """Represents a connection to a Model Context Protocol (MCP) server.

    This class implements a context manager pattern for efficient connection management,
    allowing reuse of the same connection for multiple tool calls to reduce latency.
    It handles the creation, initialization, and cleanup of MCP connections.

    The connection runs in a background thread to avoid blocking the main application thread
    while maintaining communication with the MCP service. When structured content is available
    from MCP tools, it will be returned as the last item in the content array of the ToolResult.

    Warning:
        This class implements the experimental ToolProvider interface and its methods
        are subject to change.
    """

    def __init__(
        self,
        transport_callable: Callable[[], MCPTransport],
        *,
        startup_timeout: int = 30,
        tool_filters: ToolFilters | None = None,
        prefix: str | None = None,
        elicitation_callback: Optional[ElicitationFnT] = None,
    ) -> None:
        """Initialize a new MCP Server connection.

        Args:
            transport_callable: A callable that returns an MCPTransport (read_stream, write_stream) tuple.
            startup_timeout: Timeout after which MCP server initialization should be cancelled.
                Defaults to 30.
            tool_filters: Optional filters to apply to tools.
            prefix: Optional prefix for tool names.
            elicitation_callback: Optional callback function to handle elicitation requests from the MCP server.
        """
        self._startup_timeout = startup_timeout
        self._tool_filters = tool_filters
        self._prefix = prefix
        self._elicitation_callback = elicitation_callback

        mcp_instrumentation()
        self._session_id = uuid.uuid4()
        self._log_debug_with_thread("initializing MCPClient connection")
        # Main thread blocks until future completes
        self._init_future: futures.Future[None] = futures.Future()
        # Set within the inner loop as it needs the asyncio loop
        self._close_future: asyncio.futures.Future[None] | None = None
        self._close_exception: None | Exception = None
        # Do not want to block other threads while close event is false
        self._transport_callable = transport_callable

        self._background_thread: threading.Thread | None = None
        self._background_thread_session: ClientSession | None = None
        self._background_thread_event_loop: AbstractEventLoop | None = None
        self._loaded_tools: list[MCPAgentTool] | None = None
        self._tool_provider_started = False
        self._consumers: set[Any] = set()

    def __enter__(self) -> "MCPClient":
        """Context manager entry point which initializes the MCP server connection.

        TODO: Refactor to lazy initialization pattern following idiomatic Python.
        Heavy work in __enter__ is non-idiomatic - should move connection logic to first method call instead.
        """
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
            logger.exception("client initialization timed out")
            # Pass None for exc_type, exc_val, exc_tb since this isn't a context manager exit
            self.stop(None, None, None)
            raise MCPClientInitializationError(
                f"background thread did not start in {self._startup_timeout} seconds"
            ) from e
        except Exception as e:
            logger.exception("client failed to initialize")
            # Pass None for exc_type, exc_val, exc_tb since this isn't a context manager exit
            self.stop(None, None, None)
            raise MCPClientInitializationError("the client initialization failed") from e
        return self

    # ToolProvider interface methods (experimental, as ToolProvider is experimental)
    async def load_tools(self, **kwargs: Any) -> Sequence[AgentTool]:
        """Load and return tools from the MCP server.

        This method implements the ToolProvider interface by loading tools
        from the MCP server and caching them for reuse.

        Args:
            **kwargs: Additional arguments for future compatibility.

        Returns:
            List of AgentTool instances from the MCP server.
        """
        logger.debug(
            "started=<%s>, cached_tools=<%s> | loading tools",
            self._tool_provider_started,
            self._loaded_tools is not None,
        )

        if not self._tool_provider_started:
            try:
                logger.debug("starting MCP client")
                self.start()
                self._tool_provider_started = True
                logger.debug("MCP client started successfully")
            except Exception as e:
                logger.error("error=<%s> | failed to start MCP client", e)
                raise ToolProviderException(f"Failed to start MCP client: {e}") from e

        if self._loaded_tools is None:
            logger.debug("loading tools from MCP server")
            self._loaded_tools = []
            pagination_token = None
            page_count = 0

            while True:
                logger.debug("page=<%d>, token=<%s> | fetching tools page", page_count, pagination_token)
                # Use constructor defaults for prefix and filters in load_tools
                paginated_tools = self.list_tools_sync(
                    pagination_token, prefix=self._prefix, tool_filters=self._tool_filters
                )

                # Tools are already filtered by list_tools_sync, so add them all
                for tool in paginated_tools:
                    self._loaded_tools.append(tool)

                logger.debug(
                    "page=<%d>, page_tools=<%d>, total_filtered=<%d> | processed page",
                    page_count,
                    len(paginated_tools),
                    len(self._loaded_tools),
                )

                pagination_token = paginated_tools.pagination_token
                page_count += 1

                if pagination_token is None:
                    break

            logger.debug("final_tools=<%d> | loading complete", len(self._loaded_tools))

        return self._loaded_tools

    def add_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Add a consumer to this tool provider.

        Synchronous to prevent GC deadlocks when called from Agent finalizers.
        """
        self._consumers.add(consumer_id)
        logger.debug("added provider consumer, count=%d", len(self._consumers))

    def remove_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Remove a consumer from this tool provider.

        This method is idempotent - calling it multiple times with the same ID
        has no additional effect after the first call.

        Synchronous to prevent GC deadlocks when called from Agent finalizers.
        Uses existing synchronous stop() method for safe cleanup.
        """
        self._consumers.discard(consumer_id)
        logger.debug("removed provider consumer, count=%d", len(self._consumers))

        if not self._consumers and self._tool_provider_started:
            logger.debug("no consumers remaining, cleaning up")
            try:
                self.stop(None, None, None)  # Existing sync method - safe for finalizers
                self._tool_provider_started = False
                self._loaded_tools = None
            except Exception as e:
                logger.error("error=<%s> | failed to cleanup MCP client", e)
                raise ToolProviderException(f"Failed to cleanup MCP client: {e}") from e

    # MCP-specific methods

    def stop(
        self, exc_type: Optional[BaseException], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """Signals the background thread to stop and waits for it to complete, ensuring proper cleanup of all resources.

        This method is defensive and can handle partial initialization states that may occur
        if start() fails partway through initialization.

        Resources to cleanup:
        - _background_thread: Thread running the async event loop
        - _background_thread_session: MCP ClientSession (auto-closed by context manager)
        - _background_thread_event_loop: AsyncIO event loop in background thread
        - _close_future: AsyncIO future to signal thread shutdown
        - _close_exception: Exception that caused the background thread shutdown; None if a normal shutdown occurred.
        - _init_future: Future for initialization synchronization

        Cleanup order:
        1. Signal close future to background thread (if session initialized)
        2. Wait for background thread to complete
        3. Reset all state for reuse

        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        self._log_debug_with_thread("exiting MCPClient context")

        # Only try to signal close future if we have a background thread
        if self._background_thread is not None:
            # Signal close future if event loop exists
            if self._background_thread_event_loop is not None:

                async def _set_close_event() -> None:
                    if self._close_future and not self._close_future.done():
                        self._close_future.set_result(None)

                # Not calling _invoke_on_background_thread since the session does not need to exist
                # we only need the thread and event loop to exist.
                asyncio.run_coroutine_threadsafe(coro=_set_close_event(), loop=self._background_thread_event_loop)

            self._log_debug_with_thread("waiting for background thread to join")
            self._background_thread.join()

        if self._background_thread_event_loop is not None:
            self._background_thread_event_loop.close()

        self._log_debug_with_thread("background thread is closed, MCPClient context exited")

        # Reset fields to allow instance reuse
        self._init_future = futures.Future()
        self._background_thread = None
        self._background_thread_session = None
        self._background_thread_event_loop = None
        self._session_id = uuid.uuid4()
        self._loaded_tools = None
        self._tool_provider_started = False
        self._consumers = set()

        if self._close_exception:
            exception = self._close_exception
            self._close_exception = None
            raise RuntimeError("Connection to the MCP server was closed") from exception

    def list_tools_sync(
        self,
        pagination_token: str | None = None,
        prefix: str | None = None,
        tool_filters: ToolFilters | None = None,
    ) -> PaginatedList[MCPAgentTool]:
        """Synchronously retrieves the list of available tools from the MCP server.

        This method calls the asynchronous list_tools method on the MCP session
        and adapts the returned tools to the AgentTool interface.

        Args:
            pagination_token: Optional token for pagination
            prefix: Optional prefix to apply to tool names. If None, uses constructor default.
                   If explicitly provided (including empty string), overrides constructor default.
            tool_filters: Optional filters to apply to tools. If None, uses constructor default.
                         If explicitly provided (including empty dict), overrides constructor default.

        Returns:
            List[AgentTool]: A list of available tools adapted to the AgentTool interface
        """
        self._log_debug_with_thread("listing MCP tools synchronously")
        if not self._is_session_active():
            raise MCPClientInitializationError(CLIENT_SESSION_NOT_RUNNING_ERROR_MESSAGE)

        effective_prefix = self._prefix if prefix is None else prefix
        effective_filters = self._tool_filters if tool_filters is None else tool_filters

        async def _list_tools_async() -> ListToolsResult:
            return await cast(ClientSession, self._background_thread_session).list_tools(cursor=pagination_token)

        list_tools_response: ListToolsResult = self._invoke_on_background_thread(_list_tools_async()).result()
        self._log_debug_with_thread("received %d tools from MCP server", len(list_tools_response.tools))

        mcp_tools = []
        for tool in list_tools_response.tools:
            # Apply prefix if specified
            if effective_prefix:
                prefixed_name = f"{effective_prefix}_{tool.name}"
                mcp_tool = MCPAgentTool(tool, self, name_override=prefixed_name)
                logger.debug("tool_rename=<%s->%s> | renamed tool", tool.name, prefixed_name)
            else:
                mcp_tool = MCPAgentTool(tool, self)

            # Apply filters if specified
            if self._should_include_tool_with_filters(mcp_tool, effective_filters):
                mcp_tools.append(mcp_tool)

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
            return await cast(ClientSession, self._background_thread_session).list_prompts(cursor=pagination_token)

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
            return await cast(ClientSession, self._background_thread_session).get_prompt(prompt_id, arguments=args)

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
            return await cast(ClientSession, self._background_thread_session).call_tool(
                name, arguments, read_timeout_seconds
            )

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
            return await cast(ClientSession, self._background_thread_session).call_tool(
                name, arguments, read_timeout_seconds
            )

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

        # Build a typed list of ToolResultContent.
        mapped_contents: list[ToolResultContent] = [
            mc
            for content in call_tool_result.content
            if (mc := self._map_mcp_content_to_tool_result_content(content)) is not None
        ]

        status: ToolResultStatus = "error" if call_tool_result.isError else "success"
        self._log_debug_with_thread("tool execution completed with status: %s", status)
        result = MCPToolResult(
            status=status,
            toolUseId=tool_use_id,
            content=mapped_contents,
        )

        if call_tool_result.structuredContent:
            result["structuredContent"] = call_tool_result.structuredContent
        if call_tool_result.meta:
            result["metadata"] = call_tool_result.meta

        return result

    async def _async_background_thread(self) -> None:
        """Asynchronous method that runs in the background thread to manage the MCP connection.

        This method establishes the transport connection, creates and initializes the MCP session,
        signals readiness to the main thread, and waits for a close signal.
        """
        self._log_debug_with_thread("starting async background thread for MCP connection")

        # Initialized here so that it has the asyncio loop
        self._close_future = asyncio.Future()

        try:
            async with self._transport_callable() as (read_stream, write_stream, *_):
                self._log_debug_with_thread("transport connection established")
                async with ClientSession(
                    read_stream,
                    write_stream,
                    message_handler=self._handle_error_message,
                    elicitation_callback=self._elicitation_callback,
                ) as session:
                    self._log_debug_with_thread("initializing MCP session")
                    await session.initialize()

                    self._log_debug_with_thread("session initialized successfully")
                    # Store the session for use while we await the close event
                    self._background_thread_session = session
                    # Signal that the session has been created and is ready for use
                    self._init_future.set_result(None)

                    self._log_debug_with_thread("waiting for close signal")
                    # Keep background thread running until signaled to close.
                    # Thread is not blocked as this a future
                    await self._close_future

                    self._log_debug_with_thread("close signal received")
        except Exception as e:
            # If we encounter an exception and the future is still running,
            # it means it was encountered during the initialization phase.
            if not self._init_future.done():
                self._init_future.set_exception(e)
            else:
                # _close_future is automatically cancelled by the framework which doesn't provide us with the useful
                # exception, so instead we store the exception in a different field where stop() can read it
                self._close_exception = e
                if self._close_future and not self._close_future.done():
                    self._close_future.set_result(None)

                self._log_debug_with_thread(
                    "encountered exception on background thread after initialization %s", str(e)
                )

    # Raise an exception if the underlying client raises an exception in a message
    # This happens when the underlying client has an http timeout error
    async def _handle_error_message(self, message: Exception | Any) -> None:
        if isinstance(message, Exception):
            error_msg = str(message).lower()
            if any(pattern in error_msg for pattern in _NON_FATAL_ERROR_PATTERNS):
                self._log_debug_with_thread("ignoring non-fatal MCP session error", message)
            else:
                raise message
        await anyio.lowlevel.checkpoint()

    def _background_task(self) -> None:
        """Sets up and runs the event loop in the background thread.

        This method creates a new event loop for the background thread,
        sets it as the current event loop, and runs the async_background_thread
        coroutine until completion. In this case "until completion" means until the _close_future is resolved.
        This allows for a long-running event loop.
        """
        self._log_debug_with_thread("setting up background task event loop")
        self._background_thread_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._background_thread_event_loop)
        self._background_thread_event_loop.run_until_complete(self._async_background_thread())

    def _map_mcp_content_to_tool_result_content(
        self,
        content: MCPTextContent | MCPImageContent | MCPEmbeddedResource | Any,
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
        elif isinstance(content, MCPEmbeddedResource):
            """
            TODO: Include URI information in results.
                Models may find it useful to be aware not only of the information,
                but the location of the information too.

                This may be difficult without taking an opinionated position. For example,
                a content block may need to indicate that the following Image content block
                is of particular URI.
            """

            self._log_debug_with_thread("mapping MCP embedded resource content")

            resource = content.resource
            if isinstance(resource, TextResourceContents):
                return {"text": resource.text}
            elif isinstance(resource, BlobResourceContents):
                try:
                    raw_bytes = base64.b64decode(resource.blob)
                except Exception:
                    self._log_debug_with_thread("embedded resource blob could not be decoded - dropping")
                    return None

                if resource.mimeType and (
                    resource.mimeType.startswith("text/")
                    or resource.mimeType
                    in (
                        "application/json",
                        "application/xml",
                        "application/javascript",
                        "application/yaml",
                        "application/x-yaml",
                    )
                    or resource.mimeType.endswith(("+json", "+xml"))
                ):
                    try:
                        return {"text": raw_bytes.decode("utf-8", errors="replace")}
                    except Exception:
                        pass

                if resource.mimeType in MIME_TO_FORMAT:
                    return {
                        "image": {
                            "format": MIME_TO_FORMAT[resource.mimeType],
                            "source": {"bytes": raw_bytes},
                        }
                    }

                self._log_debug_with_thread("embedded resource blob with non-textual/unknown mimeType - dropping")
                return None

            return None  # type: ignore[unreachable]  # Defensive: future MCP resource types
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
        # save a reference to this so that even if it's reset we have the original
        close_future = self._close_future

        if (
            self._background_thread_session is None
            or self._background_thread_event_loop is None
            or close_future is None
        ):
            raise MCPClientInitializationError("the client session was not initialized")

        async def run_async() -> T:
            # Fix for strands-agents/sdk-python/issues/995 - cancel all pending invocations if/when the session closes
            invoke_event = asyncio.create_task(coro)
            tasks: list[asyncio.Task | asyncio.Future] = [
                invoke_event,
                close_future,
            ]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            if done.pop() == close_future:
                self._log_debug_with_thread("event loop for the server closed before the invoke completed")
                raise RuntimeError("Connection to the MCP server was closed")
            else:
                return await invoke_event

        invoke_future = asyncio.run_coroutine_threadsafe(coro=run_async(), loop=self._background_thread_event_loop)
        return invoke_future

    def _should_include_tool(self, tool: MCPAgentTool) -> bool:
        """Check if a tool should be included based on constructor filters."""
        return self._should_include_tool_with_filters(tool, self._tool_filters)

    def _should_include_tool_with_filters(self, tool: MCPAgentTool, filters: Optional[ToolFilters]) -> bool:
        """Check if a tool should be included based on provided filters."""
        if not filters:
            return True

        # Apply allowed filter
        if "allowed" in filters:
            if not self._matches_patterns(tool, filters["allowed"]):
                return False

        # Apply rejected filter
        if "rejected" in filters:
            if self._matches_patterns(tool, filters["rejected"]):
                return False

        return True

    def _matches_patterns(self, tool: MCPAgentTool, patterns: list[_ToolMatcher]) -> bool:
        """Check if tool matches any of the given patterns."""
        for pattern in patterns:
            if callable(pattern):
                if pattern(tool):
                    return True
            elif isinstance(pattern, Pattern):
                if pattern.match(tool.mcp_tool.name):
                    return True
            elif isinstance(pattern, str):
                if pattern == tool.mcp_tool.name:
                    return True
        return False

    def _is_session_active(self) -> bool:
        return self._background_thread is not None and self._background_thread.is_alive()
