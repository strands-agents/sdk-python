"""OpenTelemetry instrumentation for Model Context Protocol (MCP) tracing.

Enables distributed tracing across MCP client-server boundaries by injecting
OpenTelemetry context into MCP request metadata (_meta field) and extracting
it on the server side, creating unified traces that span from agent calls
through MCP tool executions.

Based on: https://github.com/traceloop/openllmetry/tree/main/packages/opentelemetry-instrumentation-mcp
Related issue: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/246
"""

from contextlib import _AsyncGeneratorContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Tuple

from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCRequest
from opentelemetry import context, propagate
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper


@dataclass(slots=True, frozen=True)
class ItemWithContext:
    """Wrapper for items that need to carry OpenTelemetry context.

    Used to preserve tracing context across async boundaries in MCP sessions,
    ensuring that distributed traces remain connected even when messages are
    processed asynchronously.

    Attributes:
        item: The original item being wrapped
        ctx: The OpenTelemetry context associated with the item
    """

    item: Any
    ctx: context.Context


def mcp_instrumentation() -> None:
    """Apply OpenTelemetry instrumentation patches to MCP components.

    This function instruments three key areas of MCP communication:
    1. Client-side: Injects tracing context into tool call requests
    2. Transport-level: Extracts context from incoming messages
    3. Session-level: Manages bidirectional context flow

    The patches enable distributed tracing by:
    - Adding OpenTelemetry context to the _meta field of MCP requests
    - Extracting and activating context on the server side
    - Preserving context across async message processing boundaries
    """

    def patch_mcp_client(wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any) -> Any:
        """Patch MCP client to inject OpenTelemetry context into tool calls.

        Intercepts outgoing MCP requests and injects the current OpenTelemetry
        context into the request's _meta field for tools/call methods. This
        enables server-side context extraction and trace continuation.

        Args:
            wrapped: The original function being wrapped
            instance: The instance the method is being called on
            args: Positional arguments to the wrapped function
            kwargs: Keyword arguments to the wrapped function

        Returns:
            Result of the wrapped function call
        """
        if len(args) < 1:
            return wrapped(*args, **kwargs)

        request = args[0]
        method = getattr(request.root, "method", None)

        if method != "tools/call":
            return wrapped(*args, **kwargs)

        try:
            if hasattr(request.root, "params") and request.root.params:
                # Handle Pydantic models
                if hasattr(request.root.params, "model_dump") and hasattr(request.root.params, "model_validate"):
                    params_dict = request.root.params.model_dump()
                    # Add _meta with tracing context
                    meta = params_dict.setdefault("_meta", {})
                    propagate.get_global_textmap().inject(meta)

                    # Recreate the Pydantic model with the updated data
                    # This preserves the original model type and avoids serialization warnings
                    params_class = type(request.root.params)
                    try:
                        request.root.params = params_class.model_validate(params_dict)
                    except Exception:
                        # Fallback to dict if model recreation fails
                        request.root.params = params_dict

                elif isinstance(request.root.params, dict):
                    # Handle dict params directly
                    meta = request.root.params.setdefault("_meta", {})
                    propagate.get_global_textmap().inject(meta)

            return wrapped(*args, **kwargs)

        except Exception:
            return wrapped(*args, **kwargs)

    def transport_wrapper() -> Callable[
        [Callable[..., Any], Any, Any, Any], _AsyncGeneratorContextManager[tuple[Any, Any]]
    ]:
        """Create a wrapper for MCP transport connections.

        Returns a context manager that wraps transport read/write streams
        with context extraction capabilities. The wrapped reader will
        automatically extract OpenTelemetry context from incoming messages.

        Returns:
            An async context manager that yields wrapped transport streams
        """

        @asynccontextmanager
        async def traced_method(
            wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
        ) -> AsyncGenerator[Tuple[Any, Any], None]:
            async with wrapped(*args, **kwargs) as result:
                try:
                    read_stream, write_stream = result
                except ValueError:
                    read_stream, write_stream, _ = result
                yield TransportContextExtractingReader(read_stream), write_stream

        return traced_method

    def session_init_wrapper() -> Callable[[Any, Any, Tuple[Any, ...], dict[str, Any]], None]:
        """Create a wrapper for MCP session initialization.

        Wraps session message streams to enable bidirectional context flow.
        The reader extracts and activates context, while the writer preserves
        context for async processing.

        Returns:
            A function that wraps session initialization
        """

        def traced_method(
            wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            wrapped(*args, **kwargs)
            reader = getattr(instance, "_incoming_message_stream_reader", None)
            writer = getattr(instance, "_incoming_message_stream_writer", None)
            if reader and writer:
                instance._incoming_message_stream_reader = SessionContextAttachingReader(reader)
                instance._incoming_message_stream_writer = SessionContextSavingWriter(writer)

        return traced_method

    # Apply patches
    wrap_function_wrapper("mcp.shared.session", "BaseSession.send_request", patch_mcp_client)

    register_post_import_hook(
        lambda _: wrap_function_wrapper(
            "mcp.server.streamable_http", "StreamableHTTPServerTransport.connect", transport_wrapper()
        ),
        "mcp.server.streamable_http",
    )

    register_post_import_hook(
        lambda _: wrap_function_wrapper("mcp.server.session", "ServerSession.__init__", session_init_wrapper()),
        "mcp.server.session",
    )


class TransportContextExtractingReader(ObjectProxy):
    """A proxy reader that extracts OpenTelemetry context from MCP messages.

    Wraps an async message stream reader to automatically extract and activate
    OpenTelemetry context from the _meta field of incoming MCP requests. This
    enables server-side trace continuation from client-injected context.

    The reader handles both SessionMessage and JSONRPCMessage formats, and
    supports both dict and Pydantic model parameter structures.
    """

    def __init__(self, wrapped: Any) -> None:
        """Initialize the context-extracting reader.

        Args:
            wrapped: The original async stream reader to wrap
        """
        super().__init__(wrapped)

    async def __aenter__(self) -> Any:
        """Enter the async context manager by delegating to the wrapped object."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the async context manager by delegating to the wrapped object."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Iterate over messages, extracting and activating context as needed.

        For each incoming message, checks if it contains tracing context in
        the _meta field. If found, extracts and activates the context for
        the duration of message processing, then properly detaches it.

        Yields:
            Messages from the wrapped stream, processed under the appropriate
            OpenTelemetry context
        """
        async for item in self.__wrapped__:
            if isinstance(item, SessionMessage):
                request = item.message.root
            elif type(item) is JSONRPCMessage:
                request = item.root
            else:
                yield item
                continue

            if isinstance(request, JSONRPCRequest) and request.params:
                # Handle both dict and Pydantic model params
                if hasattr(request.params, "get"):
                    # Dict-like access
                    meta = request.params.get("_meta")
                elif hasattr(request.params, "_meta"):
                    # Direct attribute access for Pydantic models
                    meta = getattr(request.params, "_meta", None)
                else:
                    meta = None

                if meta:
                    extracted_context = propagate.extract(meta)
                    restore = context.attach(extracted_context)
                    try:
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class SessionContextSavingWriter(ObjectProxy):
    """A proxy writer that preserves OpenTelemetry context with outgoing items.

    Wraps an async message stream writer to capture the current OpenTelemetry
    context and associate it with outgoing items. This enables context
    preservation across async boundaries in MCP session processing.
    """

    def __init__(self, wrapped: Any) -> None:
        """Initialize the context-saving writer.

        Args:
            wrapped: The original async stream writer to wrap
        """
        super().__init__(wrapped)

    async def __aenter__(self) -> Any:
        """Enter the async context manager by delegating to the wrapped object."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the async context manager by delegating to the wrapped object."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        """Send an item while preserving the current OpenTelemetry context.

        Captures the current context and wraps the item with it, enabling
        the receiving side to restore the appropriate tracing context.

        Args:
            item: The item to send through the stream

        Returns:
            Result of sending the wrapped item
        """
        ctx = context.get_current()
        return await self.__wrapped__.send(ItemWithContext(item, ctx))


class SessionContextAttachingReader(ObjectProxy):
    """A proxy reader that restores OpenTelemetry context from wrapped items.

    Wraps an async message stream reader to detect ItemWithContext instances
    and restore their associated OpenTelemetry context during processing.
    This completes the context preservation cycle started by SessionContextSavingWriter.
    """

    def __init__(self, wrapped: Any) -> None:
        """Initialize the context-attaching reader.

        Args:
            wrapped: The original async stream reader to wrap
        """
        super().__init__(wrapped)

    async def __aenter__(self) -> Any:
        """Enter the async context manager by delegating to the wrapped object."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the async context manager by delegating to the wrapped object."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Iterate over items, restoring context for ItemWithContext instances.

        For items wrapped with context, temporarily activates the associated
        OpenTelemetry context during processing, then properly detaches it.
        Regular items are yielded without context modification.

        Yields:
            Unwrapped items processed under their associated OpenTelemetry context
        """
        async for item in self.__wrapped__:
            if isinstance(item, ItemWithContext):
                restore = context.attach(item.ctx)
                try:
                    yield item.item
                finally:
                    context.detach(restore)
            else:
                yield item
