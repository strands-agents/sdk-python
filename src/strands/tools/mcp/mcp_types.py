"""Type definitions for MCP integration."""

from contextlib import AbstractAsyncContextManager
from typing import Any, Dict

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import GetSessionIdCallback
from mcp.shared.memory import MessageStream
from mcp.shared.message import SessionMessage
from typing_extensions import NotRequired

from strands.types.tools import ToolResult

"""
MCPTransport defines the interface for MCP transport implementations. This abstracts
communication with an MCP server, hiding details of the underlying transport mechanism (WebSocket, stdio, etc.).

It represents an async context manager that yields a tuple of read and write streams for MCP communication.
When used with `async with`, it should establish the connection and yield the streams, then clean up
when the context is exited.

The read stream receives messages from the client (or exceptions if parsing fails), while the write
stream sends messages to the client.

Example implementation (simplified):
```python
@contextlib.asynccontextmanager
async def my_transport_implementation():
    # Set up connection
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
    
    # Start background tasks to handle actual I/O
    async with anyio.create_task_group() as tg:
        tg.start_soon(reader_task, read_stream_writer)
        tg.start_soon(writer_task, write_stream_reader)
        
        # Yield the streams to the caller
        yield (read_stream, write_stream)
```
"""
# GetSessionIdCallback was added for HTTP Streaming but was not applied to the MessageStream type
# https://github.com/modelcontextprotocol/python-sdk/blob/ed25167fa5d715733437996682e20c24470e8177/src/mcp/client/streamable_http.py#L418
_MessageStreamWithGetSessionIdCallback = tuple[
    MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage], GetSessionIdCallback
]
MCPTransport = AbstractAsyncContextManager[MessageStream | _MessageStreamWithGetSessionIdCallback]


class MCPToolResult(ToolResult):
    """Result of an MCP tool execution.

    Extends the base ToolResult with MCP-specific structured content support.
    The structuredContent field contains optional JSON data returned by MCP tools
    that provides structured results beyond the standard text/image/document content.

    Attributes:
        structuredContent: Optional JSON object containing structured data returned
            by the MCP tool. This allows MCP tools to return complex data structures
            that can be processed programmatically by agents or other tools.
    """

    structuredContent: NotRequired[Dict[str, Any]]
