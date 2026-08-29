OpenTelemetry instrumentation for Model Context Protocol (MCP) tracing.

Enables distributed tracing across MCP client-server boundaries. The client side is handled with `inject_trace_context`, which `MCPClient` uses to merge the current OpenTelemetry context into the `_meta` field of outgoing tool calls through the public `meta` parameter of the official `mcp` package.

The server side covers MCP servers hosted in the same process. It extracts client-injected context from incoming messages so server-side spans join the client’s trace. This requires patching private internals of the `mcp` package and those internals are only stable within the 1.x line, so the patches are gated to `mcp` 1.x. The `mcp` 2.x line propagates trace context natively.

Based on: [https://github.com/traceloop/openllmetry/tree/main/packages/opentelemetry-instrumentation-mcp](https://github.com/traceloop/openllmetry/tree/main/packages/opentelemetry-instrumentation-mcp) Related issue: [https://github.com/modelcontextprotocol/modelcontextprotocol/issues/246](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/246)

#### inject\_trace\_context

```python
def inject_trace_context(meta: dict[str, Any] | None) -> dict[str, Any] | None
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:41](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L41)

Merge the current OpenTelemetry context into MCP request metadata.

Injects the active trace context (for example `traceparent` and `tracestate`) into a copy of `meta` so it can be sent as the `_meta` field of an MCP request. This enables server-side context extraction and trace continuation across the client-server boundary.

Telemetry must never fail the tool call itself, so propagator errors are logged and swallowed; the caller’s metadata is still returned.

**Arguments**:

-   `meta` - Existing request metadata, or None. The input is not mutated.

**Returns**:

A new dict containing the caller’s entries plus the injected trace context, or None when there is no metadata to send.

## ItemWithContext

```python
@dataclass(slots=True, frozen=True)
class ItemWithContext()
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:68](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L68)

Wrapper for items that need to carry OpenTelemetry context.

Used to preserve tracing context across async boundaries in MCP sessions, ensuring that distributed traces remain connected even when messages are processed asynchronously.

**Attributes**:

-   `item` - The original item being wrapped
-   `ctx` - The OpenTelemetry context associated with the item

#### mcp\_instrumentation

```python
def mcp_instrumentation() -> None
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:84](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L84)

Apply OpenTelemetry instrumentation patches for in-process MCP servers.

Instruments two areas of server-side MCP communication:

1.  Transport-level: Extracts context from incoming messages
2.  Session-level: Preserves context across async message processing boundaries

Together the patches let an MCP server hosted in the same process join the trace of the client that injected context into the request’s `_meta` field. Client-side injection does not require patching; `MCPClient` injects context via `inject_trace_context`.

The patches wrap private internals of the `mcp` package that are only stable within the 1.x line, so they are applied only when `mcp` 1.x is installed. The `mcp` 2.x line propagates trace context natively.

This function is idempotent - multiple calls will not accumulate wrappers.

## TransportContextExtractingReader

```python
class TransportContextExtractingReader(ObjectProxy)
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:180](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L180)

A proxy reader that extracts OpenTelemetry context from MCP messages.

Wraps an async message stream reader to automatically extract and activate OpenTelemetry context from the \_meta field of incoming MCP requests. This enables server-side trace continuation from client-injected context.

The reader handles both SessionMessage and JSONRPCMessage formats, and supports both dict and Pydantic model parameter structures.

#### \_\_init\_\_

```python
def __init__(wrapped: Any) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:191](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L191)

Initialize the context-extracting reader.

**Arguments**:

-   `wrapped` - The original async stream reader to wrap

#### \_\_aenter\_\_

```python
async def __aenter__() -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:199](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L199)

Enter the async context manager by delegating to the wrapped object.

#### \_\_aexit\_\_

```python
async def __aexit__(exc_type: Any, exc_value: Any, traceback: Any) -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:203](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L203)

Exit the async context manager by delegating to the wrapped object.

#### \_\_aiter\_\_

```python
async def __aiter__() -> AsyncGenerator[Any, None]
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:207](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L207)

Iterate over messages, extracting and activating context as needed.

For each incoming message, checks if it contains tracing context in the \_meta field. If found, extracts and activates the context for the duration of message processing, then properly detaches it.

**Yields**:

Messages from the wrapped stream, processed under the appropriate OpenTelemetry context

## SessionContextSavingWriter

```python
class SessionContextSavingWriter(ObjectProxy)
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:249](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L249)

A proxy writer that preserves OpenTelemetry context with outgoing items.

Wraps an async message stream writer to capture the current OpenTelemetry context and associate it with outgoing items. This enables context preservation across async boundaries in MCP session processing.

#### \_\_init\_\_

```python
def __init__(wrapped: Any) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:257](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L257)

Initialize the context-saving writer.

**Arguments**:

-   `wrapped` - The original async stream writer to wrap

#### \_\_aenter\_\_

```python
async def __aenter__() -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:265](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L265)

Enter the async context manager by delegating to the wrapped object.

#### \_\_aexit\_\_

```python
async def __aexit__(exc_type: Any, exc_value: Any, traceback: Any) -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:269](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L269)

Exit the async context manager by delegating to the wrapped object.

#### send

```python
async def send(item: Any) -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:273](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L273)

Send an item while preserving the current OpenTelemetry context.

Captures the current context and wraps the item with it, enabling the receiving side to restore the appropriate tracing context.

**Arguments**:

-   `item` - The item to send through the stream

**Returns**:

Result of sending the wrapped item

## SessionContextAttachingReader

```python
class SessionContextAttachingReader(ObjectProxy)
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:289](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L289)

A proxy reader that restores OpenTelemetry context from wrapped items.

Wraps an async message stream reader to detect ItemWithContext instances and restore their associated OpenTelemetry context during processing. This completes the context preservation cycle started by SessionContextSavingWriter.

#### \_\_init\_\_

```python
def __init__(wrapped: Any) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:297](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L297)

Initialize the context-attaching reader.

**Arguments**:

-   `wrapped` - The original async stream reader to wrap

#### \_\_aenter\_\_

```python
async def __aenter__() -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:305](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L305)

Enter the async context manager by delegating to the wrapped object.

#### \_\_aexit\_\_

```python
async def __aexit__(exc_type: Any, exc_value: Any, traceback: Any) -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:309](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L309)

Exit the async context manager by delegating to the wrapped object.

#### \_\_aiter\_\_

```python
async def __aiter__() -> AsyncGenerator[Any, None]
```

Defined in: [src/strands/tools/mcp/mcp\_instrumentation.py:313](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_instrumentation.py#L313)

Iterate over items, restoring context for ItemWithContext instances.

For items wrapped with context, temporarily activates the associated OpenTelemetry context during processing, then properly detaches it. Regular items are yielded without context modification.

**Yields**:

Unwrapped items processed under their associated OpenTelemetry context