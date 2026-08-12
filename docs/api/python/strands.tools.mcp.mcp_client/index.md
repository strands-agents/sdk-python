Model Context Protocol (MCP) server connection management module.

This module provides the MCPClient class which handles connections to MCP servers. It manages the lifecycle of MCP connections, including initialization, tool discovery, tool invocation, and proper cleanup of resources. The connection runs in a background thread to avoid blocking the main application thread while maintaining communication with the MCP service.

## ToolFilters

```python
class ToolFilters(TypedDict)
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:84](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L84)

Filters for controlling which MCP tools are loaded and available.

Tools are filtered in this order:

1.  If ‘allowed’ is specified, only tools matching these patterns are included
2.  Tools matching ‘rejected’ patterns are then excluded

## MCPServerConfig

```python
class MCPServerConfig(TypedDict)
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:103](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L103)

Schema for a single MCP server entry in a load\_servers config.

Provide either ‘command’ (stdio) or ‘url’ (streamable-http/sse), not both. When ‘transport’ is omitted it is auto-detected from the fields present. String values support ’${VAR}’ / ’${env:VAR}’ interpolation, and ’~’ in ‘command’ and ‘cwd’ is expanded to the home directory.

‘disabled’ skips the server entirely. ‘continue\_on\_error’ keeps the rest of the servers usable when this one fails: a config-resolution failure (e.g. a missing env var) skips it during load\_servers instead of raising, and a connection failure yields no tools instead of raising when the agent loads them.

## MCPClient

```python
class MCPClient(ToolProvider)
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:155](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L155)

Represents a connection to a Model Context Protocol (MCP) server.

This class implements a context manager pattern for efficient connection management, allowing reuse of the same connection for multiple tool calls to reduce latency. It handles the creation, initialization, and cleanup of MCP connections.

The connection runs in a background thread to avoid blocking the main application thread while maintaining communication with the MCP service. When structured content is available from MCP tools, it will be returned as the last item in the content array of the ToolResult.

#### load\_servers

```python
@classmethod
def load_servers(cls, config: "str | dict[str, Any]") -> "list[MCPClient]"
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:168](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L168)

Create MCPClient instances from an `mcpServers` JSON config (file path or mapping).

Returns one client per enabled server. Accepts either a flat mapping of server name to config, or that mapping nested under an `mcpServers` key. Servers marked `"disabled": true` are skipped. When a server sets `"continue_on_error": true`, a failure resolving its config (e.g. a missing env var) skips that server instead of raising.

Transport is auto-detected from the fields present: `command` selects stdio and `url` selects streamable-http. Set `transport` explicitly (`"stdio"`, `"sse"`, or `"streamable-http"`) to override. String values support `$\{VAR}` / `$\{env:VAR}` interpolation against the process environment, and `~` in `command` and `cwd` is expanded to the user’s home directory.

**Arguments**:

-   `config` - A file path (with optional `file://` prefix) to a JSON config, or a dictionary mapping server names to configs (optionally under an `mcpServers` key).

**Returns**:

One MCPClient per enabled server, ready to pass to `Agent(tools=...)`.

**Raises**:

-   `FileNotFoundError` - If the config file does not exist.
-   `json.JSONDecodeError` - If the config file contains invalid JSON.
-   `ValueError` - If the overall config shape is invalid or a server entry is not a mapping. These are malformed-config errors and always raise, regardless of `continue_on_error`. A failure building an individual server (e.g. a missing env var) also raises unless that server set `continue_on_error`, in which case it is skipped.

#### \_\_init\_\_

```python
def __init__(transport_callable: Callable[[], MCPTransport],
             *,
             startup_timeout: int = 30,
             tool_filters: ToolFilters | None = None,
             prefix: str | None = None,
             application_name: str | None = None,
             application_version: str | None = None,
             continue_on_error: bool = False,
             elicitation_callback: ElicitationFnT | None = None,
             progress_callback: ProgressFnT | None = None,
             tasks_config: TasksConfig | None = None) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:223](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L223)

Initialize a new MCP Server connection.

**Arguments**:

-   `transport_callable` - A callable that returns an MCPTransport (read\_stream, write\_stream) tuple.
-   `startup_timeout` - Timeout after which MCP server initialization should be cancelled. Defaults to 30.
-   `tool_filters` - Optional filters to apply to tools.
-   `prefix` - Optional prefix for tool names.
-   `application_name` - Optional name to identify this agent via clientInfo.name. If provided, the MCP server will see this name during the initialize handshake. Defaults to None (uses the MCP SDK default “mcp”).
-   `application_version` - Optional version string to report alongside application\_name. Defaults to None (uses the Strands SDK version).
-   `continue_on_error` - When True, a connection failure during `load_tools` is logged and yields no tools instead of raising, so one unavailable server does not prevent an agent from using the others. Only the connection (`start()`) is swallowed; an error while listing tools after a successful connect still propagates. Defaults to False.
-   `elicitation_callback` - Optional callback function to handle elicitation requests from the MCP server.
-   `progress_callback` - Optional callback to receive progress notifications during tool execution. Called with `(progress, total, message)` as the server reports progress. The `total` and `message` parameters may be `None` if the server does not provide them.
-   `tasks_config` - Configuration for MCP task-augmented execution for long-running tools. If provided (not None), enables task-augmented execution for tools that support it. See TasksConfig for details. This feature is experimental and subject to change.

#### \_\_enter\_\_

```python
def __enter__() -> "MCPClient"
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:305](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L305)

Context manager entry point which initializes the MCP server connection.

TODO: Refactor to lazy initialization pattern following idiomatic Python. Heavy work in **enter** is non-idiomatic - should move connection logic to first method call instead.

#### \_\_exit\_\_

```python
def __exit__(exc_type: type[BaseException] | None,
             exc_val: BaseException | None,
             exc_tb: TracebackType | None) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:313](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L313)

Context manager exit point that cleans up resources.

#### start

```python
def start() -> "MCPClient"
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:322](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L322)

Starts the background thread and waits for initialization.

This method starts the background thread that manages the MCP connection and blocks until the connection is ready or times out.

**Returns**:

-   `self` - The MCPClient instance

**Raises**:

-   `Exception` - If the MCP connection fails to initialize within the timeout period

#### continue\_on\_error

```python
@property
def continue_on_error() -> bool
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:365](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L365)

Whether a connection failure is swallowed instead of raised (see `__init__`).

#### connection\_failed

```python
@property
def connection_failed() -> bool
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:370](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L370)

Whether a `continue_on_error` connection attempt has failed and not yet been reset.

Sticky within a connection lifecycle: stays True until teardown (removing the last consumer, or `stop()`) resets the client. Always False when `continue_on_error` is not set, since a failure raises instead.

#### load\_tools

```python
async def load_tools(**kwargs: Any) -> Sequence[AgentTool]
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:380](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L380)

Load and return tools from the MCP server.

This method implements the ToolProvider interface by loading tools from the MCP server and caching them for reuse.

**Arguments**:

-   `**kwargs` - Additional arguments for future compatibility.

**Returns**:

List of AgentTool instances from the MCP server. Empty when the connection fails and `continue_on_error` is set; the failure is sticky within a connection lifecycle and is not retried on subsequent calls. Teardown (removing the last consumer, or `stop()`) resets the client so a later consumer reconnects.

#### add\_consumer

```python
def add_consumer(consumer_id: Any, **kwargs: Any) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:453](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L453)

Add a consumer to this tool provider.

Synchronous to prevent GC deadlocks when called from Agent finalizers.

#### remove\_consumer

```python
def remove_consumer(consumer_id: Any, **kwargs: Any) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:461](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L461)

Remove a consumer from this tool provider.

This method is idempotent - calling it multiple times with the same ID has no additional effect after the first call.

Synchronous to prevent GC deadlocks when called from Agent finalizers. Uses existing synchronous stop() method for safe cleanup.

#### stop

```python
def stop(exc_type: type[BaseException] | None, exc_val: BaseException | None,
         exc_tb: TracebackType | None) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:488](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L488)

Signals the background thread to stop and waits for it to complete, ensuring proper cleanup of all resources.

This method is defensive and can handle partial initialization states that may occur if start() fails partway through initialization.

Resources to cleanup:

-   \_background\_thread: Thread running the async event loop
-   \_background\_thread\_session: MCP ClientSession (auto-closed by context manager)
-   \_background\_thread\_event\_loop: AsyncIO event loop in background thread
-   \_close\_future: AsyncIO future to signal thread shutdown
-   \_close\_exception: Exception that caused the background thread shutdown; None if a normal shutdown occurred.
-   \_init\_future: Future for initialization synchronization

Cleanup order:

1.  Signal close future to background thread (if session initialized)
2.  Wait for background thread to complete
3.  Reset all state for reuse

**Arguments**:

-   `exc_type` - Exception type if an exception was raised in the context
-   `exc_val` - Exception value if an exception was raised in the context
-   `exc_tb` - Exception traceback if an exception was raised in the context

#### list\_tools\_sync

```python
def list_tools_sync(
        pagination_token: str | None = None,
        prefix: str | None = None,
        tool_filters: ToolFilters | None = None
) -> PaginatedList[MCPAgentTool]
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:568](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L568)

Synchronously retrieves the list of available tools from the MCP server.

This method calls the asynchronous list\_tools method on the MCP session and adapts the returned tools to the AgentTool interface.

**Arguments**:

-   `pagination_token` - Optional token for pagination
-   `prefix` - Optional prefix to apply to tool names. If None, uses constructor default. If explicitly provided (including empty string), overrides constructor default.
-   `tool_filters` - Optional filters to apply to tools. If None, uses constructor default. If explicitly provided (including empty dict), overrides constructor default.

**Returns**:

-   `List[AgentTool]` - A list of available tools adapted to the AgentTool interface

#### list\_prompts\_sync

```python
def list_prompts_sync(
        pagination_token: str | None = None) -> ListPromptsResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:626](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L626)

Synchronously retrieves the list of available prompts from the MCP server.

This method calls the asynchronous list\_prompts method on the MCP session and returns the raw ListPromptsResult with pagination support.

**Arguments**:

-   `pagination_token` - Optional token for pagination

**Returns**:

-   `ListPromptsResult` - The raw MCP response containing prompts and pagination info

#### get\_prompt\_sync

```python
def get_prompt_sync(prompt_id: str, args: dict[str, Any]) -> GetPromptResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:652](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L652)

Synchronously retrieves a prompt from the MCP server.

**Arguments**:

-   `prompt_id` - The ID of the prompt to retrieve
-   `args` - Optional arguments to pass to the prompt

**Returns**:

-   `GetPromptResult` - The prompt response from the MCP server

#### list\_resources\_sync

```python
def list_resources_sync(
        pagination_token: str | None = None) -> ListResourcesResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:674](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L674)

Synchronously retrieves the list of available resources from the MCP server.

This method calls the asynchronous list\_resources method on the MCP session and returns the raw ListResourcesResult with pagination support.

**Arguments**:

-   `pagination_token` - Optional token for pagination

**Returns**:

-   `ListResourcesResult` - The raw MCP response containing resources and pagination info

#### read\_resource\_sync

```python
def read_resource_sync(uri: AnyUrl | str) -> ReadResourceResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:698](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L698)

Synchronously reads a resource from the MCP server.

**Arguments**:

-   `uri` - The URI of the resource to read

**Returns**:

-   `ReadResourceResult` - The resource content from the MCP server

#### list\_resource\_templates\_sync

```python
def list_resource_templates_sync(
        pagination_token: str | None = None) -> ListResourceTemplatesResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:721](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L721)

Synchronously retrieves the list of available resource templates from the MCP server.

Resource templates define URI patterns that can be used to access resources dynamically.

**Arguments**:

-   `pagination_token` - Optional token for pagination

**Returns**:

-   `ListResourceTemplatesResult` - The raw MCP response containing resource templates and pagination info

#### call\_tool\_sync

```python
def call_tool_sync(
        tool_use_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        meta: dict[str, Any] | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        cancel_signal: threading.Event | None = None) -> MCPToolResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:831](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L831)

Synchronously calls a tool on the MCP server.

This method automatically uses task-augmented execution when appropriate, based on server capabilities and tool-level taskSupport settings.

**Arguments**:

-   `tool_use_id` - Unique identifier for this tool use
-   `name` - Name of the tool to call
-   `arguments` - Optional arguments to pass to the tool
-   `read_timeout_seconds` - Optional timeout for the tool call
-   `meta` - Optional metadata to pass to the tool call per MCP spec (\_meta)
-   `progress_callback` - Optional callback to receive progress notifications for this call. Overrides the instance-level callback set at construction time.
-   `cancel_signal` - Optional caller-owned, thread-safe event for this call. A pre-set event cancels before the MCP request starts. If set while the request is in flight, cancellation wins over a concurrently arriving result. The returned error result has `cancelled=True`. Remote cancellation is best-effort and bounded; the shared MCP session remains reusable. Clear the event before reusing it for another call.

**Returns**:

-   `MCPToolResult` - The tool result. Locally observed cancellation returns an error result with `cancelled=True` rather than raising. Cancelling the outer asyncio task is separate and still raises `asyncio.CancelledError` for `call_tool_async`.

#### call\_tool\_async

```python
async def call_tool_async(
        tool_use_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        meta: dict[str, Any] | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        cancel_signal: threading.Event | None = None) -> MCPToolResult
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:888](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L888)

Asynchronously calls a tool on the MCP server.

This method automatically uses task-augmented execution when appropriate, based on server capabilities and tool-level taskSupport settings.

**Arguments**:

-   `tool_use_id` - Unique identifier for this tool use
-   `name` - Name of the tool to call
-   `arguments` - Optional arguments to pass to the tool
-   `read_timeout_seconds` - Optional timeout for the tool call
-   `meta` - Optional metadata to pass to the tool call per MCP spec (\_meta)
-   `progress_callback` - Optional callback to receive progress notifications for this call. Overrides the instance-level callback set at construction time.
-   `cancel_signal` - Optional caller-owned, thread-safe event for this call. A pre-set event cancels before the MCP request starts. If set while the request is in flight, cancellation wins over a concurrently arriving result. The returned error result has `cancelled=True`. Remote cancellation is best-effort and bounded; the shared MCP session remains reusable. Clear the event before reusing it for another call.

**Returns**:

-   `MCPToolResult` - The tool result. Locally observed cancellation returns an error result with `cancelled=True` rather than raising. Cancelling the outer asyncio task is separate and still raises `asyncio.CancelledError` for `call_tool_async`.

#### map\_mcp\_content\_to\_tool\_result\_content

```python
def map_mcp_content_to_tool_result_content(
    content: MCPTextContent | MCPImageContent | MCPEmbeddedResource | Any
) -> ToolResultContent | None
```

Defined in: [src/strands/tools/mcp/mcp\_client.py:1130](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_client.py#L1130)

Maps MCP content types to tool result content types.

This method converts MCP-specific content types to the generic ToolResultContent format used by the agent framework. Subclasses can override this to intercept or transform specific content blocks before they reach the model.

**Arguments**:

-   `content` - The MCP content to convert

**Returns**:

ToolResultContent or None: The converted content, or None if the content type is not supported