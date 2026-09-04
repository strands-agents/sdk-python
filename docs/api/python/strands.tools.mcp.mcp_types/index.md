Type definitions for MCP integration.

## ToolsChangedCallback

```python
class ToolsChangedCallback(Protocol)
```

Defined in: [src/strands/tools/mcp/mcp\_types.py:19](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_types.py#L19)

Called after the server announces a tool list change and the client refreshes it.

Implemented by a plain function as well — the `**kwargs` tail lets the calling convention grow new keyword arguments without breaking existing callbacks.

#### \_\_call\_\_

```python
def __call__(previous_names: list[str], refreshed_tools: list["MCPAgentTool"],
             **kwargs: Any) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_types.py:26](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_types.py#L26)

Handle a refresh, given the previous tool names and the refreshed tool instances.

#### ToolsChanged

A tools-changed handler: the `**kwargs`\-ready protocol or a plain two-argument callable.

## MCPClientCredentials

```python
class MCPClientCredentials(TypedDict)
```

Defined in: [src/strands/tools/mcp/mcp\_types.py:35](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_types.py#L35)

OAuth client credentials for machine-to-machine authentication.

Used with the `MCPClient` `auth` parameter, or the `auth` key of a server entry in a `load_servers` config, to authenticate against a streamable HTTP MCP server with the OAuth client\_credentials grant.

**Attributes**:

-   `client_id` - The OAuth client ID.
-   `client_secret` - The OAuth client secret.
-   `scopes` - OAuth scopes to request, joined with spaces. Advisory only: if the server advertises its own scopes (via the `WWW-Authenticate` header or its protected-resource / authorization-server metadata), the server’s list is used instead and this value is ignored.

## MCPToolResult

```python
class MCPToolResult(ToolResult)
```

Defined in: [src/strands/tools/mcp/mcp\_types.py:92](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_types.py#L92)

Result of an MCP tool execution.

Extends the base ToolResult with MCP-specific structured content support. The structuredContent field contains optional JSON data returned by MCP tools that provides structured results beyond the standard text/image/document content.

**Attributes**:

-   `structuredContent` - Optional JSON object containing structured data returned by the MCP tool. This allows MCP tools to return complex data structures that can be processed programmatically by agents or other tools.
-   `metadata` - Optional arbitrary metadata returned by the MCP tool. This field allows MCP servers to attach custom metadata to tool results (e.g., token usage, performance metrics, or business-specific tracking information).
-   `isError` - Whether the MCP tool reported an application-level error via `CallToolResult.isError`. `True` means the tool executed but its logic returned a failure. Absent when the tool succeeded or when the error was a protocol/client exception rather than a tool-reported failure, letting callers distinguish application errors from transport/protocol errors.
-   `cancelled` - `True` when the local per-call cancellation signal was observed. This confirms local cancellation, not that remote execution stopped.