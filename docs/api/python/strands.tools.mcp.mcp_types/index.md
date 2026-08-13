Type definitions for MCP integration.

## MCPClientCredentials

```python
class MCPClientCredentials(TypedDict)
```

Defined in: [src/strands/tools/mcp/mcp\_types.py:15](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_types.py#L15)

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

Defined in: [src/strands/tools/mcp/mcp\_types.py:72](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_types.py#L72)

Result of an MCP tool execution.

Extends the base ToolResult with MCP-specific structured content support. The structuredContent field contains optional JSON data returned by MCP tools that provides structured results beyond the standard text/image/document content.

**Attributes**:

-   `structuredContent` - Optional JSON object containing structured data returned by the MCP tool. This allows MCP tools to return complex data structures that can be processed programmatically by agents or other tools.
-   `metadata` - Optional arbitrary metadata returned by the MCP tool. This field allows MCP servers to attach custom metadata to tool results (e.g., token usage, performance metrics, or business-specific tracking information).
-   `isError` - Whether the MCP tool reported an application-level error via `CallToolResult.isError`. `True` means the tool executed but its logic returned a failure. Absent when the tool succeeded or when the error was a protocol/client exception rather than a tool-reported failure, letting callers distinguish application errors from transport/protocol errors.
-   `cancelled` - `True` when the local per-call cancellation signal was observed. This confirms local cancellation, not that remote execution stopped.