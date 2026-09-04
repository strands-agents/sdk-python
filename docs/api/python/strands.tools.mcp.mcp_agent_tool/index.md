MCP Agent Tool module for adapting Model Context Protocol tools to the agent framework.

This module provides the MCPAgentTool class which serves as an adapter between MCP (Model Context Protocol) tools and the agent framework’s tool interface. It allows MCP tools to be seamlessly integrated and used within the agent ecosystem.

## MCPAgentTool

```python
class MCPAgentTool(AgentTool)
```

Defined in: [src/strands/tools/mcp/mcp\_agent\_tool.py:25](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_agent_tool.py#L25)

Adapter class that wraps an MCP tool and exposes it as an AgentTool.

This class bridges the gap between the MCP protocol’s tool representation and the agent framework’s tool interface, allowing MCP tools to be used seamlessly within the agent framework.

#### \_\_init\_\_

```python
def __init__(mcp_tool: MCPTool,
             mcp_client: "MCPClient",
             name_override: str | None = None,
             timeout: timedelta | None = None) -> None
```

Defined in: [src/strands/tools/mcp/mcp\_agent\_tool.py:33](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_agent_tool.py#L33)

Initialize a new MCPAgentTool instance.

**Arguments**:

-   `mcp_tool` - The MCP tool to adapt
-   `mcp_client` - The MCP server connection to use for tool invocation
-   `name_override` - Optional name to use for the agent tool (for disambiguation) If None, uses the original MCP tool name
-   `timeout` - Optional timeout duration for tool execution. On the mcp 2.x line, the timeout bounds each request round of a multi round-trip tool call rather than the call as a whole.

#### tool\_name

```python
@property
def tool_name() -> str
```

Defined in: [src/strands/tools/mcp/mcp\_agent\_tool.py:58](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_agent_tool.py#L58)

Get the name of the tool.

**Returns**:

-   `str` - The agent-facing name of the tool (may be disambiguated)

#### tool\_spec

```python
@property
def tool_spec() -> ToolSpec
```

Defined in: [src/strands/tools/mcp/mcp\_agent\_tool.py:67](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_agent_tool.py#L67)

Get the specification of the tool.

This method converts the MCP tool specification to the agent framework’s ToolSpec format, including the input schema, description, optional output schema, and optional tool annotations.

**Returns**:

-   `ToolSpec` - The tool specification in the agent framework format

#### tool\_type

```python
@property
def tool_type() -> str
```

Defined in: [src/strands/tools/mcp/mcp\_agent\_tool.py:102](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_agent_tool.py#L102)

Get the type of the tool.

**Returns**:

-   `str` - The type of the tool, always “python” for MCP tools

#### stream

```python
@override
async def stream(tool_use: ToolUse, invocation_state: dict[str, Any],
                 **kwargs: Any) -> ToolGenerator
```

Defined in: [src/strands/tools/mcp/mcp\_agent\_tool.py:111](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_agent_tool.py#L111)

Stream the MCP tool.

This method delegates the tool stream to the MCP server connection, passing the tool use ID, tool name, and input arguments.

**Arguments**:

-   `tool_use` - The tool use request containing tool ID and parameters.
-   `invocation_state` - Context for the tool invocation, including agent state.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Tool events with the last being the tool result.