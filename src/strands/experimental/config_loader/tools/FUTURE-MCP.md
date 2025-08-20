# MCP Tool Loading Implementation Plan

## Overview

This document outlines the implementation plan for integrating Model Context Protocol (MCP) server tool loading into the Strands Agents ConfigLoader system. MCP is an open protocol that standardizes how applications provide context to LLMs, enabling communication between the system and locally running MCP servers that provide additional tools and resources.

## Current Tool Loading Architecture

The existing `ToolConfigLoader` supports multiple tool loading mechanisms:

1. **String-based loading**: Load tools by identifier from modules or registries
2. **Module-based loading**: Support for tools that follow the TOOL_SPEC pattern
3. **Agent-as-Tool**: Configure complete agents as reusable tools
4. **Swarm-as-Tool**: Configure swarms as tools for complex operations
5. **Graph-as-Tool**: Configure graphs as tools for workflow operations

## MCP Integration Goals

### Primary Objectives
- **Seamless Integration**: MCP tools should work identically to existing tool types
- **Configuration Consistency**: Use same configuration patterns as other tool sources
- **Dynamic Discovery**: Support runtime discovery of available MCP tools
- **Error Resilience**: Graceful handling of MCP server connectivity issues
- **Performance**: Efficient tool loading and caching mechanisms

### Secondary Objectives
- **Hot Reloading**: Support for MCP server restarts without agent restart
- **Tool Versioning**: Handle MCP tool version changes gracefully
- **Security**: Validate MCP tool specifications and inputs
- **Monitoring**: Provide visibility into MCP server health and tool usage

## Implementation Architecture

### 1. MCP Client Integration

#### MCPClient Class
```python
class MCPClient:
    """Client for communicating with MCP servers."""
    
    def __init__(self, server_config: Dict[str, Any]):
        self.server_name = server_config["name"]
        self.connection_config = server_config["connection"]
        self.transport = self._create_transport()
        self.session = None
        
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        
    async def list_tools(self) -> List[MCPToolSpec]:
        """Get list of available tools from MCP server."""
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the MCP server."""
        
    def is_connected(self) -> bool:
        """Check if connection to MCP server is active."""
```

#### MCPToolSpec Class
```python
class MCPToolSpec:
    """Specification for an MCP tool."""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.server_name = None  # Set by MCPToolWrapper
        
    def to_tool_spec(self) -> ToolSpec:
        """Convert to Strands ToolSpec format."""
```

### 2. MCP Tool Wrapper

#### MCPToolWrapper Class
```python
class MCPToolWrapper(AgentTool):
    """Wrapper that adapts MCP tools to Strands AgentTool interface."""
    
    def __init__(self, mcp_client: MCPClient, tool_spec: MCPToolSpec):
        self._mcp_client = mcp_client
        self._mcp_tool_spec = tool_spec
        self._tool_name = f"mcp_{mcp_client.server_name}_{tool_spec.name}"
        self._tool_spec = tool_spec.to_tool_spec()
        
    @property
    def tool_name(self) -> str:
        """The unique name of the tool."""
        
    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification."""
        
    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "mcp"
        
    async def stream(self, tool_use: ToolUse, invocation_state: Dict[str, Any], **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Execute the MCP tool and stream results."""
```

### 3. MCP Server Registry

#### MCPServerRegistry Class
```python
class MCPServerRegistry:
    """Registry for managing MCP server connections and tool discovery."""
    
    def __init__(self):
        self._servers: Dict[str, MCPClient] = {}
        self._tool_cache: Dict[str, MCPToolWrapper] = {}
        self._connection_pool = MCPConnectionPool()
        
    async def register_server(self, server_config: Dict[str, Any]) -> None:
        """Register and connect to an MCP server."""
        
    async def unregister_server(self, server_name: str) -> None:
        """Unregister and disconnect from an MCP server."""
        
    async def discover_tools(self, server_name: Optional[str] = None) -> List[MCPToolWrapper]:
        """Discover available tools from registered servers."""
        
    async def get_tool(self, tool_identifier: str) -> Optional[MCPToolWrapper]:
        """Get a specific tool by identifier."""
        
    async def refresh_tools(self, server_name: Optional[str] = None) -> None:
        """Refresh tool cache from MCP servers."""
        
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered MCP servers."""
```

### 4. ToolConfigLoader Integration

#### Enhanced ToolConfigLoader
```python
class ToolConfigLoader:
    """Enhanced with MCP support."""
    
    def __init__(self, registry: Optional[ToolRegistry] = None, mcp_registry: Optional[MCPServerRegistry] = None):
        # Existing initialization
        self._mcp_registry = mcp_registry or MCPServerRegistry()
        
    async def configure_mcp_servers(self, mcp_config: List[Dict[str, Any]]) -> None:
        """Configure MCP servers from configuration."""
        
    def _determine_config_type(self, config: Dict[str, Any]) -> str:
        """Enhanced to detect MCP tool configurations."""
        if "mcp_server" in config or "mcp_tool" in config:
            return "mcp"
        # Existing logic
        
    async def _load_mcp_tool(self, tool_config: Dict[str, Any]) -> AgentTool:
        """Load a tool from MCP server."""
```

## Configuration Schema

### MCP Server Configuration
```yaml
mcp_servers:
  - name: "filesystem"
    connection:
      type: "stdio"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
    
  - name: "database"
    connection:
      type: "sse"
      url: "http://localhost:3001/sse"
    
  - name: "custom_tools"
    connection:
      type: "websocket"
      url: "ws://localhost:8080/mcp"
      headers:
        Authorization: "Bearer ${MCP_TOKEN}"
```

### MCP Tool Configuration
```yaml
# Method 1: Load specific tool from MCP server
tools:
  - name: "read_file"
    mcp_server: "filesystem"
    mcp_tool: "read_file"
    
# Method 2: Load all tools from MCP server
tools:
  - mcp_server: "filesystem"
    prefix: "fs_"  # Optional prefix for tool names
    
# Method 3: Load tool with custom configuration
tools:
  - name: "custom_file_reader"
    mcp_server: "filesystem"
    mcp_tool: "read_file"
    description: "Custom description for this tool usage"
    input_schema:
      # Override or extend the MCP tool's input schema
      properties:
        file_path:
          description: "Path to file (must be within allowed directories)"
```

### Agent Configuration with MCP Tools
```yaml
agent:
  model: "us.amazon.nova-pro-v1:0"
  system_prompt: "You are a helpful assistant with file system access."
  mcp_servers:
    - name: "filesystem"
      connection:
        type: "stdio"
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
  tools:
    - mcp_server: "filesystem"
    - name: "calculator"  # Regular tool
```

## Implementation Phases

### Phase 1: Core MCP Integration
**Scope**: Basic MCP client and tool wrapper implementation

**Components**:
- `MCPClient` class with stdio transport support
- `MCPToolWrapper` class implementing `AgentTool` interface
- `MCPServerRegistry` for server management
- Basic error handling and connection management

**Configuration Support**:
- MCP server registration in agent configurations
- Simple tool loading by server name
- Basic stdio transport for MCP servers

### Phase 2: Enhanced Transport Support
**Scope**: Support for multiple MCP transport types

**Components**:
- SSE (Server-Sent Events) transport implementation
- WebSocket transport implementation
- HTTP transport implementation
- Transport factory and configuration system

**Configuration Support**:
- Multiple transport types in server configuration
- Connection parameters and authentication
- Transport-specific error handling

### Phase 3: Advanced Tool Management
**Scope**: Dynamic tool discovery and management

**Components**:
- Tool caching and invalidation strategies
- Hot reloading of MCP server tools
- Tool versioning and compatibility checking
- Performance optimization for tool discovery

**Configuration Support**:
- Tool filtering and selection criteria
- Custom tool naming and prefixing
- Tool metadata and documentation integration

### Phase 4: Production Features
**Scope**: Production-ready features and monitoring

**Components**:
- Connection pooling and resource management
- Health monitoring and alerting
- Security validation and sandboxing
- Comprehensive logging and metrics

**Configuration Support**:
- Security policies for MCP tools
- Resource limits and timeouts
- Monitoring and alerting configuration

## Error Handling Strategy

### Connection Errors
- **Server Unavailable**: Graceful degradation, tool marked as unavailable
- **Connection Lost**: Automatic reconnection with exponential backoff
- **Authentication Failed**: Clear error messages and configuration guidance

### Tool Execution Errors
- **Tool Not Found**: Clear error with available tool suggestions
- **Invalid Arguments**: Schema validation with helpful error messages
- **Execution Timeout**: Configurable timeouts with proper cleanup

### Configuration Errors
- **Invalid Server Config**: Validation with specific error messages
- **Missing Dependencies**: Clear guidance on MCP server installation
- **Schema Conflicts**: Resolution strategies for conflicting tool schemas

## Security Considerations

### MCP Server Validation
- **Allowlisted Servers**: Only connect to explicitly configured servers
- **Transport Security**: Enforce secure connections where possible
- **Authentication**: Support for various authentication mechanisms

### Tool Execution Security
- **Input Validation**: Strict validation of tool inputs against schemas
- **Output Sanitization**: Sanitize tool outputs to prevent injection attacks
- **Resource Limits**: Enforce timeouts and resource usage limits

### Configuration Security
- **Credential Management**: Secure handling of MCP server credentials
- **Path Restrictions**: Validate and restrict file system access paths
- **Network Policies**: Control network access for MCP servers

## Testing Strategy

### Unit Tests
- MCP client connection and communication
- Tool wrapper functionality and error handling
- Server registry management and caching
- Configuration parsing and validation

### Integration Tests
- End-to-end tool loading and execution
- Multiple transport type testing
- Error scenario testing and recovery
- Performance testing with multiple servers

### Mock MCP Servers
- Test servers for different transport types
- Error simulation servers for testing resilience
- Performance testing servers with various response times

## Migration and Compatibility

### Backward Compatibility
- Existing tool loading mechanisms remain unchanged
- MCP tools integrate seamlessly with existing agent configurations
- No breaking changes to current ToolConfigLoader API

### Migration Path
- MCP servers can be added incrementally to existing configurations
- Tools can be migrated from modules to MCP servers gradually
- Configuration validation ensures smooth transitions

## Documentation Requirements

### User Documentation
- MCP server setup and configuration guide
- Tool loading patterns and best practices
- Troubleshooting guide for common MCP issues
- Security best practices for MCP integration

### Developer Documentation
- MCP client API documentation
- Tool wrapper development guide
- Transport implementation guide
- Testing and debugging procedures

## Future Enhancements

### Advanced Features
- **Tool Composition**: Combine multiple MCP tools into workflows
- **Dynamic Schema Generation**: Generate tool schemas from MCP introspection
- **Tool Marketplace**: Discovery and installation of MCP tool packages
- **Cross-Server Tool Dependencies**: Tools that depend on multiple MCP servers

### Performance Optimizations
- **Parallel Tool Discovery**: Concurrent discovery across multiple servers
- **Intelligent Caching**: Smart caching based on tool usage patterns
- **Connection Multiplexing**: Efficient connection reuse for multiple tools
- **Lazy Loading**: Load tools only when first used

### Monitoring and Observability
- **Tool Usage Analytics**: Track tool usage patterns and performance
- **Server Health Dashboards**: Real-time monitoring of MCP server health
- **Performance Metrics**: Detailed metrics for tool execution times
- **Error Tracking**: Comprehensive error tracking and alerting

## Implementation Notes

### Dependencies
- MCP client library (to be determined based on available options)
- Async/await support throughout the implementation
- JSON Schema validation for tool specifications
- Transport-specific libraries (websockets, sse, etc.)

### Configuration Validation
- JSON Schema validation for MCP server configurations
- Runtime validation of MCP tool specifications
- Compatibility checking between tool versions
- Security policy validation

### Performance Considerations
- Connection pooling to minimize connection overhead
- Tool caching to reduce discovery latency
- Async operations to prevent blocking
- Resource cleanup to prevent memory leaks

This implementation plan provides a comprehensive approach to integrating MCP tool loading into the Strands Agents ConfigLoader system while maintaining consistency with existing patterns and ensuring production-ready reliability and security.
