# Strands Agents SDK - AI Agent Developer Guide

**Generated:** Tuesday, September 23, 2025 at 6:02 PM EDT  
**Repository:** https://github.com/strands-agents/sdk-python  
**Purpose:** Comprehensive guide for AI agents to understand and work with the Strands Agents Python SDK

---

## Executive Summary

Strands Agents is a production-ready Python SDK for building AI agents using a model-driven approach. The framework provides a lightweight yet powerful agent loop that scales from simple conversational assistants to complex autonomous workflows with multi-agent coordination.

**Key Metrics:**
- 84 Python source files (~17,131 lines of code)
- 9+ model provider integrations
- Native MCP (Model Context Protocol) support
- Python 3.10+ compatibility
- Apache 2.0 licensed

---

## Core Architecture Patterns

### 1. Agent-Centric Design
The `Agent` class is the primary interface with two interaction patterns:
- **Natural Language**: `agent("What is the weather?")`
- **Direct Tool Access**: `agent.tool.weather_tool(location="NYC")`

### 2. Model Abstraction Layer
All model providers implement the `Model` abstract base class with standardized methods:
- `stream()`: Streaming conversation interface
- `structured_output()`: Type-safe structured responses
- `update_config()`: Runtime configuration updates

### 3. Tool System Architecture
Tools are Python functions decorated with `@tool` that automatically:
- Extract metadata from docstrings and type hints
- Generate JSON schemas for validation
- Handle both direct calls and agent-mediated execution
- Support async/await patterns

---

## Package Structure Deep Dive

```
src/strands/
├── agent/                          # Core agent orchestration
│   ├── agent.py                   # Main Agent class (34KB)
│   ├── conversation_manager/      # Context and history management
│   ├── state.py                   # Agent state tracking
│   └── agent_result.py           # Response formatting
├── models/                         # Model provider ecosystem
│   ├── bedrock.py                 # AWS Bedrock (primary, 32KB)
│   ├── anthropic.py               # Claude models (17KB)
│   ├── openai.py                  # GPT models (16KB)
│   ├── litellm.py                 # Universal proxy (9KB)
│   ├── ollama.py                  # Local models (14KB)
│   ├── writer.py                  # Writer AI (17KB)
│   ├── llamaapi.py                # Llama access (16KB)
│   ├── mistral.py                 # Mistral AI (21KB)
│   ├── sagemaker.py               # AWS SageMaker (26KB)
│   └── model.py                   # Abstract base class
├── tools/                          # Tool system and execution
│   ├── decorator.py               # @tool decorator (25KB)
│   ├── registry.py                # Tool management (25KB)
│   ├── loader.py                  # Dynamic loading
│   ├── watcher.py                 # Hot reloading
│   ├── mcp/                       # Model Context Protocol
│   │   ├── mcp_client.py         # MCP client implementation
│   │   ├── mcp_agent_tool.py     # MCP tool wrapper
│   │   └── mcp_types.py          # Type definitions
│   └── executors/                 # Tool execution strategies
│       ├── concurrent.py         # Parallel execution
│       └── sequential.py         # Sequential execution
├── multiagent/                     # Multi-agent coordination
│   ├── swarm.py                   # Agent swarm patterns (27KB)
│   ├── graph.py                   # Graph-based workflows (30KB)
│   ├── base.py                    # Common interfaces
│   └── a2a/                       # Agent-to-Agent protocol
├── session/                        # Persistence layer
│   ├── file_session_manager.py   # File-based storage
│   ├── s3_session_manager.py     # AWS S3 storage
│   └── repository_session_manager.py # Repository pattern
├── types/                          # Type system
│   ├── content.py                 # Message and content types
│   ├── tools.py                   # Tool specifications
│   ├── streaming.py               # Stream event types
│   ├── exceptions.py              # Error handling
│   └── _events.py                 # Internal event system
├── telemetry/                      # Observability
│   ├── tracer.py                  # OpenTelemetry integration
│   └── metrics.py                 # Performance metrics
├── hooks/                          # Event system
│   ├── hook_registry.py           # Hook management
│   └── callback_handler.py        # Event callbacks
└── experimental/                   # Experimental features
    └── hooks/                     # Advanced hook patterns
```

---

## Critical Implementation Details

### Agent Class Interface

```python
class Agent:
    """Core Agent interface with dual interaction patterns."""
    
    def __init__(
        self,
        model: Optional[Model] = None,           # Model provider
        tools: Optional[list] = None,            # Available tools
        system_prompt: Optional[str] = None,     # System context
        load_tools_from_directory: bool = False, # Hot reloading
        tool_executor: Optional[ToolExecutor] = None, # Execution strategy
        session_manager: Optional[SessionManager] = None, # Persistence
        hooks: Optional[list[HookProvider]] = None, # Event hooks
        **kwargs  # See src/strands/agent/agent.py for additional parameters
    ):
        # Agent initialization with comprehensive configuration
    
    def __call__(self, input: str) -> AgentResult:
        """Natural language interface."""
        
    @property
    def tool(self) -> ToolCaller:
        """Direct tool access interface."""
        
    async def stream(self, input: str) -> AsyncGenerator[StreamEvent, None]:
        """Streaming response interface."""
```

### Tool Decorator System

```python
@tool
def example_tool(param1: str, param2: int = 42) -> dict:
    """Tool description for the LLM.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
        
    Returns:
        Structured result dictionary
    """
    return {
        "status": "success",
        "content": [{"text": f"Processed {param1} with {param2}"}]
    }
```

**Key Features:**
- Automatic schema generation from type hints
- Docstring parsing for descriptions
- Support for optional parameters with defaults
- Async function support
- Error handling and validation

### Model Provider Pattern

All model providers implement this interface:

```python
class Model(abc.ABC):
    @abc.abstractmethod
    def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream conversation with the model."""
        
    @abc.abstractmethod
    def structured_output(
        self, 
        output_model: Type[T], 
        prompt: Messages, 
        system_prompt: Optional[str] = None, 
        **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model."""
```

---

## Usage Patterns for AI Agents

### 1. Basic Agent Creation

```python
from strands import Agent
from strands_tools import calculator

# Simple agent with pre-built tools
agent = Agent(tools=[calculator])
result = agent("What is the square root of 1764?")
```

### 2. Custom Tool Development

```python
from strands import Agent, tool

@tool
def web_search(query: str, max_results: int = 10) -> dict:
    """Search the web for information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    # Implementation here
    return {"results": [...], "count": max_results}

agent = Agent(tools=[web_search])
```

### 3. Model Provider Configuration

```python
from strands import Agent
from strands.models import BedrockModel, OpenAIModel

# AWS Bedrock configuration
bedrock_model = BedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
    temperature=0.3,
    max_tokens=4096,
    streaming=True
)

# OpenAI configuration
openai_model = OpenAIModel(
    model_id="gpt-4",
    temperature=0.7,
    api_key="your-api-key"
)

agent = Agent(model=bedrock_model, tools=[...])
```

### 4. MCP Integration

```python
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

# Connect to MCP server
mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx", 
            args=["awslabs.aws-documentation-mcp-server@latest"]
        )
    )
)

with mcp_client:
    agent = Agent(tools=mcp_client.list_tools_sync())
    result = agent("Tell me about Amazon Bedrock")
```

### 5. Multi-Agent Systems

```python
from strands.multiagent import Swarm, GraphBuilder

# Swarm pattern
swarm = Swarm([researcher_agent, calculator_agent, writer_agent])
result = swarm.run("Research AI market size and calculate growth projections")

# Graph pattern
graph = (GraphBuilder()
    .add_agent("researcher", researcher_agent)
    .add_agent("calculator", calculator_agent)
    .add_edge("researcher", "calculator")
    .build())
```

### 6. Session Management

```python
from strands.session import FileSessionManager, S3SessionManager

# File-based persistence
file_session = FileSessionManager(base_path="./sessions")

# S3-based persistence
s3_session = S3SessionManager(
    bucket_name="my-agent-sessions",
    region="us-west-2"
)

agent = Agent(session_manager=file_session)
```

---

## Advanced Features

### Hot Reloading Development

```python
# Enable automatic tool reloading from ./tools/ directory
agent = Agent(load_tools_from_directory=True)

# Tools in ./tools/ directory are automatically loaded and reloaded
# when files change during development
```

### Structured Output

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_topics: list[str]

# Get type-safe structured output
async for event in agent.model.structured_output(
    output_model=AnalysisResult,
    prompt=[{"role": "user", "content": "Analyze this text..."}]
):
    if isinstance(event.get("output"), AnalysisResult):
        result = event["output"]
        print(f"Sentiment: {result.sentiment}")
```

### Event Hooks System

```python
from strands.hooks import HookProvider

class CustomHookProvider(HookProvider):
    def on_before_model_invocation(self, event: BeforeInvocationEvent):
        print(f"About to call model with: {event.messages}")
        
    def on_after_tool_invocation(self, event: AfterToolInvocationEvent):
        print(f"Tool {event.tool_name} returned: {event.result}")

agent = Agent(hooks=[CustomHookProvider()])
```

### Concurrent Tool Execution

```python
from strands.tools.executors import ConcurrentToolExecutor

# Execute multiple tools in parallel
concurrent_executor = ConcurrentToolExecutor()
agent = Agent(
    tools=[tool1, tool2, tool3],
    tool_executor=concurrent_executor
)
```

---

## Error Handling and Best Practices

*This section is reserved for future team guidance.*

---

## Testing and Development

### Integration Test Patterns

The codebase includes comprehensive integration tests demonstrating real usage:

- **`test_function_tools.py`**: Basic tool creation and usage
- **`test_multiagent_swarm.py`**: Multi-agent coordination
- **`test_mcp_client.py`**: MCP server integration
- **`test_stream_agent.py`**: Streaming responses
- **`test_session.py`**: Session persistence

### Development Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev,all]"

# Run tests
hatch test                    # Unit tests
hatch test tests_integ       # Integration tests

# Code quality
hatch fmt --formatter        # Format code
hatch fmt --linter          # Lint code
```

---

## Configuration Reference

### Model Provider Configurations

**AWS Bedrock:**
```python
BedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
    temperature=0.3,
    max_tokens=4096,
    top_p=0.9,
    streaming=True,
    guardrails_config={...}
)
```

**OpenAI:**
```python
OpenAIModel(
    model_id="gpt-4",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=2048,
    organization="org-..."
)
```

**Anthropic:**
```python
AnthropicModel(
    model_id="claude-3-sonnet-20240229",
    api_key="sk-ant-...",
    temperature=0.5,
    max_tokens=4096
)
```

### Tool Executor Options

```python
# Sequential execution (default)
from strands.tools.executors import SequentialToolExecutor
agent = Agent(tool_executor=SequentialToolExecutor())

# Concurrent execution
from strands.tools.executors import ConcurrentToolExecutor
agent = Agent(tool_executor=ConcurrentToolExecutor(max_workers=4))
```

---

---

## Documentation & Resources

- **Documentation**: https://strandsagents.com/
- **Comprehensive Documentation Index**: See https://strandsagents.com/latest/llms.txt for complete documentation links
- **GitHub Repository**: https://github.com/strands-agents/sdk-python
- **PyPI Package**: https://pypi.org/project/strands-agents/
- **Sample Projects**: https://github.com/strands-agents/samples
- **Tools Collection**: https://github.com/strands-agents/tools

---

## Summary for AI Agents

When working with the Strands Agents SDK:

1. **Start with the `Agent` class** - it's the primary interface
2. **Use `@tool` decorator** for creating custom tools
3. **Choose appropriate model providers** based on requirements
4. **Leverage MCP integration** for accessing external tools
5. **Consider multi-agent patterns** for complex workflows
6. **Implement proper error handling** for production use
7. **Use streaming for better user experience**
8. **Configure session management** for conversation persistence

The SDK is designed to be both simple for basic use cases and powerful for complex agent systems. The extensive test suite and documentation provide excellent examples of real-world usage patterns.
