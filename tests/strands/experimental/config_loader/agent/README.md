# AgentConfigLoader Tests

This directory contains comprehensive unit tests for the `AgentConfigLoader` class in `src/strands/experimental/config_loader/agent_config_loader.py`.

## Test Coverage

The test suite includes 42 test cases covering:

### Core Functionality
- **Initialization**: Testing with and without custom ToolConfigLoader
- **Agent Loading**: Creating Agent instances from dictionary configurations
- **Agent Serialization**: Converting Agent instances back to dictionary configurations
- **Caching**: Verifying agent caching behavior with cache keys

### Configuration Loading
- **Model Loading**: Support for string model IDs and dictionary configurations
- **Tool Loading**: Integration with ToolConfigLoader for dynamic tool loading
- **Message Loading**: Loading initial conversation messages
- **State Loading**: Loading and managing agent state
- **Callback Handler Loading**: Support for different callback handler types
- **Conversation Manager Loading**: Support for different conversation management strategies

### Integration Features
- **ToolConfigLoader Integration**: Lazy loading to prevent circular imports
- **Circular Reference Protection**: Safe handling of mutual dependencies
- **YAML Configuration Support**: Loading agents from YAML-like configurations
- **Roundtrip Serialization**: Serialize and deserialize agents without data loss

### Error Handling
- **Invalid Configurations**: Proper validation and error messages
- **Missing Required Fields**: Validation of required configuration fields
- **Type Validation**: Ensuring configuration values are of correct types
- **Import Errors**: Handling missing dependencies gracefully

### Advanced Features
- **Lazy Loading**: ToolConfigLoader is loaded only when needed
- **Cache Management**: Agent caching with configurable cache keys
- **Extensibility**: Support for hooks and session managers (with warnings)
- **Configuration Validation**: Comprehensive validation of all configuration options

## Sample Usage

### Basic Agent Loading
```python
from strands.experimental.config_loader.agent.agent_config_loader import AgentConfigLoader

config = {
    "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "system_prompt": "You're a helpful assistant.",
    "tools": [
        {"name": "weather_tool.weather"}
    ]
}

loader = AgentConfigLoader()
agent = loader.load_agent(config)
```

### YAML Configuration Example
```yaml
agent:
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  system_prompt: "You're a helpful assistant. You can do simple math calculation, and tell the weather."
  tools:
    - name: weather_tool.weather
```

### Agent Serialization
```python
# Serialize an existing agent
config = loader.serialize_agent(agent)

# Use the config for persistence or configuration management
import json
with open('agent_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Running the Tests

```bash
# Run all tests
python -m pytest tests/strands/experimental/config_loader/agent_config_loader/ -v

# Run specific test files
python -m pytest tests/strands/experimental/config_loader/agent_config_loader/test_agent_config_loader.py -v
python -m pytest tests/strands/experimental/config_loader/agent_config_loader/test_integration.py -v

# Run with coverage
python -m pytest tests/strands/experimental/config_loader/agent/ --cov=strands.experimental.config_loader.agent.agent_config_loader
```

## Test Structure

The tests use:
- **Mock objects** for Agent instances, tools, and external dependencies
- **Comprehensive mocking** for ToolConfigLoader integration
- **Integration tests** demonstrating real-world usage scenarios
- **Error simulation** for testing exception handling
- **Roundtrip testing** for serialization/deserialization validation

## Future Agent Integration

The AgentConfigLoader is designed to support a future enhancement to the Agent constructor:

```python
# Proposed Agent constructor enhancement
class Agent:
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        if config:
            loader = AgentConfigLoader()
            # Load configuration and apply to self
            # This would make other parameters optional when config is provided
        else:
            # Use existing initialization logic
```

This would enable agents to be created directly from configuration dictionaries while maintaining backward compatibility.

All tests are designed to be independent and can run in any order.
