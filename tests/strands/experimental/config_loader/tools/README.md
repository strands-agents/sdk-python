# ToolConfigLoader Tests

This directory contains comprehensive unit tests for the `ToolConfigLoader` class in `src/strands/experimental/config_loader/tool_config_loader.py`.

## Test Coverage

The test suite includes 80 test cases covering:

### Core Functionality
- **Initialization**: Testing with and without custom registry
- **Tool Loading**: Loading tools by identifier from various sources
- **Caching**: Verifying tool and module caching behavior
- **Multiple Tool Loading**: Loading lists of tools with different specifications

### Loading Strategies
- **Registry Loading**: Loading tools from the internal registry
- **Module Path Loading**: Loading tools from specific file paths
- **Fully Qualified Names**: Loading tools using module.tool notation
- **Search Fallback**: Searching in common locations (tools/ directory, current directory)

### Tool Extraction
- **Decorated Functions**: Extracting `@tool` decorated functions
- **AgentTool Classes**: Extracting AgentTool subclass instances
- **Direct Attributes**: Extracting tools as module attributes
- **Tool Name Matching**: Matching tools by their `tool_name` property

### Agent-as-Tool Functionality
- **Agent Tool Loading**: Loading Agents as tools from dictionary configurations
- **Argument Substitution**: Template variable substitution using `{arg_name}` format
- **Nested Agent Tools**: Support for agents that use other agents as tools
- **Mixed Tool Types**: Loading both traditional tools and agent tools together
- **YAML Configuration Support**: Compatible with YAML-style configurations

### Enhanced Args Format (NEW)
- **Legacy Format Support**: Backward compatibility with `{"arg_name": "default_value"}` format
- **Enhanced Format**: Support for `[{"name": "arg_name", "description": "...", "type_hint": "str"}]` format
- **Type Mapping**: Automatic mapping from type hints to JSON schema types
- **Default Values**: Support for explicit default values in enhanced format
- **Mixed Format Support**: Can handle both legacy and enhanced formats in the same configuration

### Error Handling
- **Import Errors**: Handling module import failures
- **Missing Tools**: Proper error messages for non-existent tools
- **Invalid Specifications**: Validation of tool specification formats
- **Class Instantiation Errors**: Handling tool class instantiation failures
- **Agent Loading Errors**: Proper error handling for agent configuration issues
- **Args Validation**: Validation of enhanced args format requirements

### Utility Functions
- **Module Import**: Dynamic module loading from file paths
- **Tool Scanning**: Discovering available tools in modules
- **Cache Management**: Clearing internal caches
- **Available Tools**: Listing available tool identifiers

## Enhanced Args Format

### Legacy Format (Still Supported)
```python
args = {"location": "NYC", "units": "metric"}
```

### Enhanced Format (NEW)
```python
args = [
    {
        "name": "my_arg",
        "description": "a description of my_arg",
        "type_hint": "str",
        "default_value": "default_value"  # optional
    },
    {
        "name": "my_other_arg",
        "description": "a description of my_other_arg",
        "type_hint": "float"
    }
]
```

### YAML Configuration Example (Enhanced Format)
```yaml
tools:
  - name: weather_tool.weather
    args:
      - name: my_arg
        description: a description of my_arg
        type_hint: str
        default_value: default_value
      - name: my_other_arg
        description: a description of my_other_arg
        type_hint: float
    agent:
      model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
      system_prompt: "You're a helpful assistant. You can tell the weather."
      tools:
        - name: http_request
```

### Type Hint Mapping
The enhanced args format supports automatic type mapping:

| Type Hint | JSON Schema Type |
|-----------|------------------|
| `str`, `string` | `string` |
| `int`, `integer` | `integer` |
| `float`, `number` | `number` |
| `bool`, `boolean` | `boolean` |
| `list`, `array` | `array` |
| `dict`, `object` | `object` |
| *unknown* | `string` (default) |

### Features of Enhanced Args
- **Rich Descriptions**: Each argument can have a detailed description
- **Type Safety**: Type hints are mapped to JSON schema types for validation
- **Default Values**: Explicit default values can be specified
- **Backward Compatibility**: Legacy format is automatically normalized to enhanced format internally
- **Template Substitution**: Works seamlessly with `{arg_name}` template substitution

## Basic Agent Tool Configuration

### Legacy Format
```python
agent_tool_config = {
    "name": "weather_assistant",
    "description": "Weather information agent",
    "args": {"location": "NYC", "units": "metric"},
    "agent": {
        "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "system_prompt": "You are a weather assistant for {location} using {units} units.",
        "tools": [{"name": "http_request"}]
    }
}
```

### Enhanced Format
```python
agent_tool_config = {
    "name": "weather_assistant",
    "description": "Weather information agent",
    "args": [
        {
            "name": "location",
            "description": "The location to get weather for",
            "type_hint": "str",
            "default_value": "NYC"
        },
        {
            "name": "units",
            "description": "Temperature units (celsius or fahrenheit)",
            "type_hint": "str",
            "default_value": "metric"
        }
    ],
    "agent": {
        "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "system_prompt": "You are a weather assistant for {location} using {units} units.",
        "tools": [{"name": "http_request"}]
    }
}

loader = ToolConfigLoader()
weather_tool = loader.load_tool(agent_tool_config)
```

### Argument Substitution
Both formats support template variable substitution:
- **Template Format**: Use `{arg_name}` in system prompts and queries
- **Default Values**: Configured in the `args` section
- **Runtime Override**: Tool callers can override default values
- **Automatic Substitution**: Variables are substituted before calling the agent

## Running the Tests

```bash
# Run all tests
python -m pytest tests/strands/experimental/config_loader/tool_config_loader/ -v

# Run specific test files
python -m pytest tests/strands/experimental/config_loader/tool_config_loader/test_tool_config_loader.py -v
python -m pytest tests/strands/experimental/config_loader/tool_config_loader/test_agent_as_tool.py -v
python -m pytest tests/strands/experimental/config_loader/tool_config_loader/test_enhanced_args.py -v
python -m pytest tests/strands/experimental/config_loader/tool_config_loader/test_enhanced_args_example.py -v
python -m pytest tests/strands/experimental/config_loader/tool_config_loader/test_product_recommendation_example.py -v

# Run with coverage
python -m pytest tests/strands/experimental/config_loader/tools/ --cov=strands.experimental.config_loader.tools.tool_config_loader
```

## Test Structure

The tests use:
- **Mock objects** for Agent instances, tools, and external dependencies
- **Comprehensive mocking** for AgentConfigLoader integration
- **Async testing** for streaming tool execution
- **Template substitution testing** for argument replacement
- **Type mapping testing** for enhanced args format
- **Integration tests** demonstrating real-world usage scenarios
- **Error simulation** for testing exception handling
- **YAML compatibility testing** for configuration format validation
- **Backward compatibility testing** for legacy format support

## Key Test Files

### `test_tool_config_loader.py`
- Original ToolConfigLoader functionality tests (37 tests)
- String-based tool loading
- Module scanning and caching
- Error handling and edge cases

### `test_agent_as_tool.py`
- Agent-as-tool functionality tests (21 tests)
- AgentAsToolWrapper class testing
- Dictionary-based tool loading
- Argument substitution and template processing
- Mixed tool type loading
- Circular reference protection

### `test_enhanced_args.py`
- Enhanced args format tests (12 tests)
- Args normalization testing
- Type hint mapping validation
- Enhanced vs legacy format comparison
- Validation error handling

### `test_enhanced_args_example.py`
- Comprehensive enhanced args examples (5 tests)
- Complete YAML configuration examples
- Type mapping demonstrations
- Mixed format compatibility
- Default value handling

### `test_product_recommendation_example.py`
- Real-world usage examples (5 tests)
- Product recommendation assistant patterns
- Complex nested agent configurations
- YAML configuration compatibility
- Argument substitution in practice

## Circular Reference Protection

The implementation includes protection against circular imports between `ToolConfigLoader` and `AgentConfigLoader`:
- **Lazy Loading**: AgentConfigLoader is imported only when needed
- **TYPE_CHECKING**: Proper type hints without runtime imports
- **Mutual References**: Both loaders can safely reference each other

All tests are designed to be independent and can run in any order.
