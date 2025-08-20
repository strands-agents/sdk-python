# Strands Agents Configuration Schema

This directory contains the comprehensive JSON Schema for validating all Strands Agents configuration files. The schema enforces proper structure and types while providing IDE support for autocompletion and validation.

## Overview

The `strands-config-schema.json` file provides validation for four types of Strands Agents configurations:

- **Agent Configuration** (`agent:`) - Single agent with tools, structured output, and advanced features
- **Graph Configuration** (`graph:`) - Multi-agent workflows with nodes, edges, and conditions  
- **Swarm Configuration** (`swarm:`) - Collaborative agent teams with autonomous coordination
- **Tools Configuration** (`tools:`) - Standalone tool definitions and configurations

## Schema Features

### ✅ **Comprehensive Type Validation**
- Enforces correct data types (string, number, boolean, array, object)
- No restrictive length or value constraints (except logical minimums)
- Supports both simple and complex configuration patterns
- Handles nested configurations (agents-as-tools, graphs-as-tools, swarms-as-tools)

### ✅ **Flexible Structure**
- Required fields enforced where necessary
- Optional fields with sensible defaults documented
- `additionalProperties: true` for extensibility
- Support for `null` values where appropriate

### ✅ **Advanced Features**
- Graph edge conditions with 6+ condition types
- Structured output schema validation
- Tool configuration with multiple formats
- Message and model configuration validation

## Configuration Types

### Agent Configuration

```yaml
# yaml-language-server: $schema=./strands-config-schema.json

agent:
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  system_prompt: "You are a helpful assistant"
  name: "MyAgent"
  tools:
    - weather_tool.weather
    - name: "custom_tool"
      description: "A custom tool"
      input_schema:
        type: object
        properties:
          query: {type: string}
  structured_output: "MySchema"

# Optional global schemas
schemas:
  - name: "MySchema"
    schema:
      type: object
      properties:
        result: {type: string}
```

### Graph Configuration

```yaml
# yaml-language-server: $schema=./strands-config-schema.json

graph:
  name: "Research Workflow"
  nodes:
    - node_id: "researcher"
      type: "agent"
      config:
        model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        system_prompt: "You are a researcher"
    - node_id: "analyst"
      type: "agent"
      config:
        model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        system_prompt: "You are an analyst"
  edges:
    - from_node: "researcher"
      to_node: "analyst"
      condition:
        type: "expression"
        expression: "state.results.get('researcher', {}).get('status') == 'complete'"
  entry_points: ["researcher"]
```

### Swarm Configuration

```yaml
# yaml-language-server: $schema=./strands-config-schema.json

swarm:
  max_handoffs: 20
  execution_timeout: 900.0
  agents:
    - name: "researcher"
      model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
      system_prompt: "You are a research specialist"
      tools: []
    - name: "writer"
      model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
      system_prompt: "You are a writing specialist"
      tools: []
```

### Tools Configuration

```yaml
# yaml-language-server: $schema=./strands-config-schema.json

tools:
  - weather_tool.weather
  - strands_tools.file_write
  - name: "custom_agent_tool"
    description: "An agent as a tool"
    agent:
      model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
      system_prompt: "You are a specialized tool agent"
    input_schema:
      type: object
      properties:
        query: {type: string}
```

## IDE Integration

### VSCode Setup

To enable YAML validation and autocompletion in VSCode:

1. **Install the YAML Extension**:
   - Install the "YAML" extension by Red Hat from the VSCode marketplace

2. **Configure Schema Association**:

   **Option A: File-level schema reference (Recommended)**
   Add this line at the top of your configuration files:
   ```yaml
   # yaml-language-server: $schema=https://strandsagents.com/schemas/config/v1
   
   agent:
     model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
     # ... rest of configuration
   ```

   **Option B: VSCode settings for `.strands.yml` files**
   Add to your VSCode `settings.json`:
   ```json
   {
     "yaml.schemas": {
       "https://strandsagents.com/schemas/config/v1": "*.strands.yml"
     }
   }
   ```

   **Option C: Local schema file reference**
   For development with local schema file:
   ```yaml
   # yaml-language-server: $schema=./path/to/strands-config-schema.json
   ```

3. **File Naming Convention**:
   While not required, using the `.strands.yml` extension makes it easy for VSCode to automatically apply the correct schema and provides clear identification of Strands configuration files.

### Other IDEs

**IntelliJ IDEA / PyCharm**:
- The schema works with the built-in YAML plugin
- Configure schema mapping in Settings → Languages & Frameworks → Schemas and DTDs → JSON Schema Mappings

**Vim/Neovim**:
- Use with `coc-yaml` or similar LSP plugins
- Configure schema association in the plugin settings

## Schema Validation Rules

### Required Fields

| Configuration Type | Required Fields |
|-------------------|----------------|
| Agent | `agent.model` |
| Graph | `graph.nodes`, `graph.edges`, `graph.entry_points` |
| Swarm | `swarm.agents` |
| Tools | `tools` (array) |

### Default Values

The schema documents these sensible defaults:

```yaml
# Agent defaults
record_direct_tool_call: true
load_tools_from_directory: false

# Graph defaults  
reset_on_revisit: false

# Swarm defaults
max_handoffs: 20
max_iterations: 20
execution_timeout: 900.0
node_timeout: 300.0
repetitive_handoff_detection_window: 0
repetitive_handoff_min_unique_agents: 0
```

### Flexible Validation

- **Timeout Values**: Accept `null` for unlimited timeouts
- **Model Configuration**: Support both string IDs and complex objects
- **Tool Definitions**: Handle simple strings and complex objects
- **Additional Properties**: Allow extension fields for future compatibility

## Condition Types

The schema supports comprehensive validation for graph edge conditions:

### Expression Conditions
```yaml
condition:
  type: "expression"
  expression: "state.results.get('node_id', {}).get('status') == 'complete'"
  description: "Check if node completed successfully"
```

### Rule Conditions
```yaml
condition:
  type: "rule"
  rules:
    - field: "results.validator.status"
      operator: "equals"
      value: "valid"
    - field: "results.validator.confidence"
      operator: "greater_than"
      value: 0.8
  logic: "and"
```

### Function Conditions
```yaml
condition:
  type: "function"
  module: "my_conditions"
  function: "check_completion"
  timeout: 5.0
  default: false
```

### Template Conditions
```yaml
condition:
  type: "template"
  template: "node_result_contains"
  parameters:
    node_id: "classifier"
    search_text: "technical"
```

### Composite Conditions
```yaml
condition:
  type: "composite"
  logic: "and"
  conditions:
    - type: "expression"
      expression: "state.execution_count < 10"
    - type: "rule"
      rules:
        - field: "status"
          operator: "equals"
          value: "ready"
```

## Validation Tools

### Command Line Validation

You can validate configurations using standard JSON Schema tools:

```bash
# Using ajv-cli
npm install -g ajv-cli
ajv validate -s strands-config-schema.json -d config.yml

# Using python jsonschema
pip install jsonschema pyyaml
python -c "
import json, yaml
from jsonschema import validate
schema = json.load(open('strands-config-schema.json'))
config = yaml.safe_load(open('config.yml'))
validate(config, schema)
print('✅ Configuration is valid')
"
```

### Online Validation

You can use online JSON Schema validators:
- [JSON Schema Validator](https://www.jsonschemavalidator.net/)
- [Schema Validator](https://jsonschemalint.com/)

## Error Messages

The schema provides clear validation error messages:

```
❌ VALIDATION ERROR: 'model' is a required property
   Path: ['agent']

❌ VALIDATION ERROR: 'invalid_type' is not one of ['agent', 'swarm', 'graph']  
   Path: ['graph', 'nodes', 0, 'type']

❌ VALIDATION ERROR: None is not of type 'string'
   Path: ['swarm', 'agents', 0, 'name']
```

## Schema Evolution

### Versioning
- Current version: `v1` (`https://strandsagents.com/schemas/config/v1`)
- Future versions will maintain backward compatibility where possible
- Breaking changes will increment the major version

### Extensibility
- The schema uses `additionalProperties: true` for extensibility
- New optional fields can be added without breaking existing configurations
- Custom properties are supported for specialized use cases

## Best Practices

### File Organization
```
project/
├── configs/
│   ├── agents/
│   │   ├── researcher.strands.yml
│   │   └── writer.strands.yml
│   ├── graphs/
│   │   ├── workflow.strands.yml
│   │   └── pipeline.strands.yml
│   └── swarms/
│       └── team.strands.yml
└── tools/
    └── custom-tools.strands.yml
```

### Configuration Management
- Use meaningful names for agents, nodes, and tools
- Include descriptions for complex configurations
- Leverage the schema's validation to catch errors early
- Use consistent naming conventions across configurations

### Development Workflow
1. Create configuration files with `.strands.yml` extension
2. Add schema reference at the top of files
3. Use IDE validation during development
4. Test configurations with ConfigLoaders
5. Validate in CI/CD pipelines if needed

## Troubleshooting

### Common Issues

**Schema not loading in VSCode**:
- Ensure the YAML extension is installed and enabled
- Check that the schema URL or path is correct
- Restart VSCode after configuration changes

**Validation errors for working configurations**:
- Ensure you're using the latest schema version
- Check that required top-level keys (`agent:`, `graph:`, etc.) are present
- Verify that all required fields are included

**Schema not found errors**:
- For local development, use relative paths to the schema file
- For production, ensure the schema URL is accessible
- Consider hosting the schema file in your project repository

## Contributing

When updating configurations or adding new features:

1. Ensure all example configurations validate against the schema
2. Update the schema if new fields or structures are added
3. Test schema changes against existing configurations
4. Update this documentation for any new features or changes

The schema serves as both validation and documentation, so keeping it accurate and comprehensive is essential for the developer experience.
