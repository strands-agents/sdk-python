# Enhanced YAML Format for Agent Tools

This document describes the enhanced YAML format for configuring agents with tools, including support for agent-as-tool configurations.

## Overview

The AgentConfigLoader now supports three types of tool configurations in the `tools` array:

1. **String Format**: Simple tool lookup by name
2. **Traditional Dictionary Format**: Tool lookup with name and optional module
3. **Agent-as-Tool Format**: Complete agent configuration as a tool

## Basic YAML Format

```yaml
agent:
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  system_prompt: "You're a helpful assistant. You can do simple math calculation, and tell the weather."
  tools:
    - calculator  # String format - tool lookup
    - name: weather_tool.weather  # Agent-as-tool format
      args:
        - name: location
          description: "Location for weather lookup"
          type: string
          required: true
        - name: units
          description: "Temperature units"
          type: string
          required: false
          default: celsius
      agent:
        model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        system_prompt: "You're a helpful assistant. You can tell the weather."
        tools:
          - http_request
```

## Tool Configuration Types

### 1. String Format (Tool Lookup)

The simplest format for referencing existing tools:

```yaml
tools:
  - calculator
  - weather_tool
  - database_query
```

### 2. Traditional Dictionary Format (Tool Lookup with Module)

For tools that need module specification:

```yaml
tools:
  - name: email_tool
    module: business.email
  - name: database_tool
    module: db.tools
```

### 3. Agent-as-Tool Format (Complete Agent Configuration)

For creating specialized agent tools:

```yaml
tools:
  - name: weather_assistant
    description: "Specialized weather information agent"
    args:
      - name: location
        description: "Location for weather lookup"
        type: string
        required: true
      - name: forecast_days
        description: "Number of forecast days"
        type: integer
        required: false
        default: 3
    agent:
      model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
      system_prompt: "Weather specialist for {location} with {forecast_days} day forecast"
      tools:
        - weather_api
        - alert_service
```

## Complete Example

```yaml
agent:
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  system_prompt: "You are a comprehensive business assistant with specialized capabilities."
  tools:
    # Basic string tools
    - calculator
    - calendar
    
    # Traditional dictionary tool
    - name: email_tool
      module: business.email
    
    # Customer service agent-as-tool
    - name: customer_service_agent
      description: "Specialized customer service assistant"
      args:
        - name: customer_id
          description: "Customer identifier"
          type: string
          required: true
        - name: priority
          description: "Service priority level"
          type: string
          required: false
          default: normal
      agent:
        model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        system_prompt: "Customer service specialist for customer {customer_id} with {priority} priority"
        tools:
          - crm_lookup
          - ticket_system
          # Nested agent-as-tool
          - name: escalation_agent
            agent:
              model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
              system_prompt: "Escalation specialist for complex issues"
              tools:
                - manager_notification
                - priority_queue
    
    # Analytics agent-as-tool
    - name: analytics_agent
      description: "Business analytics and reporting agent"
      args:
        - name: report_type
          description: "Type of report to generate"
          type: string
          required: true
        - name: date_range
          description: "Date range for the report"
          type: string
          required: false
          default: last_30_days
      agent:
        model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        system_prompt: "Analytics specialist generating {report_type} reports for {date_range}"
        tools:
          - database_query
          - chart_generator
          - report_formatter
```

## Agent-as-Tool Arguments

The `args` section in agent-as-tool configurations supports the following properties:

### Required Properties
- `name`: The parameter name
- `description`: Human-readable description of the parameter

### Optional Properties
- `type`: JSON Schema type (default: "string")
  - Supported types: `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`
- `required`: Whether the parameter is required (default: `true`)
- `default`: Default value for the parameter (optional)

### Example Args Configuration

```yaml
args:
  - name: location
    description: "Location for weather lookup"
    type: string
    required: true
    
  - name: units
    description: "Temperature units"
    type: string
    required: false
    default: celsius
    
  - name: forecast_days
    description: "Number of forecast days"
    type: integer
    required: false
    default: 3
    
  - name: include_alerts
    description: "Whether to include weather alerts"
    type: boolean
    required: false
    default: true
```

## Template Substitution

Agent-as-tool configurations support template variable substitution in system prompts using `{parameter_name}` syntax:

```yaml
agent:
  system_prompt: "Weather assistant for {location} using {units} for {forecast_days} days"
```

The parameters are substituted at runtime when the tool is invoked.

## Nested Agent Tools

Agent-as-tool configurations can contain other agent-as-tool configurations, allowing for complex hierarchical agent structures:

```yaml
- name: research_assistant
  agent:
    model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    system_prompt: "Research assistant"
    tools:
      - web_search
      - name: data_analyzer  # Nested agent-as-tool
        agent:
          model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
          system_prompt: "Data analysis specialist"
          tools:
            - statistics_tool
            - visualization_tool
```

## Backward Compatibility

The enhanced format maintains full backward compatibility with existing configurations:

```yaml
# Old format still works
tools:
  - calculator
  - name: weather_tool
  - name: database_tool
    module: db.tools
```

## Error Handling

The system provides clear error messages for invalid configurations:

- Missing `name` field in dictionary configurations
- Invalid tool configuration types
- Malformed agent-as-tool configurations

## Benefits

1. **Flexibility**: Mix and match different tool types in the same configuration
2. **Modularity**: Create specialized agent tools for specific tasks
3. **Reusability**: Agent-as-tool configurations can be reused across different agents
4. **Hierarchy**: Support for nested agent structures
5. **Type Safety**: Rich parameter definitions with types and validation
6. **Template Support**: Dynamic parameter substitution in system prompts

This enhanced format enables building complex, hierarchical agent systems while maintaining simplicity for basic use cases.
