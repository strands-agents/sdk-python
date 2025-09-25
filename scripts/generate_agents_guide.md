# Generate AGENTS.md Guide - Prompt for Strands Agent

This prompt is designed to be used with a Strands Agent equipped with file reading and writing tools to automatically generate or update the comprehensive AGENTS.md developer guide.

## Instructions for the Agent

You are tasked with creating a comprehensive `AGENTS.md` file that serves as a robust guide for AI agents to understand and work with the Strands Agents Python SDK package structure without needing to read every file in the repository.

### Your Objectives:

1. **Analyze the codebase structure** by reading key files and directories
2. **Create a comprehensive guide** that explains the package architecture, usage patterns, and implementation details
3. **Include practical code examples** from the actual codebase
4. **Provide troubleshooting guidance** and best practices
5. **Reference the llms.txt file** for additional documentation links

### Required Analysis Steps:

1. **Read the existing AGENTS.md file** (if it exists) to understand the current structure and content

2. **Read the project structure**:
2. **Read the project structure**:
   - Examine `pyproject.toml` for dependencies and configuration
   - Analyze `src/strands/` directory structure
   - Review key implementation files in each module

3. **Examine core components**:
3. **Examine core components**:
   - `src/strands/agent/agent.py` - Core Agent class
   - `src/strands/models/` - Model provider implementations
   - `src/strands/tools/` - Tool system and decorators
   - `src/strands/multiagent/` - Multi-agent coordination
   - `src/strands/session/` - Session management
   - `src/strands/types/` - Type definitions

4. **Review integration tests** for usage patterns:
4. **Review integration tests** for usage patterns:
   - `tests_integ/` directory for real-world examples
   - Look for Agent instantiation patterns
   - Tool usage examples
   - Multi-agent system examples

5. **Reference documentation**:
   - Include reference to https://strandsagents.com/latest/llms.txt for comprehensive documentation links
   - Link to official documentation at https://strandsagents.com/

### Output Requirements:

Create an `AGENTS.md` file with the following structure:

```markdown
# Strands Agents SDK - AI Agent Developer Guide

**Generated:** [Current timestamp]
**Repository:** https://github.com/strands-agents/sdk-python
**Purpose:** Comprehensive guide for AI agents to understand and work with the Strands Agents Python SDK

## Executive Summary
[Brief overview with key metrics]

## Core Architecture Patterns
[Explain the main design patterns]

## Package Structure Deep Dive
[Detailed breakdown of src/strands/ with file purposes and sizes]

## Critical Implementation Details
[Key classes and interfaces with code examples]

## Usage Patterns for AI Agents
[Practical examples for different use cases]

## Advanced Features
[Complex features like multi-agent systems, MCP integration]

## Error Handling and Best Practices
[Common issues and solutions]

## Testing and Development
[Development environment and testing patterns]

## Configuration Reference
[Model provider configurations and options]

## Troubleshooting Guide
[Common issues and solutions]

## Documentation References
[Reference to llms.txt and official docs]

## Summary for AI Agents
[Key takeaways and quick reference]
```

### Key Requirements:

- **Be comprehensive but concise** - provide enough detail for understanding without overwhelming
- **Include real code examples** from the codebase, not hypothetical ones
- **Focus on practical usage** - what an AI agent needs to know to use the SDK effectively
- **Update timestamps** to current date/time
- **Reference https://strandsagents.com/latest/llms.txt** for additional documentation
- **Maintain accuracy** - ensure all information reflects the current codebase state

### Tools You Should Use:

- `file_read` to examine files and directories
- `editor` to create or update the final AGENTS.md file

### Success Criteria:

The resulting AGENTS.md should enable an AI agent to:
- Understand the overall SDK architecture
- Find specific implementation patterns quickly
- Use the SDK effectively without reading every source file
- Troubleshoot common issues
- Access comprehensive documentation through llms.txt reference

Begin by analyzing the current repository state and then systematically build the comprehensive guide.
