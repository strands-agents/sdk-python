# AGENTS.md

This document provides context, patterns, and guidelines for AI coding assistants working in this repository. For human contributors, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Product Overview

Strands Agents is an open-source Python SDK for building AI agents with a model-driven approach. It provides a lightweight, flexible framework that scales from simple conversational assistants to complex autonomous workflows.

**Core Features:**
- Model Agnostic: Multiple model providers (Amazon Bedrock, Anthropic, OpenAI, Gemini, Ollama, etc.)
- Python-Based Tools: Simple `@tool` decorator with hot reloading
- MCP Integration: Native Model Context Protocol support
- Multi-Agent Systems: Agent-to-agent, swarms, and graph patterns
- Streaming Support: Real-time response streaming
- Hooks: Event-driven extensibility for agent lifecycle
- Session Management: Pluggable session managers (file, S3, custom)
- Observability: OpenTelemetry tracing and metrics

## Directory Structure

```
strands-py/
├── src/strands/          # All production source code
│   ├── agent/            # Core agent loop, state, conversation management
│   ├── models/           # Model provider implementations (Bedrock, OpenAI, etc.)
│   ├── tools/            # Tool system (registry, execution, MCP client)
│   ├── event_loop/       # Event loop and streaming
│   ├── multiagent/       # Multi-agent orchestration (A2A, graphs, swarm)
│   ├── session/          # Session persistence
│   ├── telemetry/        # Tracing and metrics
│   └── types/            # Shared type definitions
├── tests/                # Unit tests (mirrors src/ structure)
├── tests_integ/          # Integration tests with real providers
├── docs/                 # Developer documentation
└── pyproject.toml        # Build config, dependencies, tool settings
```

## Development Workflow

### 1. Environment Setup

```bash
hatch shell                                    # Enter dev environment
pre-commit install -t pre-commit -t commit-msg # Install hooks
```

### 2. Making Changes

1. Create feature branch
2. Implement changes following the patterns below
3. Run quality checks before committing
4. Commit with conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
5. Push and open PR

### 3. Pull Request Guidelines

When creating pull requests, you MUST follow the guidelines in PR.md. Key principles:

Focus on WHY: Explain motivation and user impact, not implementation details
Document public API changes: Show before/after code examples
Be concise: Use prose over bullet lists; avoid exhaustive checklists
Target senior engineers: Assume familiarity with the SDK
Exclude implementation details: Leave these to code comments and diffs
See PR.md for the complete guidance and template.

### 4. Quality Gates

Pre-commit hooks run automatically on commit:
- Formatting (ruff)
- Linting (ruff + mypy)
- Tests (pytest)
- Commit message validation (commitizen)

All checks must pass before commit is allowed.

## Coding Patterns and Best Practices

### Logging Style

Use structured logging with field-value pairs followed by human-readable messages:

```python
logger.debug("field1=<%s>, field2=<%s> | human readable message", field1, field2)
```

**Guidelines:**
- Add context as `FIELD=<VALUE>` pairs at the beginning
- Separate pairs with commas
- Enclose values in `<>` for readability (especially for empty values)
- Use `%s` string interpolation (not f-strings) for performance
- Use lowercase messages, no punctuation
- Separate multiple statements with pipe `|`

**Good:**
```python
logger.debug("user_id=<%s>, action=<%s> | user performed action", user_id, action)
logger.info("request_id=<%s>, duration_ms=<%d> | request completed", request_id, duration)
logger.warning("attempt=<%d>, max_attempts=<%d> | retry limit approaching", attempt, max_attempts)
```

**Bad:**
```python
logger.debug(f"User {user_id} performed action {action}")  # Don't use f-strings
logger.info("Request completed in %d ms.", duration)       # Don't add punctuation
```

### Type Annotations

All code must include type annotations:
- Function parameters and return types required
- No implicit optional types
- Use `typing` or `typing_extensions` for complex types
- Mypy strict mode enforced

```python
def process_message(content: str, max_tokens: int | None = None) -> AgentResult:
    ...
```

### Docstrings

Use Google-style docstrings for all public functions, classes, and modules:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed. This docstring is used by LLMs
    to understand the function's purpose when used as a tool.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
    """
    pass
```

### Import Organization

Imports must be at the top of the file.

Imports are automatically organized by ruff/isort:
1. Standard library imports
2. Third-party imports
3. Local application imports

Use absolute imports for cross-package references, relative imports within packages.

```python
# Standard library
import logging
from typing import Any

# Third-party
import boto3
from pydantic import BaseModel

# Local
from strands.agent import Agent
from .tools import Tool
```

### File Organization

- Each major feature in its own directory
- Base classes and interfaces defined first
- Implementation-specific code in separate files
- Private modules prefixed with `_`
- Test files prefixed with `test_`

### Naming Conventions

- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with `_`

### Error Handling

- Use custom exceptions from `strands.types.exceptions`
- Provide clear error messages with context
- Don't swallow exceptions silently

## Testing Patterns

### Unit Tests (`tests/`)

- Mirror the `src/strands/` structure exactly
- Focus on isolated component testing
- Use mocking for external dependencies (models, AWS services)
- Use fixtures from `tests/fixtures/` (e.g., `mocked_model_provider.py`)

```python
# tests/strands/agent/test_agent.py mirrors src/strands/agent/agent.py
```

### Integration Tests (`tests_integ/`)

- End-to-end testing with real model providers
- Require credentials/API keys (set via environment variables)
- Organized by feature area

### Test File Naming

- Unit tests: `test_{module}.py` in `tests/strands/{path}/`
- Integration tests: `test_{feature}.py` in `tests_integ/`

### Running Tests

```bash
hatch test                           # Run unit tests
hatch test -c                        # Run with coverage
hatch run test-integ                 # Run integration tests
hatch test tests/strands/agent/      # Run specific directory
hatch test --all                     # Test all Python versions (3.10-3.13)
```

### Writing Tests

- Use pytest fixtures for setup/teardown
- Use `moto` for mocking AWS services
- Use `pytest.mark.asyncio` for async tests
- Keep tests focused and independent
- Import packages at the top of the test files

## MCP Tasks (Experimental)

The SDK supports MCP task-augmented execution for long-running tools. This feature is experimental and aligns with the MCP specification 2025-11-25.

### Overview

Task-augmented execution allows tools to run asynchronously with a workflow:
1. Create task via `call_tool_as_task`
2. Poll for completion via `poll_task`
3. Get result via `get_task_result`

### Configuration

Enable tasks by passing a `TasksConfig` to `MCPClient`:

```python
from datetime import timedelta
from strands.tools.mcp import MCPClient, TasksConfig

# Enable with defaults (ttl=1min, poll_timeout=5min)
client = MCPClient(transport, tasks_config={})

# Or configure explicitly
client = MCPClient(
    transport,
    tasks_config=TasksConfig(
        ttl=timedelta(minutes=2),           # Task time-to-live
        poll_timeout=timedelta(minutes=10),  # Polling timeout
    ),
)
```

### Tool Support Levels

MCP tools declare their task support via `execution.taskSupport`:
- `TASK_REQUIRED`: Tool must use task-augmented execution
- `TASK_OPTIONAL`: Tool can use tasks if client opts in
- `TASK_FORBIDDEN`: Tool does not support tasks (default)

### Decision Logic

Task-augmented execution is used when ALL conditions are met:
1. Client opts in via `tasks_config` (not None)
2. Server advertises task capability (`tasks.requests.tools.call`)
3. Tool's `taskSupport` is `required` or `optional`

### Key Files

- `src/strands/tools/mcp/mcp_tasks.py` - `TasksConfig` and defaults
- `src/strands/tools/mcp/mcp_client.py` - Task execution logic (`_call_tool_as_task_and_poll_async`)
- `tests/strands/tools/mcp/test_mcp_client_tasks.py` - Unit tests
- `tests_integ/mcp/test_mcp_client_tasks.py` - Integration tests
- `tests_integ/mcp/task_echo_server.py` - Test server with task support

## Things to Do

- Use explicit return types for all functions
- Write Google-style docstrings for public APIs
- Use structured logging format
- Add type annotations everywhere
- Use relative imports within packages
- Mirror src/ structure in tests/
- Run `hatch fmt --formatter` and `hatch fmt --linter` before committing
- Follow conventional commits (`feat:`, `fix:`, `docs:`, etc.)

## Things NOT to Do

- Don't use f-strings in logging calls
- Don't use `Any` type without good reason
- Don't skip type annotations
- Don't put unit tests outside `tests/strands/` structure
- Don't commit without running pre-commit hooks
- Don't add punctuation to log messages
- Don't use implicit optional types

## Development Commands

```bash
# Environment
hatch shell                    # Enter dev environment

# Formatting & Linting
hatch fmt --formatter          # Format code
hatch fmt --linter             # Run linters (ruff + mypy)

# Testing
hatch test                     # Run unit tests
hatch test -c                  # Run with coverage
hatch run test-integ           # Run integration tests
hatch test --all               # Test all Python versions

# Pre-commit
pre-commit run --all-files     # Run all hooks manually

# Readiness Check
hatch run prepare              # Run all checks (format, lint, test)

# Build
hatch build                    # Build package
```

## Agent-Specific Notes

### Writing Code

- Make the SMALLEST reasonable changes to achieve the desired outcome
- Prefer simple, clean, maintainable solutions over clever ones
- Reduce code duplication, even if refactoring takes extra effort
- Match the style and formatting of surrounding code
- Fix broken things immediately when you find them

### Code Comments

- Comments should explain WHAT the code does or WHY it exists
- NEVER add comments about what used to be there or how something changed
- NEVER refer to temporal context ("recently refactored", "moved")
- Keep comments concise and evergreen

### Code Review Considerations

- Address all review comments
- Test changes thoroughly
- Update documentation if behavior changes
- Maintain test coverage
- Follow conventional commit format for fix commits

## Additional Resources

- [Strands Agents Documentation](https://strandsagents.com/)
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Human contributor guidelines
- [docs/](./docs/) - Developer documentation
  - [STYLE_GUIDE.md](./docs/STYLE_GUIDE.md) - Code style conventions
  - [HOOKS.md](./docs/HOOKS.md) - Hooks system guide
  - [PR.md](./docs/PR.md) - PR description guidelines
  - [MCP_CLIENT_ARCHITECTURE.md](./docs/MCP_CLIENT_ARCHITECTURE.md) - MCP threading design
