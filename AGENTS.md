# AGENTS.md

This document provides context, patterns, and guidelines for AI coding assistants working in this repository. For human contributors, see [CONTRIBUTING.md](./CONTRIBUTING.md).

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
strands-agents/
│
├── src/strands/                          # Main package source code
│   ├── agent/                            # Core agent implementation
│   │   ├── agent.py                      # Main Agent class
│   │   ├── agent_result.py               # Agent execution results
│   │   ├── base.py                       # AgentBase protocol (agent interface)
│   │   ├── a2a_agent.py                  # A2AAgent client for remote A2A agents
│   │   ├── state.py                      # Agent state management
│   │   └── conversation_manager/         # Message history strategies
│   │       ├── conversation_manager.py           # Base conversation manager
│   │       ├── null_conversation_manager.py      # No-op manager
│   │       ├── sliding_window_conversation_manager.py  # Window-based
│   │       └── summarizing_conversation_manager.py     # Summarization-based
│   │
│   ├── event_loop/                       # Agent execution loop
│   │   ├── event_loop.py                 # Main loop logic
│   │   ├── streaming.py                  # Streaming response handling
│   │   └── _recover_message_on_max_tokens_reached.py
│   │
│   ├── models/                           # Model provider implementations
│   │   ├── model.py                      # Base model interface
│   │   ├── bedrock.py                    # Amazon Bedrock
│   │   ├── anthropic.py                  # Anthropic Claude
│   │   ├── openai.py                     # OpenAI
│   │   ├── gemini.py                     # Google Gemini
│   │   ├── ollama.py                     # Ollama local models
│   │   ├── litellm.py                    # LiteLLM unified interface
│   │   ├── mistral.py                    # Mistral AI
│   │   ├── llamaapi.py                   # LlamaAPI
│   │   ├── llamacpp.py                   # llama.cpp local
│   │   ├── sagemaker.py                  # AWS SageMaker
│   │   ├── writer.py                     # Writer AI
│   │   └── _validation.py                # Validation utilities
│   │
│   ├── tools/                            # Tool system
│   │   ├── decorator.py                  # @tool decorator
│   │   ├── tools.py                      # Tool base classes
│   │   ├── registry.py                   # Tool registration
│   │   ├── loader.py                     # Dynamic tool loading
│   │   ├── watcher.py                    # Hot reload
│   │   ├── _caller.py                    # Tool invocation
│   │   ├── _validator.py                 # Tool validation
│   │   ├── _tool_helpers.py              # Helper utilities
│   │   ├── executors/                    # Tool execution environments
│   │   │   ├── _executor.py              # Base executor
│   │   │   ├── concurrent.py             # Thread/process pool
│   │   │   └── sequential.py             # Sequential execution
│   │   ├── mcp/                          # Model Context Protocol
│   │   │   ├── mcp_client.py             # MCP client implementation
│   │   │   ├── mcp_agent_tool.py         # MCP tool wrapper
│   │   │   ├── mcp_types.py              # MCP type definitions
│   │   │   └── mcp_instrumentation.py    # MCP telemetry
│   │   └── structured_output/            # Structured output handling
│   │       ├── structured_output_tool.py
│   │       ├── structured_output_utils.py
│   │       └── _structured_output_context.py
│   │
│   ├── multiagent/                       # Multi-agent patterns
│   │   ├── base.py                       # Base multi-agent classes
│   │   ├── graph.py                      # Graph-based orchestration
│   │   ├── swarm.py                      # Swarm pattern
│   │   ├── a2a/                          # Agent-to-agent protocol
│   │   │   ├── executor.py               # A2A executor
│   │   │   ├── server.py                 # A2A server
│   │   │   └── converters.py             # Strands/A2A type converters
│   │   └── nodes/                        # Graph node implementations
│   │
│   ├── types/                            # Type definitions
│   │   ├── content.py                    # Content types (text, images, etc.)
│   │   ├── tools.py                      # Tool-related types
│   │   ├── streaming.py                  # Streaming event types
│   │   ├── exceptions.py                 # Custom exceptions
│   │   ├── agent.py                      # Agent types
│   │   ├── session.py                    # Session types
│   │   ├── multiagent.py                 # Multi-agent types
│   │   ├── guardrails.py                 # Guardrail types
│   │   ├── interrupt.py                  # Interrupt types
│   │   ├── media.py                      # Media types
│   │   ├── citations.py                  # Citation types
│   │   ├── traces.py                     # Trace types
│   │   ├── event_loop.py                 # Event loop types
│   │   ├── json_dict.py                  # JSON dict utilities
│   │   ├── collections.py                # Collection types
│   │   ├── _events.py                    # Internal event types
│   │   ├── a2a.py                        # A2A protocol types
│   │   └── models/                       # Model-specific types
│   │
│   ├── session/                          # Session management
│   │   ├── session_manager.py            # Base interface
│   │   ├── file_session_manager.py       # File-based storage
│   │   ├── s3_session_manager.py         # S3 storage
│   │   ├── repository_session_manager.py # Repository pattern
│   │   └── session_repository.py         # Storage interface
│   │
│   ├── telemetry/                        # Observability (OpenTelemetry)
│   │   ├── tracer.py                     # Tracing
│   │   ├── metrics.py                    # Metrics collection
│   │   ├── metrics_constants.py          # Metric definitions
│   │   └── config.py                     # Configuration
│   │
│   ├── hooks/                            # Event hooks system
│   │   ├── events.py                     # Hook event definitions
│   │   └── registry.py                   # Hook registration
│   │
│   ├── handlers/                         # Event handlers
│   │   └── callback_handler.py           # Callback handling
│   │
│   ├── experimental/                     # Experimental features (API may change)
│   │   ├── agent_config.py               # Experimental agent config
│   │   ├── bidi/                         # Bidirectional streaming
│   │   │   ├── agent/                    # Bidi agent implementation
│   │   │   ├── io/                       # Input/output handling
│   │   │   ├── models/                   # Bidi model providers
│   │   │   ├── tools/                    # Bidi tools
│   │   │   ├── types/                    # Bidi types
│   │   │   └── _async/                   # Async utilities
│   │   ├── hooks/                        # Experimental hooks
│   │   │   ├── events.py
│   │   │   └── multiagent/
│   │   ├── steering/                     # Agent steering
│   │   │   ├── context_providers/
│   │   │   ├── core/
│   │   │   └── handlers/
│   │   └── tools/                        # Experimental tools
│   │       └── tool_provider.py
│   │
│   ├── __init__.py                       # Public API exports
│   ├── interrupt.py                      # Interrupt handling
│   ├── _async.py                         # Async utilities
│   ├── _exception_notes.py               # Exception helpers
│   ├── _identifier.py                    # ID generation
│   └── py.typed                          # PEP 561 marker
│
├── tests/                                # Unit tests (mirrors src/)
│   ├── conftest.py                       # Pytest fixtures
│   ├── fixtures/                         # Test fixtures
│   │   ├── mocked_model_provider.py      # Mock model for testing
│   │   ├── mock_agent_tool.py
│   │   ├── mock_hook_provider.py
│   │   └── ...
│   └── strands/                          # Tests mirror src/strands/
│       ├── agent/
│       ├── event_loop/
│       ├── models/
│       ├── tools/
│       ├── multiagent/
│       ├── types/
│       ├── session/
│       ├── telemetry/
│       ├── hooks/
│       ├── handlers/
│       ├── experimental/
│       └── utils/
│
├── tests_integ/                          # Integration tests
│   ├── conftest.py
│   ├── models/                           # Model provider tests
│   │   ├── test_model_bedrock.py
│   │   ├── test_model_anthropic.py
│   │   ├── test_model_openai.py
│   │   ├── test_model_gemini.py
│   │   ├── test_model_ollama.py
│   │   └── ...
│   ├── mcp/                              # MCP integration tests
│   │   ├── test_mcp_client.py
│   │   ├── echo_server.py
│   │   └── ...
│   ├── tools/                            # Tool system tests
│   ├── hooks/                            # Hook tests
│   ├── interrupts/                       # Interrupt tests
│   ├── steering/                         # Steering tests
│   ├── bidi/                             # Bidirectional streaming tests
│   ├── a2a/                              # A2A agent integration tests
│   ├── test_multiagent_graph.py
│   ├── test_multiagent_swarm.py
│   ├── test_stream_agent.py
│   ├── test_session.py
│   └── ...
│
├── docs/                                 # Developer documentation
│   ├── README.md                         # Docs folder overview
│   ├── STYLE_GUIDE.md                    # Code style conventions
│   ├── HOOKS.md                          # Hooks system guide
│   ├── PR.md                             # PR description guidelines
│   └── MCP_CLIENT_ARCHITECTURE.md        # MCP threading architecture
│
├── pyproject.toml                        # Project config (build, deps, tools)
├── AGENTS.md                             # This file
└── CONTRIBUTING.md                       # Human contributor guidelines
```

### Directory Purposes

- **`src/strands/`**: All production code
- **`tests/`**: Unit tests mirroring src/ structure
- **`tests_integ/`**: Integration tests with real model providers
- **`docs/`**: Developer documentation for contributors

**IMPORTANT**: After making changes that affect the directory structure (adding new directories, moving files, or adding significant new files), you MUST update this directory structure section to reflect the current state of the repository.

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
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Human contributor guidelines
- [docs/](./docs/) - Developer documentation
  - [STYLE_GUIDE.md](./docs/STYLE_GUIDE.md) - Code style conventions
  - [HOOKS.md](./docs/HOOKS.md) - Hooks system guide
  - [PR.md](./docs/PR.md) - PR description guidelines
  - [MCP_CLIENT_ARCHITECTURE.md](./docs/MCP_CLIENT_ARCHITECTURE.md) - MCP threading design
