# Harness Python v1.16.0

Released 2025-11-12
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.16.0 · Package: https://pypi.org/project/strands-agents/1.16.0/

## Features
- Add tool definitions to traces via semconv opt-in [otel] (https://github.com/strands-agents/sdk-python/pull/1113)
- Support string descriptions in Annotated parameters [tool] (https://github.com/strands-agents/sdk-python/pull/1089)
- allow SystemContentBlocks in LiteLLMModel [model] (https://github.com/strands-agents/sdk-python/pull/1141)

## Fixes
- handle non-JSON error messages from Gemini API [model] (https://github.com/strands-agents/sdk-python/pull/1062)
- Handle "prompt is too long" from Anthropic [model] (https://github.com/strands-agents/sdk-python/pull/1137)
- Strip argument sections out of inputSpec top-level description (https://github.com/strands-agents/sdk-python/pull/1142)
- Don't hang when MCP server returns 5xx [mcp] (https://github.com/strands-agents/sdk-python/pull/1169)
- allow setter on system\_prompt and system\_prompt\_content [model] (https://github.com/strands-agents/sdk-python/pull/1171)

## Other
- share thread context [context] (https://github.com/strands-agents/sdk-python/pull/1146)
- async hooks [hooks] (https://github.com/strands-agents/sdk-python/pull/1119)
- updated opt-in attributes to internal [otel] (https://github.com/strands-agents/sdk-python/pull/1152)
- share interrupt state (https://github.com/strands-agents/sdk-python/pull/1148)
