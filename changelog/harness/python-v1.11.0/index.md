# Harness Python v1.11.0

Released 2025-10-08
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.11.0 · Package: https://pypi.org/project/strands-agents/1.11.0/

## Features
- updated traces to match OTEL v1.37 semantic conventions [otel] (https://github.com/strands-agents/sdk-python/pull/952)
- implement concurrent message reading for session managers [sessions] (https://github.com/strands-agents/sdk-python/pull/897)

## Fixes
- GeminiModel argument in README (https://github.com/strands-agents/sdk-python/pull/955)
- removed double serialization for events [otel] (https://github.com/strands-agents/sdk-python/pull/977)
- map LiteLLM context window errors to ContextWindowOverflowException [model] (https://github.com/strands-agents/sdk-python/pull/994)

## Other
- tool - executors - concurrent - remove no-op gather [tool] (https://github.com/strands-agents/sdk-python/pull/954)
- event loop - handle model execution (https://github.com/strands-agents/sdk-python/pull/958)
- hooks - before tool call event - cancel tool [tool] (https://github.com/strands-agents/sdk-python/pull/964)

## First-time contributors
- @tosi29 (#955)
