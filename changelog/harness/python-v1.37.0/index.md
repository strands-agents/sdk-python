# Harness Python v1.37.0

Released 2026-04-22
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.37.0 · Package: https://pypi.org/project/strands-agents/1.37.0/

## Features
- introduce checkpoint in experimental [persistence] (https://github.com/strands-agents/sdk-python/pull/2181)
- add context\_window\_limit to model configs (https://github.com/strands-agents/sdk-python/pull/2176)

## Fixes
- add fallback trim point for tool-heavy conversations in SlidingWindowConversationManager [tool] (https://github.com/strands-agents/sdk-python/pull/2174)
- skip MCPClient cleanup during interpreter finalization [mcp] (https://github.com/strands-agents/sdk-python/pull/2144)
- update retired claude-3-haiku model in integration tests (https://github.com/strands-agents/sdk-python/pull/2186)

## First-time contributors
- @lufecadu (#2174)
