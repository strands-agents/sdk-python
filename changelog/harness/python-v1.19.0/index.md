# Harness Python v1.19.0

Released 2025-12-03
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.19.0 · Package: https://pypi.org/project/strands-agents/1.19.0/

## Features
- add experimental steering for modular prompting (https://github.com/strands-agents/sdk-python/pull/1280)

## Fixes
- avoid KeyError in direct tool calls with context [tool] (https://github.com/strands-agents/sdk-python/pull/1213)
- attached custom attributes to all spans (https://github.com/strands-agents/sdk-python/pull/1235)

## Other
- hooks - before node call - cancel node [hooks] (https://github.com/strands-agents/sdk-python/pull/1203)
- interrupts - support falsey responses (https://github.com/strands-agents/sdk-python/pull/1256)
- Bidirectional Streaming Agent [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/1276)
- mcp - elicitation - fix server request test [mcp] (https://github.com/strands-agents/sdk-python/pull/1281)
- adjust integ test system prompts to reduce flakiness (https://github.com/strands-agents/sdk-python/pull/1282)

## First-time contributors
- @qmays-phdata (#1213)
