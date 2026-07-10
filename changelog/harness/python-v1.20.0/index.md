# Harness Python v1.20.0

Released 2025-12-15
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.20.0 · Package: https://pypi.org/project/strands-agents/1.20.0/

## Features
- add AgentResult to AfterInvocationEvent [hooks] (https://github.com/strands-agents/sdk-python/pull/1125)
- Create agent.md and docs folder [agent] (https://github.com/strands-agents/sdk-python/pull/1312)

## Fixes
- Return structured output JSON when AgentResult has no text [agent] (https://github.com/strands-agents/sdk-python/pull/1290)
- fix broken tool spec with composition keywords [tool] (https://github.com/strands-agents/sdk-python/pull/1301)
- close mcp client event loop [mcp] (https://github.com/strands-agents/sdk-python/pull/1321)

## Other
- Remove toolResult message when toolUse is missing due to pagination in session management [sessions] (https://github.com/strands-agents/sdk-python/pull/1274)
- interrupts - swarm [multiagent] (https://github.com/strands-agents/sdk-python/pull/1193)
- bidi - fix record direct tool call [tool] (https://github.com/strands-agents/sdk-python/pull/1300)
- Update doc strings to eliminate warnings in doc build (https://github.com/strands-agents/sdk-python/pull/1284)
- bidi - tests - lint (https://github.com/strands-agents/sdk-python/pull/1307)
- bidi - fix mypy errors (https://github.com/strands-agents/sdk-python/pull/1308)
- bidi - remove python 3.11+ features (https://github.com/strands-agents/sdk-python/pull/1302)

## First-time contributors
- @davidpadbury (#1321)
