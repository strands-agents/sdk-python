# Harness Python v1.22.0

Released 2026-01-13
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.22.0 · Package: https://pypi.org/project/strands-agents/1.22.0/

## Features
- provide extra command content as the the prompt to the agent [agent] (https://github.com/strands-agents/sdk-python/pull/1419)
- add guardrail\_latest\_message option [model] (https://github.com/strands-agents/sdk-python/pull/1224)
- introduce AgentBase Protocol as the interface for agent classes to implement [agent] (https://github.com/strands-agents/sdk-python/pull/1126)
- pass invocation\_state to model providers (https://github.com/strands-agents/sdk-python/pull/1414)

## Fixes
- import errors for models with optional imports (https://github.com/strands-agents/sdk-python/pull/1384)
- UnboundLocal Exception Fix [model] (https://github.com/strands-agents/sdk-python/pull/1420)
- make calculator tool more robust to LLM output variations [tool] (https://github.com/strands-agents/sdk-python/pull/1445)
- resolve string formatting error in MCP client error handling [mcp] (https://github.com/strands-agents/sdk-python/pull/1446)
- add concurrency protection to prevent parallel invocations from corrupting agent state [agent] (https://github.com/strands-agents/sdk-python/pull/1453)
- propagate contextvars to background thread [mcp] (https://github.com/strands-agents/sdk-python/pull/1444)

## Other
- update github agent action to reference S3\_SESSION\_BUCKET [agent] (https://github.com/strands-agents/sdk-python/pull/1418)
- \[FEATURE\] add MCP resource operations in MCP Tools [mcp] (https://github.com/strands-agents/sdk-python/pull/1117)
- add BidiGeminiLiveModel and BidiOpenAIRealtimeModel to the init (https://github.com/strands-agents/sdk-python/pull/1383)
- bidi - async - remove cancelling call (https://github.com/strands-agents/sdk-python/pull/1357)
- fix! Litellm handle non streaming response fix for issue #477 [model] (https://github.com/strands-agents/sdk-python/pull/512)
- update pytest requirement from \<9.0.0,\>=8.0.0 to \>=8.0.0,\<10.0.0 in the dev-dependencies group (https://github.com/strands-agents/sdk-python/pull/1161)
- Add Security.md file (https://github.com/strands-agents/sdk-python/pull/1454)
- Update release notes sop (https://github.com/strands-agents/sdk-python/pull/1456)
- bidi - move 3.12 check to nova sonic module (https://github.com/strands-agents/sdk-python/pull/1439)
- update sphinx requirement from \<9.0.0,\>=5.0.0 to \>=5.0.0,\<10.0.0 (https://github.com/strands-agents/sdk-python/pull/1426)
- Update to opus 4.5 (https://github.com/strands-agents/sdk-python/pull/1471)

## First-time contributors
- @aiancheruk (#1224)
- @emattiza (#1420)
- @schleidl (#512)
- @tirth14 (#1414)
