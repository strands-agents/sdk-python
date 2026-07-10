# Harness Python v1.24.0

Released 2026-01-29
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.24.0 · Package: https://pypi.org/project/strands-agents/1.24.0/

## Features
- add automatic prompt caching support [model] (https://github.com/strands-agents/sdk-python/pull/1438)
- add retry mechanism for tool calls [hooks] (https://github.com/strands-agents/sdk-python/pull/1556)
- move ToolProvider out of experimental namespace [tool] (https://github.com/strands-agents/sdk-python/pull/1567)
- update AgentResult \_\_str\_\_ priority order [agent] (https://github.com/strands-agents/sdk-python/pull/1553)
- Add invocation state [hooks] (https://github.com/strands-agents/sdk-python/pull/1550)

## Fixes
- Populate tool\_args correctly for steering (https://github.com/strands-agents/sdk-python/pull/1531)

## Other
- fix flaky openai structured output test by adding Field guidance [model] (https://github.com/strands-agents/sdk-python/pull/1534)
- interrupts - multiagent - do not emit AfterNodeCallEvent on interrupt [multiagent] (https://github.com/strands-agents/sdk-python/pull/1539)
- add workflow for lambda layer publish (https://github.com/strands-agents/sdk-python/pull/870)
- interrupts - graph - agent based [multiagent] (https://github.com/strands-agents/sdk-python/pull/1533)
- refactor use\_span to be closed automatically (https://github.com/strands-agents/sdk-python/pull/1293)
- limit permission scope on lambda layer github action (https://github.com/strands-agents/sdk-python/pull/1555)
- Enable Auto-close labels on Pull requests as well. (https://github.com/strands-agents/sdk-python/pull/1552)
- Use devtools actions (https://github.com/strands-agents/sdk-python/pull/1554)
- \[FIX\] models - gemini - start and stop reasoningContent [model] (https://github.com/strands-agents/sdk-python/pull/1557)
- callback handler - fix reporting of tool when missing delta [tool] (https://github.com/strands-agents/sdk-python/pull/1573)
- Fix failing integ tests (https://github.com/strands-agents/sdk-python/pull/1580)

## First-time contributors
- @kevmyung (#1438)
