# Harness Python v1.14.0

Released 2025-10-29
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.14.0 · Package: https://pypi.org/project/strands-agents/1.14.0/

## Features
- add experimental AgentConfig with comprehensive tool management [tool] (https://github.com/strands-agents/sdk-python/pull/935)
- add multiagent hooks, add serialize & deserialize function to multiagent base & agent result [multiagent] (https://github.com/strands-agents/sdk-python/pull/1070)
- Add Structured Output as part of the agent loop [structured-output] (https://github.com/strands-agents/sdk-python/pull/943)
- add experimental agent managed connection via ToolProvider [mcp] (https://github.com/strands-agents/sdk-python/pull/895)
- skip model invocation when latest message contains ToolUse (https://github.com/strands-agents/sdk-python/pull/1068)

## Fixes
- make strands agent invoke\_agent span as INTERNAL spanKind [otel] (https://github.com/strands-agents/sdk-python/pull/1055)
- Don't bail out if there are no tool\_uses (https://github.com/strands-agents/sdk-python/pull/1087)
- enhance structured output handling [model] (https://github.com/strands-agents/sdk-python/pull/1021)

## Other
- models - litellm - start and stop reasoning [model] (https://github.com/strands-agents/sdk-python/pull/947)
- integ tests - interrupts - remove asyncio marker (https://github.com/strands-agents/sdk-python/pull/1045)
- interrupt - docstring - fix formatting (https://github.com/strands-agents/sdk-python/pull/1074)
- add pr size labeler (https://github.com/strands-agents/sdk-python/pull/1082)
- fix (bug): retry on varying Bedrock throttlingexception cases [model] (https://github.com/strands-agents/sdk-python/pull/1096)
- direct tool call - interrupt not allowed [tool] (https://github.com/strands-agents/sdk-python/pull/1097)
- mcp elicitation [mcp] (https://github.com/strands-agents/sdk-python/pull/1094)
- Transform invalid tool usages on sending, not on initial detection [tool] (https://github.com/strands-agents/sdk-python/pull/1091)

## First-time contributors
- @mr-lee (#935)
- @Arindam200 (#1021)
