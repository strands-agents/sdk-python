# Harness Python v1.30.0

Released 2026-03-11
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.30.0 · Package: https://pypi.org/project/strands-agents/1.30.0/

## Features
- add "anthropic" cache strategy to bypass model ID check [model] (https://github.com/strands-agents/sdk-python/pull/1808)
- serialize tool results as JSON when possible [tool] (https://github.com/strands-agents/sdk-python/pull/1752)
- expose server instructions from InitializeResult on MCPClient [mcp] (https://github.com/strands-agents/sdk-python/pull/1814)
- add dirty flag to skip unnecessary agent state persistence [sessions] (https://github.com/strands-agents/sdk-python/pull/1803)
- add public tool\_spec setter (https://github.com/strands-agents/sdk-python/pull/1822)
- add CancellationToken for graceful agent execution cancellation [agent] (https://github.com/strands-agents/sdk-python/pull/1772)
- optimize session manager initialization [sessions] (https://github.com/strands-agents/sdk-python/pull/1829)
- add resume flag to AfterInvocationEvent [hooks] (https://github.com/strands-agents/sdk-python/pull/1767)
- add agent skills as a plugin [agent] (https://github.com/strands-agents/sdk-python/pull/1755)
- move steering from experimental to production (https://github.com/strands-agents/sdk-python/pull/1853)

## Fixes
- summary manager using structured output [structured-output] (https://github.com/strands-agents/sdk-python/pull/1805)
- added LANGFUSE\_BASE\_URL check for additinoal attribute (https://github.com/strands-agents/sdk-python/pull/1826)
- report usage metrics in streaming mode [model] (https://github.com/strands-agents/sdk-python/pull/1697)
- use output\_text for assistant messages in multi-turn conversations (https://github.com/strands-agents/sdk-python/pull/1851)
- place cache point on last user message instead of assistant (https://github.com/strands-agents/sdk-python/pull/1821)
- break circular references so Agent cleanup doesn't hang with MCPClient [agent] (https://github.com/strands-agents/sdk-python/pull/1830)
- Set \_is\_new\_session = False at the end of each initialize\_\* method (https://github.com/strands-agents/sdk-python/pull/1859)

## First-time contributors
- @ShotaroKataoka (#1814)
- @jgoyani1 (#1772)
- @jackatorcflo (#1697)
- @giulio-leone (#1851)
