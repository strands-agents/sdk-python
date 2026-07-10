# Harness Python v1.28.0

Released 2026-02-25
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.28.0 · Package: https://pypi.org/project/strands-agents/1.28.0/

## Features
- support union types and list of types for add\_hook [hooks] (https://github.com/strands-agents/sdk-python/pull/1719)
- make pyaudio an optional dependency by lazy loading (https://github.com/strands-agents/sdk-python/pull/1731)
- add Plugin Protocol for agent extensibility [hooks] (https://github.com/strands-agents/sdk-python/pull/1733)
- add plugins parameter to Agent [agent] (https://github.com/strands-agents/sdk-python/pull/1734)
- migrate SteeringHandler from HookProvider to Plugin (https://github.com/strands-agents/sdk-python/pull/1738)

## Fixes
- update region for agentcore in our new account (https://github.com/strands-agents/sdk-python/pull/1715)
- remove test that fails for python 3.14 (https://github.com/strands-agents/sdk-python/pull/1717)
- rename init\_plugin to init\_agent (https://github.com/strands-agents/sdk-python/pull/1765)

## Other
- convert Plugin from Protocol to ABC (https://github.com/strands-agents/sdk-python/pull/1741)
- switch to Sonnet 4.6 for Anthropic provider integ tests [model] (https://github.com/strands-agents/sdk-python/pull/1754)
