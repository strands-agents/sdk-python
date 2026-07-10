# Harness Python v1.39.0

Released 2026-05-08
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.39.0 · Package: https://pypi.org/project/strands-agents/1.39.0/

## Features
- enable openai provider use aws profile [model] (https://github.com/strands-agents/sdk-python/pull/2230)
- add context window limit lookup table [context] (https://github.com/strands-agents/sdk-python/pull/2249)
- add useNativeTokenCount flag to skip token counting API calls (https://github.com/strands-agents/sdk-python/pull/2255)
- implement full A2A task lifecycle state support [a2a] (https://github.com/strands-agents/sdk-python/pull/2245)

## Fixes
- include root cause in MCPClientInitializationError message (https://github.com/strands-agents/sdk-python/pull/2238)
- fix count tokens for bedrock models [model] (https://github.com/strands-agents/sdk-python/pull/2254)
- cache unsupported models for bedrocks token counting (https://github.com/strands-agents/sdk-python/pull/2250)
- correct MCPClient.\_\_exit\_\_ and stop() type annotations (https://github.com/strands-agents/sdk-python/pull/2248)
- integration test updates (https://github.com/strands-agents/sdk-python/pull/2262)

## First-time contributors
- @aidandaly24 (#2238)
