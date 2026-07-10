# Harness Python v1.17.0

Released 2025-11-18
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.17.0 · Package: https://pypi.org/project/strands-agents/1.17.0/

## Features
- allow setting a timeout when creating MCPAgentTool (https://github.com/strands-agents/sdk-python/pull/1184)

## Fixes
- add validation for stream parameter in LiteLLM [model] (https://github.com/strands-agents/sdk-python/pull/1183)
- handle MetadataEvents without optional usage and metrics [otel] (https://github.com/strands-agents/sdk-python/pull/1187)
- base64 decode byte data before placing in ContentBlocks [a2a] (https://github.com/strands-agents/sdk-python/pull/1195)

## Other
- swarm - switch to handoff node only after current node stops [multiagent] (https://github.com/strands-agents/sdk-python/pull/1147)

## First-time contributors
- @AnirudhKonduru (#1184)
