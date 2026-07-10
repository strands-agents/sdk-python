# Harness Python v1.15.0

Released 2025-11-04
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.15.0 · Package: https://pypi.org/project/strands-agents/1.15.0/

## Features
- add multiagent session/repository management. [multiagent] (https://github.com/strands-agents/sdk-python/pull/1071)
- Add stream\_async [multiagent] (https://github.com/strands-agents/sdk-python/pull/961)
- Enable multiagent session persistent in Graph/Swarm [multiagent] (https://github.com/strands-agents/sdk-python/pull/1110)
- add SystemContentBlock support for provider-agnostic caching [model] (https://github.com/strands-agents/sdk-python/pull/1112)

## Fixes
- (bug): Drop reasoningContent from request (https://github.com/strands-agents/sdk-python/pull/1099)
- Dont initialize an agent on swarm init [multiagent] (https://github.com/strands-agents/sdk-python/pull/1107)
- Allow none structured output context in tool executors [structured-output] (https://github.com/strands-agents/sdk-python/pull/1128)
- Fix broken converstaion with orphaned toolUse (https://github.com/strands-agents/sdk-python/pull/1123)

## Other
- Fix #1077: properly redact toolResult blocks to avoid corrupting the conversation (https://github.com/strands-agents/sdk-python/pull/1080)
- linting (https://github.com/strands-agents/sdk-python/pull/1120)
- Fix input/output message not redacted when guardrails\_trace="enabled\_full" (https://github.com/strands-agents/sdk-python/pull/1072)

## First-time contributors
- @leotac (#1080)
