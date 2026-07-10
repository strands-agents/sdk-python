# Harness Python v1.7.0

Released 2025-09-02
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.7.0 · Package: https://pypi.org/project/strands-agents/1.7.0/

## Features
- Implement typed events internally (https://github.com/strands-agents/sdk-python/pull/745)
- Use TypedEvent inheritance for callback behavior (https://github.com/strands-agents/sdk-python/pull/755)
- claude citation support with BedrockModel [model] (https://github.com/strands-agents/sdk-python/pull/631)
- Enable hooks for MultiAgents [hooks] (https://github.com/strands-agents/sdk-python/pull/760)

## Fixes
- fix stop reason for bedrock model when stop\_reason [model] (https://github.com/strands-agents/sdk-python/pull/767)
- Return tool result message as part of event + expand unit test coverage [tool] (https://github.com/strands-agents/sdk-python/pull/771)
- fix loading tools with same tool name [tool] (https://github.com/strands-agents/sdk-python/pull/772)

## Other
- summarization manager - add summary prompt to messages (https://github.com/strands-agents/sdk-python/pull/698)
- Add invocation\_state to ToolContext (https://github.com/strands-agents/sdk-python/pull/761)
- Add VPC endpoint support to BedrockModel class - Add optional endpoin… [model] (https://github.com/strands-agents/sdk-python/pull/502)

## First-time contributors
- @dbavro19 (#502)
