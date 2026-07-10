# Harness Python v1.7.1

Released 2025-09-05
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.7.1 · Package: https://pypi.org/project/strands-agents/1.7.1/

## Features
- Implement async generator tools [tool] (https://github.com/strands-agents/sdk-python/pull/788)

## Fixes
- don't emit ToolStream events for non generator functions (https://github.com/strands-agents/sdk-python/pull/773)
- adjust test\_bedrock\_guardrails to account for async behavior (https://github.com/strands-agents/sdk-python/pull/785)
- replace invalid Hook names in doc comment with BeforeInvocationEvent & AfterInvocationEvent [hooks] (https://github.com/strands-agents/sdk-python/pull/782)
- Remove status field from toolResult for non-claude 3 models in Bedrock model provider [model] (https://github.com/strands-agents/sdk-python/pull/686)
- filter 'SDK\_UNKNOWN\_MEMBER' from response content (https://github.com/strands-agents/sdk-python/pull/798)
- only add signature to reasoning blocks if signature is provided (https://github.com/strands-agents/sdk-python/pull/806)

## Other
- update openai requirement from \<1.100.0 to \<1.102.0 [model] (https://github.com/strands-agents/sdk-python/pull/722)

## First-time contributors
- @deepyes02 (#782)
