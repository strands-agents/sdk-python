# Harness Python v0.3.0

Released 2025-07-11
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v0.3.0 · Package: https://pypi.org/project/strands-agents/0.3.0/

## Features
- Implement the core system of typed hooks & callbacks [hooks] (https://github.com/strands-agents/sdk-python/pull/304)
- Add hooks for before/after tool calls + allow hooks to update values [tool] (https://github.com/strands-agents/sdk-python/pull/352)
- mcp async call tool [async] (https://github.com/strands-agents/sdk-python/pull/406)
- introduce Graph multi-agent orchestrator [multiagent] (https://github.com/strands-agents/sdk-python/pull/336)

## Fixes
- handle multiple tool calls in Mistral streaming responses [model] (https://github.com/strands-agents/sdk-python/pull/384)
- add-threading-instrumentation (https://github.com/strands-agents/sdk-python/pull/394)
- Update mistral tests to avoid shared agents [model] (https://github.com/strands-agents/sdk-python/pull/398)
- Allow tool names that start with numbers [tool] (https://github.com/strands-agents/sdk-python/pull/407)

## Other
- iterative tool handler process [tool] (https://github.com/strands-agents/sdk-python/pull/340)
- remove thread pool wrapper (https://github.com/strands-agents/sdk-python/pull/339)
- updated scope name, enable setting up meter (https://github.com/strands-agents/sdk-python/pull/331)
- async model stream interface [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/306)
- allow custom agent name [agent] (https://github.com/strands-agents/sdk-python/pull/347)
- Extract hook based tests to a separate file [hooks] (https://github.com/strands-agents/sdk-python/pull/349)
- Refactor event loop to use Agent object rather than individual parameters [agent] (https://github.com/strands-agents/sdk-python/pull/359)
- models - openai - async client [model] (https://github.com/strands-agents/sdk-python/pull/353)
- models - openai - do not accept b64 images [model] (https://github.com/strands-agents/sdk-python/pull/368)
- iterative tools [tool] (https://github.com/strands-agents/sdk-python/pull/345)
- a2a streaming [a2a] (https://github.com/strands-agents/sdk-python/pull/366)
- Update A2AServer docstrings [multiagent] (https://github.com/strands-agents/sdk-python/pull/377)
- move a2a test module [a2a] (https://github.com/strands-agents/sdk-python/pull/379)
- models - mistral - async [model] (https://github.com/strands-agents/sdk-python/pull/375)
- models - ollama - async [model] (https://github.com/strands-agents/sdk-python/pull/373)
- models - anthropic - async [model] (https://github.com/strands-agents/sdk-python/pull/371)
- agent tool - remove invoke [tool] (https://github.com/strands-agents/sdk-python/pull/369)
- Add cohere client (https://github.com/strands-agents/sdk-python/pull/236)
- deps(a2a): upgrade a2a with db support [a2a] (https://github.com/strands-agents/sdk-python/pull/395)
- Writer model provider [model] (https://github.com/strands-agents/sdk-python/pull/228)
- Update integ tests to isolate provider-based tests (https://github.com/strands-agents/sdk-python/pull/396)
- Remove agent.tool\_config and update usages to use tool\_specs [agent] (https://github.com/strands-agents/sdk-python/pull/388)
- multi modal input (https://github.com/strands-agents/sdk-python/pull/367)
- async tools support [tool] (https://github.com/strands-agents/sdk-python/pull/391)
- Add basis for conformance-based tests (https://github.com/strands-agents/sdk-python/pull/403)
- Add hooks for when new messages are appended to the agent's messages [hooks] (https://github.com/strands-agents/sdk-python/pull/385)
- Add Model Invocation Hooks [hooks] (https://github.com/strands-agents/sdk-python/pull/387)
- structured output - multi-modal input [structured-output] (https://github.com/strands-agents/sdk-python/pull/405)
- \[REFACTOR\] Unify Model Interface Around Single Entry Point (model.stream) [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/400)
- Rename StartRequestEvent & EndRequestEvent events (https://github.com/strands-agents/sdk-python/pull/408)
- models - bedrock - threading [model] (https://github.com/strands-agents/sdk-python/pull/411)
- Mark hooks as non-experimental [hooks] (https://github.com/strands-agents/sdk-python/pull/410)
- models - litellm - async [model] (https://github.com/strands-agents/sdk-python/pull/414)
- models - move abstract class (https://github.com/strands-agents/sdk-python/pull/409)
- Remove event\_loop\_cycle from top level import (https://github.com/strands-agents/sdk-python/pull/415)
- Remove message processor (https://github.com/strands-agents/sdk-python/pull/417)
- Update interfaces to include kwargs to enable backwards compatibility (https://github.com/strands-agents/sdk-python/pull/413)
- Remove \_remove\_dangling\_messages from SlidingWindowConversationManager (https://github.com/strands-agents/sdk-python/pull/418)
- set Agent property load\_tools\_from\_directory to default to False [agent] (https://github.com/strands-agents/sdk-python/pull/419)

## First-time contributors
- @signoredems (#377)
- @billytrend-cohere (#236)
- @yanomaly (#228)
- @mkmeral (#400)
