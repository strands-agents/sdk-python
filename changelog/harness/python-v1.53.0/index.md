# Harness Python v1.53.0

Released 2026-08-21
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.53.0 · Package: https://pypi.org/project/strands-agents/1.53.0/

## Features
- enable prompt caching via cache\_config and cache\_tools [model] (https://github.com/strands-agents/harness-sdk/pull/3571)
- surface tool annotations in ToolSpec [mcp, tool] (https://github.com/strands-agents/harness-sdk/pull/3528)
- add client OAuth authentication for streamable HTTP [mcp] (https://github.com/strands-agents/harness-sdk/pull/3554)
- add agent-as-tool delegation [multiagent, agent] (https://github.com/strands-agents/harness-sdk/pull/3346)
- add after tool call duration [hooks, tool] (https://github.com/strands-agents/harness-sdk/pull/3589)
- add injected content behind cache points (https://github.com/strands-agents/harness-sdk/pull/3704)
- built-in SDK integrations and maintainer tiers in the catalog (https://github.com/strands-agents/harness-sdk/pull/3766)
- count only complexity a PR adds, in both SDKs (https://github.com/strands-agents/harness-sdk/pull/3771)
- support min/max score filtering in Bedrock knowledge base store [model, tool] (https://github.com/strands-agents/harness-sdk/pull/3726)
- add /community/ editorial hub and 14-lesson course (https://github.com/strands-agents/harness-sdk/pull/3520)
- add echo suppression support [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3580)
- add audio content blocks [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3862)

## Fixes
- include strandly workspace in root build and type-check (https://github.com/strands-agents/harness-sdk/pull/3804)
- omit falsy cache point TTLs [model] (https://github.com/strands-agents/harness-sdk/pull/3799)
- classify Bedrock Mantle context-overflow errors [context, model] (https://github.com/strands-agents/harness-sdk/pull/3722)
- inherit cache\_config ttl on an untimed tools cache point [model] (https://github.com/strands-agents/harness-sdk/pull/3858)
- raise on unsupported document and image formats instead of sending undeliverable requests [model, tool] (https://github.com/strands-agents/harness-sdk/pull/3790)
- emit gemini usage metadata alongside content events [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3725)
- failing tests [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3883)
- fix failing unit tests for bedrock caching [model] (https://github.com/strands-agents/harness-sdk/pull/3887)
- accumulate cache token counters in Graph and Swarm [multiagent] (https://github.com/strands-agents/harness-sdk/pull/3884)
- cache system prompt in auto mode (https://github.com/strands-agents/harness-sdk/pull/3681)

## Other
- re-enable anthropic integration tests (https://github.com/strands-agents/harness-sdk/pull/3783)
- update litellm requirement from \<=1.95.0,\>=1.75.9 to \>=1.75.9,\<=1.96.0 in /strands-py (https://github.com/strands-agents/harness-sdk/pull/3775)
- require both decorator and log when deprecating a tool [tool] (https://github.com/strands-agents/harness-sdk/pull/3599)
- update API review label names in strands-review skill (https://github.com/strands-agents/harness-sdk/pull/3805)
- add integration testing for caching [model] (https://github.com/strands-agents/harness-sdk/pull/3793)
- remove vestigial strandly workspace (https://github.com/strands-agents/harness-sdk/pull/3806)
- bump astral-sh/setup-uv from 9.0.0 to 10.0.1 (https://github.com/strands-agents/harness-sdk/pull/3848)
- apply ruff formatting (https://github.com/strands-agents/harness-sdk/pull/3866)
- file-based memory store [persistence] (https://github.com/strands-agents/harness-sdk/pull/2895)
- scope portaudio install to a gated bidi job (https://github.com/strands-agents/harness-sdk/pull/3890)
- remove unwinnable rate-limit throttling integ test [context, model] (https://github.com/strands-agents/harness-sdk/pull/3891)
- update mypy requirement from \<2.0.0,\>=1.15.0 to \>=1.15.0,\<3.0.0 in /strands-py (https://github.com/strands-agents/harness-sdk/pull/3868)
