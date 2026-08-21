# Harness TypeScript v1.14.0

Released 2026-08-21
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.14.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.14.0

## Features
- enable prompt caching via cache\_config and cache\_tools [model] (https://github.com/strands-agents/harness-sdk/pull/3571)
- surface tool annotations in ToolSpec [mcp, tool] (https://github.com/strands-agents/harness-sdk/pull/3528)
- add injected content behind cache points (https://github.com/strands-agents/harness-sdk/pull/3704)
- built-in SDK integrations and maintainer tiers in the catalog (https://github.com/strands-agents/harness-sdk/pull/3766)
- add execution-scoped cancellation to ToolContext [devx, tool] (https://github.com/strands-agents/harness-sdk/pull/3807)
- count only complexity a PR adds, in both SDKs (https://github.com/strands-agents/harness-sdk/pull/3771)
- support min/max score filtering in Bedrock knowledge base store [model, tool] (https://github.com/strands-agents/harness-sdk/pull/3726)
- add /community/ editorial hub and 14-lesson course (https://github.com/strands-agents/harness-sdk/pull/3520)
- support multiple continuation inputs [hooks, agent] (https://github.com/strands-agents/harness-sdk/pull/3837)
- thread per-call model through InvokeModelStage [model, language] (https://github.com/strands-agents/harness-sdk/pull/3875)
- add audio content blocks [model, language] (https://github.com/strands-agents/harness-sdk/pull/3865)
- add FileMemoryStore [persistence] (https://github.com/strands-agents/harness-sdk/pull/3825)
- add context manager offloading strategies [context] (https://github.com/strands-agents/harness-sdk/pull/3505)

## Fixes
- throw on undeliverable document blocks instead of silently dropping them [devx, model] (https://github.com/strands-agents/harness-sdk/pull/3786)
- deliver url-source images instead of silently dropping them [model] (https://github.com/strands-agents/harness-sdk/pull/3792)
- unwrap NumericValue metadata back to the stored number [model, persistence] (https://github.com/strands-agents/harness-sdk/pull/3655)
- include strandly workspace in root build and type-check (https://github.com/strands-agents/harness-sdk/pull/3804)
- omit falsy cache point TTLs [model] (https://github.com/strands-agents/harness-sdk/pull/3799)
- preserve initialization failures [hooks, agent] (https://github.com/strands-agents/harness-sdk/pull/3482)
- read prompt\_tokens\_details cache reads in TS OpenAI Chat [model] (https://github.com/strands-agents/harness-sdk/pull/3885)
- cache system prompt in auto mode (https://github.com/strands-agents/harness-sdk/pull/3681)

## Other
- require both decorator and log when deprecating a tool [tool] (https://github.com/strands-agents/harness-sdk/pull/3599)
- bump @aws-sdk/client-bedrock-runtime from 3.1097.0 to 3.1104.0 in the production-minor group (https://github.com/strands-agents/harness-sdk/pull/3801)
- update API review label names in strands-review skill (https://github.com/strands-agents/harness-sdk/pull/3805)
- add integration testing for caching [model] (https://github.com/strands-agents/harness-sdk/pull/3793)
- remove vestigial strandly workspace (https://github.com/strands-agents/harness-sdk/pull/3806)
- bump @aws-sdk/client-bedrock-runtime from 3.1104.0 to 3.1105.0 in the production-minor group (https://github.com/strands-agents/harness-sdk/pull/3816)
- bump @aws-sdk/client-bedrock-runtime from 3.1105.0 to 3.1106.0 in the production-minor group (https://github.com/strands-agents/harness-sdk/pull/3850)
- bump astral-sh/setup-uv from 9.0.0 to 10.0.1 (https://github.com/strands-agents/harness-sdk/pull/3848)
- bump @aws-sdk/client-bedrock-runtime from 3.1106.0 to 3.1107.0 in the production-minor group (https://github.com/strands-agents/harness-sdk/pull/3869)
- file-based memory store [persistence] (https://github.com/strands-agents/harness-sdk/pull/2895)
- scope portaudio install to a gated bidi job (https://github.com/strands-agents/harness-sdk/pull/3890)
