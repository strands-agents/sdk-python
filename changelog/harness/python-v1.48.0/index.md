# Harness Python v1.48.0

Released 2026-07-17
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.48.0 · Package: https://pypi.org/project/strands-agents/1.48.0/

## Features
- allow ports in either direction (https://github.com/strands-agents/harness-sdk/pull/3160)
- expose client\_name parameter on MCPClient for clientInfo identification [devx, mcp] (https://github.com/strands-agents/harness-sdk/pull/3113)
- add gen\_ai\_span\_attributes\_only env var [otel] (https://github.com/strands-agents/harness-sdk/pull/3191)
- add unified storage interface [persistence] (https://github.com/strands-agents/harness-sdk/pull/3259)

## Fixes
- upgrade ts release workflow to npm 12 (https://github.com/strands-agents/harness-sdk/pull/3188)
- load directory tools under a namespaced module key [tool] (https://github.com/strands-agents/harness-sdk/pull/2994)
- set Strands user agent for Nova Sonic client [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/2132)
- support vLLM v0.16.0+ reasoning field in streaming and non-streaming paths [model] (https://github.com/strands-agents/harness-sdk/pull/3252)
- detect throttling via ClientError status attribute [model] (https://github.com/strands-agents/harness-sdk/pull/3228)
- place cache point before non-PDF document blocks [model] (https://github.com/strands-agents/harness-sdk/pull/2001)
- prevent symlink attacks in FileSessionManager [persistence] (https://github.com/strands-agents/harness-sdk/pull/2937)

## Other
- bump actions/checkout from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/3138)
- group Python dev tooling by name pattern (https://github.com/strands-agents/harness-sdk/pull/3171)
- scale description length to diff size (https://github.com/strands-agents/harness-sdk/pull/3183)
- run docs CI on python and typescript changes (https://github.com/strands-agents/harness-sdk/pull/3178)
- sync bot fork with upstream before opening PRs (https://github.com/strands-agents/harness-sdk/pull/3179)
- update Nova Sonic docs for v2 and fix stale references [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3195)
- trigger changelog sync directly from the release workflows (https://github.com/strands-agents/harness-sdk/pull/3193)
- revert chore(deps): relax litellm upper bound to \<2.0.0 (#3149) [model] (https://github.com/strands-agents/harness-sdk/pull/3223)
- document Bedrock strict\_tools schema constraints [devx, model] (https://github.com/strands-agents/harness-sdk/pull/3261)
- guard sync job to upstream; document workflow-scope requirement (https://github.com/strands-agents/harness-sdk/pull/3278)
- bump actions/setup-node from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/3229)
- remove dead private code (https://github.com/strands-agents/harness-sdk/pull/3286)
- deduplicate pr guidelines summary in sdk agents files (https://github.com/strands-agents/harness-sdk/pull/3293)
