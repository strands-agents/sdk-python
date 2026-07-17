# Harness TypeScript v1.10.0

Released 2026-07-17
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.10.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.10.0

## Features
- add unified storage interface [hooks, persistence] (https://github.com/strands-agents/harness-sdk/pull/3099)
- added gen\_ai\_span\_attributes\_only var to skip event attributes [otel] (https://github.com/strands-agents/harness-sdk/pull/3194)
- auto-namespace unified Storage under offloader [context, persistence] (https://github.com/strands-agents/harness-sdk/pull/3258)

## Fixes
- place cache point before non-PDF document blocks [model] (https://github.com/strands-agents/harness-sdk/pull/2001)
- use /openai/v1 Mantle base URL for the Responses API [model] (https://github.com/strands-agents/harness-sdk/pull/3280)

## Other
- sync bot fork with upstream before opening PRs (https://github.com/strands-agents/harness-sdk/pull/3179)
- trigger changelog sync directly from the release workflows (https://github.com/strands-agents/harness-sdk/pull/3193)
- extract generic async Queue from multiagent [async, multiagent] (https://github.com/strands-agents/harness-sdk/pull/3262)
- guard sync job to upstream; document workflow-scope requirement (https://github.com/strands-agents/harness-sdk/pull/3278)
- bump actions/setup-node from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/3229)
- remove dead barrel and obsolete types package (https://github.com/strands-agents/harness-sdk/pull/3285)
- deduplicate pr guidelines summary in sdk agents files (https://github.com/strands-agents/harness-sdk/pull/3293)
- deduplicate testing guide and agents file (https://github.com/strands-agents/harness-sdk/pull/3291)
