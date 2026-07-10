# Harness TypeScript v1.9.0

Released 2026-07-10
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.9.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.9.0

## Features
- expose metrics getter on LocalAgent [devx, agent] (https://github.com/strands-agents/harness-sdk/pull/3116)
- map labels to native issue type and language field (https://github.com/strands-agents/harness-sdk/pull/2984)
- add durable identifiers to messages [sessions] (https://github.com/strands-agents/harness-sdk/pull/2836)
- publish TypeScript integ test metrics to CloudWatch (https://github.com/strands-agents/harness-sdk/pull/3134)
- port durable-execution checkpoints to TypeScript [language, persistence] (https://github.com/strands-agents/harness-sdk/pull/3103)
- allow ports in either direction (https://github.com/strands-agents/harness-sdk/pull/3160)

## Fixes
- harden npm lifecycle scripts for best practices (https://github.com/strands-agents/harness-sdk/pull/3128)
- rename LocalMemoryStore to TestMemoryStore (https://github.com/strands-agents/harness-sdk/pull/3123)
- upgrade ts release workflow to npm 12 (https://github.com/strands-agents/harness-sdk/pull/3188)

## Other
- add changelog generator and sync workflow (https://github.com/strands-agents/harness-sdk/pull/2765)
- fixed typescript release-workflow not running integ tests (https://github.com/strands-agents/harness-sdk/pull/3126)
- bump peter-evans/create-pull-request from 7.0.11 to 8.1.1 (https://github.com/strands-agents/harness-sdk/pull/3135)
- improve agent guidance on issue references in regression tests (https://github.com/strands-agents/harness-sdk/pull/3146)
- bump the development-dependencies group across 1 directory with 16 updates (https://github.com/strands-agents/harness-sdk/pull/3143)
- tweak pr-writer skill to be more concise (https://github.com/strands-agents/harness-sdk/pull/3148)
- bump @aws-sdk/client-bedrock-runtime from 3.1075.0 to 3.1078.0 in the production-minor group across 1 directory (https://github.com/strands-agents/harness-sdk/pull/3108)
- bump actions/checkout from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/3138)
- group Python dev tooling by name pattern (https://github.com/strands-agents/harness-sdk/pull/3171)
- export Checkpoint APIs only from /experimental [devx, persistence] (https://github.com/strands-agents/harness-sdk/pull/3180)
- scale description length to diff size (https://github.com/strands-agents/harness-sdk/pull/3183)
- run docs CI on python and typescript changes (https://github.com/strands-agents/harness-sdk/pull/3178)
