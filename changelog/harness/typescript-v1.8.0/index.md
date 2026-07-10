# Harness TypeScript v1.8.0

Released 2026-07-08
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.8.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.8.0

## Features
- result handling, model-state isolation, and system-prompt fidelity [hooks, model] (https://github.com/strands-agents/harness-sdk/pull/2812)
- add local memory store [persistence] (https://github.com/strands-agents/harness-sdk/pull/2859)
- add Port Request template [language] (https://github.com/strands-agents/harness-sdk/pull/3009)
- load MCP servers from JSON [mcp, config] (https://github.com/strands-agents/harness-sdk/pull/2947)
- add storage design doc proposal [persistence] (https://github.com/strands-agents/harness-sdk/pull/3080)
- preserve MCP output schemas [mcp] (https://github.com/strands-agents/harness-sdk/pull/3109)

## Fixes
- fixed memory test failure (https://github.com/strands-agents/harness-sdk/pull/3020)
- allow interface return types from tool callbacks [language, tool] (https://github.com/strands-agents/harness-sdk/pull/3025)
- emit valid json for null/undefined tool returns [language, tool] (https://github.com/strands-agents/harness-sdk/pull/3024)
- preserve body when search output exceeds max\_chars [tool] (https://github.com/strands-agents/harness-sdk/pull/3065)
- buffer graph node printer output to prevent interleaving [async, multiagent] (https://github.com/strands-agents/harness-sdk/pull/3077)
- drop blank text content blocks before send [model] (https://github.com/strands-agents/harness-sdk/pull/3029)
- guard upload-metrics on integration-test job result (https://github.com/strands-agents/harness-sdk/pull/3102)
- confine retrieved paths to the artifact directory [context, persistence] (https://github.com/strands-agents/harness-sdk/pull/3035)
- create release tag via git push instead of gh release --target (https://github.com/strands-agents/harness-sdk/pull/3115)
- exclude test artifacts from package (https://github.com/strands-agents/harness-sdk/pull/3055)

## Other
- add Strandslator design doc [language] (https://github.com/strands-agents/harness-sdk/pull/2790)
- improve agent guidance (https://github.com/strands-agents/harness-sdk/pull/2959)
- change-based selective integration testing (https://github.com/strands-agents/harness-sdk/pull/2921)
- move strandslator design docs into team/designs (https://github.com/strands-agents/harness-sdk/pull/3006)
- restrict issue responder to collaborators and pin action version (https://github.com/strands-agents/harness-sdk/pull/2991)
- add CODEOWNERS to scaffold PR assignment (https://github.com/strands-agents/harness-sdk/pull/3078)
- bump astral-sh/setup-uv from 7.6.0 to 8.2.0 (https://github.com/strands-agents/harness-sdk/pull/2974)
- bump the development-dependencies group across 1 directory with 9 updates (https://github.com/strands-agents/harness-sdk/pull/3075)
- bump commander from 14.0.3 to 15.0.0 (https://github.com/strands-agents/harness-sdk/pull/2999)
- bump dorny/paths-filter from 4.0.1 to 4.0.2 (https://github.com/strands-agents/harness-sdk/pull/3083)
- bump astral-sh/setup-uv from 8.2.0 to 8.3.0 (https://github.com/strands-agents/harness-sdk/pull/3094)
- bump the development-dependencies group across 1 directory with 10 updates (https://github.com/strands-agents/harness-sdk/pull/3098)
- added release workflow (https://github.com/strands-agents/harness-sdk/pull/2940)
- fix integration test runs (https://github.com/strands-agents/harness-sdk/pull/3112)
