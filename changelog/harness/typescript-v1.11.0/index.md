# Harness TypeScript v1.11.0

Released 2026-07-24
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.11.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.11.0

## Features
- add ToolExecutor class hierarchy [language, tool] (https://github.com/strands-agents/harness-sdk/pull/3268)
- propose bidi webrtc design [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3386)
- add stop tool [tool, agent] (https://github.com/strands-agents/harness-sdk/pull/3397)
- add sleep tool [tool] (https://github.com/strands-agents/harness-sdk/pull/3393)

## Fixes
- make the release pip-audit step actually run (https://github.com/strands-agents/harness-sdk/pull/3335)
- detect tool use from streamed content when finish\_reason is non-tool [model] (https://github.com/strands-agents/harness-sdk/pull/3206)
- surface Responses stream failures [model] (https://github.com/strands-agents/harness-sdk/pull/3290)
- replay assistant text history as valid string-content input in Responses adapters [devx, model] (https://github.com/strands-agents/harness-sdk/pull/3399)
- reject keys for s3 storage if not configured [persistence] (https://github.com/strands-agents/harness-sdk/pull/3411)
- verify aws region (https://github.com/strands-agents/harness-sdk/pull/3412)

## Other
- merge strands-agents/mcp-server into monorepo [mcp] (https://github.com/strands-agents/harness-sdk/pull/3300)
- extract shared registerNodeDefaults() to prevent src/test drift [devx] (https://github.com/strands-agents/harness-sdk/pull/3303)
- replace duplicated examples guide with reference pointer (https://github.com/strands-agents/harness-sdk/pull/3288)
- bump brace-expansion from 5.0.6 to 5.0.7 (https://github.com/strands-agents/harness-sdk/pull/3370)
- refactor TestMemoryStore to use the unified storage interface (https://github.com/strands-agents/harness-sdk/pull/3260)
- bump body-parser from 2.2.2 to 2.3.0 (https://github.com/strands-agents/harness-sdk/pull/3387)
- bump actions/setup-python from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/3352)
- bump astral-sh/setup-uv from 8.3.0 to 9.0.0 (https://github.com/strands-agents/harness-sdk/pull/3407)
- bump pypa/gh-action-pypi-publish from 1.14.0 to 1.14.1 (https://github.com/strands-agents/harness-sdk/pull/3406)
