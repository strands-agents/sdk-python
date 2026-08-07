# Harness TypeScript v1.12.0

Released 2026-08-07
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.12.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.12.0

## Features
- hydrate search snippets concurrently [async, mcp] (https://github.com/strands-agents/harness-sdk/pull/3435)
- add llm as judge tool risk classifier to hitl [hil, interventions] (https://github.com/strands-agents/harness-sdk/pull/3566)
- add MCP tool filtering and name prefixes [mcp] (https://github.com/strands-agents/harness-sdk/pull/3415)
- add context window limits for Claude 5 and GPT-5.6 families [context, model] (https://github.com/strands-agents/harness-sdk/pull/3629)
- community integration catalog with search, filtering, and curated backfill (https://github.com/strands-agents/harness-sdk/pull/3416)
- detect repetitive swarm handoffs [multiagent] (https://github.com/strands-agents/harness-sdk/pull/3461)
- add estimateUtilization method to Model base class [context, model] (https://github.com/strands-agents/harness-sdk/pull/3641)
- accept storage as optional top-level Agent parameter [agent, sessions] (https://github.com/strands-agents/harness-sdk/pull/3660)

## Fixes
- pin mcp dependency below v2.0.0 [mcp] (https://github.com/strands-agents/harness-sdk/pull/3524)
- exclude bootstrap tools from MCP audit (https://github.com/strands-agents/harness-sdk/pull/3565)
- validate snapshot scope [persistence] (https://github.com/strands-agents/harness-sdk/pull/3593)
- preserve interrupt state across cancelled resumes and middleware re-reads [hil] (https://github.com/strands-agents/harness-sdk/pull/3615)
- record gen\_ai.tool.call.arguments/result on execute\_tool spans [otel] (https://github.com/strands-agents/harness-sdk/pull/3550)
- index document content on hydration with thread safety [mcp] (https://github.com/strands-agents/harness-sdk/pull/3502)
- reject redundant languages in astro frontmatter [devx] (https://github.com/strands-agents/harness-sdk/pull/3678)
- mantle base path routing (https://github.com/strands-agents/harness-sdk/pull/3691)
- label fork PRs by verifying the artifact's claim against the PR head (https://github.com/strands-agents/harness-sdk/pull/3705)
- bump @aws-sdk/core to 3.977.6 to fix NumericValue metadata regression [model, persistence] (https://github.com/strands-agents/harness-sdk/pull/3709)

## Other
- update design doc template to uncover key information early (https://github.com/strands-agents/harness-sdk/pull/3532)
- mark agentic context management mode as experimental [context] (https://github.com/strands-agents/harness-sdk/pull/3553)
- bump postcss from 8.5.15 to 8.5.25 (https://github.com/strands-agents/harness-sdk/pull/3543)
- remove reference to agent builder after archival (https://github.com/strands-agents/harness-sdk/pull/3581)
- bump fast-uri from 3.1.2 to 3.1.4 (https://github.com/strands-agents/harness-sdk/pull/3405)
- rename sandbox-routed bash tool to shell [tool, server] (https://github.com/strands-agents/harness-sdk/pull/3574)
- run the package build test on pull requests [mcp] (https://github.com/strands-agents/harness-sdk/pull/3592)
- bump @aws-sdk/client-bedrock-runtime from 3.1078.0 to 3.1095.0 in the production-minor group across 1 directory (https://github.com/strands-agents/harness-sdk/pull/3174)
- bump hono from 4.12.32 to 4.13.0 (https://github.com/strands-agents/harness-sdk/pull/3643)
- bump the npm\_and\_yarn group across 1 directory with 1 update (https://github.com/strands-agents/harness-sdk/pull/3620)
- bump ip-address from 10.2.0 to 10.4.0 (https://github.com/strands-agents/harness-sdk/pull/3619)
- bump @aws-sdk/client-bedrock-runtime from 3.1095.0 to 3.1097.0 in the production-minor group across 1 directory (https://github.com/strands-agents/harness-sdk/pull/3625)
- bump @hono/node-server and @modelcontextprotocol/sdk (https://github.com/strands-agents/harness-sdk/pull/3596)
- extend the comment rule to require to-the-point, non-inferable content (https://github.com/strands-agents/harness-sdk/pull/3676)
- bump actions/upload-artifact from 6.0.0 to 7.0.1 (https://github.com/strands-agents/harness-sdk/pull/3702)
- bump @hono/node-server from 1.19.14 to 2.1.0 (https://github.com/strands-agents/harness-sdk/pull/3669)
- bump actions/setup-python from 6.3.0 to 7.0.0 (https://github.com/strands-agents/harness-sdk/pull/3701)
- bump fast-uri from 3.1.4 to 3.1.5 (https://github.com/strands-agents/harness-sdk/pull/3618)
- bump actions/download-artifact from 6.0.0 to 8.0.1 (https://github.com/strands-agents/harness-sdk/pull/3703)
