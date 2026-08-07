# Harness Python v1.51.0

Released 2026-08-07
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.51.0 · Package: https://pypi.org/project/strands-agents/1.51.0/

## Features
- hydrate search snippets concurrently [async, mcp] (https://github.com/strands-agents/harness-sdk/pull/3435)
- support the interrupt round trip over A2A [hil, a2a] (https://github.com/strands-agents/harness-sdk/pull/3486)
- add BeforeToolsEvent and AfterToolsEvent batch hooks [hooks, hil] (https://github.com/strands-agents/harness-sdk/pull/3508)
- add telemetry and metrics [otel, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3282)
- add \_\_str\_\_ support for MultiAgentResult and NodeResult [devx, multiagent] (https://github.com/strands-agents/harness-sdk/pull/1998)
- add should\_offload callback for selective offloading [context, hooks] (https://github.com/strands-agents/harness-sdk/pull/3267)
- add context window limits for Claude 5 and GPT-5.6 families [context, model] (https://github.com/strands-agents/harness-sdk/pull/3629)
- support Tool Choice for Gemini in Python [model] (https://github.com/strands-agents/harness-sdk/pull/3551)
- add classifier option for LLM-driven risk classification [hil, interventions] (https://github.com/strands-agents/harness-sdk/pull/3575)
- community integration catalog with search, filtering, and curated backfill (https://github.com/strands-agents/harness-sdk/pull/3416)
- add snapshot session manager to python [persistence, sessions] (https://github.com/strands-agents/harness-sdk/pull/3283)
- add estimateUtilization method to Model base class [context, model] (https://github.com/strands-agents/harness-sdk/pull/3641)
- add AgentStreamStage and AgentStreamContext types [hooks, hil] (https://github.com/strands-agents/harness-sdk/pull/3635)

## Fixes
- refresh eviction cycle on retrieve for unified Storage backends [context] (https://github.com/strands-agents/harness-sdk/pull/3487)
- pin mcp dependency below v2.0.0 [mcp] (https://github.com/strands-agents/harness-sdk/pull/3524)
- exclude bootstrap tools from MCP audit (https://github.com/strands-agents/harness-sdk/pull/3565)
- skip guardContent wrap for unsupported image formats [model, interventions] (https://github.com/strands-agents/harness-sdk/pull/3607)
- count usage from hook-retried model calls [otel, hooks] (https://github.com/strands-agents/harness-sdk/pull/3627)
- record gen\_ai.tool.call.arguments/result on execute\_tool spans [otel] (https://github.com/strands-agents/harness-sdk/pull/3550)
- index document content on hydration with thread safety [mcp] (https://github.com/strands-agents/harness-sdk/pull/3502)
- resolve a default region for the store's clients so it works in cloud envs [model, persistence] (https://github.com/strands-agents/harness-sdk/pull/3583)
- correct crash-restart resume for handoff and completed swarms [multiagent] (https://github.com/strands-agents/harness-sdk/pull/3391)
- reject redundant languages in astro frontmatter [devx] (https://github.com/strands-agents/harness-sdk/pull/3678)
- mantle base path routing (https://github.com/strands-agents/harness-sdk/pull/3691)
- label fork PRs by verifying the artifact's claim against the PR head (https://github.com/strands-agents/harness-sdk/pull/3705)

## Other
- update design doc template to uncover key information early (https://github.com/strands-agents/harness-sdk/pull/3532)
- bump cedarpy from 4.8.6 to 4.8.7 in /strands-py (https://github.com/strands-agents/harness-sdk/pull/3214)
- mark agentic context management mode as experimental [context] (https://github.com/strands-agents/harness-sdk/pull/3553)
- bump postcss from 8.5.15 to 8.5.25 (https://github.com/strands-agents/harness-sdk/pull/3543)
- remove reference to agent builder after archival (https://github.com/strands-agents/harness-sdk/pull/3581)
- bump fast-uri from 3.1.2 to 3.1.4 (https://github.com/strands-agents/harness-sdk/pull/3405)
- rename sandbox-routed bash tool to shell [tool, server] (https://github.com/strands-agents/harness-sdk/pull/3574)
- update litellm requirement from \<=1.91.1,\>=1.75.9 to \>=1.75.9,\<=1.93.0 in /strands-py (https://github.com/strands-agents/harness-sdk/pull/3587)
- run the package build test on pull requests [mcp] (https://github.com/strands-agents/harness-sdk/pull/3592)
- mark make\_bash with typing\_extensions.deprecated [devx, tool] (https://github.com/strands-agents/harness-sdk/pull/3595)
- update litellm requirement from \<=1.93.0,\>=1.75.9 to \>=1.75.9,\<=1.95.0 in /strands-py (https://github.com/strands-agents/harness-sdk/pull/3649)
- bump hono from 4.12.32 to 4.13.0 (https://github.com/strands-agents/harness-sdk/pull/3643)
- bump the npm\_and\_yarn group across 1 directory with 1 update (https://github.com/strands-agents/harness-sdk/pull/3620)
- bump ip-address from 10.2.0 to 10.4.0 (https://github.com/strands-agents/harness-sdk/pull/3619)
- bump @hono/node-server and @modelcontextprotocol/sdk (https://github.com/strands-agents/harness-sdk/pull/3596)
- label PRs by cognitive complexity, stop counting tests toward size (https://github.com/strands-agents/harness-sdk/pull/3634)
- wait for the retained task, not the log line [async, mcp] (https://github.com/strands-agents/harness-sdk/pull/3537)
- extend the comment rule to require to-the-point, non-inferable content (https://github.com/strands-agents/harness-sdk/pull/3676)
