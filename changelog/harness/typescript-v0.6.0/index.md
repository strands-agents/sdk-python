# Harness TypeScript v0.6.0

Released 2026-03-11
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v0.6.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/0.6.0

## Features
- add CitationsBlock for document citation support (https://github.com/strands-agents/sdk-typescript/pull/568)
- add session manager implementation and related tests [sessions] (https://github.com/strands-agents/sdk-typescript/pull/569)
- make tasks opt-in via tasksConfig [mcp] (https://github.com/strands-agents/sdk-typescript/pull/516)
- swarm orchestration pattern [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/606)
- browser-compatible tracer with    NodeTracerProvider auto-detection [otel] (https://github.com/strands-agents/sdk-typescript/pull/622)
- add getTracer to telemetry api surface [otel] (https://github.com/strands-agents/sdk-typescript/pull/604)
- support both Zod and JSON schemas for tool() factory [tool] (https://github.com/strands-agents/sdk-typescript/pull/617)
- add guardrail redaction support with input/output hand… [model] (https://github.com/strands-agents/sdk-typescript/pull/631)
- graph orchestration pattern [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/632)
- local metrics tracking for agent loop [otel] (https://github.com/strands-agents/sdk-typescript/pull/597)
- add delete session & list pagination API, update tests [sessions] (https://github.com/strands-agents/sdk-typescript/pull/623)

## Fixes
- remove circular import of barrel index.js in agent.ts [agent] (https://github.com/strands-agents/sdk-typescript/pull/605)
- remove deprecated eslint-env comments incompatible with flat config (https://github.com/strands-agents/sdk-typescript/pull/611)
- use source import for Agent in swarm integ tests [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/628)
- add warn log when node execution fails in multi-agent orchestration [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/640)

## Other
- bump express-rate-limit from 8.2.1 to 8.3.0 (https://github.com/strands-agents/sdk-typescript/pull/607)
- remove color assertions from image/video integ tests (https://github.com/strands-agents/sdk-typescript/pull/609)
- tidy registry tests in to separate \_\_tests\_\_ dir (https://github.com/strands-agents/sdk-typescript/pull/612)
- simplify ToolRegistry to name-based CRUDL interface (https://github.com/strands-agents/sdk-typescript/pull/616)
- bump actions/upload-artifact from 6 to 7 (https://github.com/strands-agents/sdk-typescript/pull/580)
- bump aws-actions/configure-aws-credentials from 5 to 6 (https://github.com/strands-agents/sdk-typescript/pull/496)
- bump actions/github-script from 7 to 8 (https://github.com/strands-agents/sdk-typescript/pull/525)
- bump amannn/action-semantic-pull-request from 5 to 6 (https://github.com/strands-agents/sdk-typescript/pull/526)
- npm audit fix (https://github.com/strands-agents/sdk-typescript/pull/638)
