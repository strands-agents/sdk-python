# Harness TypeScript v1.6.0

Released 2026-06-16
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.6.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.6.0

## Features
- copy middleware context inputs to prevent accidental mutation [hooks, agent] (https://github.com/strands-agents/harness-sdk/pull/2742)
- add turn-based eviction to InMemoryStorage [context] (https://github.com/strands-agents/harness-sdk/pull/2648)
- memory injection [context, agent] (https://github.com/strands-agents/harness-sdk/pull/2631)
- add internal middleware system for InvokeModelStage [hooks, model] (https://github.com/strands-agents/harness-sdk/pull/2760)
- add cedar vended intervention handler [interventions] (https://github.com/strands-agents/harness-sdk/pull/2365)
- add agentic context management with model-driven compression tools [context, agent] (https://github.com/strands-agents/harness-sdk/pull/2754)
- add memory injection (https://github.com/strands-agents/harness-sdk/pull/2797)
- port agentic context management to python [context] (https://github.com/strands-agents/harness-sdk/pull/2808)

## Fixes
- remove pin messaging barrel export [context, devx] (https://github.com/strands-agents/harness-sdk/pull/2767)
- reduce workflow noise from deployment history and label churn (https://github.com/strands-agents/harness-sdk/pull/2766)
- correct fixture path in integration test (https://github.com/strands-agents/harness-sdk/pull/2810)

## Other
- add message pinning section to conversation management page [context] (https://github.com/strands-agents/harness-sdk/pull/2749)
- design: add context management presets design doc [context] (https://github.com/strands-agents/harness-sdk/pull/2756)
- add Syntax component for inline language-specific identifiers [documentation] (https://github.com/strands-agents/harness-sdk/pull/2751)
- improve monorepo clarity for releases and issues (https://github.com/strands-agents/harness-sdk/pull/2759)
- bump esbuild version (https://github.com/strands-agents/harness-sdk/pull/2785)
- add integration test for memory manager [context] (https://github.com/strands-agents/harness-sdk/pull/2764)
- consolidate dev docs into a consistent home (https://github.com/strands-agents/harness-sdk/pull/2791)
- point PR reviewers at docs skills for site/ changes (https://github.com/strands-agents/harness-sdk/pull/2794)
- point PR reviewers at docs skills for site/ changes (https://github.com/strands-agents/harness-sdk/pull/2807)
- add CLAUDE.md files (https://github.com/strands-agents/harness-sdk/pull/2809)
- isolate KB search/add tool tests from auto-injection [tool] (https://github.com/strands-agents/harness-sdk/pull/2825)
- remove lockfile drift check to unblock Dependabot and audit fixes (https://github.com/strands-agents/harness-sdk/pull/2841)
- bump hono to 4.12.25 to fix high-severity audit failure (https://github.com/strands-agents/harness-sdk/pull/2843)
- assert S3 sidecar metadata and scope round-trip [persistence] (https://github.com/strands-agents/harness-sdk/pull/2840)
- inline tool prefixing into getTools implementations [server] (https://github.com/strands-agents/harness-sdk/pull/2806)
