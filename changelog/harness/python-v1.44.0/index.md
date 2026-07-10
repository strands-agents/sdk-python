# Harness Python v1.44.0

Released 2026-06-16
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.44.0 · Package: https://pypi.org/project/strands-agents/1.44.0/

## Features
- add turn-based eviction to InMemoryStorage [context] (https://github.com/strands-agents/harness-sdk/pull/2648)
- add internal middleware system for InvokeModelStage [hooks, model] (https://github.com/strands-agents/harness-sdk/pull/2760)
- pass invocation\_state to edge condition calls [multiagent, hil] (https://github.com/strands-agents/harness-sdk/pull/2642)
- port memory manager and extraction to Python [async, hooks, persistence] (https://github.com/strands-agents/harness-sdk/pull/2740)
- add GoalLoop vended plugin with docs [hooks, structured-output] (https://github.com/strands-agents/harness-sdk/pull/2738)
- add Agent memory\_manager param with configurable sync auto-flush [persistence, agent] (https://github.com/strands-agents/harness-sdk/pull/2795)
- add memory injection (https://github.com/strands-agents/harness-sdk/pull/2797)
- add default for memory extraction trigger (https://github.com/strands-agents/harness-sdk/pull/2811)
- port agentic context management to python [context] (https://github.com/strands-agents/harness-sdk/pull/2808)
- port HumanInTheLoop vended intervention to Python [hil, interventions] (https://github.com/strands-agents/harness-sdk/pull/2750)
- integrate Sandbox with Agent [server, agent] (https://github.com/strands-agents/harness-sdk/pull/2762)
- add Cedar authorization handler for Python [tool, interventions] (https://github.com/strands-agents/harness-sdk/pull/2802)
- port BedrockKnowledgeBaseStore to strands-py [model, persistence] (https://github.com/strands-agents/harness-sdk/pull/2834)
- vended tools/plugins for sandbox [tool, server] (https://github.com/strands-agents/harness-sdk/pull/2835)

## Fixes
- remove pin messaging barrel export [context, devx] (https://github.com/strands-agents/harness-sdk/pull/2767)
- reduce workflow noise from deployment history and label churn (https://github.com/strands-agents/harness-sdk/pull/2766)
- consolidate PR guidelines and update pr-writer skill (https://github.com/strands-agents/harness-sdk/pull/2772)
- allow sync or async InterventionHandler lifecycle overrides [async, interventions] (https://github.com/strands-agents/harness-sdk/pull/2800)
- mark internal memory functions as private (https://github.com/strands-agents/harness-sdk/pull/2817)

## Other
- add message pinning section to conversation management page [context] (https://github.com/strands-agents/harness-sdk/pull/2749)
- design: add context management presets design doc [context] (https://github.com/strands-agents/harness-sdk/pull/2756)
- add Syntax component for inline language-specific identifiers [documentation] (https://github.com/strands-agents/harness-sdk/pull/2751)
- improve monorepo clarity for releases and issues (https://github.com/strands-agents/harness-sdk/pull/2759)
- bump esbuild version (https://github.com/strands-agents/harness-sdk/pull/2785)
- consolidate dev docs into a consistent home (https://github.com/strands-agents/harness-sdk/pull/2791)
- suppress verbose logging in tests with large text payloads [devx] (https://github.com/strands-agents/harness-sdk/pull/2773)
- point PR reviewers at docs skills for site/ changes (https://github.com/strands-agents/harness-sdk/pull/2794)
- apply ruff format to fix pre-existing drift in strands-py (https://github.com/strands-agents/harness-sdk/pull/2799)
- point PR reviewers at docs skills for site/ changes (https://github.com/strands-agents/harness-sdk/pull/2807)
- add CLAUDE.md files (https://github.com/strands-agents/harness-sdk/pull/2809)
- represent passed-in config types as TypedDicts [devx] (https://github.com/strands-agents/harness-sdk/pull/2824)
- represent ExtractionConfig as a TypedDict [devx] (https://github.com/strands-agents/harness-sdk/pull/2827)
- drop redundant \`| None\` from optional config fields [devx] (https://github.com/strands-agents/harness-sdk/pull/2832)
- remove lockfile drift check to unblock Dependabot and audit fixes (https://github.com/strands-agents/harness-sdk/pull/2841)
- bump hono to 4.12.25 to fix high-severity audit failure (https://github.com/strands-agents/harness-sdk/pull/2843)
- assert S3 sidecar metadata and scope round-trip [persistence] (https://github.com/strands-agents/harness-sdk/pull/2840)
