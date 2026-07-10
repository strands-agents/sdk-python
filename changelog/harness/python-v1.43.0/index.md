# Harness Python v1.43.0

Released 2026-06-12
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.43.0 · Package: https://pypi.org/project/strands-agents/1.43.0/

## Features
- add python wasm release workflow (https://github.com/strands-agents/harness-sdk/pull/2409)
- add optional hook order [hooks] (https://github.com/strands-agents/harness-sdk/pull/2559)
- wire checkpointing into agent event loop [persistence] (https://github.com/strands-agents/harness-sdk/pull/2190)
- make get/set app state sync [server] (https://github.com/strands-agents/harness-sdk/pull/2610)
- wire vended tools through the WASM guest [server] (https://github.com/strands-agents/harness-sdk/pull/2607)
- add test-infra/ CDK stack for long-running integ test resources (https://github.com/strands-agents/harness-sdk/pull/2650)
- add Sandbox core abstraction (TS→Python port, core only 1/N) [server] (https://github.com/strands-agents/harness-sdk/pull/2665)
- add context\_manager="auto" facade on Agent [context] (https://github.com/strands-agents/harness-sdk/pull/2643)
- add claude-opus-4-8 to context window limits [context] (https://github.com/strands-agents/harness-sdk/pull/2676)
- add model\_state as a snapshot field [persistence] (https://github.com/strands-agents/harness-sdk/pull/2680)
- generate smart preview links for changed doc pages [documentation] (https://github.com/strands-agents/harness-sdk/pull/2692)
- add message pinning to conversation managers [context] (https://github.com/strands-agents/harness-sdk/pull/2644)
- add Docker/SSH Sandbox implementations [server] (https://github.com/strands-agents/harness-sdk/pull/2691)
- add pr-create and pr-writer agent skills [devx] (https://github.com/strands-agents/harness-sdk/pull/2646)
- add pr-feedback agent skill for triaging PR review comments (https://github.com/strands-agents/harness-sdk/pull/2712)
- allow same-account IAM roles to assume the integ test role (https://github.com/strands-agents/harness-sdk/pull/2723)
- implement intervention primitive in python with cancellation support [interventions] (https://github.com/strands-agents/harness-sdk/pull/2693)

## Fixes
- resolve dependabot alerts and bump CI actions (https://github.com/strands-agents/harness-sdk/pull/2546)
- workflow\_dispatch step for python wasm release (https://github.com/strands-agents/harness-sdk/pull/2560)
- add maintain role to auto-strands-review allowed-roles (https://github.com/strands-agents/harness-sdk/pull/2633)
- include json blocks in counting tokens [context] (https://github.com/strands-agents/harness-sdk/pull/2639)
- introduce agent factory for isolating agent context from different callers [a2a] (https://github.com/strands-agents/harness-sdk/pull/2628)
- update stale GitHub links from old repos to harness-sdk [documentation] (https://github.com/strands-agents/harness-sdk/pull/2698)
- isolate conversation state per context in TypeScript [a2a, sessions] (https://github.com/strands-agents/harness-sdk/pull/2696)
- default to us-east-1 for integration tests (https://github.com/strands-agents/harness-sdk/pull/2720)
- label npm updates typescript instead of javascript (https://github.com/strands-agents/harness-sdk/pull/2726)
- remove stale consuming-repos reference from PR template (https://github.com/strands-agents/harness-sdk/pull/2727)
- remove flaky test\_graph\_parallel\_execution test [multiagent] (https://github.com/strands-agents/harness-sdk/pull/2743)
- stop over-applying area-community and chore (https://github.com/strands-agents/harness-sdk/pull/2725)

## Other
- commit LICENSE and NOTICE directly into package directories (https://github.com/strands-agents/harness-sdk/pull/2392)
- update repository references to harness-sdk (https://github.com/strands-agents/harness-sdk/pull/2618)
- document per-invocation limits feature [documentation] (https://github.com/strands-agents/harness-sdk/pull/2638)
- add missing policies to integ test role (https://github.com/strands-agents/harness-sdk/pull/2670)
- fixup test-infra constructs (https://github.com/strands-agents/harness-sdk/pull/2673)
- use local monorepo source instead of cloning SDKs [documentation] (https://github.com/strands-agents/harness-sdk/pull/2677)
- use native model\_state snapshot field [a2a] (https://github.com/strands-agents/harness-sdk/pull/2694)
- add explicit permissions to workflow jobs (https://github.com/strands-agents/harness-sdk/pull/2388)
- add automated issue labeling workflow (https://github.com/strands-agents/harness-sdk/pull/2359)
- bump actions/checkout from 4 to 6 (https://github.com/strands-agents/harness-sdk/pull/2100)
- add PR labeling to issue-labeler workflow (https://github.com/strands-agents/harness-sdk/pull/2702)
- add README to .agents/skills/ explaining each skill's purpose (https://github.com/strands-agents/harness-sdk/pull/2714)
- allow \`blog\` type in PR title validation (https://github.com/strands-agents/harness-sdk/pull/2718)
- enforce API review label requirement before merge (https://github.com/strands-agents/harness-sdk/pull/2716)
- add AI contribution guidance to CONTRIBUTING and PR template (https://github.com/strands-agents/harness-sdk/pull/2728)
