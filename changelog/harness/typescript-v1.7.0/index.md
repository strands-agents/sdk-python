# Harness TypeScript v1.7.0

Released 2026-06-25
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.7.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.7.0

## Features
- add pre-push skill mirroring the CI merge gate locally [devx] (https://github.com/strands-agents/harness-sdk/pull/2856)
- add namespace option for namespaced Cedar policies [interventions] (https://github.com/strands-agents/harness-sdk/pull/2896)
- support managed Bedrock knowledge bases in retrieve and ACL [model, tool] (https://github.com/strands-agents/harness-sdk/pull/2909)
- telemetry for memory manager [otel] (https://github.com/strands-agents/harness-sdk/pull/2858)

## Fixes
- report Responses prompt-cache tokens (TypeScript) [otel, model] (https://github.com/strands-agents/harness-sdk/pull/2782)
- handle non-string error code in classifyOpenAIError [devx, model] (https://github.com/strands-agents/harness-sdk/pull/2850)
- disambiguate Gemini tool-result part displayNames [model] (https://github.com/strands-agents/harness-sdk/pull/2881)
- prevent false-positive test failures when output is piped [agent] (https://github.com/strands-agents/harness-sdk/pull/2963)
- sha-pin third-party GitHub Actions (https://github.com/strands-agents/harness-sdk/pull/2964)
- run bedrock-kb store test in node only (https://github.com/strands-agents/harness-sdk/pull/2966)
- filter graph dependency reasoning blocks [multiagent, model] (https://github.com/strands-agents/harness-sdk/pull/2883)

## Other
- add memory manager docs (https://github.com/strands-agents/harness-sdk/pull/2758)
- require 2 maintainer reviews on bot/agent PRs (https://github.com/strands-agents/harness-sdk/pull/2755)
- raise timeout for sandbox isolation test [server] (https://github.com/strands-agents/harness-sdk/pull/2842)
- add SSH sandbox integration tests for Py and TS [server] (https://github.com/strands-agents/harness-sdk/pull/2863)
- bump codecov/codecov-action from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/2657)
- bump @aws-sdk/client-bedrock-runtime from 3.1058.0 to 3.1066.0 in the production-minor group across 1 directory (https://github.com/strands-agents/harness-sdk/pull/2708)
- add timeouts and apt retries to test-lint workflow (https://github.com/strands-agents/harness-sdk/pull/2874)
- bump actions/checkout from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/2876)
- update dependencies and refactor dev tooling internals (https://github.com/strands-agents/harness-sdk/pull/2870)
- opt back into fork-PR checkout under pull\_request\_target (https://github.com/strands-agents/harness-sdk/pull/2879)
- remove abandoned wasm TypeScript-to-Python experiment (https://github.com/strands-agents/harness-sdk/pull/2838)
- gate dependency vulns on PR-introduced changes only (https://github.com/strands-agents/harness-sdk/pull/2872)
- tighten dependabot versioning strategy (https://github.com/strands-agents/harness-sdk/pull/2899)
- bump actions/dependency-review-action from 4 to 5 (https://github.com/strands-agents/harness-sdk/pull/2903)
- add a reasoning bridge to the design doc template (https://github.com/strands-agents/harness-sdk/pull/2907)
- bump @aws-sdk/client-bedrock-runtime from 3.1066.0 to 3.1069.0 in the production-minor group across 1 directory (https://github.com/strands-agents/harness-sdk/pull/2877)
- bump actions/checkout from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/2904)
- unify Dependabot grouping and cooldown across all ecosystems (https://github.com/strands-agents/harness-sdk/pull/2901)
- bump bulk minors and patches (https://github.com/strands-agents/harness-sdk/pull/2933)
- bump dev-dependency majors (https://github.com/strands-agents/harness-sdk/pull/2935)
- add TESTING.md for python (https://github.com/strands-agents/harness-sdk/pull/2961)
- improve testing instructions (https://github.com/strands-agents/harness-sdk/pull/2965)
- skip SDK test matrices on markdown-only changes (https://github.com/strands-agents/harness-sdk/pull/2967)
- bump uuid from 14.0.0 to 14.0.1 in the production-minor group across 1 directory (https://github.com/strands-agents/harness-sdk/pull/2957)
- revert pinning virtualenv now that hatch 1.16.5 is out (https://github.com/strands-agents/harness-sdk/pull/1780)
