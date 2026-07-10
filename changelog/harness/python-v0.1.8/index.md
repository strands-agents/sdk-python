# Harness Python v0.1.8

Released 2025-06-18
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v0.1.8 · Package: https://pypi.org/project/strands-agents/0.1.8/

## Features
- implement summarizing conversation manager (https://github.com/strands-agents/sdk-python/pull/112)
- Simplify contribution template + pr scripts to run (https://github.com/strands-agents/sdk-python/pull/221)

## Fixes
- Enable underscores in direct method invocations to match hyphens (https://github.com/strands-agents/sdk-python/pull/178)
- add inference profile to litellm test and remove ownership check… [model] (https://github.com/strands-agents/sdk-python/pull/209)
- Update PR Integration Test Workflow (https://github.com/strands-agents/sdk-python/pull/237)
- remove unused dependency swagger-parser (https://github.com/strands-agents/sdk-python/pull/220)
- Update throttling logic to use exponential back-off (https://github.com/strands-agents/sdk-python/pull/223)

## Other
- moved truncation logic to conversation manager and added should\_truncate\_results (https://github.com/strands-agents/sdk-python/pull/192)
- Disallow similar tool names in the tool registry [tool] (https://github.com/strands-agents/sdk-python/pull/193)
- add integration test workflow (https://github.com/strands-agents/sdk-python/pull/201)
- allow custom tracer provider to Agent [agent] (https://github.com/strands-agents/sdk-python/pull/207)
- add a2a deps and mitigate otel conflict [a2a] (https://github.com/strands-agents/sdk-python/pull/232)
- raise exception if exporter unavailable [otel] (https://github.com/strands-agents/sdk-python/pull/234)
- docstring parser (https://github.com/strands-agents/sdk-python/pull/239)

## First-time contributors
- @stefanoamorelli (#112)
- @poshinchen (#192)
- @jer96 (#232)
- @AdnaneKhan (#237)
