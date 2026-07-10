# Harness Python v1.42.0

Released 2026-06-01
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.42.0 · Package: https://pypi.org/project/strands-agents/1.42.0/

## Features
- add endpoint\_url parameter to S3SessionManager (https://github.com/strands-agents/sdk-python/pull/1934)
- plumb through cache tokens in metadata events [model] (https://github.com/strands-agents/sdk-python/pull/2287)
- add \`agent\_card\_url\` property to \`A2AServer\` for customizable \`url\` in \`AgentCard\` [a2a] (https://github.com/strands-agents/sdk-python/pull/2003)
- use call\_async for true async streaming [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/2361)
- add Limits and support it during invoke/stream [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/2360)
- pass invocation\_state to edge condition calls [multiagent, hooks] (https://github.com/strands-agents/sdk-python/pull/2305)
- make variant arms inherit from container (https://github.com/strands-agents/sdk-python/pull/2386)
- promote content-to-tool-result method to public API [devx, mcp] (https://github.com/strands-agents/sdk-python/pull/2370)
- add DecoratedTool for host-side Python tools [tool] (https://github.com/strands-agents/sdk-python/pull/2412)

## Fixes
- fix flaky tests to accept string or number (https://github.com/strands-agents/sdk-python/pull/2319)
- handle None text in message content sanitization [model] (https://github.com/strands-agents/sdk-python/pull/1920)
- make MetricsClient singleton thread-safe [otel] (https://github.com/strands-agents/sdk-python/pull/2349)
- handle safety-blocked metadata [model] (https://github.com/strands-agents/sdk-python/pull/2353)
- read vllm reasoning deltas [model] (https://github.com/strands-agents/sdk-python/pull/2354)
- downgrade validation failure log from error to debug [structured-output] (https://github.com/strands-agents/sdk-python/pull/2368)
- scope authorization-check job permissions to contents: read (https://github.com/strands-agents/sdk-python/pull/2367)
- use separate READMEs for Python and TypeScript packages (https://github.com/strands-agents/sdk-python/pull/2384)
- keep concurrent tool results in request order [tool] (https://github.com/strands-agents/sdk-python/pull/2340)
- realign provider context-overflow patterns and drop MIT dual-license [context] (https://github.com/strands-agents/sdk-python/pull/2394)
- fix bootstrap ordering and update README (https://github.com/strands-agents/sdk-python/pull/2402)
- update vitest to ^4.1.6 (https://github.com/strands-agents/sdk-python/pull/2534)

## Other
- prepare directory layout for monorepo convergence (https://github.com/strands-agents/sdk-python/pull/2317)
- merge strands-agents/docs into monorepo (https://github.com/strands-agents/sdk-python/pull/2339)
- address fast-follow items from docs monorepo merge (https://github.com/strands-agents/sdk-python/pull/2348)
- merge strands-agents/sdk-typescript into monorepo (https://github.com/strands-agents/sdk-python/pull/2350)
- update stale references for monorepo consolidation (https://github.com/strands-agents/sdk-python/pull/2358)
- use env var for repository name in integration test workflow (https://github.com/strands-agents/sdk-python/pull/2371)
- sync strands-agents/sdk-typescript into monorepo (https://github.com/strands-agents/sdk-python/pull/2363)
- revert "feat: pass invocation\_state to edge condition calls (#2305)" (https://github.com/strands-agents/sdk-python/pull/2389)
- add security warnings to http\_request and file\_editor vended tools [tool] (https://github.com/strands-agents/sdk-python/pull/2391)
- allow design type in PR title validation (https://github.com/strands-agents/sdk-python/pull/2395)

## First-time contributors
- @tealgreen0503 (#1934)
- @yatszhash (#2287)
- @yoppi (#1920)
- @gtholpadi (#2349)
- @he-yufeng (#2353)
- @yananym (#2305)
