# Harness TypeScript v1.17.0

Released 2026-09-08
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.17.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.17.0

## Features
- require Node.js 22+, drop Node 20 support (https://github.com/strands-agents/harness-sdk/pull/4145)
- porting classifier routing strategy from python [model, language] (https://github.com/strands-agents/harness-sdk/pull/3987)
- add L1 stash for durable storage of offloaded content [context, persistence] (https://github.com/strands-agents/harness-sdk/pull/3924)
- support caching for vercel provider [model] (https://github.com/strands-agents/harness-sdk/pull/4055)
- add session support for L1 [persistence, sessions] (https://github.com/strands-agents/harness-sdk/pull/4118)
- add prefix\_with\_server\_name and continue\_on\_error defaults to load\_servers [mcp] (https://github.com/strands-agents/harness-sdk/pull/4177)
- export context manager types as experimental [context, devx] (https://github.com/strands-agents/harness-sdk/pull/4231)

## Fixes
- include cached tokens in input\_tokens [model] (https://github.com/strands-agents/harness-sdk/pull/4080)
- reject backslashes in storage keys [persistence] (https://github.com/strands-agents/harness-sdk/pull/4126)
- close the model invoke span when the consumer breaks the stream [otel, model] (https://github.com/strands-agents/harness-sdk/pull/3898)
- remove @tobilu/qmd from optional dependencies (https://github.com/strands-agents/harness-sdk/pull/4178)
- emit tool results before user text to preserve tool\_use/tool\_result adjacency [model] (https://github.com/strands-agents/harness-sdk/pull/4138)

## Other
- design: propose shared agent and model types [devx, agent] (https://github.com/strands-agents/harness-sdk/pull/3855)
- cancel superseded PR runs to reduce runner concurrency (https://github.com/strands-agents/harness-sdk/pull/4134)
- use strands-agents 4-core runner pool (https://github.com/strands-agents/harness-sdk/pull/4141)
- move codecov config to repo root so Codecov discovers it again (https://github.com/strands-agents/harness-sdk/pull/4174)
- add install-size guard to pack test (https://github.com/strands-agents/harness-sdk/pull/4179)
