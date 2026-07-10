# Harness TypeScript v1.1.0

Released 2026-05-08
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v1.1.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.1.0

## Features
- add hook fields on Before/After Tool Call Event and AfterInvocationEvent [tool] (https://github.com/strands-agents/sdk-typescript/pull/957)
- add tsc type-checking to strands-wasm (https://github.com/strands-agents/sdk-typescript/pull/979)
- auto-populate contextWindowLimit from model ID lookup tables (https://github.com/strands-agents/sdk-typescript/pull/954)
- expose model on local agent [agent] (https://github.com/strands-agents/sdk-typescript/pull/938)
- add agent guide for wasm feature development [agent] (https://github.com/strands-agents/sdk-typescript/pull/992)
- add result offload plugin [context] (https://github.com/strands-agents/sdk-typescript/pull/974)
- implement interrupt system for human-in-the-loop workflows (https://github.com/strands-agents/sdk-typescript/pull/784)
- add structured output implementation for wasm [structured-output] (https://github.com/strands-agents/sdk-typescript/pull/1000)
- surface server logs, enable failOpen, metadata getters [mcp] (https://github.com/strands-agents/sdk-typescript/pull/1010)
- add timeouts to graph/swarm; bedrock request timeout [model] (https://github.com/strands-agents/sdk-typescript/pull/1008)
- add endTurn decision field to AfterToolsEvent [hooks] (https://github.com/strands-agents/sdk-typescript/pull/982)
- add useNativeTokenCount flag to skip token counting API calls (https://github.com/strands-agents/sdk-typescript/pull/1009)
- add Symbol.asyncDispose to McpClient to enable await-using cleanup [mcp] (https://github.com/strands-agents/sdk-typescript/pull/1016)
- add optional hook order [hooks] (https://github.com/strands-agents/sdk-typescript/pull/1005)
- add proactive context compression to conversation managers [context] (https://github.com/strands-agents/sdk-typescript/pull/965)
- normalize invalid tool names [tool] (https://github.com/strands-agents/sdk-typescript/pull/1017)
- add DefaultModelRetryStrategy, ModelRetryStrategy, and BackoffStrategy (https://github.com/strands-agents/sdk-typescript/pull/888)

## Fixes
- migrate LifecycleBridge from deleted HookProvider to Plugin (https://github.com/strands-agents/sdk-typescript/pull/967)
- format entry.ts (https://github.com/strands-agents/sdk-typescript/pull/975)
- fix stale imports and ToolRegistry API in WASM bridge (https://github.com/strands-agents/sdk-typescript/pull/973)
- add missing root script delegations (https://github.com/strands-agents/sdk-typescript/pull/972)
- run generate before type-check in strands-wasm (https://github.com/strands-agents/sdk-typescript/pull/987)
- paginate listTools(), improve fallback description [mcp] (https://github.com/strands-agents/sdk-typescript/pull/984)
- mapEvent interrupt guard, broken test imports (https://github.com/strands-agents/sdk-typescript/pull/1003)
- remove type checking for wasm temporarily (https://github.com/strands-agents/sdk-typescript/pull/1007)
- cache unsupported models for bedrocks token counting (https://github.com/strands-agents/sdk-typescript/pull/999)
- allow browser env in OpenAI responses integ fixture [model] (https://github.com/strands-agents/sdk-typescript/pull/1014)

## Other
- bump @anthropic-ai/sdk from 0.89.0 to 0.92.0 [model] (https://github.com/strands-agents/sdk-typescript/pull/978)
- add contract tests for the WASM bridge (https://github.com/strands-agents/sdk-typescript/pull/983)
- eliminate type safety gaps in entry.ts bridge (https://github.com/strands-agents/sdk-typescript/pull/988)
- decompose mapEvent into typed functions (https://github.com/strands-agents/sdk-typescript/pull/989)
- update tenacity requirement from \>=8.0 to \>=9.1.4 in /strands-py (https://github.com/strands-agents/sdk-typescript/pull/842)
- update docstring-parser requirement from \>=0.16 to \>=0.18.0 in /strands-py (https://github.com/strands-agents/sdk-typescript/pull/843)
- bump commander from 12.1.0 to 14.0.3 (https://github.com/strands-agents/sdk-typescript/pull/845)
- bump fast-xml-parser and @aws-sdk/xml-builder (https://github.com/strands-agents/sdk-typescript/pull/940)
- bump uuid from 13.0.0 to 14.0.0 (https://github.com/strands-agents/sdk-typescript/pull/962)
- update boto3 requirement from \>=1.42.92 to \>=1.43.2 in /strands-py (https://github.com/strands-agents/sdk-typescript/pull/986)
- bump @aws-sdk/client-bedrock-runtime from 3.1033.0 to 3.1037.0 in the production-minor group across 1 directory [model] (https://github.com/strands-agents/sdk-typescript/pull/993)
- rename hook order to SDK first/last [hooks] (https://github.com/strands-agents/sdk-typescript/pull/1024)

## First-time contributors
- @mathpal (#967)
