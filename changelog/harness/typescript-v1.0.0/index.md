# Harness TypeScript v1.0.0

Released 2026-04-30
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v1.0.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.0.0

## Features
- add prepack script so git installs build dist/ (https://github.com/strands-agents/sdk-typescript/pull/874)
- add mcp tool result multimodal support [mcp] (https://github.com/strands-agents/sdk-typescript/pull/865)
- add concise inline comments in the wit contract (https://github.com/strands-agents/sdk-typescript/pull/878)
- add countTokens() heuristic to Model base class (https://github.com/strands-agents/sdk-typescript/pull/853)
- add elicitation callback support [mcp] (https://github.com/strands-agents/sdk-typescript/pull/876)
- make tool result mutable [tool] (https://github.com/strands-agents/sdk-typescript/pull/907)
- add cancellation support to BeforeInvocationEvent and BeforeModelCallEvent (https://github.com/strands-agents/sdk-typescript/pull/908)
- centralize model defaults; emit warnings when defaults are used… (https://github.com/strands-agents/sdk-typescript/pull/909)
- estimate input tokens before model calls (https://github.com/strands-agents/sdk-typescript/pull/890)
- override countTokens with native token counting for supported providers (https://github.com/strands-agents/sdk-typescript/pull/886)
- add concurrent tool execution [tool] (https://github.com/strands-agents/sdk-typescript/pull/854)
- add bridge in wasm for conversation manager (https://github.com/strands-agents/sdk-typescript/pull/880)
- add invocation state; apply to all events (https://github.com/strands-agents/sdk-typescript/pull/887)
- add openai-responses model provider and stateful model support [model] (https://github.com/strands-agents/sdk-typescript/pull/820)
- add optional session token to browser-agent Bedrock settings [model] (https://github.com/strands-agents/sdk-typescript/pull/960)
- concurrent tool execution strategy by default [tool] (https://github.com/strands-agents/sdk-typescript/pull/970)

## Fixes
- add version field to root package.json so downstream file: insta… (https://github.com/strands-agents/sdk-typescript/pull/875)
- remove internal ProviderTokenCountError from public exports (https://github.com/strands-agents/sdk-typescript/pull/937)
- change token counting fallback log from warn to debug (https://github.com/strands-agents/sdk-typescript/pull/942)
- fix transport type cast for StreamableHTTPClientTransport [mcp] (https://github.com/strands-agents/sdk-typescript/pull/939)
- add prepare script to examples for standalone install (https://github.com/strands-agents/sdk-typescript/pull/961)
- include README and LICENSE files in published npm package (https://github.com/strands-agents/sdk-typescript/pull/969)

## Other
- update wasm content with guides (https://github.com/strands-agents/sdk-typescript/pull/879)
