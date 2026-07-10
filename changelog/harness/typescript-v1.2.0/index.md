# Harness TypeScript v1.2.0

Released 2026-05-14
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v1.2.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.2.0

## Features
- refine sliding window coversation manager logic (https://github.com/strands-agents/sdk-typescript/pull/1018)
- add npm-pack test to ci-cd (https://github.com/strands-agents/sdk-typescript/pull/996)
- cache AccessDenied error for count tokens (https://github.com/strands-agents/sdk-typescript/pull/1032)
- handle toolsChanged notifications [mcp] (https://github.com/strands-agents/sdk-typescript/pull/1038)
- WIT-first SDK contract and strands-py 2.0.0a1 rewrite (https://github.com/strands-agents/sdk-typescript/pull/1034)
- expose takeSnapshot and loadSnapshot on Agent [agent] (https://github.com/strands-agents/sdk-typescript/pull/1045)
- add official Discord link (https://github.com/strands-agents/sdk-typescript/pull/1051)
- normalize tool-name and update correct MessageAddedEvent behavior [tool] (https://github.com/strands-agents/sdk-typescript/pull/1048)
- make totalDuration and averageCycleTime O(1) on AgentMetrics (https://github.com/strands-agents/sdk-typescript/pull/1063)
- support multi-agent interrupts; add InterruptEvent [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/1044)
- support url + auth fields on McpClientConfig [mcp] (https://github.com/strands-agents/sdk-typescript/pull/1059)
- forward agent cancel signal to MCP server [mcp] (https://github.com/strands-agents/sdk-typescript/pull/1069)

## Fixes
- npm security audit fix (https://github.com/strands-agents/sdk-typescript/pull/1041)
- align context overflow detection patterns(#894) [context] (https://github.com/strands-agents/sdk-typescript/pull/966)
- default useNativeTokenCount to false (https://github.com/strands-agents/sdk-typescript/pull/1056)
- structured tool output user/assistant bug fix [tool] (https://github.com/strands-agents/sdk-typescript/pull/1049)
- use correct 'citation' delta key for streaming citations in Bedrock provider [model] (https://github.com/strands-agents/sdk-typescript/pull/1058)
- give maintainers auto integ tests (https://github.com/strands-agents/sdk-typescript/pull/1064)
- replace Node 22+ globSync with readdirSync in strands-dev CLI (https://github.com/strands-agents/sdk-typescript/pull/1062)

## Other
- serialized interrupts and structuredOutput as JSON and citationsBlock (https://github.com/strands-agents/sdk-typescript/pull/1043)
- persist guardrails redaction (https://github.com/strands-agents/sdk-typescript/pull/1040)
- update AGENTS.MD (https://github.com/strands-agents/sdk-typescript/pull/1057)

## First-time contributors
- @Luffy2208 (#966)
