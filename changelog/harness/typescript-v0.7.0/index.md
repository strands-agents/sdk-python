# Harness TypeScript v0.7.0

Released 2026-03-19
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v0.7.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/0.7.0

## Features
- add guardLatestUserMessage guardrail option [model] (https://github.com/strands-agents/sdk-typescript/pull/635)
- implement Plugin system to replace HookProvider (https://github.com/strands-agents/sdk-typescript/pull/619)
- add A2A protocol support with AgentBase interface [a2a] (https://github.com/strands-agents/sdk-typescript/pull/601)
- add otel meter [otel] (https://github.com/strands-agents/sdk-typescript/pull/655)
- make Swarm start optional, defaulting to first node [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/657)
- add promptcaching for bedrock model provider [model] (https://github.com/strands-agents/sdk-typescript/pull/595)
- add agents-as-tools example [tool] (https://github.com/strands-agents/sdk-typescript/pull/662)
- replace agentId with id and add id/name/description to AgentBase (https://github.com/strands-agents/sdk-typescript/pull/663)
- support documentblock, imageblock, videoblock in model providers that support it (https://github.com/strands-agents/sdk-typescript/pull/576)
- strongly type the conversation-manager (https://github.com/strands-agents/sdk-typescript/pull/664)
- add MultiAgentState to remaining multi-agent streaming events [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/661)
- align S3 location pattern with Python SDK (https://github.com/strands-agents/sdk-typescript/pull/679)
- add TTFB metric, Langfuse detection, system prompt on chat spans [otel] (https://github.com/strands-agents/sdk-typescript/pull/681)

## Fixes
- delete package-lock.json (https://github.com/strands-agents/sdk-typescript/pull/649)
- fix build errors locally and on actions (https://github.com/strands-agents/sdk-typescript/pull/653)
- migrate plugins to be an interface (https://github.com/strands-agents/sdk-typescript/pull/654)
- resolve peer dependency type errors for consumers with skipLibCheck: false (https://github.com/strands-agents/sdk-typescript/pull/671)
- export LocalAgent and MultiAgent types for plugin authors [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/683)
- narrow multi-agent input type to exclude Message\[\] and MessageData\[\] [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/684)
- fix export type bug (https://github.com/strands-agents/sdk-typescript/pull/674)
- fix model sliently overwrites syntaxerror when both maxtoken& syntax occur (https://github.com/strands-agents/sdk-typescript/pull/680)
- fix file editor replace bug (https://github.com/strands-agents/sdk-typescript/pull/688)
- fix agent retry pass in same arg [agent] (https://github.com/strands-agents/sdk-typescript/pull/687)

## Other
- add concrete metric assertions and usage support to MockMessageModel (https://github.com/strands-agents/sdk-typescript/pull/644)
- add multi-agent orchestration documentation and examples [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/648)
- extract MIME type utilities into dedicated mime module (https://github.com/strands-agents/sdk-typescript/pull/656)
- widen AgentNode and orchestrators to accept AgentBase with type discriminators (https://github.com/strands-agents/sdk-typescript/pull/665)
- make StateSerializable use symbols for private api implementations (https://github.com/strands-agents/sdk-typescript/pull/667)
- rename vended tools modules from snake\_case to kebab-case [tool] (https://github.com/strands-agents/sdk-typescript/pull/672)
- split agent interfaces into InvokableAgent and LocalAgent, rename MultiAgentBase to MultiAgent [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/670)
- document NodeStreamUpdateInnerEvent source values (https://github.com/strands-agents/sdk-typescript/pull/677)
- remove v1 issue template (https://github.com/strands-agents/sdk-typescript/pull/686)
- rename AppState to StateStore and Agent.state to Agent.appState [agent] (https://github.com/strands-agents/sdk-typescript/pull/685)
