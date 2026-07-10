# Harness TypeScript v1.0.0-rc.4

Released 2026-04-17
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v1.0.0-rc.4 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.0.0-rc.4

## Features
- add swarm+session manager resume logic,unit tests, integration test [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/800)
- support custom ClientFactory in A2AAgent for authenticated requests [a2a] (https://github.com/strands-agents/sdk-typescript/pull/810)
- track agent.messages token size [context] (https://github.com/strands-agents/sdk-typescript/pull/790)
- add graph+session manager integration + tests [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/809)
- add agent skills plugin [agent] (https://github.com/strands-agents/sdk-typescript/pull/807)
- expose metrics/usage on message metadata [otel] (https://github.com/strands-agents/sdk-typescript/pull/815)

## Fixes
- evaluate all incoming edge handlers in Graph.\_findReady [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/804)
- added function replacer for notebook\_tool replace [tool] (https://github.com/strands-agents/sdk-typescript/pull/814)

## Other
- add package-lock.json (https://github.com/strands-agents/sdk-typescript/pull/813)

## First-time contributors
- @cogwirrel (#810)
