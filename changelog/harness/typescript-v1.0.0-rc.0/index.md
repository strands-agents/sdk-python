# Harness TypeScript v1.0.0-rc.0

Released 2026-03-26
Release: https://github.com/strands-agents/sdk-typescript/releases/tag/v1.0.0-rc.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.0.0-rc.0

## Features
- prevent self-handoffs in Swarm [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/697)
- update default model to Claude Sonnet 4 (claude-sonnet-4-6) (https://github.com/strands-agents/sdk-typescript/pull/692)
- add before tool cancellation support [hooks] (https://github.com/strands-agents/sdk-typescript/pull/696)
- add local traces into agentResult [otel] (https://github.com/strands-agents/sdk-typescript/pull/620)
- add multi-agent traces [multiagent, otel] (https://github.com/strands-agents/sdk-typescript/pull/666)
- add model subpath exports, rename GeminiModel to GoogleModel, and add api field to OpenAIModel (https://github.com/strands-agents/sdk-typescript/pull/711)
- add toJSON() to multiagent and a2a streaming events for wire-safe serialization [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/741)
- add toJSON() to all streaming events for wire-safe serialization [bidirectional-streaming] (https://github.com/strands-agents/sdk-typescript/pull/708)

## Fixes
- gemini model should handle throttling correctly [model] (https://github.com/strands-agents/sdk-typescript/pull/691)
- use logger instead of console log that bypass logging system (https://github.com/strands-agents/sdk-typescript/pull/698)
- remove vi.restoreAllMocks() breaking Anthropic mock in browser tests [model] (https://github.com/strands-agents/sdk-typescript/pull/700)
- swarm maxstep throws when finish normally [multiagent] (https://github.com/strands-agents/sdk-typescript/pull/678)
- correctly restore null systemPrompt in loadSnapshot (https://github.com/strands-agents/sdk-typescript/pull/704)
- add newline after printing agent response [agent] (https://github.com/strands-agents/sdk-typescript/pull/705)
- update anthropic log line to follow structured logging convention [model] (https://github.com/strands-agents/sdk-typescript/pull/706)
- use undefined rather than falsy system prompt check (https://github.com/strands-agents/sdk-typescript/pull/707)
- clarify A2AAgent log message for non-text content stripping (https://github.com/strands-agents/sdk-typescript/pull/718)
- sliding window conversation manager treats windowSize 0 as no-op (https://github.com/strands-agents/sdk-typescript/pull/716)
- standardize log messages to follow structured logging format (https://github.com/strands-agents/sdk-typescript/pull/722)
- update default OpenAI model IDs to current generation [model] (https://github.com/strands-agents/sdk-typescript/pull/723)
- inner node status should propagate (https://github.com/strands-agents/sdk-typescript/pull/726)
- add SessionManager guard rails and widen snapshot types to LocalAgent (https://github.com/strands-agents/sdk-typescript/pull/730)
- add persistence to vended bash tool [tool] (https://github.com/strands-agents/sdk-typescript/pull/738)
- force slidingWindowConversationManager to use user message (https://github.com/strands-agents/sdk-typescript/pull/739)
- guarantee after-events fire during hook errors and stream cleanup [hooks] (https://github.com/strands-agents/sdk-typescript/pull/737)
- move A2AExpressServer to dedicated subpath export for browser compatibility (https://github.com/strands-agents/sdk-typescript/pull/721)
- allow pre-release versions in NPM publish workflow (https://github.com/strands-agents/sdk-typescript/pull/745)
- add --tag latest to npm publish for pre-release versions (https://github.com/strands-agents/sdk-typescript/pull/747)

## Other
- bump uuid from 10.0.0 to 13.0.0 (https://github.com/strands-agents/sdk-typescript/pull/625)
- simplify structured output internals and fix infinite loop bug [structured-output] (https://github.com/strands-agents/sdk-typescript/pull/709)

## First-time contributors
- @notowen333 (#738)
- @agent-of-mkmeral (#708)
