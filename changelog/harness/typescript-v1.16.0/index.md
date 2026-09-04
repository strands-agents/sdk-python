# Harness TypeScript v1.16.0

Released 2026-08-31
Release: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.16.0 · Package: https://www.npmjs.com/package/@strands-agents/sdk/v/1.16.0

## Features
- add internal BackgroundTaskManager interface + InProcessTaskManager impl [async, multiagent] (https://github.com/strands-agents/harness-sdk/pull/3876)
- add QmdSearchStrategy with BM25 full-text search [persistence] (https://github.com/strands-agents/harness-sdk/pull/3897)
- add backgroundTasks to Agent [async, agent] (https://github.com/strands-agents/harness-sdk/pull/4026)

## Fixes
- stop str\_replace and insert from rewriting untouched bytes [tool] (https://github.com/strands-agents/harness-sdk/pull/4014)
- close cycle telemetry when the stream iterator exits early [otel] (https://github.com/strands-agents/harness-sdk/pull/4054)

## Other
- align 0016 with shipped ClassifierStrategy (https://github.com/strands-agents/harness-sdk/pull/4046)
