# Harness Python v1.4.0

Released 2025-08-08
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.4.0 · Package: https://pypi.org/project/strands-agents/1.4.0/

## Features
- Add additional intructions for contributors to find issues that are ready to be worked on (https://github.com/strands-agents/sdk-python/pull/595)
- configurable request handler [a2a] (https://github.com/strands-agents/sdk-python/pull/601)

## Fixes
- added mcp tracing context propagation [otel] (https://github.com/strands-agents/sdk-python/pull/569)
- ensure tool\_use content blocks are valid after max\_tokens to prevent unrecoverable state (https://github.com/strands-agents/sdk-python/pull/607)
- do not modify conversation\_history when prompt is passed [structured-output] (https://github.com/strands-agents/sdk-python/pull/628)

## Other
- Change max\_tokens type to int to match Anthropic API [model] (https://github.com/strands-agents/sdk-python/pull/588)
- update host per AppSec recommendation [a2a] (https://github.com/strands-agents/sdk-python/pull/619)

## First-time contributors
- @vinc3m1 (#588)
