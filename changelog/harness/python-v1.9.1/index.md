# Harness Python v1.9.1

Released 2025-09-19
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.9.1 · Package: https://pypi.org/project/strands-agents/1.9.1/

## Features
- decouple Strands ContentBlock and BedrockModel (https://github.com/strands-agents/sdk-python/pull/836)

## Fixes
- Invoke callback handler for structured\_output [structured-output] (https://github.com/strands-agents/sdk-python/pull/857)
- Update prepare to use format instead of test-format (https://github.com/strands-agents/sdk-python/pull/858)
- add explicit permissions to auto-close workflow (https://github.com/strands-agents/sdk-python/pull/893)
- make mcp\_instrumentation idempotent to prevent recursion errors (https://github.com/strands-agents/sdk-python/pull/892)
- Fix github workflow to use fmt instead of hatch run (https://github.com/strands-agents/sdk-python/pull/898)
- make tool\_choice an optional keyword arg instead positional [model] (https://github.com/strands-agents/sdk-python/pull/899)
