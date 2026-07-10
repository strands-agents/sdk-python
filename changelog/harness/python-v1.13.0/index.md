# Harness Python v1.13.0

Released 2025-10-17
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.13.0 · Package: https://pypi.org/project/strands-agents/1.13.0/

## Features
- replace kwargs with invocation\_state in agent APIs [agent] (https://github.com/strands-agents/sdk-python/pull/966)
- updated semantic conventions, added timeToFirstByteMs into spans and metrics [otel] (https://github.com/strands-agents/sdk-python/pull/997)
- Support adding exception notes for Python 3.10 (https://github.com/strands-agents/sdk-python/pull/1034)

## Fixes
- validate ToolContext parameter name and raise clear error (https://github.com/strands-agents/sdk-python/pull/1028)

## Other
- added gen\_ai.tool.description and gen\_ai.tool.json\_schema [otel] (https://github.com/strands-agents/sdk-python/pull/1027)
- integ tests - fix flaky structured output test [structured-output] (https://github.com/strands-agents/sdk-python/pull/1030)
- hooks - before tool call event - interrupt [tool] (https://github.com/strands-agents/sdk-python/pull/987)
- multiagents - temporarily raise exception when interrupted (https://github.com/strands-agents/sdk-python/pull/1038)
- interrupts - decorated tools [tool] (https://github.com/strands-agents/sdk-python/pull/1041)
