# Harness Python v1.29.0

Released 2026-03-04
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.29.0 · Package: https://pypi.org/project/strands-agents/1.29.0/

## Features
- improve tool result truncation strategy [tool] (https://github.com/strands-agents/sdk-python/pull/1756)
- improve plugin creation devex with @hook and @tool decorators [tool] (https://github.com/strands-agents/sdk-python/pull/1740)
- add OpenAI Responses API model implementation [model] (https://github.com/strands-agents/sdk-python/pull/975)

## Fixes
- added latest semantic conventions as span attributes for langfuse [otel] (https://github.com/strands-agents/sdk-python/pull/1768)
- preserve guardrail\_latest\_message wrapping after tool execution [tool] (https://github.com/strands-agents/sdk-python/pull/1658)
- throw exceptions from ConcurrentToolExecutor (#1796) (https://github.com/strands-agents/sdk-python/pull/1797)

## Other
- pin virtualenv to \<21 for hatch bug (https://github.com/strands-agents/sdk-python/pull/1771)
- bump actions/upload-artifact from 6 to 7 (https://github.com/strands-agents/sdk-python/pull/1777)
- bump actions/download-artifact from 7 to 8 (https://github.com/strands-agents/sdk-python/pull/1776)

## First-time contributors
- @austinmw (#1658)
