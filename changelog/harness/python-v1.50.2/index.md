# Harness Python v1.50.2

Released 2026-07-27
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.50.2 · Package: https://pypi.org/project/strands-agents/1.50.2/

## Features
- add per-call MCP tool cancellation [mcp, hil] (https://github.com/strands-agents/harness-sdk/pull/3402)
- add context manager class design doc [context] (https://github.com/strands-agents/harness-sdk/pull/3307)

## Fixes
- gemini live mp to use updated api [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3424)
- deduplicate inverted-index postings [mcp] (https://github.com/strands-agents/harness-sdk/pull/3417)
- consume reasoning signature per content block [model, agent] (https://github.com/strands-agents/harness-sdk/pull/3472)
- legacy file storage accepts bare filenames and stems [persistence] (https://github.com/strands-agents/harness-sdk/pull/3495)

## Other
- bump google-genai floor to \>=1.67.0 [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3478)
- correct search tool contracts [mcp] (https://github.com/strands-agents/harness-sdk/pull/3456)
- update context offloader comments to deprecate legacy storage [persistence] (https://github.com/strands-agents/harness-sdk/pull/3476)
- remove security features, accept httpx.AsyncClient [devx, tool] (https://github.com/strands-agents/harness-sdk/pull/3491)
