# Harness Python v1.21.0

Released 2026-01-02
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.21.0 · Package: https://pypi.org/project/strands-agents/1.21.0/

## Features
- support passing additional keyword arguments to FastAPI and Starlette constructors [a2a] (https://github.com/strands-agents/sdk-python/pull/1250)
- add replace method to ToolRegistry [tool] (https://github.com/strands-agents/sdk-python/pull/1182)
- add meta field support to MCP tool results [mcp] (https://github.com/strands-agents/sdk-python/pull/1237)
- Add support for web and search result citations (https://github.com/strands-agents/sdk-python/pull/1344)
- add gemini\_tools field to GeminiModel with validation and tests (https://github.com/strands-agents/sdk-python/pull/1050)
- allow custom-client for OpenAIModel and GeminiModel (https://github.com/strands-agents/sdk-python/pull/1366)
- add api check to github workflow (https://github.com/strands-agents/sdk-python/pull/1348)
- add per\_turn parameter to SlidingWindowConversationManager (https://github.com/strands-agents/sdk-python/pull/1374)
- added agent\_invocations (https://github.com/strands-agents/sdk-python/pull/1387)
- allow hooks to retry model invocations on exceptions [hooks] (https://github.com/strands-agents/sdk-python/pull/1405)

## Fixes
- remove unnecessary None from dict.get() calls (https://github.com/strands-agents/sdk-python/pull/956)
- CitationLocation is UnionType, and correctly joining citation chunks when streaming is being used [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/1341)
- prevent double counting of usage metrics [otel] (https://github.com/strands-agents/sdk-python/pull/1327)
- Pass CODECOV\_TOKENS through for code-coverage stats (https://github.com/strands-agents/sdk-python/pull/1385)
- check api breaking change against main (https://github.com/strands-agents/sdk-python/pull/1397)
- support tools returning image content [model] (https://github.com/strands-agents/sdk-python/pull/1079)
- emit deprecation warning only when deprecated aliases are accessed (https://github.com/strands-agents/sdk-python/pull/1380)

## Other
- Add issue-responder action agent [agent] (https://github.com/strands-agents/sdk-python/pull/1319)
- Expose Status from .base for easier imports (https://github.com/strands-agents/sdk-python/pull/1356)
- Port PR guidelines from sdk-typescript (https://github.com/strands-agents/sdk-python/pull/1373)
- bump actions/checkout from 5 to 6 (https://github.com/strands-agents/sdk-python/pull/1222)
- update pytest-asyncio requirement from \<1.3.0,\>=1.0.0 to \>=1.0.0,\<1.4.0 (https://github.com/strands-agents/sdk-python/pull/1166)
- bump actions/upload-artifact from 4 to 6 (https://github.com/strands-agents/sdk-python/pull/1332)
- bump actions/download-artifact from 5 to 7 (https://github.com/strands-agents/sdk-python/pull/1333)
- update pre-commit requirement from \<4.4.0,\>=3.2.0 to \>=3.2.0,\<4.6.0 (https://github.com/strands-agents/sdk-python/pull/1242)
- bump aws-actions/configure-aws-credentials from 4 to 5 (https://github.com/strands-agents/sdk-python/pull/1352)
- update ruff requirement from \<0.14.0,\>=0.13.0 to \>=0.13.0,\<0.15.0 (https://github.com/strands-agents/sdk-python/pull/1004)
- bump astral-sh/setup-uv from 6 to 7 (https://github.com/strands-agents/sdk-python/pull/1390)
- bump actions/checkout from 5 to 6 (https://github.com/strands-agents/sdk-python/pull/1389)
- Port TypeScript agents into Python (https://github.com/strands-agents/sdk-python/pull/1403)

## First-time contributors
- @snooyen (#1250)
- @ericfzhu (#1341)
- @rajib76 (#1327)
- @danilop (#1344)
- @pshiko (#1050)
- @jsamuel1 (#1380)
