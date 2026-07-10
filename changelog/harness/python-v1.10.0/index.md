# Harness Python v1.10.0

Released 2025-09-29
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.10.0 · Package: https://pypi.org/project/strands-agents/1.10.0/

## Features
- add optional outputSchema support for tool specifications [tool] (https://github.com/strands-agents/sdk-python/pull/818)
- add Gemini model provider [model] (https://github.com/strands-agents/sdk-python/pull/725)
- add supports\_hot\_reload property to PythonAgentTool (https://github.com/strands-agents/sdk-python/pull/928)
- Mark ModelCall and ToolCall events as non-experimental [hooks] (https://github.com/strands-agents/sdk-python/pull/926)
- Create a new HookEvent for Multiagent [multiagent] (https://github.com/strands-agents/sdk-python/pull/925)

## Fixes
- Fix event loop closed error from Gemini asyncio [model] (https://github.com/strands-agents/sdk-python/pull/932)
- Fix mcp timeout issue [mcp] (https://github.com/strands-agents/sdk-python/pull/922)

## Other
- Improve OpenAI error handling [model] (https://github.com/strands-agents/sdk-python/pull/918)
- update sphinx-autodoc-typehints requirement from \<2.0.0,\>=1.12.0 to \>=1.12.0,\<4.0.0 (https://github.com/strands-agents/sdk-python/pull/903)
- update sphinx requirement from \<6.0.0,\>=5.0.0 to \>=5.0.0,\<9.0.0 (https://github.com/strands-agents/sdk-python/pull/904)
- update openai requirement from \<1.108.0,\>=1.68.0 to \>=1.68.0,\<1.110.0 [model] (https://github.com/strands-agents/sdk-python/pull/916)
- update pytest-asyncio requirement from \<1.2.0,\>=1.0.0 to \>=1.0.0,\<1.3.0 (https://github.com/strands-agents/sdk-python/pull/861)

## First-time contributors
- @notgitika (#725)
