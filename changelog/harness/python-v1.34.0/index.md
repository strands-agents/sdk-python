# Harness Python v1.34.0

Released 2026-03-31
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.34.0 · Package: https://pypi.org/project/strands-agents/1.34.0/

## Features
- add AgentAsTool (https://github.com/strands-agents/sdk-python/pull/1932)
- auto-wrap Agent instances passed in tools list [tool] (https://github.com/strands-agents/sdk-python/pull/1997)
- emit system prompt on chat spans per GenAI semconv [otel] (https://github.com/strands-agents/sdk-python/pull/1818)
- add support for MCP elicitation -32042 error handling [mcp] (https://github.com/strands-agents/sdk-python/pull/1745)
- add stateful model support for server-side conversation management (https://github.com/strands-agents/sdk-python/pull/2004)
- add built-in tool support for OpenAI Responses API [model] (https://github.com/strands-agents/sdk-python/pull/2011)

## Fixes
- ollama input/output token count [model] (https://github.com/strands-agents/sdk-python/pull/2008)
- handle reasoning content in OpenAIResponsesModel request formatting (https://github.com/strands-agents/sdk-python/pull/2013)

## Other
- remove Cohere from required integ test providers (https://github.com/strands-agents/sdk-python/pull/1967)

## First-time contributors
- @sanjeed5 (#1818)
- @Christian-kam (#1745)
