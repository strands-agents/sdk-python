# Harness Python v1.36.0

Released 2026-04-17
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.36.0 · Package: https://pypi.org/project/strands-agents/1.36.0/

## Features
- accept callable hook callbacks in Agent constructor [hooks] (https://github.com/strands-agents/sdk-python/pull/1992)
- add client\_config param and deprecate a2a\_client\_factory [a2a] (https://github.com/strands-agents/sdk-python/pull/2103)
- plumb through cache tokens in metadata events [model] (https://github.com/strands-agents/sdk-python/pull/2116)
- add take\_snapshot() and load\_snapshot() methods [agent] (https://github.com/strands-agents/sdk-python/pull/1948)
- support loading skills from URLs (https://github.com/strands-agents/sdk-python/pull/2091)
- add metadata field to messages for stateful context tracking [context] (https://github.com/strands-agents/sdk-python/pull/2125)
- support request\_state stop\_event\_loop flag [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/1954)

## Fixes
- handle missing optional fields in non-streaming citation conversion [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/2098)
- add common gen\_ai attributes to event loop cycle spans [otel] (https://github.com/strands-agents/sdk-python/pull/1973)
- use per-invocation usage in agent span attributes [otel] (https://github.com/strands-agents/sdk-python/pull/2017)
- clear leaked running loop in MCP client background thread [mcp] (https://github.com/strands-agents/sdk-python/pull/2111)
- preserve Gemini thought\_signature in LiteLLM multi-turn tool calls [model] (https://github.com/strands-agents/sdk-python/pull/2129)
- normalize empty toolResult content arrays in \_format\_bedrock\_messages [model] (https://github.com/strands-agents/sdk-python/pull/2123)
- remove force\_flush in tracer [otel] (https://github.com/strands-agents/sdk-python/pull/2142)

## First-time contributors
- @en-yao (#2017)
- @ghhamel (#2123)
