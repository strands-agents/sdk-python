# Harness Python v1.38.0

Released 2026-04-30
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.38.0 · Package: https://pypi.org/project/strands-agents/1.38.0/

## Features
- preserve CallToolResult.isError flag in MCPToolResult [mcp] (https://github.com/strands-agents/sdk-python/pull/2118)
- add \`count\_token\` method to model with naive estimation using tiktoken [context] (https://github.com/strands-agents/sdk-python/pull/2031)
- add TTL support to CachePoint for prompt caching (https://github.com/strands-agents/sdk-python/pull/1660)
- large tool result offload [tool] (https://github.com/strands-agents/sdk-python/pull/2162)
- override count\_tokens with native token counting for supported providers (https://github.com/strands-agents/sdk-python/pull/2189)
- add ProviderTokenCountError for native token counting failures (https://github.com/strands-agents/sdk-python/pull/2211)
- estimate input tokens before model calls (https://github.com/strands-agents/sdk-python/pull/2221)
- return explicit paths in preview and auto-enable retrieval (https://github.com/strands-agents/sdk-python/pull/2222)
- add strict\_tools config with auto-inject of additional… [model] (https://github.com/strands-agents/sdk-python/pull/2213)

## Fixes
- forward ttl field from CachePoint in \_format\_system\_messages [model] (https://github.com/strands-agents/sdk-python/pull/2153)
- preserve cache points in system prompt during skills inj… (https://github.com/strands-agents/sdk-python/pull/2134)
- generate unique toolUseId instead of reusing tool name [model] (https://github.com/strands-agents/sdk-python/pull/2053)
- use non-interactive flag for Nova Sonic history and system promp… (https://github.com/strands-agents/sdk-python/pull/2188)
- upgrade default model to Claude Sonnet 4.5 [model] (https://github.com/strands-agents/sdk-python/pull/2193)
- handle window\_size=0 and reject negative values (https://github.com/strands-agents/sdk-python/pull/2208)
- change token counting fallback log from warning to debug (https://github.com/strands-agents/sdk-python/pull/2220)
- do not synthesize exception for cancelled tools [tool] (https://github.com/strands-agents/sdk-python/pull/2106)
- update tests to use non-EOL'd model (https://github.com/strands-agents/sdk-python/pull/2226)

## Other
- added warning for default model awareness and is subject to change (https://github.com/strands-agents/sdk-python/pull/2164)
- update litellm requirement from \<=1.82.6,\>=1.75.9 to \>=1.75.9,\<=1.83.13 [model] (https://github.com/strands-agents/sdk-python/pull/2197)
- update pre-commit requirement from \<4.6.0,\>=3.2.0 to \>=3.2.0,\<4.7.0 (https://github.com/strands-agents/sdk-python/pull/2185)
- update style guide for tool spec navigation [tool] (https://github.com/strands-agents/sdk-python/pull/2203)

## First-time contributors
- @Zelys-DFKH (#2118)
- @ElliottJW (#2153)
- @Ratansairohith (#2053)
- @kpx-dev (#1660)
- @prettyprettyprettygood (#2188)
- @SuperMarioYL (#2208)
- @Gastly (#2106)
- @kaghatim (#2213)
