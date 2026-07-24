# Harness Python v1.50.0

Released 2026-07-24
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.50.0 · Package: https://pypi.org/project/strands-agents/1.50.0/

## Features
- add ExecuteToolStage with middleware-initiated interrupts [hooks, tool] (https://github.com/strands-agents/harness-sdk/pull/3233)
- configurable retry exceptions (#1597) [devx, agent] (https://github.com/strands-agents/harness-sdk/pull/3340)
- propose bidi webrtc design [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3386)
- add http\_request to strands-py [tool] (https://github.com/strands-agents/harness-sdk/pull/3395)
- add stop tool [tool, agent] (https://github.com/strands-agents/harness-sdk/pull/3397)
- add sleep tool [tool] (https://github.com/strands-agents/harness-sdk/pull/3393)

## Fixes
- make the release pip-audit step actually run (https://github.com/strands-agents/harness-sdk/pull/3335)
- replay assistant text history as valid string-content input in Responses adapters [devx, model] (https://github.com/strands-agents/harness-sdk/pull/3399)
- keep fan-in node out of resume while a parallel sibling is in-flight [multiagent] (https://github.com/strands-agents/harness-sdk/pull/3390)
- reject keys for s3 storage if not configured [persistence] (https://github.com/strands-agents/harness-sdk/pull/3411)
- verify aws region (https://github.com/strands-agents/harness-sdk/pull/3412)
- send llama.cpp sampler params at the top level, not under extra\_body [model] (https://github.com/strands-agents/harness-sdk/pull/3423)
- preserve shared context and cumulative accounting across serialize/deserialize [context, multiagent] (https://github.com/strands-agents/harness-sdk/pull/3396)
- surface Responses stream failures [model] (https://github.com/strands-agents/harness-sdk/pull/3427)

## Other
- merge strands-agents/mcp-server into monorepo [mcp] (https://github.com/strands-agents/harness-sdk/pull/3300)
- replace duplicated examples guide with reference pointer (https://github.com/strands-agents/harness-sdk/pull/3288)
- bump brace-expansion from 5.0.6 to 5.0.7 (https://github.com/strands-agents/harness-sdk/pull/3370)
- refactor TestMemoryStore to use the unified storage interface (https://github.com/strands-agents/harness-sdk/pull/3260)
- bump body-parser from 2.2.2 to 2.3.0 (https://github.com/strands-agents/harness-sdk/pull/3387)
- bump actions/setup-python from 6 to 7 (https://github.com/strands-agents/harness-sdk/pull/3352)
- bump astral-sh/setup-uv from 8.3.0 to 9.0.0 (https://github.com/strands-agents/harness-sdk/pull/3407)
- bump pypa/gh-action-pypi-publish from 1.14.0 to 1.14.1 (https://github.com/strands-agents/harness-sdk/pull/3406)
