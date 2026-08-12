# Harness Python v1.52.0

Released 2026-08-12
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.52.0 · Package: https://pypi.org/project/strands-agents/1.52.0/

## Features
- add AgentStreamStage with middleware-initiated interrupts [hooks, hil] (https://github.com/strands-agents/harness-sdk/pull/3594)
- add top level storage [persistence, agent] (https://github.com/strands-agents/harness-sdk/pull/3743)
- add ModelRouter and accept it via Agent(model=) [model, agent] (https://github.com/strands-agents/harness-sdk/pull/3474)

## Fixes
- send document content as file\_data on the Responses API [model] (https://github.com/strands-agents/harness-sdk/pull/3576)
- honor a cache point placed in the last user message [model] (https://github.com/strands-agents/harness-sdk/pull/3677)
- update bidi google-genai version floor [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3740)
- keep the pre-rename name on the deprecated bash aliases [devx, tool] (https://github.com/strands-agents/harness-sdk/pull/3751)
- move count\_tokens fixture off retired gemini-2.0-flash [model] (https://github.com/strands-agents/harness-sdk/pull/3755)
- report retrieval failures as tool errors (https://github.com/strands-agents/harness-sdk/pull/3680)
- send tool-result documents as file\_data, not file\_url [model, tool] (https://github.com/strands-agents/harness-sdk/pull/3674)
- emit thought signature as its own reasoning delta [model] (https://github.com/strands-agents/harness-sdk/pull/3306)
- move count\_tokens fixture to gemini-3.1-flash-lite [model] (https://github.com/strands-agents/harness-sdk/pull/3763)
- retry incompatible tool-result turns [model] (https://github.com/strands-agents/harness-sdk/pull/3622)
- log warning when tool input JSON is malformed [tool, agent] (https://github.com/strands-agents/harness-sdk/pull/2054)

## Other
- bump actions/upload-artifact from 6.0.0 to 7.0.1 (https://github.com/strands-agents/harness-sdk/pull/3702)
- bump actions/setup-python from 6.3.0 to 7.0.0 (https://github.com/strands-agents/harness-sdk/pull/3701)
- bump actions/download-artifact from 6.0.0 to 8.0.1 (https://github.com/strands-agents/harness-sdk/pull/3703)
- bump dorny/paths-filter from 4.0.2 to 4.0.3 (https://github.com/strands-agents/harness-sdk/pull/3729)
- skipped mcp otel instrumentation for mcp v2 [mcp, otel] (https://github.com/strands-agents/harness-sdk/pull/3611)
- retry inconclusive Mantle routing probes [model] (https://github.com/strands-agents/harness-sdk/pull/3747)
- add prescriptive guidance for keeping cognitive complexity low (https://github.com/strands-agents/harness-sdk/pull/3741)
- post release notes to the announcements discussion board (https://github.com/strands-agents/harness-sdk/pull/3707)
- treat bash as its own deprecated tool [tool] (https://github.com/strands-agents/harness-sdk/pull/3756)
- link catalog and standalone page changes in the preview comment (https://github.com/strands-agents/harness-sdk/pull/3774)
