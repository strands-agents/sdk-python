# Harness Python v1.47.0

Released 2026-07-10
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.47.0 · Package: https://pypi.org/project/strands-agents/1.47.0/

## Features
- map labels to native issue type and language field (https://github.com/strands-agents/harness-sdk/pull/2984)
- add durable identifiers to messages [sessions] (https://github.com/strands-agents/harness-sdk/pull/2836)
- publish TypeScript integ test metrics to CloudWatch (https://github.com/strands-agents/harness-sdk/pull/3134)
- add continue\_on\_error to MCP client [devx, mcp] (https://github.com/strands-agents/harness-sdk/pull/3101)
- added span redaction [otel] (https://github.com/strands-agents/harness-sdk/pull/3111)

## Fixes
- validate the AWS region before building the Nova Sonic endpoint URL [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/2990)
- remove duplicate client creation in Nova Sonic start() [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3124)
- fix typo and inconsistent error messages across model providers [devx, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3125)
- declarative rebuild for \_fix\_broken\_tool\_use [context, sessions] (https://github.com/strands-agents/harness-sdk/pull/3119)
- export BidiConnectionRestartEvent and add 8kHz sample rate support [devx, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3127)
- encode bytes in SessionAgent.to\_dict() for JSON serialization [persistence, sessions] (https://github.com/strands-agents/harness-sdk/pull/3117)
- harden npm lifecycle scripts for best practices (https://github.com/strands-agents/harness-sdk/pull/3128)
- rename LocalMemoryStore to TestMemoryStore (https://github.com/strands-agents/harness-sdk/pull/3123)
- handle tool usage after reasoning content [model, tool] (https://github.com/strands-agents/harness-sdk/pull/1647)
- handle tool use metadata in contentBlockDelta for non-standard models [model, agent] (https://github.com/strands-agents/harness-sdk/pull/2077)

## Other
- add changelog generator and sync workflow (https://github.com/strands-agents/harness-sdk/pull/2765)
- fixed typescript release-workflow not running integ tests (https://github.com/strands-agents/harness-sdk/pull/3126)
- bump peter-evans/create-pull-request from 7.0.11 to 8.1.1 (https://github.com/strands-agents/harness-sdk/pull/3135)
- improve agent guidance on issue references in regression tests (https://github.com/strands-agents/harness-sdk/pull/3146)
- route message appends through Agent.\_append\_messages [agent] (https://github.com/strands-agents/harness-sdk/pull/3131)
- tweak pr-writer skill to be more concise (https://github.com/strands-agents/harness-sdk/pull/3148)
- update litellm requirement from \<=1.91.0,\>=1.75.9 to \>=1.75.9,\<=1.91.1 in /strands-py (https://github.com/strands-agents/harness-sdk/pull/3142)
- relax litellm upper bound to \<2.0.0 [model] (https://github.com/strands-agents/harness-sdk/pull/3149)
