# Harness Python v1.23.0

Released 2026-01-21
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.23.0 · Package: https://pypi.org/project/strands-agents/1.23.0/

## Features
- override service name by OTEL\_SERVICE\_NAME env (https://github.com/strands-agents/sdk-python/pull/1400)
- allow steering on AfterModelCallEvents (https://github.com/strands-agents/sdk-python/pull/1429)
- add configurable retry\_strategy for model calls [agent] (https://github.com/strands-agents/sdk-python/pull/1424)
- graduate multiagent hook events from experimental [multiagent] (https://github.com/strands-agents/sdk-python/pull/1498)

## Fixes
- prevent agent hang by checking session closure state [mcp] (https://github.com/strands-agents/sdk-python/pull/1396)
- extract text from citationsContent in AgentResult.\_\_str\_\_ [agent] (https://github.com/strands-agents/sdk-python/pull/1489)
- Swap unit test sleeps with explicit signaling (https://github.com/strands-agents/sdk-python/pull/1497)
- disable thinking mode when forcing tool\_choice [model] (https://github.com/strands-agents/sdk-python/pull/1495)
- use a2a artifact update event [a2a] (https://github.com/strands-agents/sdk-python/pull/1401)
- provide unique toolUseId for gemini models [model] (https://github.com/strands-agents/sdk-python/pull/1201)
- handle missing usage attribute on ModelResponseStream [model] (https://github.com/strands-agents/sdk-python/pull/1520)
- accumulate execution\_time across interrupt/resume cycles [multiagent] (https://github.com/strands-agents/sdk-python/pull/1502)
- reduce flakiness in guardrail redact output test (https://github.com/strands-agents/sdk-python/pull/1505)

## Other
- update sphinx-rtd-theme requirement from \<2.0.0,\>=1.0.0 to \>=1.0.0,\<4.0.0 (https://github.com/strands-agents/sdk-python/pull/1466)
- update websockets requirement from \<16.0.0,\>=15.0.0 to \>=15.0.0,\<17.0.0 (https://github.com/strands-agents/sdk-python/pull/1451)
- Update ruff configuration to apply pyupgrade to modernize python syntax (https://github.com/strands-agents/sdk-python/pull/1336)
- Expose input messages to BeforeInvocationEvent hook [hooks] (https://github.com/strands-agents/sdk-python/pull/1474)
- interrupts - graph - hook based [multiagent] (https://github.com/strands-agents/sdk-python/pull/1478)
- Fix PEP 563 incompatibility with @tool decorated tools [tool] (https://github.com/strands-agents/sdk-python/pull/1494)
- Add parallel reading support to S3SessionManager.list\_messages() (https://github.com/strands-agents/sdk-python/pull/1186)
- gemini - tool\_use\_id\_to\_name - local [model] (https://github.com/strands-agents/sdk-python/pull/1521)
- Nova Sonic 2 support for BidiAgent (https://github.com/strands-agents/sdk-python/pull/1476)

## First-time contributors
- @maxrabin (#1336)
- @tmokmss (#1489)
- @okamototk (#1400)
- @strands-agent (#1495)
- @brycewcole (#1401)
- @CrysisDeu (#1186)
- @AirswitchAsa (#1201)
- @lanazhang (#1476)
