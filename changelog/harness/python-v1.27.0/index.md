# Harness Python v1.27.0

Released 2026-02-19
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.27.0 · Package: https://pypi.org/project/strands-agents/1.27.0/

## Features
- Propagate exceptions to AfterToolCallEvent for decorated tools (#1565) [tool] (https://github.com/strands-agents/sdk-python/pull/1566)
- add conventional commit workflow in PR (https://github.com/strands-agents/sdk-python/pull/1645)
- add concurrent\_invocation\_mode parameter [agent] (https://github.com/strands-agents/sdk-python/pull/1707)
- add add\_hook convenience method for hook callback registration [agent] (https://github.com/strands-agents/sdk-python/pull/1706)

## Fixes
- the A2AAgent returns empty AgentResult content (https://github.com/strands-agents/sdk-python/pull/1675)
- correct output reference for approval-env in integration test (https://github.com/strands-agents/sdk-python/pull/1685)
- update approval env var for strands agent workflows [agent] (https://github.com/strands-agents/sdk-python/pull/1701)
- update allowed roles to include maintainer (https://github.com/strands-agents/sdk-python/pull/1704)
- propagate reasoningSignature on Gemini tool use [model] (https://github.com/strands-agents/sdk-python/pull/1703)
- handle OpenAI model responses with tool calls and no other assistant content [model] (https://github.com/strands-agents/sdk-python/pull/1562)
- Update finalize condition for workflow execution (https://github.com/strands-agents/sdk-python/pull/1708)
- upgrade mcp minimum dependency to 1.23.0 for Tasks support [mcp] (https://github.com/strands-agents/sdk-python/pull/1674)

## Other
- auto run review workflow on maintainer PR (https://github.com/strands-agents/sdk-python/pull/1673)
- bump actions/github-script from 7 to 8 (https://github.com/strands-agents/sdk-python/pull/1699)
- bump amannn/action-semantic-pull-request from 5 to 6 (https://github.com/strands-agents/sdk-python/pull/1684)
- coverage for python 3.14 (https://github.com/strands-agents/sdk-python/pull/1178)
