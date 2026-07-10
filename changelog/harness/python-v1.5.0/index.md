# Harness Python v1.5.0

Released 2025-08-19
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.5.0 · Package: https://pypi.org/project/strands-agents/1.5.0/

## Features
- Add configuration option to MCP Client for server init timeout [mcp] (https://github.com/strands-agents/sdk-python/pull/657)
- add structured\_output\_span (https://github.com/strands-agents/sdk-python/pull/655)
- expose tool\_use and agent through ToolContext to decorated tools [tool] (https://github.com/strands-agents/sdk-python/pull/557)
- add cached token metrics support for Amazon Bedrock [model] (https://github.com/strands-agents/sdk-python/pull/531)

## Fixes
- Properly handle prompt=None & avoid agent hanging [agent] (https://github.com/strands-agents/sdk-python/pull/643)
- only set signature in message if signature was provided by the model (https://github.com/strands-agents/sdk-python/pull/682)
- Add openai dependency to sagemaker dependency group [model] (https://github.com/strands-agents/sdk-python/pull/678)
- append blank text content if assistant content is empty (https://github.com/strands-agents/sdk-python/pull/677)

## Other
- feature(graph): Allow cyclic graphs [multiagent] (https://github.com/strands-agents/sdk-python/pull/497)
- request to include code snippet section (https://github.com/strands-agents/sdk-python/pull/654)
- litellm - set 1.73.1 as minimum version [model] (https://github.com/strands-agents/sdk-python/pull/668)
- session manager - prevent file path injection [sessions] (https://github.com/strands-agents/sdk-python/pull/680)
- Have \[all\] group reference the other optional dependency groups by name (https://github.com/strands-agents/sdk-python/pull/674)

## First-time contributors
- @fhwilton55 (#657)
- @oaltagar-aws (#531)
