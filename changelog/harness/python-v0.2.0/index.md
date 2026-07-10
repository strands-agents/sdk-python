# Harness Python v0.2.0

Released 2025-07-02
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v0.2.0 · Package: https://pypi.org/project/strands-agents/0.2.0/

## Features
- Add reasoning content for openai model provider [model] (https://github.com/strands-agents/sdk-python/pull/187)
- tools as skills [a2a] (https://github.com/strands-agents/sdk-python/pull/287)
- Add Mistral model support to strands [model] (https://github.com/strands-agents/sdk-python/pull/284)
- add debug logging for model converse requests (https://github.com/strands-agents/sdk-python/pull/297)
- Add reproduction test for #320 (https://github.com/strands-agents/sdk-python/pull/322)
- Agent State [agent] (https://github.com/strands-agents/sdk-python/pull/292)

## Fixes
- correcting incorrect docstring in tracer.py - non-existing argument documented (https://github.com/strands-agents/sdk-python/pull/293)
- Fix docs warnings (https://github.com/strands-agents/sdk-python/pull/303)
- Migrate Mistral structured\_output to an iterator [model] (https://github.com/strands-agents/sdk-python/pull/305)

## Other
- iterative event loop (https://github.com/strands-agents/sdk-python/pull/268)
- Add additional exception information for common bedrock errors [model] (https://github.com/strands-agents/sdk-python/pull/290)
- iterative structured output [structured-output] (https://github.com/strands-agents/sdk-python/pull/291)
- tools - do not remove $defs [tool] (https://github.com/strands-agents/sdk-python/pull/294)
- refactor tracer (https://github.com/strands-agents/sdk-python/pull/286)
- iterative agent [agent] (https://github.com/strands-agents/sdk-python/pull/295)
- Use region from boto3 session when possible [sessions] (https://github.com/strands-agents/sdk-python/pull/299)
- update spanKind and attributes for tokens (https://github.com/strands-agents/sdk-python/pull/296)
- remove kwargs spread after agent call [agent] (https://github.com/strands-agents/sdk-python/pull/289)
- allow custom tracer\_provider and chain setup (https://github.com/strands-agents/sdk-python/pull/316)
- stop passing around callback handler (https://github.com/strands-agents/sdk-python/pull/323)
- Remove unused code (https://github.com/strands-agents/sdk-python/pull/326)
- updated semantic conventions on Generative AI spans (https://github.com/strands-agents/sdk-python/pull/319)
- Consolidate agent state unit tests [agent] (https://github.com/strands-agents/sdk-python/pull/334)
- Remove FunctionTool as a breaking change (https://github.com/strands-agents/sdk-python/pull/325)
- executor - run tools - yield [tool] (https://github.com/strands-agents/sdk-python/pull/328)

## First-time contributors
- @siddhantwaghjale (#284)
- @RingoIngo2 (#297)
