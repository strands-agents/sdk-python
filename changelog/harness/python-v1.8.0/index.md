# Harness Python v1.8.0

Released 2025-09-10
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.8.0 · Package: https://pypi.org/project/strands-agents/1.8.0/

## Features
- improve structured output tool circular reference handling [structured-output] (https://github.com/strands-agents/sdk-python/pull/817)
- add default read timeout to Bedrock config [model] (https://github.com/strands-agents/sdk-python/pull/829)
- add support for Bedrock/Anthropic ToolChoice to structured\_output [model] (https://github.com/strands-agents/sdk-python/pull/720)
- allow callers of swarm and graph to pass kwargs to executors [multiagent] (https://github.com/strands-agents/sdk-python/pull/816)
- add region-aware default model ID for Bedrock [model] (https://github.com/strands-agents/sdk-python/pull/835)

## Fixes
- fix cyclic graph behavior [multiagent] (https://github.com/strands-agents/sdk-python/pull/768)
- filter reasoningContent in Bedrock requests using DeepSeek [model] (https://github.com/strands-agents/sdk-python/pull/652)
- do not block asyncio event loop between retries (https://github.com/strands-agents/sdk-python/pull/805)
- load and register all decorated @tool functions from file path [tool] (https://github.com/strands-agents/sdk-python/pull/742)
- patch litellm bug to honor passing in use\_litellm\_proxy as client\_args [model] (https://github.com/strands-agents/sdk-python/pull/808)

## Other
- Moved tool\_spec retrieval to after the before model invocation callback (https://github.com/strands-agents/sdk-python/pull/786)
- cleanup docs so the yields section renders correctly (https://github.com/strands-agents/sdk-python/pull/820)
- Warn on unknown model configuration properties (https://github.com/strands-agents/sdk-python/pull/819)
- llama.cpp model provider support [model] (https://github.com/strands-agents/sdk-python/pull/585)
- fix(llama.cpp) - add ToolChoice and validation of model config values [model] (https://github.com/strands-agents/sdk-python/pull/838)

## First-time contributors
- @pghazanfari (#786)
- @aryan835-datainflexion (#652)
- @afarntrog (#820)
- @osdemah (#805)
- @Ratish1 (#742)
- @liushang1997 (#720)
- @westonbrown (#585)
