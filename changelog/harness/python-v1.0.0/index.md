# Harness Python v1.0.0

Released 2025-07-15
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.0.0 · Package: https://pypi.org/project/strands-agents/1.0.0/

## Features
- add pagination to mcp\_client list\_tools\_sync (https://github.com/strands-agents/sdk-python/pull/436)
- Graph - support multi-modal inputs [multiagent] (https://github.com/strands-agents/sdk-python/pull/430)
- redact content from a message in a session [sessions] (https://github.com/strands-agents/sdk-python/pull/446)
- added swarm and graph spans [multiagent] (https://github.com/strands-agents/sdk-python/pull/451)
- Store conversation manager in session [sessions] (https://github.com/strands-agents/sdk-python/pull/441)
- introduce Swarm multi-agent orchestrator [multiagent] (https://github.com/strands-agents/sdk-python/pull/416)
- add Swarm tracing [multiagent] (https://github.com/strands-agents/sdk-python/pull/461)
- Expose OpenTelemetry exporter init arguments in API [otel] (https://github.com/strands-agents/sdk-python/pull/365)
- Add kwargs to session interfaces for future extensibility [sessions] (https://github.com/strands-agents/sdk-python/pull/464)

## Fixes
- session manager tracks all agent last message [sessions] (https://github.com/strands-agents/sdk-python/pull/455)
- Fix session manager agent init [sessions] (https://github.com/strands-agents/sdk-python/pull/458)
- Plumb system\_prompt through to structured\_output [structured-output] (https://github.com/strands-agents/sdk-python/pull/466)
- Fix various docstring issues (https://github.com/strands-agents/sdk-python/pull/469)
- raise ValueError for unsupported Graph and Swarm agent features [multiagent] (https://github.com/strands-agents/sdk-python/pull/472)

## Other
- configurable host and port and remove excessive logging [a2a] (https://github.com/strands-agents/sdk-python/pull/423)
- models - bedrock - remove signaling [model] (https://github.com/strands-agents/sdk-python/pull/429)
- deps(a2a): upper bound a2a sdk dep [a2a] (https://github.com/strands-agents/sdk-python/pull/432)
- models - ollama - init async client per request [model] (https://github.com/strands-agents/sdk-python/pull/433)
- models - mistral - init client on every request [model] (https://github.com/strands-agents/sdk-python/pull/434)
- models - ollama - clean up in tests [model] (https://github.com/strands-agents/sdk-python/pull/435)
- Session persistence [sessions] (https://github.com/strands-agents/sdk-python/pull/302)
- update span names [otel] (https://github.com/strands-agents/sdk-python/pull/440)
- models - openai - null usage [model] (https://github.com/strands-agents/sdk-python/pull/442)
- upper bound deps + remove from multiagent submodule [a2a] (https://github.com/strands-agents/sdk-python/pull/447)
- Expand additional $refs for structured\_output [structured-output] (https://github.com/strands-agents/sdk-python/pull/439)
- docstrings - fix formatting (https://github.com/strands-agents/sdk-python/pull/456)
- add kwargs to multiagent interfaces [multiagent] (https://github.com/strands-agents/sdk-python/pull/454)
- multiagent - use invoke\_async instead of stream\_async [multiagent] (https://github.com/strands-agents/sdk-python/pull/463)
- correct naming in registry.py (https://github.com/strands-agents/sdk-python/pull/425)
- Update default model to be Claude 4 Sonnet (https://github.com/strands-agents/sdk-python/pull/467)
- Swarm - Remove unnecessary complete\_swarm\_task tool [multiagent] (https://github.com/strands-agents/sdk-python/pull/473)
- remove preview from README.md (https://github.com/strands-agents/sdk-python/pull/459)

## First-time contributors
- @mrtj (#365)
- @akshseh (#425)
