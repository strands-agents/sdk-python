# Harness Python v1.31.0

Released 2026-03-19
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.31.0 · Package: https://pypi.org/project/strands-agents/1.31.0/

## Features
- pass A2A request context metadata as invocation state [a2a] (https://github.com/strands-agents/sdk-python/pull/1854)
- widen openai dependency to support 2.x for litellm compatibility [model] (https://github.com/strands-agents/sdk-python/pull/1793)

## Fixes
- s3session manager bug [sessions] (https://github.com/strands-agents/sdk-python/pull/1915)
- only evaluate outbound edges from completed nodes [multiagent] (https://github.com/strands-agents/sdk-python/pull/1846)
- always use string content for tool messages [model] (https://github.com/strands-agents/sdk-python/pull/1878)
- typeError when serializing multimodal prompts with binary content in Graph/Swarm session persistence [multiagent] (https://github.com/strands-agents/sdk-python/pull/1870)
- lowercase the python language in code snippet (https://github.com/strands-agents/sdk-python/pull/1929)
- openai repsonses api error handling [model] (https://github.com/strands-agents/sdk-python/pull/1931)

## First-time contributors
- @BV-Venky (#1793)
