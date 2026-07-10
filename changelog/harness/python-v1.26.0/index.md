# Harness Python v1.26.0

Released 2026-02-11
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.26.0 · Package: https://pypi.org/project/strands-agents/1.26.0/

## Features
- Implement basic support for Tasks [mcp] (https://github.com/strands-agents/sdk-python/pull/1475)

## Fixes
- set empty text part data in \`parts\` for \`Artifact\` [multiagent] (https://github.com/strands-agents/sdk-python/pull/1643)
- use model stream to generate summary [bidirectional-streaming] (https://github.com/strands-agents/sdk-python/pull/1653)
- add 'prompt is too long' to context window overflow mes… [model] (https://github.com/strands-agents/sdk-python/pull/1663)
- fix mcp tests [mcp] (https://github.com/strands-agents/sdk-python/pull/1664)

## Other
- bump aws-actions/configure-aws-credentials from 5 to 6 (https://github.com/strands-agents/sdk-python/pull/1632)
- add guidance on using Protocol instead of Callable for extensible interfaces (https://github.com/strands-agents/sdk-python/pull/1637)

## First-time contributors
- @punkyoon (#1643)
- @eladb3 (#1663)
