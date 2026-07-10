# Harness Python v1.9.0

Released 2025-09-17
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.9.0 · Package: https://pypi.org/project/strands-agents/1.9.0/

## Features
- add cache usage metrics to OpenTelemetry spans [otel] (https://github.com/strands-agents/sdk-python/pull/825)
- Make entry point configurable [multiagent] (https://github.com/strands-agents/sdk-python/pull/851)
- add automated issue auto-close workflows with dry-run testing (https://github.com/strands-agents/sdk-python/pull/832)

## Fixes
- Add type to tool\_input (https://github.com/strands-agents/sdk-python/pull/854)
- Clean up pyproject.toml (https://github.com/strands-agents/sdk-python/pull/844)
- Updating documentation in decorator.py (https://github.com/strands-agents/sdk-python/pull/852)
- correctly label tool result messages in OpenTelemetry events [tool] (https://github.com/strands-agents/sdk-python/pull/839)
- litellm structured\_output test with more descriptive model [model] (https://github.com/strands-agents/sdk-python/pull/871)
- auto cleanup on exceptions occurring in \_\_enter\_\_ [mcp] (https://github.com/strands-agents/sdk-python/pull/833)
- do not verify \_background\_session is present in stop() [mcp] (https://github.com/strands-agents/sdk-python/pull/876)

## Other
- improve docstring formatting (https://github.com/strands-agents/sdk-python/pull/846)
- bump actions/setup-python from 5 to 6 (https://github.com/strands-agents/sdk-python/pull/796)
- bump actions/github-script from 7 to 8 (https://github.com/strands-agents/sdk-python/pull/801)
- bump aws-actions/configure-aws-credentials from 4 to 5 (https://github.com/strands-agents/sdk-python/pull/795)
- update ruff requirement from \<0.13.0,\>=0.12.0 to \>=0.12.0,\<0.14.0 (https://github.com/strands-agents/sdk-python/pull/840)
- update openai requirement from \<1.102.0,\>=1.68.0 to \>=1.68.0,\<1.108.0 [model] (https://github.com/strands-agents/sdk-python/pull/827)
- models - openai - use client context [model] (https://github.com/strands-agents/sdk-python/pull/856)
- Feature: Handle Bedrock redactedContent [model] (https://github.com/strands-agents/sdk-python/pull/848)
- models - openai - client context comment [model] (https://github.com/strands-agents/sdk-python/pull/864)
- fix links and imports (https://github.com/strands-agents/sdk-python/pull/837)

## First-time contributors
- @vamgan (#825)
- @waitasecant (#846)
- @prabhuteja12 (#852)
