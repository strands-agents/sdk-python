# Harness Python v1.6.0

Released 2025-08-26
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.6.0 · Package: https://pypi.org/project/strands-agents/1.6.0/

## Features
- support A2A FileParts and DataParts [a2a] (https://github.com/strands-agents/sdk-python/pull/596)
- Add \_\_call\_\_ implementation to MultiAgentBase [multiagent] (https://github.com/strands-agents/sdk-python/pull/645)
- Add support for agent invoke with no input, or Message input [agent] (https://github.com/strands-agents/sdk-python/pull/653)

## Fixes
- fix non-serializable parameter of agent from toolUse block [agent] (https://github.com/strands-agents/sdk-python/pull/568)
- add system\_prompt to structured\_output\_span before adding input\_messages (https://github.com/strands-agents/sdk-python/pull/709)
- prevent path traversal for message\_id in file\_session\_manager (https://github.com/strands-agents/sdk-python/pull/728)
- Add AgentInput TypeAlias (https://github.com/strands-agents/sdk-python/pull/738)
- Move AgentInput to types submodule (https://github.com/strands-agents/sdk-python/pull/746)

## Other
- Add .DS\_Store to .gitignore (https://github.com/strands-agents/sdk-python/pull/681)
- update pre-commit requirement from \<4.2.0,\>=3.2.0 to \>=3.2.0,\<4.4.0 (https://github.com/strands-agents/sdk-python/pull/706)
- update ruff requirement from \<0.5.0,\>=0.4.4 to \>=0.4.4,\<0.13.0 (https://github.com/strands-agents/sdk-python/pull/704)
- update pytest-asyncio requirement from \<0.27.0,\>=0.26.0 to \>=0.26.0,\<1.2.0 (https://github.com/strands-agents/sdk-python/pull/708)
- Update pydantic minimum version (https://github.com/strands-agents/sdk-python/pull/723)
- tool executors [tool] (https://github.com/strands-agents/sdk-python/pull/658)
- bump actions/checkout from 4 to 5 (https://github.com/strands-agents/sdk-python/pull/711)
- bump actions/download-artifact from 4 to 5 (https://github.com/strands-agents/sdk-python/pull/712)
- update pytest-cov requirement from \<5.0.0,\>=4.1.0 to \>=4.1.0,\<7.0.0 (https://github.com/strands-agents/sdk-python/pull/705)
- @dependabot\[bot\] made their first contribution (https://github.com/strands-agents/sdk-python/pull/706)

## First-time contributors
- @vawsgit (#681)
- @chengweitsai (#709)
