# Evals v1.2.0

Released 2026-08-21
Release: https://github.com/strands-agents/evals/releases/tag/v1.2.0 · Package: https://pypi.org/project/strands-agents-evals/1.2.0/

## Features
- add Claude Agents OpenInference integration tests [tracing] (https://github.com/strands-agents/evals/pull/353)
- add skill-level evaluators for skill-equipped agents [evaluators] (https://github.com/strands-agents/evals/pull/330)
- add OpenAI Agents SDK support to OpenInference mapper [tracing] (https://github.com/strands-agents/evals/pull/366)

## Fixes
- select root agent span by earliest start\_time in multi-agent traces [tracing] (https://github.com/strands-agents/evals/pull/371)
- reduce flakiness for integration tests targeting new session mappers [evaluators] (https://github.com/strands-agents/evals/pull/374)
- change bridge\_parent\_gaps to return new spans instead of mutating in place [tracing] (https://github.com/strands-agents/evals/pull/375)

## Other
- update mypy requirement from \<2.0.0 to \<3.0.0 (https://github.com/strands-agents/evals/pull/368)
- update opentelemetry-instrumentation-langchain requirement from \<0.62.0,\>=0.40.0 to \>=0.40.0,\<0.63.0 (https://github.com/strands-agents/evals/pull/369)
