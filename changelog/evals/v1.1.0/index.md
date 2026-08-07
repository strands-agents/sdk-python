# Evals v1.1.0

Released 2026-08-07
Release: https://github.com/strands-agents/evals/releases/tag/v1.1.0 · Package: https://pypi.org/project/strands-agents-evals/1.1.0/

## Features
- allow custom tools on judge-based evaluators (Trajectory, Output, Multimodal) [evaluators] (https://github.com/strands-agents/evals/pull/324)
- add ADK mapper [tracing] (https://github.com/strands-agents/evals/pull/326)
- add integration tests for ADK mapper [tracing] (https://github.com/strands-agents/evals/pull/339)

## Fixes
- include prior tool results in session\_history for tool-level evaluators [evaluators, tracing] (https://github.com/strands-agents/evals/pull/338)
- scope tools to owning agent in multi-agent traces [evaluators, tracing] (https://github.com/strands-agents/evals/pull/336)
- update ADK multi-agent integ test for single-trace design [tracing] (https://github.com/strands-agents/evals/pull/352)
- reset target session between PAIR/SequentialBreak iterations [redteam] (https://github.com/strands-agents/evals/pull/292)

## Other
- update ruff requirement from \<0.16.0,\>=0.13.0 to \>=0.13.0,\<0.17.0 (https://github.com/strands-agents/evals/pull/327)
