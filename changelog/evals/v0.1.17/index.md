# Evals v0.1.17

Released 2026-05-08
Release: https://github.com/strands-agents/evals/releases/tag/v0.1.17 · Package: https://pypi.org/project/strands-agents-evals/0.1.17/

## Features
- add multimodal evaluators and prompt templates for image-to-text evaluation (https://github.com/strands-agents/evals/pull/187)
- added analyze\_root\_cause [detectors] (https://github.com/strands-agents/evals/pull/179)
- integrated rca into evaluation workflow [detectors] (https://github.com/strands-agents/evals/pull/210)
- added refusalEvaluator, stereotypingEvaluator, insructionFollowingEvaluator [evaluators] (https://github.com/strands-agents/evals/pull/213)
- add optional tools parameter to ToolSimulator (#208) [tool] (https://github.com/strands-agents/evals/pull/209)

## Fixes
- preserve input order in run\_evaluations\_async (https://github.com/strands-agents/evals/pull/214)
- update default judge model to Claude Sonnet 4.6 [evaluators] (https://github.com/strands-agents/evals/pull/215)

## Other
- included more fields to the RCAItem [detectors] (https://github.com/strands-agents/evals/pull/211)
- updated confidencelevel and diagnose\_trigger to enum [detectors] (https://github.com/strands-agents/evals/pull/212)
- formatting (https://github.com/strands-agents/evals/pull/217)

## First-time contributors
- @sangminwoo (#187)
