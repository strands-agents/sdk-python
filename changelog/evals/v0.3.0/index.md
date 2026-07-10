# Evals v0.3.0

Released 2026-06-12
Release: https://github.com/strands-agents/evals/releases/tag/v0.3.0 · Package: https://pypi.org/project/strands-agents-evals/0.3.0/

## Features
- add built-in red teaming support [redteam] (https://github.com/strands-agents/evals/pull/184)
- add chaos resilience evaluators (failure communication, partial completion, recovery strategy) (https://github.com/strands-agents/evals/pull/236)
- add Crescendo multi-turn attack strategy [redteam] (https://github.com/strands-agents/evals/pull/245)
- added strands-evals cli [cli] (https://github.com/strands-agents/evals/pull/243)
- add LLM issue labeler for area and type [cli] (https://github.com/strands-agents/evals/pull/255)
- add Bad Likert Judge multi-turn attack strategy [redteam] (https://github.com/strands-agents/evals/pull/248)

## Fixes
- join all toolResult.content blocks to fix faithfulness false negatives (https://github.com/strands-agents/evals/pull/240)
- correct doc link and clean up issue/PR templates (https://github.com/strands-agents/evals/pull/256)

## Other
- allow importing EvaluationReport from root (https://github.com/strands-agents/evals/pull/238)
- added trace-based evaluators into defaults (https://github.com/strands-agents/evals/pull/244)
- always return flattened report (https://github.com/strands-agents/evals/pull/241)
- bumped strands-agents-version to the latest (https://github.com/strands-agents/evals/pull/246)
- added evaluator name and evaluator\_type for report (https://github.com/strands-agents/evals/pull/249)
- added single case evaluation command [cli] (https://github.com/strands-agents/evals/pull/252)
- add AI contribution guidance to CONTRIBUTING and PR template (https://github.com/strands-agents/evals/pull/257)
- add high-quality PR guidance to AGENTS.md [agent] (https://github.com/strands-agents/evals/pull/258)
- add community and character guidance to AGENTS.md [agent] (https://github.com/strands-agents/evals/pull/261)
- added generate command for experiment generation [cli] (https://github.com/strands-agents/evals/pull/260)

## First-time contributors
- @yeomjiwonyeom (#245)
