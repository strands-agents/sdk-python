# Evals v0.1.14

Released 2026-04-08
Release: https://github.com/strands-agents/evals/releases/tag/v0.1.14 · Package: https://pypi.org/project/strands-agents-evals/0.1.14/

## Features
- add ground truth assertion support to Goal Success Rate evaluator (https://github.com/strands-agents/evals/pull/180)

## Other
- devx to allow for passing the Provider directly to evaluations with creating a wrapper task (https://github.com/strands-agents/evals/pull/183)

### Major Features

#### Ground Truth Assertion Support for Goal Success Rate Evaluator — [PR#180](https://github.com/strands-agents/evals/pull/180)

The `GoalSuccessRateEvaluator` now supports a second evaluation mode: assertion-based evaluation. When `expected_assertion` is provided on the evaluation case, the judge LLM evaluates whether the agent’s behavior satisfies explicit success assertions rather than inferring goals from the conversation. This enables precise, reproducible evaluation with human-authored success criteria.

```python
from strands_evals import Case
from strands_evals.evaluators import GoalSuccessRateEvaluator

# Basic mode (existing) — judge infers goals from conversation
case_basic = Case(
    session_id="session-1",
    goal="Help user book a flight",
)

# Assertion mode (new) — judge evaluates against explicit criteria
case_assertion = Case(
    session_id="session-2",
    goal="Help user book a flight",
    expected_assertion="Agent confirmed departure city, destination, and date before searching for flights",
)

evaluator = GoalSuccessRateEvaluator()
```

Basic mode uses a Yes/No scoring rubric (Yes=1.0, No=0.0). Assertion mode uses SUCCESS/FAILURE scoring (SUCCESS=1.0, FAILURE=0.0).

#### Provider as Task Callable — [PR#183](https://github.com/strands-agents/evals/pull/183)

`TraceProvider` now exposes an `as_task()` method that returns a task callable, eliminating the need to write wrapper functions when running evaluations against traced sessions.

```python
from strands_evals.providers import TraceProvider

provider = TraceProvider(...)

# Before: manual wrapper
task = lambda case: provider.get_evaluation_data(case.session_id)

# After: built-in convenience
task = provider.as_task()
```
