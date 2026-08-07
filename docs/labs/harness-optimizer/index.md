[Harness Optimizer](https://github.com/strands-labs/harness-optimizer) is a framework for optimizing an LLM agent’s harness through **Formulas**. The core idea: enhance the agent dynamically with tunable Formulas (for example, system prompts), then improve those Formulas with optimizers based on collected agent rollout trajectories.

```python
from strands import Agent
from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent

# Create a Formula
formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")

# Attach to a Strands agent
agent = Agent(model=model)
apply_formulas_on_strands_agent(agent, [formula])

# Get / update parameters (e.g. after optimization)
params = formula.get_tunable_params()
formula.update_params({"system_prompt": "You are an expert coding assistant."})
```

## Getting started

```bash
pip install strands-harness-optimizer
```

## How it works

The training loop is a small set of composable interfaces:

1.  **DataLoader** yields batches of task samples from a Dataset.
2.  **AgentRolloutEngine** executes the agent on each sample using the current Formula parameters, producing rollouts. An **Adapter** (`apply_formulas_on_strands_agent`) bridges Formula parameters to the agent.
3.  **RewardFunction** scores each rollout.
4.  Rollouts, data, and rewards are collected into a batch.
5.  **FormulaOptimizer** analyzes the batch to propose new Formula parameters, following PyTorch’s pattern: `add_rollouts()`, `add_rewards()`, `step()`, `zero()`.
6.  The **Formula** updates its parameters and the loop repeats.

Built-in optimizers include a `ContrastiveReflectionOptimizer`, and rollout engines cover local execution and AgentCore.

## Design decisions

-   **Dict-based data** throughout - no wrapper classes for context or results.
-   **Minimal dependencies** - core depends on `strands-agents`, `strands-agents-tools`, `jinja2`, and `botocore`.
-   **PyTorch-style Dataset / DataLoader** reused from stdlib, with no PyTorch dependency.
-   **Minimal Trainer** - the built-in training loop is just two nested for-loops; bring your own if you prefer.

## Links

-   [GitHub repository](https://github.com/strands-labs/harness-optimizer)
-   [PyPI package](https://pypi.org/project/strands-harness-optimizer/)