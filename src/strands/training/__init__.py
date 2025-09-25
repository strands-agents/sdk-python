"""Training capabilities for Strands Agents.

This package provides functionality for training agents through continuous learning,
including trajectory capture, reward functions, and integration with RL/SFT frameworks.

Example usage matching the feature request API:

```python
from strands.training import StrandsAgent, StrandsEnv, AgentTrainer, math_reward_fn
from strands_tools import calculator

agent_args = {"tools": [calculator], 
              "system_prompt": "You are a helpful assistant."}

trainer = AgentTrainer(
    agent_class=StrandsAgent,
    env_class=StrandsEnv,
    agent_args=agent_args,
    env_args={"reward_fn": math_reward_fn()},
    config=training_config,
    train_dataset=dataset,
    val_dataset=validation_dataset,
)

trainer.train()
```
"""

from .agent_trainer import AgentTrainer
from .env import StrandsEnv
from .integration import (
    AgentTrainer as AgentTrainerWrapper,
    StrandsAgent,
    StrandsEnv as StrandsEnvWrapper,
    coding_reward_fn,
    general_reward_fn,
    math_reward_fn,
)
from .reward_functions import (
    RewardFunction, 
    RewardFunctionRegistry,
    TaskCompletionReward,
    EfficiencyReward,
    ToolUsageReward,
    CompositeRewardFunction,
)
from .trajectory_capture import TrajectoryCapture, TrajectoryData, TrajectoryStep

# Export the main API classes
AgentTrainer = AgentTrainerWrapper
StrandsEnv = StrandsEnvWrapper

__all__ = [
    "AgentTrainer",
    "StrandsAgent", 
    "StrandsEnv",
    "RewardFunction",
    "RewardFunctionRegistry",
    "TaskCompletionReward",
    "EfficiencyReward",
    "ToolUsageReward",
    "CompositeRewardFunction",
    "TrajectoryCapture",
    "TrajectoryData",
    "TrajectoryStep",
    "math_reward_fn",
    "coding_reward_fn",
    "general_reward_fn",
]
