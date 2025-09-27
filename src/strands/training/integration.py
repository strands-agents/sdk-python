"""Integration module for RL/SFT frameworks.

This module provides integration with external training frameworks like rLLM and veRL,
implementing the exact API specified in the feature request.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from ..agent import Agent
from .agent_trainer import AgentTrainer
from .env import StrandsEnv
from .reward_functions import RewardFunction, RewardFunctionRegistry

logger = logging.getLogger(__name__)


# Re-export the main classes for the API specified in the issue
class StrandsAgent(Agent):
    """Strands Agent class for training integration.
    
    This is a thin wrapper around the main Agent class to provide
    compatibility with training frameworks.
    """
    
    def __init__(self, **kwargs: Any):
        """Initialize Strands Agent for training."""
        super().__init__(**kwargs)
        logger.debug(
            "agent_id=<%s>, model=<%s> | initialized StrandsAgent for training",
            self.agent_id,
            getattr(self.model, 'model_id', 'unknown'),
        )


class StrandsEnvWrapper(StrandsEnv):
    """Environment wrapper for Strands Agents.
    
    This provides the exact interface expected by training frameworks.
    """
    
    def __init__(self, reward_fn: Optional[RewardFunction] = None, **kwargs: Any):
        """Initialize environment wrapper.
        
        Args:
            reward_fn: Reward function for training
            **kwargs: Additional environment arguments
        """
        # Extract agent from kwargs if provided
        agent = kwargs.pop("agent", None)
        if agent is None:
            raise ValueError("Agent must be provided in kwargs")
        
        super().__init__(
            agent=agent,
            reward_function=reward_fn,
            **kwargs,
        )
        
        logger.debug(
            "agent_id=<%s>, reward_function=<%s> | initialized StrandsEnvWrapper",
            agent.agent_id,
            reward_fn.__class__.__name__ if reward_fn else "None",
        )


class AgentTrainerWrapper(AgentTrainer):
    """Wrapper for AgentTrainer that matches the exact API from the issue.
    
    This provides the exact interface specified in the feature request:
    
    ```python
    from rllm.agents import StrandsAgent
    from rllm.environments.tools.strands_env import StrandsEnv
    from rllm.rewards.reward_fn import math_reward_fn
    from rllm.trainer.agent_trainer import AgentTrainer
    
    from strands_tools import python_repl, calculator
    
    agent_args = {"tools": [calculator], 
                  "system_prompt": "You are a helpful assistant."}
    
    trainer = AgentTrainer(
        agent_class=StrandsAgent,
        env_class=StrandsEnv,
        agent_args=agent_args,
        env_args={"reward_fn": reward_function},
        config=training_config,
        train_dataset=dataset,
        val_dataset=validation_dataset,
    )
    
    trainer.train()
    ```
    """
    
    def __init__(
        self,
        agent_class: Type[Agent],
        env_class: Type[StrandsEnv],
        agent_args: Dict[str, Any],
        env_args: Dict[str, Any],
        config: Dict[str, Any],
        train_dataset: Optional[List[Dict[str, Any]]] = None,
        val_dataset: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the agent trainer with the exact API from the issue.
        
        Args:
            agent_class: Class of the agent to train (e.g., StrandsAgent)
            env_class: Class of the environment wrapper (e.g., StrandsEnv)
            agent_args: Arguments for creating agent instances
            env_args: Arguments for creating environment instances
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        # Extract reward function from env_args
        reward_fn = env_args.pop("reward_fn", None)
        
        super().__init__(
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            reward_function=reward_fn,
        )
        
        logger.debug(
            "agent_class=<%s>, env_class=<%s>, train_samples=<%d>, val_samples=<%d> | initialized AgentTrainerWrapper",
            agent_class.__name__,
            env_class.__name__,
            len(train_dataset) if train_dataset else 0,
            len(val_dataset) if val_dataset else 0,
        )


# Convenience functions for common reward functions
def math_reward_fn() -> RewardFunction:
    """Create a reward function suitable for math problems.
    
    This function provides a reward function that rewards:
    - Correct mathematical answers
    - Efficient problem solving
    - Appropriate tool usage
    """
    from .reward_functions import CompositeRewardFunction, TaskCompletionReward, EfficiencyReward, ToolUsageReward
    
    # Create composite reward function for math problems
    reward_functions = [
        TaskCompletionReward(success_reward=2.0, failure_reward=-1.0),
        EfficiencyReward(max_steps=5, max_duration=30.0),
        ToolUsageReward(tool_use_bonus=0.1, correct_tool_bonus=0.2),
    ]
    
    weights = [0.6, 0.2, 0.2]  # Emphasize task completion
    
    return CompositeRewardFunction(
        reward_functions=reward_functions,
        weights=weights,
        name="math_reward_function",
    )


def coding_reward_fn() -> RewardFunction:
    """Create a reward function suitable for coding problems.
    
    This function provides a reward function that rewards:
    - Correct code solutions
    - Efficient debugging
    - Appropriate tool usage (like python_repl)
    """
    from .reward_functions import CompositeRewardFunction, TaskCompletionReward, EfficiencyReward, ToolUsageReward
    
    # Create composite reward function for coding problems
    reward_functions = [
        TaskCompletionReward(success_reward=3.0, failure_reward=-1.0),
        EfficiencyReward(max_steps=10, max_duration=60.0),
        ToolUsageReward(tool_use_bonus=0.2, correct_tool_bonus=0.3),
    ]
    
    weights = [0.5, 0.2, 0.3]  # Emphasize tool usage for coding
    
    return CompositeRewardFunction(
        reward_functions=reward_functions,
        weights=weights,
        name="coding_reward_function",
    )


def general_reward_fn() -> RewardFunction:
    """Create a general-purpose reward function.
    
    This function provides a balanced reward function suitable for
    general conversational tasks.
    """
    from .reward_functions import CompositeRewardFunction, TaskCompletionReward, EfficiencyReward, ToolUsageReward
    
    # Create composite reward function for general tasks
    reward_functions = [
        TaskCompletionReward(success_reward=1.0, failure_reward=-0.5),
        EfficiencyReward(max_steps=8, max_duration=45.0),
        ToolUsageReward(tool_use_bonus=0.05, correct_tool_bonus=0.1),
    ]
    
    weights = [0.7, 0.2, 0.1]  # Emphasize task completion
    
    return CompositeRewardFunction(
        reward_functions=reward_functions,
        weights=weights,
        name="general_reward_function",
    )


# Export the main classes for the API
__all__ = [
    "StrandsAgent",
    "StrandsEnv", 
    "AgentTrainer",
    "math_reward_fn",
    "coding_reward_fn", 
    "general_reward_fn",
]
