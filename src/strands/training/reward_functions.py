"""Reward function framework for training Strands Agents.

This module provides a flexible framework for defining reward functions that can
evaluate agent performance and provide feedback for training.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .trajectory_capture import TrajectoryData

logger = logging.getLogger(__name__)


class RewardFunction(ABC):
    """Abstract base class for reward functions.
    
    Reward functions evaluate agent trajectories and return numerical rewards
    that can be used for training. They can be based on various criteria such
    as task completion, efficiency, correctness, etc.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize reward function.
        
        Args:
            name: Optional name for this reward function
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute_reward(
        self,
        trajectory: TrajectoryData,
        **kwargs: Any,
    ) -> float:
        """Compute reward for a given trajectory.
        
        Args:
            trajectory: The trajectory to evaluate
            **kwargs: Additional context for reward computation
            
        Returns:
            Numerical reward value (higher is better)
        """
        pass
    
    def __call__(self, trajectory: TrajectoryData, **kwargs: Any) -> float:
        """Make the reward function callable."""
        return self.compute_reward(trajectory, **kwargs)


class CompositeRewardFunction(RewardFunction):
    """Combines multiple reward functions with weighted scores.
    
    This allows for complex reward functions that consider multiple factors
    with different importance weights.
    """
    
    def __init__(
        self,
        reward_functions: List[RewardFunction],
        weights: Optional[List[float]] = None,
        name: Optional[str] = None,
    ):
        """Initialize composite reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Optional weights for each reward function (defaults to equal weights)
            name: Optional name for this composite function
        """
        super().__init__(name)
        self.reward_functions = reward_functions
        
        if weights is None:
            self.weights = [1.0 / len(reward_functions)] * len(reward_functions)
        else:
            if len(weights) != len(reward_functions):
                raise ValueError("Number of weights must match number of reward functions")
            self.weights = weights
    
    def compute_reward(
        self,
        trajectory: TrajectoryData,
        **kwargs: Any,
    ) -> float:
        """Compute weighted combination of all reward functions."""
        total_reward = 0.0
        
        for reward_func, weight in zip(self.reward_functions, self.weights):
            try:
                reward = reward_func.compute_reward(trajectory, **kwargs)
                total_reward += weight * reward
                
                logger.debug(
                    "reward_function=<%s>, weight=<%f>, reward=<%f> | computed component reward",
                    reward_func.name,
                    weight,
                    reward,
                )
            except Exception as e:
                logger.warning(
                    "reward_function=<%s>, error=<%s> | failed to compute reward",
                    reward_func.name,
                    e,
                )
        
        logger.debug(
            "composite_reward=<%f>, num_functions=<%d> | computed composite reward",
            total_reward,
            len(self.reward_functions),
        )
        
        return total_reward


class TaskCompletionReward(RewardFunction):
    """Reward function based on task completion.
    
    Provides positive reward for successful task completion and negative
    reward for failures or errors.
    """
    
    def __init__(
        self,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        partial_reward: float = 0.5,
    ):
        """Initialize task completion reward function.
        
        Args:
            success_reward: Reward for successful completion
            failure_reward: Reward for failure
            partial_reward: Reward for partial completion
        """
        super().__init__("TaskCompletionReward")
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.partial_reward = partial_reward
    
    def compute_reward(
        self,
        trajectory: TrajectoryData,
        **kwargs: Any,
    ) -> float:
        """Compute reward based on task completion."""
        if not trajectory.final_result:
            return self.failure_reward
        
        # Check if trajectory ended successfully
        final_step = trajectory.steps[-1] if trajectory.steps else None
        if not final_step:
            return self.failure_reward
        
        success = final_step.output_data.get("success", False)
        
        if success:
            return self.success_reward
        else:
            return self.failure_reward


class EfficiencyReward(RewardFunction):
    """Reward function based on efficiency metrics.
    
    Rewards shorter execution times and fewer tool calls while maintaining
    task completion quality.
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        max_duration: float = 60.0,
        step_penalty: float = 0.1,
        duration_penalty: float = 0.01,
    ):
        """Initialize efficiency reward function.
        
        Args:
            max_steps: Maximum expected steps for full reward
            max_duration: Maximum expected duration in seconds for full reward
            step_penalty: Penalty per step beyond max_steps
            duration_penalty: Penalty per second beyond max_duration
        """
        super().__init__("EfficiencyReward")
        self.max_steps = max_steps
        self.max_duration = max_duration
        self.step_penalty = step_penalty
        self.duration_penalty = duration_penalty
    
    def compute_reward(
        self,
        trajectory: TrajectoryData,
        **kwargs: Any,
    ) -> float:
        """Compute reward based on efficiency."""
        if not trajectory.steps:
            return 0.0
        
        # Calculate duration
        if trajectory.end_time:
            duration = (trajectory.end_time - trajectory.start_time).total_seconds()
        else:
            duration = 0.0
        
        # Calculate step count
        step_count = len(trajectory.steps)
        
        # Start with base reward
        reward = 1.0
        
        # Apply step penalty
        if step_count > self.max_steps:
            excess_steps = step_count - self.max_steps
            reward -= excess_steps * self.step_penalty
        
        # Apply duration penalty
        if duration > self.max_duration:
            excess_duration = duration - self.max_duration
            reward -= excess_duration * self.duration_penalty
        
        # Ensure reward is non-negative
        reward = max(0.0, reward)
        
        logger.debug(
            "steps=<%d>, duration=<%f>, reward=<%f> | computed efficiency reward",
            step_count,
            duration,
            reward,
        )
        
        return reward


class ToolUsageReward(RewardFunction):
    """Reward function based on appropriate tool usage.
    
    Rewards agents for using tools effectively and appropriately.
    """
    
    def __init__(
        self,
        tool_use_bonus: float = 0.1,
        correct_tool_bonus: float = 0.2,
        unnecessary_tool_penalty: float = 0.1,
    ):
        """Initialize tool usage reward function.
        
        Args:
            tool_use_bonus: Bonus for using tools
            correct_tool_bonus: Bonus for using correct tools
            unnecessary_tool_penalty: Penalty for unnecessary tool usage
        """
        super().__init__("ToolUsageReward")
        self.tool_use_bonus = tool_use_bonus
        self.correct_tool_bonus = correct_tool_bonus
        self.unnecessary_tool_penalty = unnecessary_tool_penalty
    
    def compute_reward(
        self,
        trajectory: TrajectoryData,
        **kwargs: Any,
    ) -> float:
        """Compute reward based on tool usage."""
        reward = 0.0
        tool_calls = 0
        successful_tool_calls = 0
        
        # Count tool calls and their success
        for step in trajectory.steps:
            if step.step_type == "message_assistant":
                tool_calls_in_step = len(step.output_data.get("tool_calls", []))
                tool_calls += tool_calls_in_step
                
                if tool_calls_in_step > 0:
                    reward += tool_calls_in_step * self.tool_use_bonus
            
            elif step.step_type == "message_user":
                tool_results = step.output_data.get("tool_results", [])
                for tool_result in tool_results:
                    if tool_result.get("status") == "success":
                        successful_tool_calls += 1
                        reward += self.correct_tool_bonus
        
        # Apply penalty for excessive tool usage
        if tool_calls > 5:  # Arbitrary threshold
            excess_calls = tool_calls - 5
            reward -= excess_calls * self.unnecessary_tool_penalty
        
        logger.debug(
            "tool_calls=<%d>, successful_calls=<%d>, reward=<%f> | computed tool usage reward",
            tool_calls,
            successful_tool_calls,
            reward,
        )
        
        return reward


class RewardFunctionRegistry:
    """Registry for managing reward functions."""
    
    def __init__(self):
        """Initialize registry."""
        self._functions: Dict[str, RewardFunction] = {}
    
    def register(self, name: str, reward_function: RewardFunction) -> None:
        """Register a reward function.
        
        Args:
            name: Name to register the function under
            reward_function: The reward function to register
        """
        self._functions[name] = reward_function
        logger.debug(
            "name=<%s>, function=<%s> | registered reward function",
            name,
            reward_function.__class__.__name__,
        )
    
    def get(self, name: str) -> Optional[RewardFunction]:
        """Get a registered reward function.
        
        Args:
            name: Name of the function to get
            
        Returns:
            The reward function or None if not found
        """
        return self._functions.get(name)
    
    def list_functions(self) -> List[str]:
        """List all registered reward function names."""
        return list(self._functions.keys())
    
    def create_composite(
        self,
        name: str,
        function_names: List[str],
        weights: Optional[List[float]] = None,
    ) -> CompositeRewardFunction:
        """Create a composite reward function from registered functions.
        
        Args:
            name: Name for the composite function
            function_names: Names of functions to combine
            weights: Optional weights for each function
            
        Returns:
            Composite reward function
            
        Raises:
            ValueError: If any function name is not found
        """
        functions = []
        for func_name in function_names:
            func = self.get(func_name)
            if func is None:
                raise ValueError(f"Reward function '{func_name}' not found")
            functions.append(func)
        
        composite = CompositeRewardFunction(functions, weights, name)
        self.register(name, composite)
        return composite


# Global registry instance
_reward_registry = RewardFunctionRegistry()


def get_reward_registry() -> RewardFunctionRegistry:
    """Get the global reward function registry."""
    return _reward_registry


def register_reward_function(name: str, reward_function: RewardFunction) -> None:
    """Register a reward function in the global registry."""
    _reward_registry.register(name, reward_function)


def get_reward_function(name: str) -> Optional[RewardFunction]:
    """Get a reward function from the global registry."""
    return _reward_registry.get(name)
