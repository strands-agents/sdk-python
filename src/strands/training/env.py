"""Training environment wrapper for Strands Agents.

This module provides a training environment that wraps Strands Agents to make them
compatible with RL/SFT frameworks like rLLM and veRL.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ..agent import Agent
from ..types.content import ContentBlock, Message, Messages
from .reward_functions import RewardFunction
from .trajectory_capture import TrajectoryCapture, TrajectoryData

logger = logging.getLogger(__name__)


class StrandsEnv:
    """Training environment wrapper for Strands Agents.
    
    This class provides a gym-like interface for training Strands Agents,
    making them compatible with RL/SFT frameworks.
    """
    
    def __init__(
        self,
        agent: Agent,
        reward_function: Optional[RewardFunction] = None,
        max_steps: int = 20,
        trajectory_capture: Optional[TrajectoryCapture] = None,
        **kwargs: Any,
    ):
        """Initialize the training environment.
        
        Args:
            agent: The Strands Agent to wrap
            reward_function: Function to compute rewards
            max_steps: Maximum steps per episode
            trajectory_capture: Optional trajectory capture system
            **kwargs: Additional configuration
        """
        self.agent = agent
        self.reward_function = reward_function
        self.max_steps = max_steps
        self.trajectory_capture = trajectory_capture or TrajectoryCapture()
        
        # Add trajectory capture to agent if not already present
        # Note: We'll add it during initialization instead
        
        # Environment state
        self.current_step = 0
        self.current_trajectory: Optional[TrajectoryData] = None
        self.episode_reward = 0.0
        self.episode_done = False
        
        logger.debug(
            "agent_id=<%s>, max_steps=<%d>, reward_function=<%s> | initialized training environment",
            agent.agent_id,
            max_steps,
            reward_function.__class__.__name__ if reward_function else "None",
        )
    
    def reset(self, initial_prompt: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment for a new episode.
        
        Args:
            initial_prompt: Optional initial prompt for the episode
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset agent state
        self.agent.messages.clear()
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        
        # Set initial prompt if provided
        if initial_prompt:
            initial_message: Message = {
                "role": "user",
                "content": [{"text": initial_prompt}]
            }
            self.agent.messages.append(initial_message)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(
            "episode=<%d>, initial_prompt=<%s> | reset environment",
            self.current_step,
            initial_prompt[:50] + "..." if initial_prompt and len(initial_prompt) > 50 else initial_prompt,
        )
        
        return observation, info
    
    def step(self, action: Union[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (prompt string or action dict)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        self.current_step += 1
        
        # Execute action
        try:
            if isinstance(action, str):
                # Direct prompt
                result = self.agent(action)
            elif isinstance(action, dict):
                # Structured action
                prompt = action.get("prompt", "")
                invocation_args = action.get("invocation_args", {})
                result = self.agent(prompt, invocation_args=invocation_args)
            else:
                raise ValueError(f"Invalid action type: {type(action)}")
            
            # Check if episode should terminate
            terminated = self._is_terminated(result)
            truncated = self.current_step >= self.max_steps
            
            # Compute reward
            reward = self._compute_reward(result)
            self.episode_reward += reward
            
            # Update episode state
            if terminated or truncated:
                self.episode_done = True
            
            # Get observation and info
            observation = self._get_observation()
            info = self._get_info()
            
            logger.debug(
                "step=<%d>, reward=<%f>, terminated=<%s>, truncated=<%s> | executed step",
                self.current_step,
                reward,
                terminated,
                truncated,
            )
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(
                "step=<%d>, error=<%s> | step execution failed",
                self.current_step,
                e,
                exc_info=True,
            )
            
            # Return failure state
            terminated = True
            truncated = False
            reward = -1.0  # Negative reward for errors
            self.episode_reward += reward
            self.episode_done = True
            
            observation = self._get_observation()
            info = self._get_info()
            
            return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation state."""
        return {
            "messages": self.agent.messages,
            "step": self.current_step,
            "agent_id": self.agent.agent_id,
            "available_tools": list(self.agent.tool_registry.registry.keys()),
            "system_prompt": self.agent.system_prompt,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        return {
            "episode_reward": self.episode_reward,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "episode_done": self.episode_done,
            "trajectory_id": (
                self.current_trajectory.trajectory_id 
                if self.current_trajectory 
                else None
            ),
        }
    
    def _is_terminated(self, result: Any) -> bool:
        """Check if the episode should terminate based on the result."""
        # Check stop reason
        if hasattr(result, 'stop_reason'):
            stop_reason = result.stop_reason
            # Terminate on certain stop reasons
            if stop_reason in ["end_turn", "max_tokens"]:
                return True
        
        # Check for explicit termination signals
        if hasattr(result, 'message') and result.message:
            content = result.message.get("content", [])
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text = block["text"].lower()
                    if any(term in text for term in ["done", "complete", "finished", "terminate"]):
                        return True
        
        return False
    
    def _compute_reward(self, result: Any) -> float:
        """Compute reward for the current step."""
        if not self.reward_function:
            return 0.0
        
        # Get current trajectory
        trajectory = self.trajectory_capture.get_current_trajectory()
        if not trajectory:
            return 0.0
        
        try:
            reward = self.reward_function.compute_reward(trajectory)
            
            # Set reward on trajectory
            self.trajectory_capture.set_reward(reward)
            
            return reward
            
        except Exception as e:
            logger.warning(
                "error=<%s> | failed to compute reward",
                e,
            )
            return 0.0
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current state of the environment.
        
        Args:
            mode: Rendering mode ("human" for console output, "text" for string)
            
        Returns:
            Rendered string if mode is "text", None otherwise
        """
        if mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"Messages: {len(self.agent.messages)}")
            if self.agent.messages:
                last_message = self.agent.messages[-1]
                content = last_message.get("content", [])
                if content and isinstance(content[0], dict) and "text" in content[0]:
                    print(f"Last Message: {content[0]['text'][:100]}...")
            return None
        
        elif mode == "text":
            return f"Step: {self.current_step}/{self.max_steps}, Reward: {self.episode_reward:.2f}"
        
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self) -> None:
        """Clean up the environment."""
        # Get final trajectory if available
        trajectory = self.trajectory_capture.get_current_trajectory()
        if trajectory:
            trajectory.finalize()
        
        logger.debug(
            "episode_reward=<%f>, total_steps=<%d> | closed environment",
            self.episode_reward,
            self.current_step,
        )
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """Get the action space description."""
        return {
            "type": "text",
            "description": "Text prompt or structured action dict",
        }
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """Get the observation space description."""
        return {
            "messages": "List of conversation messages",
            "step": "Current step number",
            "agent_id": "Agent identifier",
            "available_tools": "List of available tool names",
            "system_prompt": "System prompt text",
        }
