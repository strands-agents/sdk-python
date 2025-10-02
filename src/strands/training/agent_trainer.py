"""Agent trainer for continuous learning with Strands Agents.

This module provides the main AgentTrainer class that integrates with RL/SFT
frameworks to enable continuous learning for Strands Agents.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ..agent import Agent
from .env import StrandsEnv
from .reward_functions import RewardFunction, RewardFunctionRegistry
from .trajectory_capture import TrajectoryCapture

logger = logging.getLogger(__name__)


class AgentTrainer:
    """Main trainer class for continuous learning with Strands Agents.
    
    This class provides a high-level interface for training agents using
    various RL/SFT frameworks. It handles dataset management, training
    configuration, and integration with external training frameworks.
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
        reward_function: Optional[RewardFunction] = None,
        trajectory_capture: Optional[TrajectoryCapture] = None,
    ):
        """Initialize the agent trainer.
        
        Args:
            agent_class: Class of the agent to train
            env_class: Class of the environment wrapper
            agent_args: Arguments for creating agent instances
            env_args: Arguments for creating environment instances
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            reward_function: Reward function for training
            trajectory_capture: Trajectory capture system
        """
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args
        self.env_args = env_args
        self.config = config
        self.train_dataset = train_dataset or []
        self.val_dataset = val_dataset or []
        self.reward_function = reward_function
        self.trajectory_capture = trajectory_capture or TrajectoryCapture()
        
        # Training state
        self.training_history: List[Dict[str, Any]] = []
        self.current_epoch = 0
        self.is_training = False
        
        logger.debug(
            "agent_class=<%s>, env_class=<%s>, train_samples=<%d>, val_samples=<%d> | initialized trainer",
            agent_class.__name__,
            env_class.__name__,
            len(self.train_dataset),
            len(self.val_dataset),
        )
    
    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the agent using the configured datasets and parameters.
        
        Args:
            epochs: Number of training epochs (overrides config)
            batch_size: Training batch size (overrides config)
            learning_rate: Learning rate (overrides config)
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Update config with provided parameters
        training_config = self.config.copy()
        if epochs is not None:
            training_config["epochs"] = epochs
        if batch_size is not None:
            training_config["batch_size"] = batch_size
        if learning_rate is not None:
            training_config["learning_rate"] = learning_rate
        training_config.update(kwargs)
        
        self.is_training = True
        self.current_epoch = 0
        
        logger.info(
            "epochs=<%d>, batch_size=<%d>, learning_rate=<%f> | starting training",
            training_config.get("epochs", 1),
            training_config.get("batch_size", 1),
            training_config.get("learning_rate", 0.001),
        )
        
        try:
            # Training loop
            for epoch in range(training_config.get("epochs", 1)):
                self.current_epoch = epoch
                
                # Train on training dataset
                train_metrics = self._train_epoch(self.train_dataset, training_config)
                
                # Validate on validation dataset
                val_metrics = self._validate_epoch(self.val_dataset, training_config)
                
                # Record epoch results
                epoch_result = {
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": training_config,
                }
                self.training_history.append(epoch_result)
                
                logger.info(
                    "epoch=<%d>, train_reward=<%f>, val_reward=<%f> | completed epoch",
                    epoch,
                    train_metrics.get("avg_reward", 0.0),
                    val_metrics.get("avg_reward", 0.0),
                )
                
                # Check for early stopping
                if self._should_stop_early(val_metrics, training_config):
                    logger.info("epoch=<%d> | early stopping triggered", epoch)
                    break
            
            # Final results
            results = {
                "training_history": self.training_history,
                "final_train_metrics": self.training_history[-1]["train_metrics"] if self.training_history else {},
                "final_val_metrics": self.training_history[-1]["val_metrics"] if self.training_history else {},
                "total_epochs": len(self.training_history),
            }
            
            logger.info(
                "total_epochs=<%d>, final_train_reward=<%f>, final_val_reward=<%f> | training completed",
                len(self.training_history),
                results["final_train_metrics"].get("avg_reward", 0.0),
                results["final_val_metrics"].get("avg_reward", 0.0),
            )
            
            return results
            
        finally:
            self.is_training = False
    
    def _train_epoch(self, dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Train for one epoch on the given dataset."""
        batch_size = config.get("batch_size", 1)
        total_reward = 0.0
        total_steps = 0
        successful_episodes = 0
        
        # Process dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            
            for sample in batch:
                try:
                    # Create agent and environment for this sample
                    agent = self.agent_class(**self.agent_args)
                    env_args = self.env_args.copy()
                    env_args["agent"] = agent
                    env_args["reward_function"] = self.reward_function
                    env_args["trajectory_capture"] = self.trajectory_capture
                    
                    env = self.env_class(**env_args)
                    
                    # Run episode
                    episode_reward, episode_steps = self._run_episode(env, sample)
                    
                    total_reward += episode_reward
                    total_steps += episode_steps
                    successful_episodes += 1
                    
                    logger.debug(
                        "sample=<%d>, episode_reward=<%f>, episode_steps=<%d> | completed training episode",
                        i,
                        episode_reward,
                        episode_steps,
                    )
                    
                except Exception as e:
                    logger.warning(
                        "sample=<%d>, error=<%s> | training episode failed",
                        i,
                        e,
                    )
                
                finally:
                    if 'env' in locals():
                        env.close()
        
        # Calculate metrics
        avg_reward = total_reward / max(successful_episodes, 1)
        avg_steps = total_steps / max(successful_episodes, 1)
        
        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "total_reward": total_reward,
            "total_steps": total_steps,
            "successful_episodes": successful_episodes,
            "total_samples": len(dataset),
        }
    
    def _validate_epoch(self, dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate for one epoch on the given dataset."""
        if not dataset:
            return {"avg_reward": 0.0, "avg_steps": 0.0, "total_reward": 0.0, "total_steps": 0, "successful_episodes": 0, "total_samples": 0}
        
        batch_size = config.get("batch_size", 1)
        total_reward = 0.0
        total_steps = 0
        successful_episodes = 0
        
        # Process dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            
            for sample in batch:
                try:
                    # Create agent and environment for this sample
                    agent = self.agent_class(**self.agent_args)
                    env_args = self.env_args.copy()
                    env_args["agent"] = agent
                    env_args["reward_function"] = self.reward_function
                    env_args["trajectory_capture"] = self.trajectory_capture
                    
                    env = self.env_class(**env_args)
                    
                    # Run episode (no training updates)
                    episode_reward, episode_steps = self._run_episode(env, sample, training=False)
                    
                    total_reward += episode_reward
                    total_steps += episode_steps
                    successful_episodes += 1
                    
                    logger.debug(
                        "sample=<%d>, episode_reward=<%f>, episode_steps=<%d> | completed validation episode",
                        i,
                        episode_reward,
                        episode_steps,
                    )
                    
                except Exception as e:
                    logger.warning(
                        "sample=<%d>, error=<%s> | validation episode failed",
                        i,
                        e,
                    )
                
                finally:
                    if 'env' in locals():
                        env.close()
        
        # Calculate metrics
        avg_reward = total_reward / max(successful_episodes, 1)
        avg_steps = total_steps / max(successful_episodes, 1)
        
        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "total_reward": total_reward,
            "total_steps": total_steps,
            "successful_episodes": successful_episodes,
            "total_samples": len(dataset),
        }
    
    def _run_episode(
        self,
        env: StrandsEnv,
        sample: Dict[str, Any],
        training: bool = True,
    ) -> tuple[float, int]:
        """Run a single episode in the environment."""
        # Reset environment
        initial_prompt = sample.get("prompt", "")
        observation, info = env.reset(initial_prompt)
        
        episode_reward = 0.0
        episode_steps = 0
        
        # Run episode
        done = False
        while not done:
            # Get action (could be from policy, random, or sample)
            action = self._get_action(observation, sample, training)
            
            # Execute step
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            done = terminated or truncated
        
        return episode_reward, episode_steps
    
    def _get_action(
        self,
        observation: Dict[str, Any],
        sample: Dict[str, Any],
        training: bool = True,
    ) -> Union[str, Dict[str, Any]]:
        """Get action for the current observation."""
        # For now, use simple action selection
        # In a real implementation, this would use a trained policy
        
        if "action" in sample:
            return sample["action"]
        
        # Default to using the agent's response
        return sample.get("prompt", "")
    
    def _should_stop_early(self, val_metrics: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if training should stop early."""
        early_stopping_patience = config.get("early_stopping_patience", None)
        if early_stopping_patience is None:
            return False
        
        # Simple early stopping based on validation reward
        if len(self.training_history) < early_stopping_patience:
            return False
        
        # Check if validation reward has improved in the last N epochs
        recent_rewards = [
            epoch["val_metrics"]["avg_reward"]
            for epoch in self.training_history[-early_stopping_patience:]
        ]
        
        if len(recent_rewards) < early_stopping_patience:
            return False
        
        # Check if reward has been decreasing
        if all(recent_rewards[i] >= recent_rewards[i + 1] for i in range(len(recent_rewards) - 1)):
            return True
        
        return False
    
    def save_model(self, path: str) -> None:
        """Save the trained model to a file."""
        # This would save the agent's state/weights
        # Implementation depends on the specific agent type
        logger.info("path=<%s> | saving model", path)
        # TODO: Implement model saving
    
    def load_model(self, path: str) -> None:
        """Load a trained model from a file."""
        # This would load the agent's state/weights
        # Implementation depends on the specific agent type
        logger.info("path=<%s> | loading model", path)
        # TODO: Implement model loading
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the training history."""
        return self.training_history.copy()
    
    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Get the best model based on validation metrics."""
        if not self.training_history:
            return None
        
        # Find epoch with best validation reward
        best_epoch = max(
            self.training_history,
            key=lambda epoch: epoch["val_metrics"].get("avg_reward", 0.0)
        )
        
        return best_epoch
