"""Tests for training functionality.

This module contains comprehensive tests for the training capabilities
including trajectory capture, reward functions, and agent training.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from strands.agent import Agent
from strands.training import (
    AgentTrainer,
    StrandsAgent,
    StrandsEnv,
    RewardFunction,
    TrajectoryCapture,
    TrajectoryData,
    TrajectoryStep,
    math_reward_fn,
    coding_reward_fn,
    general_reward_fn,
)
from strands.training.reward_functions import (
    TaskCompletionReward,
    EfficiencyReward,
    ToolUsageReward,
    CompositeRewardFunction,
)


class TestTrajectoryData:
    """Test TrajectoryData class."""
    
    def test_trajectory_creation(self):
        """Test trajectory data creation."""
        trajectory = TrajectoryData(
            agent_id="test_agent",
            session_id="test_session",
        )
        
        assert trajectory.agent_id == "test_agent"
        assert trajectory.session_id == "test_session"
        assert trajectory.trajectory_id is not None
        assert len(trajectory.steps) == 0
        assert trajectory.reward is None
    
    def test_add_step(self):
        """Test adding steps to trajectory."""
        trajectory = TrajectoryData()
        step = TrajectoryStep(
            step_type="test_step",
            input_data={"test": "input"},
            output_data={"test": "output"},
        )
        
        trajectory.add_step(step)
        
        assert len(trajectory.steps) == 1
        assert trajectory.steps[0] == step
    
    def test_finalize(self):
        """Test trajectory finalization."""
        trajectory = TrajectoryData()
        final_result = {"status": "completed"}
        
        trajectory.finalize(final_result)
        
        assert trajectory.end_time is not None
        assert trajectory.final_result == final_result
    
    def test_to_dict(self):
        """Test trajectory to dictionary conversion."""
        trajectory = TrajectoryData(agent_id="test")
        trajectory_dict = trajectory.to_dict()
        
        assert isinstance(trajectory_dict, dict)
        assert trajectory_dict["agent_id"] == "test"
        assert "trajectory_id" in trajectory_dict
        assert "steps" in trajectory_dict


class TestTrajectoryStep:
    """Test TrajectoryStep class."""
    
    def test_step_creation(self):
        """Test trajectory step creation."""
        step = TrajectoryStep(
            step_type="test_step",
            input_data={"input": "data"},
            output_data={"output": "data"},
            metadata={"meta": "data"},
        )
        
        assert step.step_type == "test_step"
        assert step.input_data == {"input": "data"}
        assert step.output_data == {"output": "data"}
        assert step.metadata == {"meta": "data"}
        assert step.step_id is not None
        assert isinstance(step.timestamp, datetime)


class TestTrajectoryCapture:
    """Test TrajectoryCapture class."""
    
    def test_initialization(self):
        """Test trajectory capture initialization."""
        capture = TrajectoryCapture()
        
        assert capture.capture_tool_calls is True
        assert capture.capture_model_responses is True
        assert capture.current_trajectory is None
    
    def test_hook_registration(self):
        """Test hook registration."""
        capture = TrajectoryCapture()
        registry = Mock()
        
        capture.register_hooks(registry)
        
        # Should register callbacks for BeforeInvocationEvent, MessageAddedEvent, AfterInvocationEvent
        assert registry.add_callback.call_count == 3
    
    def test_set_reward(self):
        """Test setting reward on current trajectory."""
        capture = TrajectoryCapture()
        trajectory = TrajectoryData()
        capture.current_trajectory = trajectory
        
        capture.set_reward(1.5)
        
        assert trajectory.reward == 1.5
    
    def test_set_reward_no_trajectory(self):
        """Test setting reward when no current trajectory."""
        capture = TrajectoryCapture()
        
        # Should not raise error
        capture.set_reward(1.5)


class TestRewardFunctions:
    """Test reward function classes."""
    
    def test_task_completion_reward_success(self):
        """Test task completion reward for success."""
        reward_func = TaskCompletionReward()
        trajectory = TrajectoryData()
        
        # Add final step indicating success
        final_step = TrajectoryStep(
            step_type="invocation_end",
            output_data={"success": True},
        )
        trajectory.add_step(final_step)
        trajectory.finalize({"status": "success"})
        
        reward = reward_func.compute_reward(trajectory)
        
        assert reward == 1.0  # success_reward
    
    def test_task_completion_reward_failure(self):
        """Test task completion reward for failure."""
        reward_func = TaskCompletionReward()
        trajectory = TrajectoryData()
        
        # Add final step indicating failure
        final_step = TrajectoryStep(
            step_type="invocation_end",
            output_data={"success": False},
        )
        trajectory.add_step(final_step)
        trajectory.finalize({"status": "failure"})
        
        reward = reward_func.compute_reward(trajectory)
        
        assert reward == -1.0  # failure_reward
    
    def test_efficiency_reward(self):
        """Test efficiency reward function."""
        reward_func = EfficiencyReward(max_steps=5, max_duration=30.0)
        trajectory = TrajectoryData()
        
        # Add steps within limits
        for i in range(3):
            step = TrajectoryStep(step_type=f"step_{i}")
            trajectory.add_step(step)
        
        trajectory.finalize()
        
        reward = reward_func.compute_reward(trajectory)
        
        assert reward > 0.0  # Should be positive for efficient execution
    
    def test_tool_usage_reward(self):
        """Test tool usage reward function."""
        reward_func = ToolUsageReward()
        trajectory = TrajectoryData()
        
        # Add step with tool calls
        step = TrajectoryStep(
            step_type="message_assistant",
            output_data={"tool_calls": [{"name": "test_tool"}]},
        )
        trajectory.add_step(step)
        
        reward = reward_func.compute_reward(trajectory)
        
        assert reward > 0.0  # Should be positive for tool usage
    
    def test_composite_reward_function(self):
        """Test composite reward function."""
        task_reward = TaskCompletionReward()
        efficiency_reward = EfficiencyReward()
        
        composite = CompositeRewardFunction(
            reward_functions=[task_reward, efficiency_reward],
            weights=[0.7, 0.3],
        )
        
        trajectory = TrajectoryData()
        final_step = TrajectoryStep(
            step_type="invocation_end",
            output_data={"success": True},
        )
        trajectory.add_step(final_step)
        trajectory.finalize()
        
        reward = composite.compute_reward(trajectory)
        
        assert isinstance(reward, float)  # Should be a float


class TestStrandsEnv:
    """Test StrandsEnv training environment."""
    
    def test_initialization(self):
        """Test environment initialization."""
        agent = Agent()
        env = StrandsEnv(agent)
        
        assert env.agent == agent
        assert env.max_steps == 20
        assert env.current_step == 0
        assert not env.episode_done
    
    def test_reset(self):
        """Test environment reset."""
        agent = Agent()
        env = StrandsEnv(agent)
        
        observation, info = env.reset("Test prompt")
        
        assert env.current_step == 0
        assert not env.episode_done
        assert "messages" in observation
        assert "step" in observation
        assert observation["step"] == 0
    
    def test_step_with_string_action(self):
        """Test step with string action."""
        agent = Agent()
        env = StrandsEnv(agent)
        
        observation, info = env.reset("Test prompt")
        
        # Mock the agent call to avoid actual model inference
        with patch.object(agent, '__call__') as mock_call:
            mock_result = Mock()
            mock_result.stop_reason = "end_turn"
            mock_result.message = {"role": "assistant", "content": [{"text": "Test response"}]}
            mock_call.return_value = mock_result
            
            observation, reward, terminated, truncated, info = env.step("Test action")
            
            assert env.current_step == 1
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
    
    def test_step_with_dict_action(self):
        """Test step with dictionary action."""
        agent = Agent()
        env = StrandsEnv(agent)
        
        observation, info = env.reset("Test prompt")
        
        # Mock the agent call
        with patch.object(agent, '__call__') as mock_call:
            mock_result = Mock()
            mock_result.stop_reason = "end_turn"
            mock_result.message = {"role": "assistant", "content": [{"text": "Test response"}]}
            mock_call.return_value = mock_result
            
            action = {"prompt": "Test action", "invocation_args": {}}
            observation, reward, terminated, truncated, info = env.step(action)
            
            assert env.current_step == 1
            assert isinstance(reward, float)


class TestAgentTrainer:
    """Test AgentTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = AgentTrainer(
            agent_class=StrandsAgent,
            env_class=StrandsEnv,
            agent_args={"system_prompt": "Test prompt"},
            env_args={},
            config={"epochs": 1, "batch_size": 1},
            train_dataset=[],
            val_dataset=[],
        )
        
        assert trainer.agent_class == StrandsAgent
        assert trainer.env_class == StrandsEnv
        assert trainer.current_epoch == 0
        assert not trainer.is_training
    
    def test_train_empty_dataset(self):
        """Test training with empty dataset."""
        trainer = AgentTrainer(
            agent_class=StrandsAgent,
            env_class=StrandsEnv,
            agent_args={"system_prompt": "Test prompt"},
            env_args={},
            config={"epochs": 1, "batch_size": 1},
            train_dataset=[],
            val_dataset=[],
        )
        
        results = trainer.train()
        
        assert "training_history" in results
        assert "total_epochs" in results
        assert results["total_epochs"] >= 0  # Should handle empty datasets gracefully
    
    def test_train_with_dataset(self):
        """Test training with sample dataset."""
        trainer = AgentTrainer(
            agent_class=StrandsAgent,
            env_class=StrandsEnv,
            agent_args={"system_prompt": "Test prompt"},
            env_args={},
            config={"epochs": 1, "batch_size": 1},
            train_dataset=[{"prompt": "Test prompt 1"}],
            val_dataset=[{"prompt": "Test prompt 2"}],
        )
        
        # Mock the environment to avoid actual agent execution
        with patch('strands.training.agent_trainer.StrandsEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = ({}, {})
            mock_env.step.return_value = ({}, 1.0, True, False, {})
            mock_env_class.return_value = mock_env
            
            results = trainer.train()
            
            assert "training_history" in results
            assert results["total_epochs"] == 1


class TestConvenienceFunctions:
    """Test convenience reward functions."""
    
    def test_math_reward_fn(self):
        """Test math reward function creation."""
        reward_func = math_reward_fn()
        
        assert isinstance(reward_func, CompositeRewardFunction)
        assert reward_func.name == "math_reward_function"
        assert len(reward_func.reward_functions) == 3
    
    def test_coding_reward_fn(self):
        """Test coding reward function creation."""
        reward_func = coding_reward_fn()
        
        assert isinstance(reward_func, CompositeRewardFunction)
        assert reward_func.name == "coding_reward_function"
        assert len(reward_func.reward_functions) == 3
    
    def test_general_reward_fn(self):
        """Test general reward function creation."""
        reward_func = general_reward_fn()
        
        assert isinstance(reward_func, CompositeRewardFunction)
        assert reward_func.name == "general_reward_function"
        assert len(reward_func.reward_functions) == 3


class TestIntegrationAPI:
    """Test the integration API matches the feature request."""
    
    def test_api_imports(self):
        """Test that the API can be imported as specified in the issue."""
        from strands.training import StrandsAgent, StrandsEnv, AgentTrainer, math_reward_fn
        
        # Test that classes exist and are callable
        assert StrandsAgent is not None
        assert StrandsEnv is not None
        assert AgentTrainer is not None
        assert math_reward_fn is not None
        
        # Test that they can be instantiated
        agent = StrandsAgent(system_prompt="Test")
        assert isinstance(agent, Agent)
        
        reward_func = math_reward_fn()
        assert isinstance(reward_func, RewardFunction)
    
    def test_trainer_api_example(self):
        """Test the exact API example from the feature request."""
        from strands.training import StrandsAgent, StrandsEnv, AgentTrainer, math_reward_fn
        
        # This should match the exact API from the issue
        agent_args = {"system_prompt": "You are a helpful assistant."}
        
        trainer = AgentTrainer(
            agent_class=StrandsAgent,
            env_class=StrandsEnv,
            agent_args=agent_args,
            env_args={"reward_fn": math_reward_fn()},
            config={"epochs": 1, "batch_size": 1},
            train_dataset=[],
            val_dataset=[],
        )
        
        assert isinstance(trainer, AgentTrainer)
        assert trainer.agent_class == StrandsAgent
        assert trainer.env_class == StrandsEnv
