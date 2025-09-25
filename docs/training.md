# Training Strands Agents

This guide covers how to use the training capabilities in Strands Agents for continuous learning through model fine-tuning on captured agent trajectories.

## Overview

The training system enables:

- **Trajectory Data Utilization**: Collect agent execution traces into training datasets
- **Trajectory-based Training**: Fine-tune models using real agent execution data
- **Continuous Learning**: Build domain-specific agents that outperform generic models

## Quick Start

```python
from strands.training import StrandsAgent, StrandsEnv, AgentTrainer, math_reward_fn
from strands_tools import calculator

# Define agent configuration
agent_args = {
    "tools": [calculator], 
    "system_prompt": "You are a helpful assistant."
}

# Create trainer
trainer = AgentTrainer(
    agent_class=StrandsAgent,
    env_class=StrandsEnv,
    agent_args=agent_args,
    env_args={"reward_fn": math_reward_fn()},
    config={
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 0.001,
    },
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
)

# Start training
results = trainer.train()
```

## Core Components

### 1. Trajectory Capture

The `TrajectoryCapture` class automatically records agent interactions, tool calls, and outcomes:

```python
from strands.training import TrajectoryCapture

# Create trajectory capture
capture = TrajectoryCapture(
    capture_tool_calls=True,
    capture_model_responses=True,
    capture_metadata=True,
)

# Add to agent
agent = Agent()
agent.hooks.add_provider(capture)

# Trajectories are automatically captured during agent execution
result = agent("What is 2 + 2?")
trajectory = capture.get_current_trajectory()
```

### 2. Reward Functions

Reward functions evaluate agent performance and provide feedback for training:

```python
from strands.training import (
    TaskCompletionReward,
    EfficiencyReward,
    ToolUsageReward,
    CompositeRewardFunction,
)

# Individual reward functions
task_reward = TaskCompletionReward(success_reward=2.0, failure_reward=-1.0)
efficiency_reward = EfficiencyReward(max_steps=5, max_duration=30.0)
tool_reward = ToolUsageReward(tool_use_bonus=0.1, correct_tool_bonus=0.2)

# Composite reward function
composite_reward = CompositeRewardFunction(
    reward_functions=[task_reward, efficiency_reward, tool_reward],
    weights=[0.6, 0.2, 0.2],
)
```

### 3. Training Environment

The `StrandsEnv` class provides a gym-like interface for training:

```python
from strands.training import StrandsEnv, math_reward_fn

# Create environment
env = StrandsEnv(
    agent=agent,
    reward_function=math_reward_fn(),
    max_steps=20,
)

# Reset environment
observation, info = env.reset("Solve this math problem: 5 * 7")

# Execute steps
action = "Let me calculate 5 * 7"
observation, reward, terminated, truncated, info = env.step(action)

# Render current state
env.render(mode="human")
```

### 4. Agent Trainer

The `AgentTrainer` class orchestrates the training process:

```python
from strands.training import AgentTrainer

trainer = AgentTrainer(
    agent_class=StrandsAgent,
    env_class=StrandsEnv,
    agent_args={"tools": [calculator]},
    env_args={"reward_fn": math_reward_fn()},
    config={
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 0.001,
        "early_stopping_patience": 3,
    },
    train_dataset=train_data,
    val_dataset=val_data,
)

# Train the agent
results = trainer.train()

# Access training history
history = trainer.get_training_history()
best_model = trainer.get_best_model()
```

## Pre-built Reward Functions

### Math Problems

```python
from strands.training import math_reward_fn

reward_func = math_reward_fn()
# Rewards: correct answers, efficient solving, appropriate tool usage
```

### Coding Problems

```python
from strands.training import coding_reward_fn

reward_func = coding_reward_fn()
# Rewards: correct code, efficient debugging, tool usage (like python_repl)
```

### General Tasks

```python
from strands.training import general_reward_fn

reward_func = general_reward_fn()
# Rewards: task completion, efficiency, balanced tool usage
```

## Custom Reward Functions

Create custom reward functions by extending the `RewardFunction` base class:

```python
from strands.training import RewardFunction
from strands.training.trajectory_capture import TrajectoryData

class CustomRewardFunction(RewardFunction):
    def compute_reward(self, trajectory: TrajectoryData, **kwargs) -> float:
        # Your custom reward logic here
        reward = 0.0
        
        # Example: reward based on conversation length
        if len(trajectory.steps) < 5:
            reward += 1.0
        
        # Example: reward for specific tool usage
        for step in trajectory.steps:
            if step.step_type == "message_assistant":
                tool_calls = step.output_data.get("tool_calls", [])
                if any(call.get("name") == "calculator" for call in tool_calls):
                    reward += 0.5
        
        return reward

# Use custom reward function
custom_reward = CustomRewardFunction()
```

## Dataset Format

Training datasets should be lists of dictionaries with the following structure:

```python
train_dataset = [
    {
        "prompt": "What is the square root of 144?",
        "expected_tools": ["calculator"],
        "difficulty": "easy",
    },
    {
        "prompt": "Calculate the area of a circle with radius 5",
        "expected_tools": ["calculator"],
        "difficulty": "medium",
    },
    # ... more samples
]

validation_dataset = [
    {
        "prompt": "What is 15% of 200?",
        "expected_tools": ["calculator"],
        "difficulty": "easy",
    },
    # ... more samples
]
```

## Training Configuration

The training configuration supports various parameters:

```python
config = {
    # Training parameters
    "epochs": 10,                    # Number of training epochs
    "batch_size": 4,                 # Batch size for training
    "learning_rate": 0.001,          # Learning rate
    
    # Early stopping
    "early_stopping_patience": 3,    # Stop if no improvement for N epochs
    
    # Environment parameters
    "max_steps": 20,                 # Maximum steps per episode
    "max_duration": 60.0,           # Maximum duration per episode (seconds)
    
    # Reward function parameters
    "reward_weights": [0.6, 0.2, 0.2],  # Weights for composite rewards
}
```

## Advanced Usage

### Custom Environment

```python
from strands.training import StrandsEnv

class CustomStrandsEnv(StrandsEnv):
    def _get_action(self, observation, sample, training=True):
        # Custom action selection logic
        if training:
            return self._get_training_action(observation, sample)
        else:
            return self._get_evaluation_action(observation, sample)
    
    def _get_training_action(self, observation, sample):
        # Implement your training action selection
        return sample.get("prompt", "")
    
    def _get_evaluation_action(self, observation, sample):
        # Implement your evaluation action selection
        return sample.get("prompt", "")
```

### Custom Agent Class

```python
from strands.training import StrandsAgent

class CustomStrandsAgent(StrandsAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
    
    def __call__(self, prompt, **kwargs):
        # Add custom preprocessing
        processed_prompt = self._preprocess_prompt(prompt)
        
        # Call parent method
        result = super().__call__(processed_prompt, **kwargs)
        
        # Add custom postprocessing
        return self._postprocess_result(result)
```

### Integration with External Frameworks

The training system is designed to integrate with external RL/SFT frameworks:

```python
# Example integration with rLLM
from rllm.trainer import RLHFTrainer
from strands.training import StrandsAgent, StrandsEnv

# Create Strands-compatible agent and environment
agent_class = StrandsAgent
env_class = StrandsEnv

# Use with rLLM trainer
trainer = RLHFTrainer(
    agent_class=agent_class,
    env_class=env_class,
    # ... other rLLM parameters
)
```

## Best Practices

1. **Start Small**: Begin with simple tasks and small datasets
2. **Monitor Training**: Use the training history to track progress
3. **Validate Regularly**: Use validation datasets to prevent overfitting
4. **Customize Rewards**: Tailor reward functions to your specific use case
5. **Iterative Improvement**: Start with basic rewards and refine based on results

## Troubleshooting

### Common Issues

1. **Low Rewards**: Check if reward functions are appropriate for your task
2. **Training Instability**: Reduce learning rate or batch size
3. **Poor Performance**: Ensure training dataset is representative
4. **Memory Issues**: Reduce batch size or dataset size

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check trajectory capture
trajectory = capture.get_current_trajectory()
print(f"Trajectory steps: {len(trajectory.steps)}")
print(f"Trajectory reward: {trajectory.reward}")

# Monitor training progress
for epoch_result in trainer.get_training_history():
    print(f"Epoch {epoch_result['epoch']}: "
          f"Train reward: {epoch_result['train_metrics']['avg_reward']:.3f}, "
          f"Val reward: {epoch_result['val_metrics']['avg_reward']:.3f}")
```

## Examples

See the `examples/training/` directory for complete examples including:

- Math problem solving
- Code generation and debugging
- General conversation tasks
- Custom reward functions
- Multi-agent training scenarios
