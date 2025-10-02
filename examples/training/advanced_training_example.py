"""Example showing custom reward functions and advanced training features.

This example demonstrates how to create custom reward functions and use
advanced training features.
"""

from strands.training import (
    StrandsAgent, 
    StrandsEnv, 
    AgentTrainer, 
    RewardFunction,
    CompositeRewardFunction,
    TaskCompletionReward,
    EfficiencyReward,
    ToolUsageReward,
)
from strands.training.trajectory_capture import TrajectoryData
from strands_tools import calculator, python_repl

# Custom reward function for coding problems
class CodingRewardFunction(RewardFunction):
    """Custom reward function for coding tasks."""
    
    def __init__(self):
        super().__init__("CodingRewardFunction")
    
    def compute_reward(self, trajectory: TrajectoryData, **kwargs) -> float:
        """Compute reward based on coding-specific criteria."""
        reward = 0.0
        
        # Check for successful completion
        if trajectory.final_result and trajectory.final_result.get("status") == "success":
            reward += 2.0
        
        # Reward for using python_repl tool
        python_repl_used = False
        for step in trajectory.steps:
            if step.step_type == "message_assistant":
                tool_calls = step.output_data.get("tool_calls", [])
                for tool_call in tool_calls:
                    if tool_call.get("name") == "python_repl":
                        python_repl_used = True
                        reward += 0.5
        
        # Penalty for too many steps (inefficient debugging)
        if len(trajectory.steps) > 15:
            reward -= 0.1 * (len(trajectory.steps) - 15)
        
        # Bonus for clean, efficient solutions
        if len(trajectory.steps) < 8:
            reward += 0.3
        
        return reward

# Create composite reward function
def create_coding_reward_function():
    """Create a composite reward function for coding tasks."""
    return CompositeRewardFunction(
        reward_functions=[
            TaskCompletionReward(success_reward=3.0, failure_reward=-1.0),
            EfficiencyReward(max_steps=10, max_duration=60.0),
            ToolUsageReward(tool_use_bonus=0.2, correct_tool_bonus=0.3),
            CodingRewardFunction(),
        ],
        weights=[0.4, 0.2, 0.2, 0.2],
        name="advanced_coding_reward",
    )

# Example dataset for coding problems
train_dataset = [
    {"prompt": "Write a Python function to calculate factorial"},
    {"prompt": "Create a function that reverses a string"},
    {"prompt": "Write code to find the largest number in a list"},
    {"prompt": "Create a function to check if a number is prime"},
]

validation_dataset = [
    {"prompt": "Write a function to calculate fibonacci numbers"},
    {"prompt": "Create a function to sort a list of numbers"},
]

# Training configuration
training_config = {
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 0.0005,
    "early_stopping_patience": 2,
}

# Agent configuration with coding tools
agent_args = {
    "tools": [calculator, python_repl], 
    "system_prompt": "You are a helpful coding assistant. Use the python_repl tool to test your code."
}

# Environment configuration with custom reward function
env_args = {
    "reward_fn": create_coding_reward_function(),
    "max_steps": 15,
}

# Create trainer
trainer = AgentTrainer(
    agent_class=StrandsAgent,
    env_class=StrandsEnv,
    agent_args=agent_args,
    env_args=env_args,
    config=training_config,
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
)

# Train the agent
print("Starting advanced training with custom reward function...")
results = trainer.train()

# Print detailed results
print(f"\nAdvanced training completed!")
print(f"Total epochs: {results['total_epochs']}")
print(f"Final train reward: {results['final_train_metrics']['avg_reward']:.3f}")
print(f"Final validation reward: {results['final_val_metrics']['avg_reward']:.3f}")

# Show detailed training history
print("\nDetailed Training History:")
for epoch_result in results['training_history']:
    epoch = epoch_result['epoch']
    train_metrics = epoch_result['train_metrics']
    val_metrics = epoch_result['val_metrics']
    
    print(f"Epoch {epoch}:")
    print(f"  Train: reward={train_metrics['avg_reward']:.3f}, "
          f"steps={train_metrics['avg_steps']:.1f}, "
          f"episodes={train_metrics['successful_episodes']}")
    print(f"  Val:   reward={val_metrics['avg_reward']:.3f}, "
          f"steps={val_metrics['avg_steps']:.1f}, "
          f"episodes={val_metrics['successful_episodes']}")

# Demonstrate trajectory capture
print("\nTrajectory Capture Example:")
from strands.training import TrajectoryCapture

# Create a simple agent with trajectory capture
agent = StrandsAgent(tools=[python_repl], system_prompt="You are a coding assistant.")
capture = TrajectoryCapture()
agent.hooks.add_provider(capture)

# Run a simple interaction
result = agent("Write a function to add two numbers")
trajectory = capture.get_current_trajectory()

if trajectory:
    print(f"Captured trajectory with {len(trajectory.steps)} steps")
    print(f"Trajectory ID: {trajectory.trajectory_id}")
    print(f"Agent ID: {trajectory.agent_id}")
    
    # Show step types
    step_types = [step.step_type for step in trajectory.steps]
    print(f"Step types: {step_types}")

print("\nAdvanced training example completed successfully!")
