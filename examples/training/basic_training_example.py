"""Example demonstrating the exact API from the feature request.

This example shows how to use the training capabilities with the exact API
specified in issue #923.
"""

from strands.training import StrandsAgent, StrandsEnv, AgentTrainer, math_reward_fn
from strands_tools import calculator

# Example dataset for math problems
train_dataset = [
    {"prompt": "What is 2 + 2?"},
    {"prompt": "Calculate 15 * 8"},
    {"prompt": "What is the square root of 144?"},
    {"prompt": "Find 25% of 200"},
    {"prompt": "What is 3 to the power of 4?"},
]

validation_dataset = [
    {"prompt": "What is 7 * 9?"},
    {"prompt": "Calculate 12 / 3"},
    {"prompt": "What is the square root of 81?"},
]

# Training configuration
training_config = {
    "epochs": 5,
    "batch_size": 2,
    "learning_rate": 0.001,
    "early_stopping_patience": 2,
}

# Agent configuration
agent_args = {
    "tools": [calculator], 
    "system_prompt": "You are a helpful math assistant. Use the calculator tool when needed."
}

# Environment configuration
env_args = {
    "reward_fn": math_reward_fn(),
    "max_steps": 10,
}

# Create trainer using the exact API from the issue
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
print("Starting training...")
results = trainer.train()

# Print results
print(f"\nTraining completed!")
print(f"Total epochs: {results['total_epochs']}")
print(f"Final train reward: {results['final_train_metrics']['avg_reward']:.3f}")
print(f"Final validation reward: {results['final_val_metrics']['avg_reward']:.3f}")

# Show training history
print("\nTraining History:")
for epoch_result in results['training_history']:
    epoch = epoch_result['epoch']
    train_reward = epoch_result['train_metrics']['avg_reward']
    val_reward = epoch_result['val_metrics']['avg_reward']
    print(f"Epoch {epoch}: Train={train_reward:.3f}, Val={val_reward:.3f}")

# Get best model
best_model = trainer.get_best_model()
if best_model:
    print(f"\nBest model from epoch {best_model['epoch']}")
    print(f"Best validation reward: {best_model['val_metrics']['avg_reward']:.3f}")

print("\nTraining example completed successfully!")
