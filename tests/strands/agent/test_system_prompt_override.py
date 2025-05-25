"""
Test for system prompt override functionality.
"""

import pytest
from strands.types.models.model import Model
from strands.agent.agent import Agent
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec
from typing import Any, Iterable, Optional


class MockModel(Model):
    """Mock model that captures system prompt for verification."""
    
    def __init__(self):
        self.captured_system_prompts = []
        
    def update_config(self, **model_config: Any) -> None:
        """Mock implementation - no configuration to update."""
        pass
        
    def get_config(self) -> Any:
        return {}
        
    def format_request(self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None) -> Any:
        self.captured_system_prompts.append(system_prompt)
        return {
            "messages": messages,
            "tool_specs": tool_specs, 
            "system_prompt": system_prompt,
        }
        
    def format_chunk(self, event: Any) -> StreamEvent:
        return {"messageStart": {"role": "assistant"}}
        
    def stream(self, request: Any) -> Iterable[Any]:
        yield {"contentBlockDelta": {"delta": {"text": "Mock response"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing."""
    return MockModel()


def test_agent_uses_default_system_prompt(mock_model):
    """Test that agent uses the default system prompt when no override is provided."""
    default_prompt = "You are a helpful assistant."
    agent = Agent(system_prompt=default_prompt, model=mock_model)
    
    agent("Hello")
    
    assert len(mock_model.captured_system_prompts) == 1
    assert mock_model.captured_system_prompts[0] == default_prompt


def test_agent_system_prompt_override(mock_model):
    """Test that agent can override system prompt on a per-call basis."""
    default_prompt = "You are a helpful assistant."
    override_prompt = "You are a pirate who speaks only in seafaring terms."
    
    agent = Agent(system_prompt=default_prompt, model=mock_model)
    
    # Call with override
    agent("Hello", system_prompt=override_prompt)
    
    assert len(mock_model.captured_system_prompts) == 1
    assert mock_model.captured_system_prompts[0] == override_prompt


def test_agent_system_prompt_override_then_default(mock_model):
    """Test that agent reverts to default system prompt after override."""
    default_prompt = "You are a helpful assistant."
    override_prompt = "You are a pirate."
    
    agent = Agent(system_prompt=default_prompt, model=mock_model)
    
    # Call with override
    agent("Hello", system_prompt=override_prompt)
    
    # Call without override (should use default)
    agent("Hello again")
    
    assert len(mock_model.captured_system_prompts) == 2
    assert mock_model.captured_system_prompts[0] == override_prompt
    assert mock_model.captured_system_prompts[1] == default_prompt


def test_agent_multiple_system_prompt_overrides(mock_model):
    """Test that agent can handle multiple different system prompt overrides."""
    default_prompt = "You are a helpful assistant."
    
    agent = Agent(system_prompt=default_prompt, model=mock_model)
    
    # Multiple calls with different overrides
    agent("Hello", system_prompt="You are a poet.")
    agent("Hello", system_prompt="You are a robot.")
    agent("Hello")  # Should use default
    
    assert len(mock_model.captured_system_prompts) == 3
    assert mock_model.captured_system_prompts[0] == "You are a poet."
    assert mock_model.captured_system_prompts[1] == "You are a robot."
    assert mock_model.captured_system_prompts[2] == default_prompt


def test_agent_system_prompt_override_none(mock_model):
    """Test that agent handles None system prompt override correctly."""
    default_prompt = "You are a helpful assistant."
    
    agent = Agent(system_prompt=default_prompt, model=mock_model)
    
    # Call with None override
    agent("Hello", system_prompt=None)
    
    assert len(mock_model.captured_system_prompts) == 1
    assert mock_model.captured_system_prompts[0] is None


def test_agent_system_prompt_override_empty_string(mock_model):
    """Test that agent handles empty string system prompt override correctly."""
    default_prompt = "You are a helpful assistant."
    
    agent = Agent(system_prompt=default_prompt, model=mock_model)
    
    # Call with empty string override
    agent("Hello", system_prompt="")
    
    assert len(mock_model.captured_system_prompts) == 1
    assert mock_model.captured_system_prompts[0] == ""


def test_agent_no_default_system_prompt(mock_model):
    """Test that agent works correctly when no default system prompt is provided."""
    agent = Agent(model=mock_model)  # No system prompt
    
    # Call without override
    agent("Hello")
    
    assert len(mock_model.captured_system_prompts) == 1
    assert mock_model.captured_system_prompts[0] is None
    
    # Call with override
    agent("Hello", system_prompt="You are helpful.")
    
    assert len(mock_model.captured_system_prompts) == 2
    assert mock_model.captured_system_prompts[1] == "You are helpful."
