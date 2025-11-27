"""Unit tests for BidiAgent."""

import unittest.mock
import asyncio
import pytest
from uuid import uuid4

from strands.experimental.bidi.agent.agent import BidiAgent
from strands.experimental.bidi.models.novasonic import BidiNovaSonicModel
from strands.experimental.bidi.types.events import (
    BidiTextInputEvent,
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiTranscriptStreamEvent,
    BidiConnectionStartEvent,
    BidiConnectionCloseEvent,
)

class MockBidiModel:
    """Mock bidirectional model for testing."""

    def __init__(self, config=None, model_id="mock-model"):
        self.config = config or {"audio": {"input_rate": 16000, "output_rate": 24000, "channels": 1}}
        self.model_id = model_id
        self._connection_id = None
        self._started = False
        self._events_to_yield = []

    async def start(self, system_prompt=None, tools=None, messages=None, **kwargs):
        if self._started:
            raise RuntimeError("model already started | call stop before starting again")
        self._connection_id = str(uuid4())
        self._started = True

    async def stop(self):
        if self._started:
            self._started = False
            self._connection_id = None

    async def send(self, content):
        if not self._started:
            raise RuntimeError("model not started | call start before sending/receiving")
        # Mock implementation - in real tests, this would trigger events

    async def receive(self):
        """Async generator yielding mock events."""
        if not self._started:
            raise RuntimeError("model not started | call start before sending/receiving")
        
        # Yield connection start event
        yield BidiConnectionStartEvent(connection_id=self._connection_id, model=self.model_id)
        
        # Yield any configured events
        for event in self._events_to_yield:
            yield event
            
        # Yield connection end event
        yield BidiConnectionCloseEvent(connection_id=self._connection_id, reason="complete")

    def set_events(self, events):
        """Helper to set events this mock model will yield."""
        self._events_to_yield = events

@pytest.fixture
def mock_model():
    """Create a mock BidiModel instance."""
    return MockBidiModel()

@pytest.fixture
def mock_tool_registry():
    """Mock tool registry with some basic tools."""
    registry = unittest.mock.Mock()
    registry.get_all_tool_specs.return_value = [
        {
            "name": "calculator",
            "description": "Perform calculations", 
            "inputSchema": {"json": {"type": "object", "properties": {}}}
        }
    ]
    registry.get_all_tools_config.return_value = {"calculator": {}}
    return registry


@pytest.fixture  
def mock_tool_caller():
    """Mock tool caller for testing tool execution."""
    caller = unittest.mock.AsyncMock()
    caller.call_tool = unittest.mock.AsyncMock()
    return caller


@pytest.fixture
def agent(mock_model, mock_tool_registry, mock_tool_caller):
    """Create a BidiAgent instance for testing."""
    with unittest.mock.patch("strands.experimental.bidi.agent.agent.ToolRegistry") as mock_registry_class:
        mock_registry_class.return_value = mock_tool_registry
        
        with unittest.mock.patch("strands.experimental.bidi.agent.agent._ToolCaller") as mock_caller_class:
            mock_caller_class.return_value = mock_tool_caller
            
            # Don't pass tools to avoid real tool loading
            agent = BidiAgent(model=mock_model)
            return agent

def test_agent_initialization():
    """Test agent initialization with various configurations."""
    # Test default initialization
    mock_model = MockBidiModel()
    agent = BidiAgent(model=mock_model)
    
    assert agent.model == mock_model
    assert agent.system_prompt is None
    assert not agent._started
    assert agent.model._connection_id is None
    
    # Test with configuration
    system_prompt = "You are a helpful assistant."
    agent_with_config = BidiAgent(
        model=mock_model,
        system_prompt=system_prompt, 
        agent_id="test_agent"
    )
    
    assert agent_with_config.system_prompt == system_prompt
    assert agent_with_config.agent_id == "test_agent"
    
    # Test with string model ID
    model_id = "amazon.nova-sonic-v1:0"
    agent_with_string = BidiAgent(model=model_id)
    
    assert isinstance(agent_with_string.model, BidiNovaSonicModel)
    assert agent_with_string.model.model_id == model_id
    
    # Test model config access
    config = agent.model.config
    assert config["audio"]["input_rate"] == 16000
    assert config["audio"]["output_rate"] == 24000
    assert config["audio"]["channels"] == 1

@pytest.mark.asyncio
async def test_agent_lifecycle(agent):
    """Test agent start/stop lifecycle and state management."""
    # Initial state
    assert not agent._started
    assert agent.model._connection_id is None
    
    # Start agent
    await agent.start()
    assert agent._started
    assert agent.model._connection_id is not None
    connection_id = agent.model._connection_id
    
    # Double start should error
    with pytest.raises(RuntimeError, match="agent already started"):
        await agent.start()
    
    # Stop agent
    await agent.stop()
    assert not agent._started
    assert agent.model._connection_id is None
    
    # Multiple stops should be safe
    await agent.stop()
    await agent.stop()
    
    # Restart should work with new connection ID
    await agent.start()
    assert agent._started
    assert agent.model._connection_id != connection_id

@pytest.mark.asyncio
async def test_send_methods(agent):
    """Test sending various input types through agent.send()."""
    await agent.start()
    
    # Test text input with TypedEvent
    text_input = BidiTextInputEvent(text="Hello", role="user")
    await agent.send(text_input)
    assert len(agent.messages) == 1
    assert agent.messages[0]["content"][0]["text"] == "Hello"
    
    # Test string input (shorthand)
    await agent.send("World")
    assert len(agent.messages) == 2
    assert agent.messages[1]["content"][0]["text"] == "World"
    
    # Test audio input (doesn't add to messages)
    audio_input = BidiAudioInputEvent(
        audio="dGVzdA==",  # base64 "test"
        format="pcm",
        sample_rate=16000,
        channels=1
    )
    await agent.send(audio_input)
    assert len(agent.messages) == 2  # Still 2, audio doesn't add
    
    # Test concurrent sends
    sends = [
        agent.send(BidiTextInputEvent(text=f"Message {i}", role="user"))
        for i in range(3)
    ]
    await asyncio.gather(*sends)
    assert len(agent.messages) == 5  # 2 + 3 new messages

@pytest.mark.asyncio
async def test_receive_methods(agent):
    """Test receiving events from model."""
    # Configure mock model to yield events
    events = [
        BidiAudioStreamEvent(
            audio="dGVzdA==",
            format="pcm", 
            sample_rate=24000,
            channels=1
        ),
        BidiTranscriptStreamEvent(
            text="Hello world",
            role="assistant",
            is_final=True,
            delta={"text": "Hello world"},
            current_transcript="Hello world"
        )
    ]
    agent.model.set_events(events)
    
    await agent.start()
    
    received_events = []
    async for event in agent.receive():
        received_events.append(event)
        if len(received_events) >= 4:  # Stop after getting expected events
            break
        
    # Verify event types and order
    assert len(received_events) >= 3
    assert isinstance(received_events[0], BidiConnectionStartEvent)
    assert isinstance(received_events[1], BidiAudioStreamEvent)
    assert isinstance(received_events[2], BidiTranscriptStreamEvent)
    
    # Test empty events
    agent.model.set_events([])
    await agent.stop()
    await agent.start()
    
    empty_events = []
    async for event in agent.receive():
        empty_events.append(event)
        if len(empty_events) >= 2:
            break
    
    assert len(empty_events) >= 1
    assert isinstance(empty_events[0], BidiConnectionStartEvent)

def test_agent_tools(agent, mock_tool_registry):
    """Test agent tool integration and properties."""
    # Test tool property access
    assert hasattr(agent, 'tool')
    assert agent.tool is not None
    assert agent.tool == agent._tool_caller
    
    # Test tool names property
    mock_tool_registry.get_all_tools_config.return_value = {
        "calculator": {}, 
        "weather": {}
    }
    
    tool_names = agent.tool_names
    assert isinstance(tool_names, list)
    assert len(tool_names) == 2
    assert "calculator" in tool_names
    assert "weather" in tool_names

@pytest.mark.asyncio
async def test_error_handling(agent):
    """Test error handling in various scenarios."""
    # Test send before start
    with pytest.raises(RuntimeError, match="call start before"):
        await agent.send(BidiTextInputEvent(text="Hello", role="user"))
    
    # Test receive before start
    with pytest.raises(RuntimeError, match="call start before"):
        async for event in agent.receive():
            pass
    
    # Test send after stop
    await agent.start()
    await agent.stop()
    with pytest.raises(RuntimeError, match="call start before"):
        await agent.send(BidiTextInputEvent(text="Hello", role="user"))
    
    # Test receive after stop
    with pytest.raises(RuntimeError, match="call start before"):
        async for event in agent.receive():
            pass


@pytest.mark.asyncio
async def test_model_error_propagation():
    """Test that model errors are properly propagated."""
    # Test model start error
    mock_model = MockBidiModel()
    mock_model.start = unittest.mock.AsyncMock(side_effect=Exception("Connection failed"))
    error_agent = BidiAgent(model=mock_model)
    
    with pytest.raises(Exception, match="Connection failed"):
        await error_agent.start()
    
    # Test model receive error
    mock_model2 = MockBidiModel()
    agent2 = BidiAgent(model=mock_model2)
    await agent2.start()
    
    async def failing_receive():
        yield BidiConnectionStartEvent(connection_id="test", model="test-model")
        raise Exception("Receive failed")
    
    agent2.model.receive = failing_receive
    with pytest.raises(Exception, match="Receive failed"):
        async for event in agent2.receive():
            pass

@pytest.mark.asyncio
async def test_agent_state_consistency(agent):
    """Test that agent state remains consistent across operations."""
    # Initial state
    assert not agent._started
    assert agent.model._connection_id is None
    
    # Start
    await agent.start()
    assert agent._started
    assert agent.model._connection_id is not None
    connection_id = agent.model._connection_id
    
    # Send operations shouldn't change connection state
    await agent.send(BidiTextInputEvent(text="Hello", role="user"))
    assert agent._started
    assert agent.model._connection_id == connection_id
    
    # Stop
    await agent.stop()
    assert not agent._started
    assert agent.model._connection_id is None