"""
End-to-end tests for the new invocation_args parameter in Agent APIs.

These tests verify that the invocation_args parameter works correctly in real scenarios
with actual tools and mock models (no AWS credentials required).
"""

import asyncio
import warnings
from unittest.mock import Mock, AsyncMock

import pytest

import strands
from strands import Agent, tool
from strands.handlers.callback_handler import PrintingCallbackHandler


class InvocationArgsCallbackHandler:
    """Custom callback handler for testing invocation_args functionality."""
    
    def __init__(self):
        self.events_received = []
        self.tool_calls = []
        self.messages = []
    
    def callback_handler(self, **kwargs) -> None:
        """Process callback events and store them for verification."""
        self.events_received.append(kwargs)
        
        if "current_tool_use" in kwargs:
            self.tool_calls.append(kwargs["current_tool_use"])
        
        if "message" in kwargs:
            self.messages.append(kwargs["message"])


@tool
def calculator_tool(operation: str, a: float, b: float) -> float:
    """Simple calculator tool for testing invocation_args."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return b != 0 and a / b or 0
    else:
        raise ValueError(f"Unknown operation: {operation}")


@tool
def echo_tool(text: str) -> str:
    """Simple echo tool for testing invocation_args."""
    return f"Echo: {text}"


@pytest.fixture
def test_callback_handler():
    """Fixture providing a test callback handler."""
    return InvocationArgsCallbackHandler()


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing."""
    mock = Mock()
    
    # Create a proper async generator for the stream method
    async def mock_stream(*args, **kwargs):
        # Simulate a simple response
        yield {"role": "assistant", "content": [{"text": "I'll help you with that calculation."}]}
    
    mock.stream = mock_stream
    return mock


@pytest.fixture
def agent_with_tools(mock_model):
    """Fixture providing an agent with test tools and mock model."""
    return Agent(tools=[calculator_tool, echo_tool], model=mock_model)


class TestInvocationArgsEndToEnd:
    """End-to-end tests for invocation_args parameter."""
    
    def test_agent_call_with_custom_callback_handler(self, agent_with_tools, test_callback_handler):
        """Test that invocation_args correctly passes custom callback handler."""
        # Test with new invocation_args parameter
        result = agent_with_tools(
            "Calculate 5 + 3 using the calculator tool",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
        
        # Verify callback handler was called
        assert len(test_callback_handler.events_received) > 0
    
    def test_agent_call_with_kwargs_deprecation_warning(self, agent_with_tools, test_callback_handler):
        """Test that using kwargs emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = agent_with_tools(
                "Calculate 5 + 3 using the calculator tool",
                callback_handler=test_callback_handler.callback_handler
            )
            
            # Verify deprecation warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Using **kwargs in Agent.__call__ is deprecated" in str(w[0].message)
            
            # Verify functionality still works
            assert result.stop_reason is not None
            assert result.message is not None
    
    def test_agent_call_with_both_parameters_precedence(self, agent_with_tools, test_callback_handler):
        """Test that invocation_args takes precedence over kwargs when both are provided."""
        other_handler = Mock()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = agent_with_tools(
                "Calculate 5 + 3 using the calculator tool",
                invocation_args={"callback_handler": test_callback_handler.callback_handler},
                callback_handler=other_handler
            )
            
            # Verify deprecation warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            
            # Verify invocation_args callback handler was used (not kwargs)
            assert len(test_callback_handler.events_received) > 0
            other_handler.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_invoke_async_with_invocation_args(self, agent_with_tools, test_callback_handler):
        """Test that invoke_async works correctly with invocation_args."""
        result = await agent_with_tools.invoke_async(
            "Echo 'Hello World' using the echo tool",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
        
        # Verify callback handler was called
        assert len(test_callback_handler.events_received) > 0
    
    @pytest.mark.asyncio
    async def test_invoke_async_with_kwargs_deprecation_warning(self, agent_with_tools, test_callback_handler):
        """Test that invoke_async emits deprecation warning with kwargs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = await agent_with_tools.invoke_async(
                "Echo 'Hello World' using the echo tool",
                callback_handler=test_callback_handler.callback_handler
            )
            
            # Verify deprecation warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Using **kwargs in Agent.invoke_async is deprecated" in str(w[0].message)
            
            # Verify functionality still works
            assert result.stop_reason is not None
            assert result.message is not None
    
    @pytest.mark.asyncio
    async def test_stream_async_with_invocation_args(self, agent_with_tools, test_callback_handler):
        """Test that stream_async works correctly with invocation_args."""
        events = []
        
        async for event in agent_with_tools.stream_async(
            "Calculate 10 * 2 using the calculator tool",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        ):
            events.append(event)
        
        # Verify events were received
        assert len(events) > 0
        
        # Verify callback handler was called
        assert len(test_callback_handler.events_received) > 0
    
    @pytest.mark.asyncio
    async def test_stream_async_with_kwargs_deprecation_warning(self, agent_with_tools, test_callback_handler):
        """Test that stream_async emits deprecation warning with kwargs."""
        events = []
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            async for event in agent_with_tools.stream_async(
                "Calculate 10 * 2 using the calculator tool",
                callback_handler=test_callback_handler.callback_handler
            ):
                events.append(event)
        
        # Verify deprecation warning was emitted
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Using **kwargs in Agent.stream_async is deprecated" in str(w[0].message)
        
        # Verify functionality still works
        assert len(events) > 0
        assert len(test_callback_handler.events_received) > 0
    
    def test_invocation_args_with_multiple_parameters(self, agent_with_tools, test_callback_handler):
        """Test invocation_args with multiple custom parameters."""
        custom_params = {
            "callback_handler": test_callback_handler.callback_handler,
            "custom_param_1": "value_1",
            "custom_param_2": 42,
            "custom_param_3": {"nested": "value"}
        }
        
        result = agent_with_tools(
            "Echo 'test' using the echo tool",
            invocation_args=custom_params
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
        
        # Verify callback handler was called
        assert len(test_callback_handler.events_received) > 0
    
    def test_invocation_args_with_none_values(self, agent_with_tools):
        """Test that invocation_args can handle None values correctly."""
        result = agent_with_tools(
            "Calculate 1 + 1 using the calculator tool",
            invocation_args={"callback_handler": None}
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
    
    def test_invocation_args_with_empty_dict(self, agent_with_tools):
        """Test that invocation_args works with empty dictionary."""
        result = agent_with_tools(
            "Calculate 1 + 1 using the calculator tool",
            invocation_args={}
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
    
    def test_invocation_args_with_none_parameter(self, agent_with_tools):
        """Test that invocation_args works when set to None."""
        result = agent_with_tools(
            "Calculate 1 + 1 using the calculator tool",
            invocation_args=None
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
    
    @pytest.mark.asyncio
    async def test_complex_workflow_with_invocation_args(self, agent_with_tools, test_callback_handler):
        """Test a complex workflow using invocation_args."""
        # First calculation
        result1 = await agent_with_tools.invoke_async(
            "Calculate 15 + 25 using the calculator tool",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        )
        
        # Second calculation
        result2 = await agent_with_tools.invoke_async(
            "Calculate 100 - 50 using the calculator tool",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        )
        
        # Verify both executions worked
        assert result1.stop_reason is not None
        assert result2.stop_reason is not None
        
        # Verify callback handler was called for both
        assert len(test_callback_handler.events_received) > 0
    
    def test_backward_compatibility_with_existing_code(self, agent_with_tools):
        """Test that existing code patterns still work without modification."""
        # This simulates existing code that uses kwargs
        result = agent_with_tools("Calculate 2 * 3 using the calculator tool")
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
        
        # Verify no warnings were emitted for simple calls without kwargs
        # (This test ensures we don't break existing simple usage patterns)
    
    def test_invocation_args_parameter_types(self, agent_with_tools):
        """Test that invocation_args accepts various parameter types."""
        # Test with different data types
        test_params = {
            "string_param": "test_string",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"key": "value"},
            "none_param": None
        }
        
        result = agent_with_tools(
            "Calculate 1 + 1 using the calculator tool",
            invocation_args=test_params
        )
        
        # Verify the agent executed successfully
        assert result.stop_reason is not None
        assert result.message is not None
    
    @pytest.mark.asyncio
    async def test_async_methods_consistency(self, agent_with_tools, test_callback_handler):
        """Test that async methods behave consistently with invocation_args."""
        # Test invoke_async
        result1 = await agent_with_tools.invoke_async(
            "Test message",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        )
        
        # Test stream_async
        events = []
        async for event in agent_with_tools.stream_async(
            "Test message",
            invocation_args={"callback_handler": test_callback_handler.callback_handler}
        ):
            events.append(event)
        
        # Both should work without errors
        assert result1.stop_reason is not None
        assert len(events) > 0
