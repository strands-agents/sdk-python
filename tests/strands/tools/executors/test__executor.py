"""Test cases for the base ToolExecutor class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.hooks import BeforeToolCallEvent
from strands.telemetry.metrics import Trace
from strands.tools.executors._executor import ToolExecutor
from strands.types._events import ToolResultEvent, ToolStreamEvent


# Create a concrete implementation for testing
class TestExecutor(ToolExecutor):
    """Concrete implementation of ToolExecutor for testing."""

    async def _execute(
        self,
        agent,
        tool_uses,
        tool_results,
        cycle_trace,
        cycle_span,
        invocation_state,
        structured_output_context,
    ):
        """Mock implementation of _execute."""
        for tool_use in tool_uses:
            async for event in self._stream_with_trace(
                agent,
                tool_use,
                tool_results,
                cycle_trace,
                cycle_span,
                invocation_state,
                structured_output_context,
            ):
                yield event


class TestToolExecutor:
    """Test the ToolExecutor base class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with required attributes."""
        agent = MagicMock()
        agent.tool_registry.dynamic_tools = {}
        agent.tool_registry.registry = {}
        agent.tool_registry.get_all_tool_specs = MagicMock(return_value=[])
        agent.model = MagicMock()
        agent.messages = []
        agent.system_prompt = "Test prompt"
        agent.hooks = MagicMock()
        agent.event_loop_metrics = MagicMock()
        return agent

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = AsyncMock()

        async def mock_stream(*args, **kwargs):
            """Mock stream generator."""
            yield "streaming data"
            yield {"toolUseId": "test-id", "status": "success", "content": [{"text": "result"}]}

        tool.stream = mock_stream
        return tool

    @pytest.fixture
    def tool_use(self):
        """Create a sample tool use."""
        return {"name": "test_tool", "toolUseId": "test-id", "input": {"key": "value"}}

    @pytest.fixture
    def structured_output_context(self):
        """Create a mock structured output context."""
        context = MagicMock()
        context.is_enabled = False
        return context

    @pytest.fixture
    def executor(self):
        """Create a test executor instance."""
        return TestExecutor()

    @pytest.mark.asyncio
    async def test_stream_successful_tool_execution(self, mock_agent, mock_tool, tool_use, structured_output_context):
        """Test successful tool execution through _stream."""
        # Setup
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hooks to return events unchanged
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify
        assert len(events) == 3  # 2 ToolStreamEvents and 1 ToolResultEvent
        assert isinstance(events[0], ToolStreamEvent)
        assert isinstance(events[1], ToolStreamEvent)
        assert isinstance(events[2], ToolResultEvent)
        assert len(tool_results) == 1
        assert tool_results[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_stream_tool_not_found(self, mock_agent, tool_use, structured_output_context):
        """Test _stream when tool is not found in registry."""
        # Setup
        tool_results = []
        invocation_state = {}

        # Setup hooks to return events unchanged
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert tool_results[0]["status"] == "error"
        assert "Unknown tool" in tool_results[0]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_stream_with_dynamic_tool(self, mock_agent, mock_tool, tool_use, structured_output_context):
        """Test _stream with dynamic tool."""
        # Setup
        mock_agent.tool_registry.dynamic_tools["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify dynamic tool was used
        assert len(events) == 3  # 2 ToolStreamEvents and 1 ToolResultEvent
        assert isinstance(events[2], ToolResultEvent)
        assert tool_results[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_stream_with_structured_output_enabled(
        self, mock_agent, mock_tool, tool_use, structured_output_context
    ):
        """Test _stream when structured output is enabled."""
        # Setup
        structured_output_context.is_enabled = True
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Track kwargs passed to tool.stream
        stream_kwargs = {}

        async def mock_stream_with_kwargs(*args, **kwargs):
            stream_kwargs.update(kwargs)
            yield "streaming data"
            yield {"toolUseId": "test-id", "status": "success", "content": [{"text": "result"}]}

        mock_tool.stream = mock_stream_with_kwargs

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        async for _ in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            pass

        # Verify structured_output_context was passed
        assert "structured_output_context" in stream_kwargs
        assert stream_kwargs["structured_output_context"] == structured_output_context

    @pytest.mark.asyncio
    async def test_stream_with_exception(self, mock_agent, tool_use, structured_output_context):
        """Test _stream when tool execution raises an exception."""
        # Setup
        mock_tool = AsyncMock()

        async def mock_stream_error(*args, **kwargs):
            yield "some data"
            raise ValueError("Test error")

        mock_tool.stream = mock_stream_error
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify error handling
        assert len(events) == 2  # Stream event and error result
        assert isinstance(events[-1], ToolResultEvent)
        assert tool_results[0]["status"] == "error"
        assert "Error: Test error" in tool_results[0]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_stream_with_tool_result_event(self, mock_agent, tool_use, structured_output_context):
        """Test _stream when tool yields ToolResultEvent directly."""
        # Setup
        mock_tool = AsyncMock()

        async def mock_stream_with_result_event(*args, **kwargs):
            result = {"toolUseId": "test-id", "status": "success", "content": [{"text": "direct result"}]}
            yield ToolResultEvent(result)

        mock_tool.stream = mock_stream_with_result_event
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert tool_results[0]["content"][0]["text"] == "direct result"

    @pytest.mark.asyncio
    async def test_stream_with_tool_stream_event(self, mock_agent, tool_use, structured_output_context):
        """Test _stream when tool yields ToolStreamEvent directly."""
        # Setup
        mock_tool = AsyncMock()

        async def mock_stream_with_stream_event(*args, **kwargs):
            yield ToolStreamEvent(tool_use, "stream event data")
            yield {"toolUseId": "test-id", "status": "success", "content": [{"text": "result"}]}

        mock_tool.stream = mock_stream_with_stream_event
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify
        assert (
            len(events) == 3
        )  # ToolStreamEvent from first yield, ToolStreamEvent from second yield, and ToolResultEvent
        assert isinstance(events[0], ToolStreamEvent)
        assert isinstance(events[1], ToolStreamEvent)
        assert isinstance(events[2], ToolResultEvent)

    @pytest.mark.asyncio
    async def test_stream_invocation_state_update(self, mock_agent, mock_tool, tool_use, structured_output_context):
        """Test that invocation_state is properly updated."""
        # Setup
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {"custom": "value"}

        # Track invocation state in hook
        captured_state = None

        def capture_state(event):
            nonlocal captured_state
            if isinstance(event, BeforeToolCallEvent):
                captured_state = event.invocation_state.copy()
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = capture_state

        # Execute
        async for _ in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            pass

        # Verify invocation state was updated
        assert captured_state is not None
        assert "model" in captured_state
        assert "messages" in captured_state
        assert "system_prompt" in captured_state
        assert "tool_config" in captured_state
        assert captured_state["custom"] == "value"  # Original value preserved

    @pytest.mark.asyncio
    async def test_stream_with_trace(self, mock_agent, mock_tool, tool_use, structured_output_context):
        """Test _stream_with_trace method."""
        # Setup
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}
        cycle_trace = MagicMock(spec=Trace)
        cycle_trace.id = "trace-id"
        cycle_span = MagicMock()

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute with mocked tracer
        with patch("strands.tools.executors._executor.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_tracer.start_tool_call_span.return_value = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            with patch("strands.tools.executors._executor.trace_api.use_span"):
                events = []
                async for event in ToolExecutor._stream_with_trace(
                    mock_agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    structured_output_context,
                ):
                    events.append(event)

        # Verify
        assert len(events) == 3  # 2 ToolStreamEvents and 1 ToolResultEvent
        assert isinstance(events[-1], ToolResultEvent)
        mock_tracer.start_tool_call_span.assert_called_once_with(tool_use, cycle_span)
        mock_tracer.end_tool_call_span.assert_called_once()
        mock_agent.event_loop_metrics.add_tool_usage.assert_called_once()
        cycle_trace.add_child.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_abstract_method(self, executor):
        """Test that _execute is abstract and must be implemented."""
        # Verify the concrete test implementation works
        mock_agent = MagicMock()
        tool_uses = []
        tool_results = []
        cycle_trace = MagicMock()
        cycle_span = MagicMock()
        invocation_state = {}
        structured_output_context = MagicMock()

        # Should not raise since TestExecutor implements _execute
        gen = executor._execute(
            mock_agent,
            tool_uses,
            tool_results,
            cycle_trace,
            cycle_span,
            invocation_state,
            structured_output_context,
        )
        assert gen is not None

    def test_base_executor_interface(self):
        """Test the base executor interface and abstract nature."""
        # Verify ToolExecutor is abstract
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ToolExecutor()

        # Verify TestExecutor can be instantiated
        executor = TestExecutor()
        assert isinstance(executor, ToolExecutor)

    @pytest.mark.asyncio
    async def test_hook_modification_of_tool_use(self, mock_agent, mock_tool, tool_use, structured_output_context):
        """Test that hooks can modify tool_use."""
        # Setup
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hook to modify tool_use
        def modify_tool_use(event):
            if isinstance(event, BeforeToolCallEvent):
                event.tool_use["modified"] = True
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = modify_tool_use

        # Execute
        async for _ in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            pass

        # The hook should have been called
        assert mock_agent.hooks.invoke_callbacks.called

    @pytest.mark.asyncio
    async def test_hook_returns_none_tool(self, mock_agent, mock_tool, tool_use, structured_output_context):
        """Test behavior when hook returns None for selected_tool."""
        # Setup
        mock_agent.tool_registry.registry["test_tool"] = mock_tool
        tool_results = []
        invocation_state = {}

        # Setup hook to return None tool
        def return_none_tool(event):
            if isinstance(event, BeforeToolCallEvent):
                event.selected_tool = None
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = return_none_tool

        # Execute
        events = []
        async for event in ToolExecutor._stream(
            mock_agent, tool_use, tool_results, invocation_state, structured_output_context
        ):
            events.append(event)

        # Verify error result
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert tool_results[0]["status"] == "error"

    @pytest.mark.asyncio
    async def test_stream_with_trace_failed_tool(self, mock_agent, tool_use, structured_output_context):
        """Test _stream_with_trace when tool execution fails."""
        # Setup - no tool in registry
        tool_results = []
        invocation_state = {}
        cycle_trace = MagicMock(spec=Trace)
        cycle_trace.id = "trace-id"
        cycle_span = MagicMock()

        # Setup hooks
        def hook_side_effect(event):
            return event

        mock_agent.hooks.invoke_callbacks.side_effect = hook_side_effect

        # Execute with mocked tracer
        with patch("strands.tools.executors._executor.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_tracer.start_tool_call_span.return_value = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            with patch("strands.tools.executors._executor.trace_api.use_span"):
                events = []
                async for event in ToolExecutor._stream_with_trace(
                    mock_agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    structured_output_context,
                ):
                    events.append(event)

        # Verify error handling with trace
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert tool_results[0]["status"] == "error"
        # Metrics should still be recorded for failed tool
        mock_agent.event_loop_metrics.add_tool_usage.assert_called_once()
        # Tool success should be False
        call_args = mock_agent.event_loop_metrics.add_tool_usage.call_args
        assert call_args[0][3] is False  # 4th positional arg is tool_success
