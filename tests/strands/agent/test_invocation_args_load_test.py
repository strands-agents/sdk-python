"""
End-to-end load test for the new invocation_args parameter in Agent APIs.

This test verifies that the invocation_args parameter performs well under load
with concurrent requests, memory usage, and various parameter combinations.
"""

import asyncio
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock
import pytest

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import strands
from strands import Agent, tool
from strands.handlers.callback_handler import PrintingCallbackHandler


class LoadTestCallbackHandler:
    """Callback handler for load testing that tracks performance metrics."""
    
    def __init__(self):
        self.call_count = 0
        self.total_events = 0
        self.start_times = {}
        self.end_times = {}
    
    def callback_handler(self, **kwargs) -> None:
        """Track callback events for performance monitoring."""
        self.call_count += 1
        self.total_events += 1
        
        # Track timing for specific events
        if "init_event_loop" in kwargs:
            self.start_times[self.call_count] = time.time()
        elif "complete" in kwargs and kwargs.get("complete"):
            self.end_times[self.call_count] = time.time()


@tool
def load_test_calculator(operation: str, a: float, b: float) -> float:
    """Calculator tool for load testing."""
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
def load_test_echo(text: str) -> str:
    """Echo tool for load testing."""
    return f"Echo: {text}"


@pytest.fixture
def load_test_callback_handler():
    """Fixture providing a load test callback handler."""
    return LoadTestCallbackHandler()


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for load testing."""
    mock = Mock()
    
    async def mock_stream(*args, **kwargs):
        # Simulate processing time
        await asyncio.sleep(0.01)  # 10ms processing time
        yield {"role": "assistant", "content": [{"text": "Load test response"}]}
    
    mock.stream = mock_stream
    return mock


@pytest.fixture
def load_test_agent(mock_model):
    """Fixture providing an agent for load testing."""
    return Agent(tools=[load_test_calculator, load_test_echo], model=mock_model)


class TestInvocationArgsLoadTest:
    """Load tests for invocation_args parameter."""
    
    def test_concurrent_agent_calls_with_invocation_args(self, load_test_agent, load_test_callback_handler):
        """Test concurrent agent calls using invocation_args parameter."""
        num_concurrent_calls = 10
        start_time = time.time()
        
        def make_agent_call(call_id):
            """Make a single agent call with invocation_args."""
            try:
                result = load_test_agent(
                    f"Calculate {call_id} + {call_id} using the calculator tool",
                    invocation_args={
                        "callback_handler": load_test_callback_handler.callback_handler,
                        "call_id": call_id,
                        "test_param": f"value_{call_id}"
                    }
                )
                return {"call_id": call_id, "success": True, "result": result}
            except Exception as e:
                return {"call_id": call_id, "success": False, "error": str(e)}
        
        # Execute concurrent calls
        with ThreadPoolExecutor(max_workers=num_concurrent_calls) as executor:
            futures = [executor.submit(make_agent_call, i) for i in range(num_concurrent_calls)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all calls succeeded
        successful_calls = [r for r in results if r["success"]]
        assert len(successful_calls) == num_concurrent_calls, f"Expected {num_concurrent_calls} successful calls, got {len(successful_calls)}"
        
        # Verify callback handler was called
        assert load_test_callback_handler.call_count > 0, "Callback handler should have been called"
        
        # Verify performance (should complete within reasonable time)
        assert total_time < 5.0, f"Load test took too long: {total_time:.2f}s"
        
        print(f"✅ Load test completed: {num_concurrent_calls} concurrent calls in {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_async_calls_with_invocation_args(self, load_test_agent, load_test_callback_handler):
        """Test concurrent async agent calls using invocation_args parameter."""
        num_concurrent_calls = 15
        
        async def make_async_call(call_id):
            """Make a single async agent call with invocation_args."""
            try:
                result = await load_test_agent.invoke_async(
                    f"Echo 'Load test {call_id}' using the echo tool",
                    invocation_args={
                        "callback_handler": load_test_callback_handler.callback_handler,
                        "call_id": call_id,
                        "async_test": True
                    }
                )
                return {"call_id": call_id, "success": True, "result": result}
            except Exception as e:
                return {"call_id": call_id, "success": False, "error": str(e)}
        
        start_time = time.time()
        
        # Execute concurrent async calls
        tasks = [make_async_call(i) for i in range(num_concurrent_calls)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all calls succeeded
        successful_calls = [r for r in results if r["success"]]
        assert len(successful_calls) == num_concurrent_calls, f"Expected {num_concurrent_calls} successful calls, got {len(successful_calls)}"
        
        # Verify callback handler was called
        assert load_test_callback_handler.call_count > 0, "Callback handler should have been called"
        
        # Verify performance
        assert total_time < 3.0, f"Async load test took too long: {total_time:.2f}s"
        
        print(f"✅ Async load test completed: {num_concurrent_calls} concurrent calls in {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_calls_with_invocation_args(self, load_test_agent, load_test_callback_handler):
        """Test concurrent streaming calls using invocation_args parameter."""
        num_concurrent_calls = 8
        
        async def make_streaming_call(call_id):
            """Make a single streaming call with invocation_args."""
            try:
                events = []
                async for event in load_test_agent.stream_async(
                    f"Calculate {call_id} * 2 using the calculator tool",
                    invocation_args={
                        "callback_handler": load_test_callback_handler.callback_handler,
                        "call_id": call_id,
                        "streaming_test": True
                    }
                ):
                    events.append(event)
                
                return {"call_id": call_id, "success": True, "event_count": len(events)}
            except Exception as e:
                return {"call_id": call_id, "success": False, "error": str(e)}
        
        start_time = time.time()
        
        # Execute concurrent streaming calls
        tasks = [make_streaming_call(i) for i in range(num_concurrent_calls)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all calls succeeded
        successful_calls = [r for r in results if r["success"]]
        assert len(successful_calls) == num_concurrent_calls, f"Expected {num_concurrent_calls} successful calls, got {len(successful_calls)}"
        
        # Verify events were received
        total_events = sum(r["event_count"] for r in successful_calls)
        assert total_events > 0, "Should have received streaming events"
        
        # Verify callback handler was called
        assert load_test_callback_handler.call_count > 0, "Callback handler should have been called"
        
        # Verify performance
        assert total_time < 4.0, f"Streaming load test took too long: {total_time:.2f}s"
        
        print(f"✅ Streaming load test completed: {num_concurrent_calls} concurrent calls, {total_events} events in {total_time:.2f}s")
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_usage_with_invocation_args(self, load_test_agent, load_test_callback_handler):
        """Test memory usage with invocation_args parameter."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple calls with various invocation_args
        num_calls = 50
        large_params = {
            "large_data": "x" * 1000,  # 1KB string
            "numbers": list(range(100)),  # List of 100 numbers
            "nested_dict": {f"key_{i}": f"value_{i}" for i in range(50)}
        }
        
        for i in range(num_calls):
            result = load_test_agent(
                f"Calculate {i} + 1 using the calculator tool",
                invocation_args={
                    "callback_handler": load_test_callback_handler.callback_handler,
                    "call_id": i,
                    **large_params
                }
            )
            assert result.stop_reason is not None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 50 calls)
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.2f}MB"
        
        print(f"✅ Memory test completed: {memory_increase:.2f}MB increase for {num_calls} calls")
    
    def test_mixed_api_usage_load_test(self, load_test_agent, load_test_callback_handler):
        """Test mixed usage of new and deprecated APIs under load."""
        num_calls = 20
        warnings_caught = []
        
        def make_mixed_call(call_id):
            """Make calls using both new and deprecated APIs."""
            try:
                if call_id % 2 == 0:
                    # Use new invocation_args API
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args={"callback_handler": load_test_callback_handler.callback_handler}
                    )
                else:
                    # Use deprecated kwargs API (should emit warning)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        result = load_test_agent(
                            f"Calculate {call_id} + 1 using the calculator tool",
                            callback_handler=load_test_callback_handler.callback_handler
                        )
                        if w:
                            warnings_caught.extend(w)
                
                return {"call_id": call_id, "success": True, "result": result}
            except Exception as e:
                return {"call_id": call_id, "success": False, "error": str(e)}
        
        start_time = time.time()
        
        # Execute mixed calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_mixed_call, i) for i in range(num_calls)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all calls succeeded
        successful_calls = [r for r in results if r["success"]]
        assert len(successful_calls) == num_calls, f"Expected {num_calls} successful calls, got {len(successful_calls)}"
        
        # Verify deprecation warnings were emitted for kwargs usage
        deprecation_warnings = [w for w in warnings_caught if issubclass(w.category, DeprecationWarning)]
        expected_warnings = num_calls // 2  # Half the calls use kwargs
        assert len(deprecation_warnings) >= expected_warnings, f"Expected at least {expected_warnings} deprecation warnings, got {len(deprecation_warnings)}"
        
        # Verify performance
        assert total_time < 3.0, f"Mixed API load test took too long: {total_time:.2f}s"
        
        print(f"✅ Mixed API load test completed: {num_calls} calls, {len(deprecation_warnings)} warnings in {total_time:.2f}s")
    
    def test_parameter_precedence_load_test(self, load_test_agent, load_test_callback_handler):
        """Test parameter precedence under load conditions."""
        num_calls = 15
        other_handler = Mock()
        
        def make_precedence_call(call_id):
            """Test parameter precedence with both invocation_args and kwargs."""
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args={"callback_handler": load_test_callback_handler.callback_handler},
                        callback_handler=other_handler  # This should be ignored
                    )
                    
                    # Verify deprecation warning was emitted
                    deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
                    assert len(deprecation_warnings) == 1, "Should emit exactly one deprecation warning"
                
                return {"call_id": call_id, "success": True, "result": result}
            except Exception as e:
                return {"call_id": call_id, "success": False, "error": str(e)}
        
        start_time = time.time()
        
        # Execute precedence test calls
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(make_precedence_call, i) for i in range(num_calls)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all calls succeeded
        successful_calls = [r for r in results if r["success"]]
        assert len(successful_calls) == num_calls, f"Expected {num_calls} successful calls, got {len(successful_calls)}"
        
        # Verify invocation_args handler was used (not kwargs handler)
        assert load_test_callback_handler.call_count > 0, "invocation_args callback handler should have been called"
        other_handler.assert_not_called(), "kwargs callback handler should not have been called"
        
        # Verify performance
        assert total_time < 2.5, f"Precedence load test took too long: {total_time:.2f}s"
        
        print(f"✅ Precedence load test completed: {num_calls} calls in {total_time:.2f}s")
    
    def test_edge_cases_load_test(self, load_test_agent):
        """Test edge cases under load conditions."""
        num_calls = 25
        
        def make_edge_case_call(call_id):
            """Test various edge cases with invocation_args."""
            try:
                # Test different edge cases
                if call_id % 5 == 0:
                    # None invocation_args
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args=None
                    )
                elif call_id % 5 == 1:
                    # Empty dict invocation_args
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args={}
                    )
                elif call_id % 5 == 2:
                    # None callback_handler
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args={"callback_handler": None}
                    )
                elif call_id % 5 == 3:
                    # Large parameter set
                    large_params = {f"param_{i}": f"value_{i}" for i in range(20)}
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args=large_params
                    )
                else:
                    # Normal case
                    result = load_test_agent(
                        f"Calculate {call_id} + 1 using the calculator tool",
                        invocation_args={"test_param": f"value_{call_id}"}
                    )
                
                return {"call_id": call_id, "success": True, "result": result}
            except Exception as e:
                return {"call_id": call_id, "success": False, "error": str(e)}
        
        start_time = time.time()
        
        # Execute edge case calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_edge_case_call, i) for i in range(num_calls)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all calls succeeded
        successful_calls = [r for r in results if r["success"]]
        assert len(successful_calls) == num_calls, f"Expected {num_calls} successful calls, got {len(successful_calls)}"
        
        # Verify performance
        assert total_time < 3.0, f"Edge case load test took too long: {total_time:.2f}s"
        
        print(f"✅ Edge case load test completed: {num_calls} calls in {total_time:.2f}s")
