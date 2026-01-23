#!/usr/bin/env python3
"""Integration tests for tool retry hook mechanism.

Tests that the AfterToolCallEvent.retry field works correctly with real agents,
allowing hooks to trigger tool re-execution on failures.
"""

from strands import Agent, tool
from strands.hooks import AfterToolCallEvent
from strands.hooks.registry import HookProvider, HookRegistry
from tests_integ.conftest import retry_on_flaky


def make_failing_tool(fail_times: int = 1):
    """Create a tool that fails a configurable number of times before succeeding."""
    state = {"call_count": 0, "fail_times": fail_times}

    @tool(name="flaky_tool")
    def _flaky_tool(message: str) -> str:
        """A tool that fails a few times before succeeding.

        Args:
            message: A message to include in the response.
        """
        state["call_count"] += 1
        if state["call_count"] <= state["fail_times"]:
            raise RuntimeError(f"Simulated failure {state['call_count']}")
        return f"Success on attempt {state['call_count']}: {message}"

    _flaky_tool.state = state
    return _flaky_tool


class SimpleRetryHook(HookProvider):
    """A simple hook that retries failed tool calls up to max_attempts times."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self._attempts: dict[str, int] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self._handle_after_tool_call)

    def _handle_after_tool_call(self, event: AfterToolCallEvent) -> None:
        tool_use_id = str(event.tool_use.get("toolUseId", ""))

        # Track attempts per tool_use_id
        current_attempt = self._attempts.get(tool_use_id, 0) + 1
        self._attempts[tool_use_id] = current_attempt

        # Check for error status - @tool decorator catches exceptions and returns error results
        is_error = event.result.get("status") == "error"

        # If there was an error and we haven't exceeded max attempts, retry
        if is_error and current_attempt < self.max_attempts:
            event.retry = True


@retry_on_flaky("LLM responses may vary in tool calling behavior")
def test_tool_retry_hook_retries_on_failure():
    """Test that a hook can trigger tool retry on failure."""
    flaky_tool = make_failing_tool(fail_times=1)
    retry_hook = SimpleRetryHook(max_attempts=3)

    agent = Agent(tools=[flaky_tool], hooks=[retry_hook])

    # Ask the agent to use the tool
    result = agent("Use the flaky_tool with message 'hello'")

    # Tool should have been called twice (1 failure + 1 success)
    assert flaky_tool.state["call_count"] == 2

    # The result should contain the success message
    assert "Success on attempt 2" in str(result)


@retry_on_flaky("LLM responses may vary in tool calling behavior")
def test_tool_retry_hook_respects_max_attempts():
    """Test that retry hook respects max_attempts limit."""
    # Tool that always fails
    flaky_tool = make_failing_tool(fail_times=100)
    retry_hook = SimpleRetryHook(max_attempts=3)

    agent = Agent(tools=[flaky_tool], hooks=[retry_hook])

    # Ask the agent to use the tool - it will fail but be retried up to max_attempts
    agent("Use the flaky_tool with message 'test'")

    # Tool should have been called exactly max_attempts times
    assert flaky_tool.state["call_count"] == 3


def test_tool_retry_hook_direct_tool_invocation():
    """Test retry hook works with direct tool invocation."""
    flaky_tool = make_failing_tool(fail_times=2)
    retry_hook = SimpleRetryHook(max_attempts=5)

    agent = Agent(tools=[flaky_tool], hooks=[retry_hook])

    # Call tool directly
    result = agent.tool.flaky_tool(message="direct call")

    # Tool should have been called 3 times (2 failures + 1 success)
    assert flaky_tool.state["call_count"] == 3
    assert result["status"] == "success"
    assert "Success on attempt 3" in result["content"][0]["text"]


def test_tool_retry_hook_no_retry_on_success():
    """Test that successful tool calls are not retried."""
    call_count = {"count": 0}

    @tool(name="success_tool")
    def success_tool(value: str) -> str:
        """A tool that always succeeds.

        Args:
            value: A value to return.
        """
        call_count["count"] += 1
        return f"Result: {value}"

    retry_hook = SimpleRetryHook(max_attempts=3)
    agent = Agent(tools=[success_tool], hooks=[retry_hook])

    # Call tool directly
    result = agent.tool.success_tool(value="test")

    # Tool should have been called only once
    assert call_count["count"] == 1
    assert result["status"] == "success"
