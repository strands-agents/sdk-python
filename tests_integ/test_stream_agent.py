"""
Test script for Strands' custom callback handler functionality.
Demonstrates different patterns of callback handling and processing.
"""

import logging
import threading
import time

from strands import Agent, tool

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


@tool
def wait(seconds: int) -> None:
    """Waits x seconds based on the user input.
    Seconds - seconds to wait"""
    time.sleep(seconds)


class ToolCountingCallbackHandler:
    def __init__(self):
        self.tool_count = 0
        self.message_count = 0

    def callback_handler(self, **kwargs) -> None:
        """
        Custom callback handler that processes and displays different types of events.

        Args:
            **kwargs: Callback event data including:
                - data: Regular output
                - complete: Completion status
                - message: Message processing
                - current_tool_use: Tool execution
        """
        # Extract event data
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        message = kwargs.get("message", {})
        current_tool_use = kwargs.get("current_tool_use", {})

        # Handle regular data output
        if data:
            print(f"🔄 Data: {data}")

        # Handle tool execution events
        if current_tool_use:
            self.tool_count += 1
            tool_name = current_tool_use.get("name", "")
            tool_input = current_tool_use.get("input", {})
            print(f"🛠️ Tool Execution #{self.tool_count}\nTool: {tool_name}\nInput: {tool_input}")

        # Handle message processing
        if message:
            self.message_count += 1
            print(f"📝 Message #{self.message_count}")

        # Handle completion
        if complete:
            self.console.print("✨ Callback Complete", style="bold green")


def test_basic_interaction():
    """Test basic AGI interaction with custom callback handler."""
    print("\nTesting Basic Interaction")

    # Initialize agent with custom handler
    agent = Agent(
        callback_handler=ToolCountingCallbackHandler().callback_handler,
        load_tools_from_directory=False,
    )

    # Simple prompt to test callbacking
    agent("Tell me a short joke from your general knowledge")

    print("\nBasic Interaction Complete")


def test_parallel_async_interaction():
    """Test that concurrent agent invocations are not allowed"""

    # Initialize agent
    agent = Agent(
        callback_handler=ToolCountingCallbackHandler().callback_handler, load_tools_from_directory=False, tools=[wait]
    )

    # Track results from both threads
    results = {"thread1": None, "thread2": None, "exception": None}

    def invoke_agent_1():
        """First invocation - should succeed"""
        try:
            result = agent("wait 5 seconds")
            results["thread1"] = result
        except Exception as e:
            results["thread1"] = e

    def invoke_agent_2():
        """Second invocation - should fail with exception"""
        try:
            result = agent("wait 5 seconds")
            results["thread2"] = result
        except Exception as e:
            results["thread2"] = e
            results["exception"] = e

    # Start first invocation
    thread1 = threading.Thread(target=invoke_agent_1)
    thread1.start()

    # Give it time to start and begin waiting
    time.sleep(1)

    # Try second invocation while first is still running
    thread2 = threading.Thread(target=invoke_agent_2)
    thread2.start()

    thread1.join()
    thread2.join()

    # Assertions
    assert results["thread1"] is not None, "First invocation should complete"
    assert not isinstance(results["thread1"], Exception), "First invocation should succeed"

    assert results["exception"] is not None, "Second invocation should throw exception"
    assert isinstance(results["thread2"], Exception), "Second invocation should fail"

    expected_message = "Agent is already processing a request. Concurrent invocations are not supported"
    assert expected_message in str(results["thread2"]), (
        f"Exception message should contain '{expected_message}', but got: {str(results['thread2'])}"
    )
