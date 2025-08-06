import threading
import unittest.mock

import pytest

from strands import tool as sa_tool
from strands.experimental.hooks import AfterToolInvocationEvent as SAAfterToolInvocationEvent
from strands.experimental.hooks import BeforeToolInvocationEvent as SABeforeToolInvocationEvent
from strands.hooks import HookRegistry as SAHookRegistry
from strands.tools.registry import ToolRegistry as SAToolRegistry


@pytest.fixture
def hook_events():
    return []


@pytest.fixture
def tool_hook(hook_events):
    def callback(event):
        hook_events.append(event)
        return event

    return callback


@pytest.fixture
def hook_registry(tool_hook):
    registry = SAHookRegistry()
    registry.add_callback(SABeforeToolInvocationEvent, tool_hook)
    registry.add_callback(SAAfterToolInvocationEvent, tool_hook)
    return registry


@pytest.fixture
def tool_events():
    return []


@pytest.fixture
def weather_tool():
    @sa_tool(name="weather_tool")
    def func():
        return "sunny"

    return func


@pytest.fixture
def temperature_tool():
    @sa_tool(name="temperature_tool")
    def func():
        return "75F"

    return func


@pytest.fixture
def exception_tool():
    @sa_tool(name="exception_tool")
    def func():
        pass

    async def mock_stream(_tool_use, _invocation_state):
        raise RuntimeError("Tool error")
        yield  # make generator

    func.stream = mock_stream
    return func


@pytest.fixture
def thread_tool(tool_events):
    @sa_tool(name="thread_tool")
    def func():
        tool_events.append({"thread_name": threading.current_thread().name})
        return "threaded"

    return func


@pytest.fixture
def tool_registry(weather_tool, temperature_tool, exception_tool, thread_tool):
    registry = SAToolRegistry()
    registry.register_tool(weather_tool)
    registry.register_tool(temperature_tool)
    registry.register_tool(exception_tool)
    registry.register_tool(thread_tool)
    return registry


@pytest.fixture
def agent(tool_registry, hook_registry):
    mock_agent = unittest.mock.Mock()
    mock_agent.tool_registry = tool_registry
    mock_agent.hooks = hook_registry
    return mock_agent


@pytest.fixture
def invocation_state():
    return {
        "tool_results": [],
        "event_loop_cycle_span": unittest.mock.Mock(),
        "event_loop_cycle_trace": unittest.mock.Mock(),
    }
