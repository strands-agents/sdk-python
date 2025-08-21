import threading
import unittest.mock

import pytest

import strands
from strands.experimental.hooks import AfterToolInvocationEvent, BeforeToolInvocationEvent
from strands.hooks import HookRegistry
from strands.tools.registry import ToolRegistry


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
    registry = HookRegistry()
    registry.add_callback(BeforeToolInvocationEvent, tool_hook)
    registry.add_callback(AfterToolInvocationEvent, tool_hook)
    return registry


@pytest.fixture
def tool_events():
    return []


@pytest.fixture
def weather_tool():
    @strands.tool(name="weather_tool")
    def func():
        return "sunny"

    return func


@pytest.fixture
def temperature_tool():
    @strands.tool(name="temperature_tool")
    def func():
        return "75F"

    return func


@pytest.fixture
def exception_tool():
    @strands.tool(name="exception_tool")
    def func():
        pass

    async def mock_stream(_tool_use, _invocation_state):
        raise RuntimeError("Tool error")
        yield  # make generator

    func.stream = mock_stream
    return func


@pytest.fixture
def thread_tool(tool_events):
    @strands.tool(name="thread_tool")
    def func():
        tool_events.append({"thread_name": threading.current_thread().name})
        return "threaded"

    return func


@pytest.fixture
def tool_registry(weather_tool, temperature_tool, exception_tool, thread_tool):
    registry = ToolRegistry()
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
def tool_results():
    return []


@pytest.fixture
def cycle_trace():
    return unittest.mock.Mock()


@pytest.fixture
def cycle_span():
    return unittest.mock.Mock()


@pytest.fixture
def invocation_state():
    return {}
