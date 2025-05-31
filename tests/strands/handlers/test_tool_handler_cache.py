import unittest.mock

import pytest

from strands import Agent
from strands.handlers.tool_handler import AgentToolHandler
from strands.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry():
    """Fixture providing a tool registry"""
    return ToolRegistry()


@pytest.fixture
def tool_handler(tool_registry):
    """Fixture providing a tool handler"""
    return AgentToolHandler(tool_registry=tool_registry)


@pytest.fixture
def mock_tool():
    """Fixture providing a mock tool"""
    mock = unittest.mock.Mock()
    mock.tool_name = "mock_tool"
    mock.invoke.return_value = {"toolUseId": "test_id", "status": "success", "content": [{"text": "Test result"}]}
    return mock


@pytest.fixture
def agent_with_cache():
    """Fixture providing an Agent instance with caching enabled"""
    agent = Agent(enable_tool_cache=True, tool_cache_size=10, tool_cache_ttl=60)
    return agent


def test_tool_handler_process_with_cache(tool_handler, tool_registry, mock_tool, agent_with_cache):
    """Test that tool handler utilizes the cache"""
    # Register mock tool in the registry
    tool_registry.registry["mock_tool"] = mock_tool

    # Tool use request
    tool_use = {"toolUseId": "test_id", "name": "mock_tool", "input": {"param": "value"}}

    # First call - cache miss
    result1 = tool_handler.process(
        tool=tool_use,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent_with_cache,
    )

    # Verify mock tool's invoke method was called
    mock_tool.invoke.assert_called_once()
    mock_tool.invoke.reset_mock()

    # Second call - cache hit
    result2 = tool_handler.process(
        tool=tool_use,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent_with_cache,
    )

    # Verify mock tool's invoke method was not called (result from cache)
    mock_tool.invoke.assert_not_called()

    # Verify results are the same
    assert result1 == result2

    # Verify cache statistics
    stats = agent_with_cache.get_tool_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_tool_handler_process_with_cacheable_tools(tool_handler, tool_registry, mock_tool):
    """Test that cacheable tools list works correctly"""
    # Register mock tools in the registry
    tool_registry.registry["mock_tool"] = mock_tool
    tool_registry.registry["other_tool"] = unittest.mock.Mock()

    # Create agent with cacheable tools list
    agent = Agent(
        enable_tool_cache=True,
        cacheable_tools=["mock_tool"],
    )

    # mock_tool call
    tool_use1 = {"toolUseId": "test_id1", "name": "mock_tool", "input": {"param": "value"}}

    # other_tool call
    tool_use2 = {"toolUseId": "test_id2", "name": "other_tool", "input": {"param": "value"}}

    # mock_tool call - should be cached
    tool_handler.process(
        tool=tool_use1,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent,
    )

    # other_tool call - should not be cached
    tool_handler.process(
        tool=tool_use2,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent,
    )

    # Verify cache statistics
    stats = agent.get_tool_cache_stats()
    assert stats["size"] == 1  # Only mock_tool should be cached


def test_tool_handler_process_with_uncacheable_tools(tool_handler, tool_registry, mock_tool):
    """Test that uncacheable tools list works correctly"""
    # Register mock tools in the registry
    tool_registry.registry["mock_tool"] = mock_tool

    # Create and configure other_tool mock
    other_tool = unittest.mock.Mock()
    other_tool.tool_name = "other_tool"
    other_tool.invoke.return_value = {
        "toolUseId": "test_id2",
        "status": "success",
        "content": [{"text": "Other tool result"}],
    }
    tool_registry.registry["other_tool"] = other_tool

    # Create agent with uncacheable tools list
    agent = Agent(
        enable_tool_cache=True,
        uncacheable_tools=["mock_tool"],  # Only mock_tool is uncacheable
    )

    # mock_tool call
    tool_use1 = {"toolUseId": "test_id1", "name": "mock_tool", "input": {"param": "value"}}

    # other_tool call
    tool_use2 = {"toolUseId": "test_id2", "name": "other_tool", "input": {"param": "value"}}

    # mock_tool call - should not be cached
    tool_handler.process(
        tool=tool_use1,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent,
    )

    # other_tool call - should be cached
    tool_handler.process(
        tool=tool_use2,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent,
    )

    # Verify cache statistics
    stats = agent.get_tool_cache_stats()
    assert stats["size"] == 1  # Only other_tool should be cached


def test_tool_handler_process_error_not_cached(tool_handler, tool_registry, mock_tool, agent_with_cache):
    """Test that error results are not cached"""
    # Register mock tool in the registry
    tool_registry.registry["mock_tool"] = mock_tool

    # Configure mock tool to return an error result
    error_result = {"toolUseId": "test_id", "status": "error", "content": [{"text": "Error result"}]}
    mock_tool.invoke.return_value = error_result

    # Tool use request
    tool_use = {"toolUseId": "test_id", "name": "mock_tool", "input": {"param": "value"}}

    # First call - should invoke the tool and get an error result
    result1 = tool_handler.process(
        tool=tool_use,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent_with_cache,
    )

    # Verify the result is an error
    assert result1["status"] == "error"

    # Reset the mock to track the next call
    mock_tool.invoke.reset_mock()

    # Second call with the same parameters - should invoke the tool again
    result2 = tool_handler.process(
        tool=tool_use,
        model=unittest.mock.Mock(),
        system_prompt="Test prompt",
        messages=[],
        tool_config={},
        callback_handler=unittest.mock.Mock(),
        agent=agent_with_cache,
    )

    # Verify the tool was invoked again
    mock_tool.invoke.assert_called_once()

    # Verify the results are the same
    assert result1 == result2
