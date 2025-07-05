import unittest.mock

import pytest

import strands


@pytest.fixture
def tool_registry():
    return strands.tools.registry.ToolRegistry()


@pytest.fixture
def tool_handler(tool_registry):
    return strands.handlers.tool_handler.AgentToolHandler(tool_registry)


@pytest.fixture
def tool_use_identity(tool_registry):
    @strands.tools.tool
    def identity(a: int):
        yield {"event": "abc"}
        return a

    tool_registry.register_tool(identity)

    return {"toolUseId": "identity", "name": "identity", "input": {"a": 1}}


def test_process(tool_handler, tool_use_identity, generate):
    process = tool_handler.process(
        tool_use_identity,
        model=unittest.mock.Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    tru_events, tru_result = generate(process)
    exp_events = [{"event": "abc"}]
    exp_result = {"toolUseId": "identity", "status": "success", "content": [{"text": "1"}]}

    assert tru_events == exp_events and tru_result == exp_result


def test_process_missing_tool(tool_handler, generate):
    process = tool_handler.process(
        tool_use={"toolUseId": "missing", "name": "missing", "input": {}},
        model=unittest.mock.Mock(),
        system_prompt="p1",
        messages=[],
        tool_config={},
        kwargs={},
    )

    tru_events, tru_result = generate(process)
    exp_events = []
    exp_result = {
        "toolUseId": "missing",
        "status": "error",
        "content": [{"text": "Unknown tool: missing"}],
    }

    assert tru_events == exp_events and tru_result == exp_result
