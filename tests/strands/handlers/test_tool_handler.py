import pytest

import strands
from strands.handlers.tool_handler import AgentToolHandler


@pytest.fixture
def agent():
    return strands.Agent()


@pytest.fixture
def tool_handler(agent):
    return AgentToolHandler(agent)


@pytest.fixture
def tool_use_identity(agent):
    @strands.tools.tool
    def identity(a: int) -> int:
        return a

    agent.tool_registry.register_tool(identity)

    return {"toolUseId": "identity", "name": "identity", "input": {"a": 1}}


def test_process(tool_handler, tool_use_identity, generate):
    process = tool_handler.run_tool(
        tool_use_identity,
        kwargs={},
    )

    _, tru_result = generate(process)
    exp_result = {"toolUseId": "identity", "status": "success", "content": [{"text": "1"}]}

    assert tru_result == exp_result


def test_process_missing_tool(tool_handler, generate):
    process = tool_handler.run_tool(
        tool={"toolUseId": "missing", "name": "missing", "input": {}},
        kwargs={},
    )

    _, tru_result = generate(process)
    exp_result = {
        "toolUseId": "missing",
        "status": "error",
        "content": [{"text": "Unknown tool: missing"}],
    }

    assert tru_result == exp_result
