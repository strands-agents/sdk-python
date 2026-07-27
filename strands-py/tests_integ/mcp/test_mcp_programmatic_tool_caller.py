"""Integration tests for the programmatic tool caller driving real MCP tools.

These spawn a real MCP server over stdio and verify that agent-authored code run
by ``programmatic_tool_caller`` can call MCP tools -- sequentially, in loops, and
concurrently -- including MCP tools whose names are not valid Python identifiers.

No model is invoked: the caller is exercised as a direct tool call.
"""

import pytest
from mcp import StdioServerParameters, stdio_client

from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient
from strands.vended_tools import programmatic_tool_caller
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def mcp_agent():
    """Build an agent with the programmatic tool caller plus real MCP tools."""
    client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="python",
                args=["tests_integ/mcp/programmatic_tool_caller_server.py"],
            )
        )
    )
    with client:
        mcp_tools = client.list_tools_sync()
        yield Agent(model=MockedModelProvider([]), tools=[programmatic_tool_caller, *mcp_tools])


def _run(agent, code):
    return agent.tool.programmatic_tool_caller(code=code, record_direct_tool_call=False)


def _text(result):
    return result["content"][0]["text"]


def test_calls_mcp_tool(mcp_agent):
    result = _run(mcp_agent, 'print(await ptc_echo(text="hello"))')
    assert result["status"] == "success"
    assert _text(result) == "echo:hello"


def test_loops_over_mcp_tool_calls(mcp_agent):
    result = _run(mcp_agent, "for index in range(3):\n    print(await ptc_add(a=index, b=10))")
    assert result["status"] == "success"
    assert _text(result).split() == ["10", "11", "12"]


def test_runs_mcp_tools_concurrently(mcp_agent):
    result = _run(
        mcp_agent,
        "results = await asyncio.gather(ptc_add(a=1, b=1), ptc_add(a=2, b=2), ptc_echo(text='p'))\nprint(results)",
    )
    assert result["status"] == "success"
    assert _text(result) == "['2', '4', 'echo:p']"


def test_chains_mcp_tool_results(mcp_agent):
    result = _run(
        mcp_agent,
        "total = await ptc_add(a=20, b=22)\nprint(await ptc_echo(text=f'total={total}'))",
    )
    assert result["status"] == "success"
    assert _text(result) == "echo:total=42"


def test_mcp_tool_error_is_catchable(mcp_agent):
    result = _run(
        mcp_agent,
        "try:\n    await ptc_boom()\nexcept RuntimeError as error:\n    print(f'caught: {error}')",
    )
    assert result["status"] == "success"
    assert "caught:" in _text(result)
    assert "mcp tool exploded" in _text(result)


def test_hyphenated_mcp_tool_is_callable(mcp_agent):
    """MCP servers commonly use hyphens; the code must still be able to call them."""
    result = _run(mcp_agent, 'print(await ptc_dash(value="x"))')
    assert result["status"] == "success"
    assert _text(result) == "dash:x"


def test_dotted_mcp_tool_is_callable(mcp_agent):
    result = _run(mcp_agent, 'print(await ptc_dot(value="y"))')
    assert result["status"] == "success"
    assert _text(result) == "dot:y"


def test_only_printed_output_is_returned(mcp_agent):
    """An MCP result that is never printed must not leak into the agent's context."""
    result = _run(mcp_agent, 'await ptc_echo(text="secret")')
    assert result["status"] == "success"
    assert _text(result) == "(no output)"
    assert "secret" not in _text(result)
