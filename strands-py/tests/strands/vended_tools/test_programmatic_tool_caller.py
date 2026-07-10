"""Tests for the programmatic tool caller.

The tool runs agent-authored Python code in which the agent's other tools are
exposed as ``async`` functions; only ``print()`` output is returned. The tests
drive it through a real ``Agent`` (with a mocked model, which is never invoked
for direct tool calls) so the full tool-caller/registry path is exercised.
"""

import pytest

from strands import Agent, tool
from strands.types.tools import ToolContext
from strands.vended_tools.programmatic_tool_caller import (
    make_programmatic_tool_caller,
    programmatic_tool_caller,
)
from strands.vended_tools.programmatic_tool_caller.types import (
    DEFAULT_PROGRAMMATIC_TOOL_CALLER_DESCRIPTION,
)
from tests.fixtures.mocked_model_provider import MockedModelProvider


@tool
def echo(text: str) -> str:
    """Return the given text."""
    return text


@tool
def adder(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(name="boom")
def boom() -> str:
    """Always fail."""
    raise ValueError("kaboom")


def _agent(*tools):
    """Build an Agent whose model is never invoked (direct tool calls only)."""
    return Agent(model=MockedModelProvider([]), tools=[*tools])


def _call(caller_tool, agent, code):
    """Invoke the programmatic tool caller as a direct tool call, return its ToolResult."""
    # register the caller tool on the agent so it can be excluded from its own namespace
    agent.tool_registry.register_tool(caller_tool)
    return getattr(agent.tool, caller_tool.tool_name)(code=code, record_direct_tool_call=False)


def _text(result):
    return result["content"][0]["text"]


class TestExecution:
    """End-to-end execution through a real Agent."""

    @pytest.mark.asyncio
    async def test_single_tool_call_prints_result(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            'result = await echo(text="hello")\nprint(result)',
        )
        assert result["status"] == "success"
        assert _text(result) == "hello"

    @pytest.mark.asyncio
    async def test_only_print_output_is_returned(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, 'await echo(text="secret")')
        assert result["status"] == "success"
        # The tool result was awaited but never printed, so it must not leak out.
        assert _text(result) == "(no output)"
        assert "secret" not in _text(result)

    @pytest.mark.asyncio
    async def test_loop_over_tool_calls(self):
        agent = _agent(adder)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            "for i in range(3):\n    print(await adder(a=i, b=10))",
        )
        assert result["status"] == "success"
        assert _text(result).split() == ["10", "11", "12"]

    @pytest.mark.asyncio
    async def test_parallel_execution_with_gather(self):
        agent = _agent(adder)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            "results = await asyncio.gather(adder(a=1, b=1), adder(a=2, b=2))\nprint(results)",
        )
        assert result["status"] == "success"
        assert _text(result) == "['2', '4']"

    @pytest.mark.asyncio
    async def test_stdlib_import_inside_code(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            "import math\nprint(math.factorial(5))",
        )
        assert result["status"] == "success"
        assert _text(result) == "120"


class TestToolExposure:
    """Which tools are visible to the executed code."""

    @pytest.mark.asyncio
    async def test_caller_does_not_expose_itself(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            "print(programmatic_tool_caller)",
        )
        # Referencing the caller by name should be a NameError (it isn't exposed).
        assert result["status"] == "error"
        assert "NameError" in _text(result)

    @pytest.mark.asyncio
    async def test_allowed_tools_filters_exposed_set(self):
        agent = _agent(echo, adder)
        tool_caller = make_programmatic_tool_caller(allowed_tools=["adder"])
        # adder is allowed...
        allowed = _call(tool_caller, agent, "print(await adder(a=1, b=2))")
        assert allowed["status"] == "success"
        assert _text(allowed) == "3"
        # ...echo is not.
        denied = _call(tool_caller, agent, 'await echo(text="x")')
        assert denied["status"] == "error"
        assert "NameError" in _text(denied)

    @pytest.mark.asyncio
    async def test_unknown_allowed_tool_is_ignored(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller(allowed_tools=["echo", "does_not_exist"])
        result = _call(tool_caller, agent, 'print(await echo(text="ok"))')
        assert result["status"] == "success"
        assert _text(result) == "ok"


class TestErrorHandling:
    """User-code failures come back as error results, not raised exceptions."""

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "print('unterminated")
        assert result["status"] == "error"
        assert "Syntax error" in _text(result)

    @pytest.mark.asyncio
    async def test_runtime_error_includes_traceback(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "raise RuntimeError('nope')")
        assert result["status"] == "error"
        assert "Execution error" in _text(result)
        assert "nope" in _text(result)

    @pytest.mark.asyncio
    async def test_tool_error_surfaces_as_exception_in_code(self):
        agent = _agent(boom)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            "try:\n    await boom()\nexcept RuntimeError as error:\n    print(f'caught: {error}')",
        )
        assert result["status"] == "success"
        assert "caught:" in _text(result)
        assert "kaboom" in _text(result)

    @pytest.mark.asyncio
    async def test_system_exit_is_caught(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "raise SystemExit(1)")
        assert result["status"] == "error"
        assert "SystemExit" in _text(result)

    @pytest.mark.asyncio
    async def test_no_agent_context_returns_error(self):
        tool_caller = make_programmatic_tool_caller()
        context = ToolContext(
            tool_use={"name": tool_caller.tool_name, "toolUseId": "id", "input": {}},
            agent=None,
            invocation_state={},
        )
        result = await tool_caller(code="print(1)", tool_context=context)
        assert result["status"] == "error"
        assert "requires an agent" in _text(result)


class TestNamespaceClash:
    """A tool that would shadow a reserved name or an extra module is rejected."""

    @pytest.mark.asyncio
    async def test_clash_with_reserved_name(self):
        @tool(name="asyncio")
        def asyncio_tool() -> str:
            """A tool whose name collides with the reserved ``asyncio`` entry."""
            return "x"

        agent = _agent(asyncio_tool)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "print(1)")
        assert result["status"] == "error"
        assert "asyncio" in _text(result)
        assert "conflict" in _text(result)

    @pytest.mark.asyncio
    async def test_clash_with_extra_module(self):
        @tool(name="json")
        def json_tool() -> str:
            """A tool whose name collides with an injected extra module."""
            return "x"

        agent = _agent(json_tool)
        tool_caller = make_programmatic_tool_caller(extra_modules=["json"])
        result = _call(tool_caller, agent, "print(1)")
        assert result["status"] == "error"
        assert "json" in _text(result)
        assert "conflict" in _text(result)


class TestToolMetadata:
    """Tool names, descriptions, and input schema."""

    def test_default_name(self):
        assert programmatic_tool_caller.tool_name == "programmatic_tool_caller"

    def test_custom_name(self):
        assert make_programmatic_tool_caller(name="run_code").tool_name == "run_code"

    def test_default_description(self):
        assert make_programmatic_tool_caller().tool_spec["description"] == (
            DEFAULT_PROGRAMMATIC_TOOL_CALLER_DESCRIPTION
        )

    def test_custom_description(self):
        assert make_programmatic_tool_caller(description="custom").tool_spec["description"] == "custom"

    def test_schema_exposes_code_but_not_context(self):
        props = programmatic_tool_caller.tool_spec["inputSchema"]["json"]["properties"]
        assert "code" in props
        assert "tool_context" not in props
