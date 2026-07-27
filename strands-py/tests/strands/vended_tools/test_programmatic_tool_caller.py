"""Tests for the programmatic tool caller.

The tool runs agent-authored Python code in which the agent's other tools are
exposed as ``async`` functions; only ``print()`` output is returned. The tests
drive it through a real ``Agent`` (with a mocked model, which is never invoked
for direct tool calls) so the full tool-caller/registry path is exercised.
"""

import asyncio

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

    @pytest.mark.asyncio
    async def test_multiline_string_is_passed_through_verbatim(self):
        # Regression: the code must not be reindented, or continuation lines inside
        # string literals would be silently corrupted with leading whitespace.
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(
            tool_caller,
            agent,
            'msg = """line one\nline two\nline three"""\nprint(await echo(text=msg))',
        )
        assert result["status"] == "success"
        assert _text(result) == "line one\nline two\nline three"

    @pytest.mark.asyncio
    async def test_code_without_top_level_await(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "print(2 + 2)")
        assert result["status"] == "success"
        assert _text(result) == "4"

    @pytest.mark.asyncio
    async def test_comment_only_code_is_a_no_op(self):
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "# nothing to do here")
        assert result["status"] == "success"
        assert _text(result) == "(no output)"

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_cross_contaminate_print_output(self):
        """Print capture must be per-call, not a process-global ``sys.stdout`` redirect.

        A global redirect would let one in-flight call's output land in another's
        result (and steal it from the real stream), so overlap two calls whose
        print windows interleave and require each result to hold only its own output.
        """
        agent = _agent(echo)
        tool_caller = make_programmatic_tool_caller()
        agent.tool_registry.register_tool(tool_caller)
        invoke = getattr(agent.tool, tool_caller.tool_name)

        code_a = "print('A1')\nawait asyncio.sleep(0.1)\nprint('A2')\n"
        code_b = "await asyncio.sleep(0.02)\nprint('B1')\nawait asyncio.sleep(0.1)\nprint('B2')\n"

        result_a, result_b = await asyncio.gather(
            asyncio.to_thread(invoke, code=code_a, record_direct_tool_call=False),
            asyncio.to_thread(invoke, code=code_b, record_direct_tool_call=False),
        )
        assert _text(result_a) == "A1\nA2"
        assert _text(result_b) == "B1\nB2"


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


class TestNonIdentifierToolNames:
    """Tools whose registry names are not valid Python identifiers (e.g. MCP tools)."""

    @pytest.mark.asyncio
    async def test_hyphenated_tool_reachable_via_alias(self):
        @tool(name="dash-tool")
        def dash_tool(value: str) -> str:
            """A tool whose registry name contains a hyphen."""
            return f"dash:{value}"

        agent = _agent(dash_tool)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, 'print(await dash_tool(value="x"))')
        assert result["status"] == "success"
        assert _text(result) == "dash:x"

    @pytest.mark.asyncio
    async def test_dotted_tool_reachable_via_alias(self):
        @tool(name="ns.fetch")
        def ns_fetch(value: str) -> str:
            """A tool whose registry name contains a dot."""
            return f"fetch:{value}"

        agent = _agent(ns_fetch)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, 'print(await ns_fetch(value="y"))')
        assert result["status"] == "success"
        assert _text(result) == "fetch:y"

    @pytest.mark.asyncio
    async def test_raw_name_remains_available(self):
        @tool(name="dash-tool")
        def dash_tool(value: str) -> str:
            """A tool whose registry name contains a hyphen."""
            return f"dash:{value}"

        agent = _agent(dash_tool)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, 'print(await globals()["dash-tool"](value="raw"))')
        assert result["status"] == "success"
        assert _text(result) == "dash:raw"

    @pytest.mark.asyncio
    async def test_alias_does_not_shadow_a_real_tool(self):
        # The registry already rejects two tools differing only by '-'/'_', but a dotted
        # name normalizes onto a real underscore name, so the alias must yield to it.
        @tool(name="ns.fetch")
        def ns_dotted(value: str) -> str:
            """Dotted tool whose alias collides with a real tool name."""
            return f"dotted:{value}"

        @tool(name="ns_fetch")
        def ns_real(value: str) -> str:
            """Real tool that already owns the alias identifier."""
            return f"real:{value}"

        agent = _agent(ns_dotted, ns_real)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, 'print(await ns_fetch(value="v"))')
        assert result["status"] == "success"
        # The real ``ns_fetch`` must win; the alias must not overwrite it.
        assert _text(result) == "real:v"

    @pytest.mark.asyncio
    async def test_alias_does_not_shadow_a_reserved_name(self):
        # "..builtins.." normalizes to __builtins__, which the executed code relies on.
        @tool(name="..builtins..")
        def sneaky(value: str) -> str:
            """Tool whose normalized alias would collide with a reserved namespace entry."""
            return f"sneaky:{value}"

        agent = _agent(sneaky)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "print(len([1, 2, 3]))")
        # Builtins must still work, i.e. the alias was skipped rather than injected.
        assert result["status"] == "success"
        assert _text(result) == "3"

    @pytest.mark.asyncio
    async def test_ambiguous_alias_is_not_injected(self):
        @tool(name="dup-tool")
        def dup_hyphen(value: str) -> str:
            """One of two tools normalizing to the same identifier."""
            return f"hyphen:{value}"

        @tool(name="dup.tool")
        def dup_dot(value: str) -> str:
            """The other tool normalizing to the same identifier."""
            return f"dot:{value}"

        agent = _agent(dup_hyphen, dup_dot)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, 'print(await dup_tool(value="v"))')
        # Ambiguous: no alias is injected rather than silently picking one.
        assert result["status"] == "error"
        assert "NameError" in _text(result)


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
    async def test_clash_with_builtins(self):
        @tool(name="__builtins__")
        def builtins_tool() -> str:
            """A tool whose name collides with the reserved ``__builtins__`` entry."""
            return "x"

        agent = _agent(builtins_tool)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "print(1)")
        assert result["status"] == "error"
        assert "__builtins__" in _text(result)
        assert "conflict" in _text(result)

    @pytest.mark.asyncio
    async def test_clash_with_print(self):
        @tool(name="print")
        def print_tool() -> str:
            """A tool whose name collides with the reserved ``print`` entry."""
            return "x"

        agent = _agent(print_tool)
        tool_caller = make_programmatic_tool_caller()
        result = _call(tool_caller, agent, "print(1)")
        assert result["status"] == "error"
        assert "print" in _text(result)
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
