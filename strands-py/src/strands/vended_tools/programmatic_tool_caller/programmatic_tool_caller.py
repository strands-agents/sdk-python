"""Tool that lets an agent orchestrate its other tools with Python code.

Provides :func:`make_programmatic_tool_caller` (a factory) and
:data:`programmatic_tool_caller` (the default instance). The tool runs
agent-authored Python code in which every other registered tool is exposed as
an ``async`` function, so the model can chain, loop over, and parallelize tool
calls in a single turn instead of one tool call per model round-trip. Only text
the code sends to ``print()`` is returned to the model; individual tool results
stay in the code's local scope unless printed.

This tool runs model-authored code in the host process. It does not sandbox
that code, so it should only be used where the code the agent
produces is trusted (or the tool set it can reach is itself constrained via
``allowed_tools``). Tools that raise interrupts (human-in-the-loop) are not
supported here, because interrupts cannot be raised from a direct tool call.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib
import io
import logging
import traceback
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import DEFAULT_PROGRAMMATIC_TOOL_CALLER_DESCRIPTION

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)

# Names the injected tools must not shadow: ``asyncio`` is always available to user code,
# ``__builtins__`` holds the builtins the code relies on, and ``__name__`` is the base namespace key.
_RESERVED_NAMESPACE_NAMES = frozenset({"asyncio", "__builtins__", "__name__"})

_USER_CODE_FILENAME = "<programmatic_tool_caller>"


def _error_result(message: str) -> dict[str, Any]:
    """Build an error-status tool result carrying ``message`` as text."""
    return {"status": "error", "content": [{"text": message}]}


def _execute_tool(agent: Any, tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Invoke a tool through the agent and unwrap its result for user code.

    Calls ``agent.tool.<tool_name>`` with ``record_direct_tool_call=False`` so
    the orchestration does not pollute the agent's message history. A successful
    result's text content blocks are joined into a single string (falling back
    to ``str(result)`` when there is no text); an error result is surfaced as a
    ``RuntimeError`` so it propagates like a normal exception in the user code.

    Args:
        agent: The agent whose tools are being invoked.
        tool_name: Registry name of the tool to call.
        tool_input: Keyword arguments to pass to the tool.

    Returns:
        The tool's text output as a string, or the raw result when it has no
        text content blocks.

    Raises:
        RuntimeError: If the tool returns an error result or the call fails.
    """
    try:
        tool_caller = getattr(agent.tool, tool_name)
        result = tool_caller(record_direct_tool_call=False, **tool_input)
    except Exception as error:
        raise RuntimeError(f"Failed to execute tool '{tool_name}': {error}") from error

    if not isinstance(result, dict):
        return result

    if result.get("status") == "error":
        content = result.get("content") or [{"text": "Unknown error"}]
        error_text = content[0].get("text", "Unknown error") if content else "Unknown error"
        raise RuntimeError(f"Tool '{tool_name}' error: {error_text}")

    content = result.get("content", [])
    text_parts = [block["text"] for block in content if isinstance(block, dict) and "text" in block]
    if text_parts:
        return "\n".join(text_parts)
    return str(result)


def _make_async_tool_function(agent: Any, tool_name: str) -> Callable[..., Awaitable[Any]]:
    """Wrap a tool as an ``async`` callable for the execution namespace.

    The underlying direct tool call is blocking, so it is offloaded to a worker
    thread; this keeps the event loop free and lets user code parallelize calls
    with ``asyncio.gather(...)``.
    """

    async def tool_function(**kwargs: Any) -> Any:
        return await asyncio.to_thread(_execute_tool, agent, tool_name, kwargs)

    return tool_function


def _resolve_available_tools(agent: Any, allowed_tools: list[str] | None, self_name: str) -> set[str]:
    """Determine which tools to expose to user code.

    The tool never exposes itself. When ``allowed_tools`` is ``None`` every other
    registered tool is exposed; otherwise the exposed set is the intersection of
    the registered tools with ``allowed_tools`` (names not registered are
    silently ignored).
    """
    registered = set(agent.tool_registry.registry.keys()) - {self_name}
    if allowed_tools is None:
        return registered
    return registered & set(allowed_tools)


def _build_namespace(available_tools: set[str], agent: Any, extra_modules: list[str]) -> dict[str, Any]:
    """Build the code execution namespace: base entries, extra modules, and tools.

    The base namespace mirrors a fresh module (``__name__``) plus ``asyncio``
    (always needed for the async tool wrappers). Each name in ``extra_modules``
    is imported and injected; a module that cannot be imported is skipped with a
    warning. Tools are then injected as ``async`` callables.

    Args:
        available_tools: Registry names of the tools to inject.
        agent: The agent whose tools are being wrapped.
        extra_modules: Extra module names to import into the namespace.

    Returns:
        A namespace dict for executing the code.

    Raises:
        ValueError: If a tool name collides with a reserved name (``asyncio``,
            ``__name__``) or with an injected extra module, which would shadow
            it in the namespace.
    """
    namespace: dict[str, Any] = {"__name__": "__main__", "asyncio": asyncio}

    reserved_names = set(_RESERVED_NAMESPACE_NAMES)
    for module_name in extra_modules:
        try:
            namespace[module_name] = importlib.import_module(module_name)
        except ImportError:
            logger.warning("module_name=<%s> | could not import extra module, skipping", module_name)
            continue
        reserved_names.add(module_name)

    clashing_tools = available_tools & reserved_names
    if clashing_tools:
        raise ValueError(
            f"Tool name(s) {sorted(clashing_tools)} conflict with reserved namespace entries "
            f"{sorted(reserved_names)}. Rename the tool(s) or restrict the exposed set via allowed_tools."
        )

    for tool_name in available_tools:
        namespace[tool_name] = _make_async_tool_function(agent, tool_name)

    return namespace


def make_programmatic_tool_caller(
    *,
    allowed_tools: list[str] | None = None,
    extra_modules: list[str] | None = None,
    name: str = "programmatic_tool_caller",
    description: str = DEFAULT_PROGRAMMATIC_TOOL_CALLER_DESCRIPTION,
) -> DecoratedFunctionTool:
    """Create a programmatic-tool-caller tool.

    The returned tool executes agent-authored Python code in which the agent's
    other tools are exposed as ``async`` functions (``await tool_name(...)``).
    Only ``print()`` output is returned to the model.

    Args:
        allowed_tools: Registry names of the tools to expose to the code. When
            ``None`` (the default), every other registered tool is exposed. The
            caller tool never exposes itself.
        extra_modules: Extra Python module names (e.g. ``["json", "math"]``) to
            import into the execution namespace. Defaults to none; ``asyncio`` is
            always available. Code may also ``import`` modules itself.
        name: Tool name. Defaults to ``"programmatic_tool_caller"``.
        description: Tool description shown to the model.

    Returns:
        A decorated tool that runs code with access to the agent's other tools.
    """
    resolved_allowed_tools = list(allowed_tools) if allowed_tools is not None else None
    resolved_extra_modules = list(extra_modules) if extra_modules is not None else []

    @tool(name=name, description=description, context="tool_context")
    async def programmatic_tool_caller_tool(code: str, tool_context: ToolContext) -> dict[str, Any]:
        """Execute Python code with the agent's other tools as async functions.

        Tools are available as ``async`` functions, so call them with ``await``.
        The code already runs in an async context; ``asyncio`` is available for
        patterns like ``asyncio.gather(...)``. Only text passed to ``print()`` is
        returned; a tool's return value stays local unless you print it.

        Example:
            ```python
            result = await calculator(expression="2 + 2")
            print(result)

            # Parallel execution
            results = await asyncio.gather(
                calculator(expression="1 + 1"),
                calculator(expression="2 + 2"),
            )
            print(results)
            ```

        Args:
            code: Python code to execute. Use ``await tool_name(...)`` to call tools.
            tool_context: Injected by the framework. Not user-facing.

        Returns:
            A tool result whose text is the code's captured ``print()`` output on
            success, or the error/traceback on failure.
        """
        agent = tool_context.agent
        if agent is None:
            return _error_result("No agent available. The programmatic tool caller requires an agent context.")

        available_tools = _resolve_available_tools(agent, resolved_allowed_tools, name)

        try:
            exec_namespace = _build_namespace(available_tools, agent, resolved_extra_modules)
        except ValueError as error:
            return _error_result(str(error))

        # Compile with top-level-await support so the model can ``await`` tools directly. This avoids
        # wrapping/indenting the code (which would corrupt continuation lines inside string literals)
        # and keeps tracebacks pointing at the model's own line numbers.
        try:
            compiled = compile(code, _USER_CODE_FILENAME, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError:
            return _error_result(f"Syntax error:\n{traceback.format_exc()}")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # ``eval`` runs synchronous code inline and returns a coroutine only when the code
                # contains a top-level ``await``; running agent-authored code is this tool's purpose.
                maybe_coroutine = eval(compiled, exec_namespace)
                if maybe_coroutine is not None:
                    await maybe_coroutine
        # SystemExit/KeyboardInterrupt raised by user code are caught so they do not tear down the host.
        except (SystemExit, KeyboardInterrupt) as error:
            return _error_result(f"Execution error: {type(error).__name__}: {error}")
        except Exception:
            return _error_result(f"Execution error:\n{traceback.format_exc()}")

        captured_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            captured_output += f"\n[stderr]\n{stderr_output}"

        text = captured_output.strip() or "(no output)"
        return {"status": "success", "content": [{"text": text}]}

    return programmatic_tool_caller_tool


programmatic_tool_caller = make_programmatic_tool_caller()
"""Default programmatic tool caller. Exposes every other registered tool."""
