"""CodeAct plugin implementation.

Implements the CodeAct paradigm where the agent responds with Python code
instead of JSON tool calls. The plugin hooks into the agent lifecycle to:

1. Modify the system prompt to instruct the model to respond with code
2. Parse code blocks from model responses
3. Execute code locally with tool wrappers injected
4. Feed execution results back via ``AfterInvocationEvent.resume``

The loop terminates when the model calls ``final_answer()`` or responds
without a code block.

References:
    - Apple ML Research: CodeAct (https://machinelearning.apple.com/research/codeact)
    - HuggingFace smolagents (https://huggingface.co/docs/smolagents/en/index)
    - Anthropic advanced tool use (https://www.anthropic.com/engineering/advanced-tool-use)
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import re
import textwrap
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any

from ...hooks.events import AfterInvocationEvent, BeforeInvocationEvent
from ...plugins import Plugin, hook

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10

_CODEACT_SYSTEM_PROMPT_PREFIX = """You are a CodeAct agent. Instead of calling tools via JSON, you write Python code to accomplish tasks.

## How to respond

When you need to take action, write Python code inside a ```python code block. Your code can:
- Call available tools as async functions (e.g., `result = await shell(command="ls -la")`)
- Use loops, conditionals, and data transformations
- Store intermediate results in variables
- Use `print()` to output information (printed output will be shown back to you)

When you have the final answer, call `final_answer("your answer here")` in your code.

If you can answer directly without tools (e.g., for simple questions), respond in plain text without a code block.

## Available tools

The following tools are available as async functions in your code:

"""

_CODEACT_OBSERVATION_PREFIX = "**Observation:**\n```\n"
_CODEACT_OBSERVATION_SUFFIX = "\n```"


class CodeActPlugin(Plugin):
    """Plugin that implements the CodeAct paradigm.

    CodeAct replaces standard tool calling with code-based orchestration.
    The model generates Python code that calls tools as async functions,
    and the plugin executes this code and feeds results back to the model.

    The plugin maintains a persistent namespace across turns, so variables
    set in one code block are available in the next. This enables multi-step
    reasoning with state accumulation.

    Code executes locally in the host process. Tool calls within the code
    are routed through the agent's tool caller (``agent.tool.X()``).

    Args:
        max_iterations: Maximum number of code execution rounds before
            forcing termination. Defaults to 10.
        allowed_modules: Optional set of module names that can be imported
            in generated code. If None, a default safe set is used.

    Example:
        ```python
        from strands import Agent
        from strands.vended_plugins.codeact import CodeActPlugin

        agent = Agent(
            tools=[shell, calculator, http_request],
            plugins=[CodeActPlugin(max_iterations=15)],
        )

        result = agent("Fetch HN front page, extract titles, save to file")
        ```
    """

    name = "codeact"

    _DEFAULT_ALLOWED_MODULES = frozenset({
        "math",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "datetime",
        "os.path",
        "pathlib",
        "textwrap",
        "statistics",
        "string",
        "random",
        "hashlib",
        "base64",
        "urllib.parse",
        "csv",
        "io",
    })

    def __init__(
        self,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        allowed_modules: set[str] | None = None,
    ) -> None:
        """Initialize the CodeAct plugin.

        Args:
            max_iterations: Maximum number of code execution rounds before
                forcing termination.
            allowed_modules: Optional set of module names that can be imported
                in generated code. If None, a default safe set is used.
        """
        self._max_iterations = max_iterations
        self._allowed_modules = (
            frozenset(allowed_modules) if allowed_modules is not None else self._DEFAULT_ALLOWED_MODULES
        )
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Initialize the plugin with an agent instance.

        Args:
            agent: The agent instance to extend with CodeAct support.
        """
        logger.debug(
            "max_iterations=<%d>, allowed_modules=<%d> | codeact plugin initialized",
            self._max_iterations,
            len(self._allowed_modules),
        )

    @hook
    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Inject CodeAct instructions and tool signatures into the system prompt.

        Modifies the system prompt to instruct the model to respond with Python
        code blocks instead of JSON tool calls. Injects function signatures for
        all available tools so the model knows how to call them.

        Also initializes per-invocation state (namespace, iteration counter).

        Args:
            event: The before-invocation event containing the agent reference.
        """
        agent = event.agent

        # Initialize CodeAct state in invocation_state
        event.invocation_state.setdefault("codeact_namespace", self._build_initial_namespace(agent))
        event.invocation_state.setdefault("codeact_iteration", 0)

        # Build tool signatures
        tool_signatures = self._build_tool_signatures(agent)

        # Inject CodeAct instructions into system prompt
        current_prompt = agent.system_prompt or ""

        # Remove previously injected CodeAct block if present
        state_data = agent.state.get("codeact")
        last_injected = state_data.get("last_injected_prompt") if isinstance(state_data, dict) else None
        if last_injected and last_injected in current_prompt:
            current_prompt = current_prompt.replace(last_injected, "")

        codeact_block = _CODEACT_SYSTEM_PROMPT_PREFIX + tool_signatures
        injection = f"\n\n{codeact_block}"
        new_prompt = f"{current_prompt}{injection}" if current_prompt else codeact_block

        # Track what we injected for cleanup
        new_injected = injection if current_prompt else codeact_block
        self._set_state_field(agent, "last_injected_prompt", new_injected)

        agent.system_prompt = new_prompt

    @hook
    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Parse code from model response, execute it, and resume with results.

        Extracts Python code blocks from the model's response, executes them
        in a persistent namespace with tool wrappers available, and sets
        ``event.resume`` with the execution output so the agent loops back
        for another turn.

        The loop terminates when:
        - The model calls ``final_answer()``
        - The model responds without a code block
        - Maximum iterations are reached

        Args:
            event: The after-invocation event containing the agent result.
        """
        if event.result is None:
            return

        # Get iteration count
        iteration = event.invocation_state.get("codeact_iteration", 0)
        if iteration >= self._max_iterations:
            logger.warning(
                "iteration=<%d>, max=<%d> | codeact max iterations reached, stopping",
                iteration,
                self._max_iterations,
            )
            return

        # Get the model's response text
        response_text = self._extract_response_text(event.result)
        if not response_text:
            return

        # Parse code block from response
        code = self._parse_code_block(response_text)
        if not code:
            # No code block — model is responding in plain text, we're done
            logger.debug("iteration=<%d> | no code block found, codeact loop complete", iteration)
            return

        # Get or create namespace
        namespace = event.invocation_state.get("codeact_namespace", {})

        # Check if code has already set final_answer before execution
        # (in case of re-entry)
        if namespace.get("__final_answer__") is not None:
            logger.debug("iteration=<%d> | final_answer already set, stopping", iteration)
            return

        # Validate the code before execution
        validation_error = self._validate_code(code)
        if validation_error:
            output = f"Code validation error: {validation_error}"
            event.invocation_state["codeact_iteration"] = iteration + 1
            event.resume = f"{_CODEACT_OBSERVATION_PREFIX}{output}{_CODEACT_OBSERVATION_SUFFIX}"
            return

        # Execute the code
        logger.debug("iteration=<%d>, code_length=<%d> | executing codeact code block", iteration, len(code))
        output = self._execute_code(code, namespace)

        # Check if final_answer was called
        if namespace.get("__final_answer__") is not None:
            final = namespace["__final_answer__"]
            logger.debug("iteration=<%d> | final_answer called", iteration)
            # Set resume with final answer so the model can produce a clean response
            event.resume = (
                f"{_CODEACT_OBSERVATION_PREFIX}{output}\n\n"
                f"final_answer was called with: {final}{_CODEACT_OBSERVATION_SUFFIX}\n\n"
                f"Provide the final answer to the user based on the above result. "
                f"Do NOT write any more code."
            )
            # Clear namespace to prevent re-triggering
            namespace["__final_answer_delivered__"] = True
            return

        # Check if this is a post-final-answer turn (model should just respond)
        if namespace.get("__final_answer_delivered__"):
            return

        # Resume with execution output for next iteration
        event.invocation_state["codeact_iteration"] = iteration + 1
        event.resume = f"{_CODEACT_OBSERVATION_PREFIX}{output}{_CODEACT_OBSERVATION_SUFFIX}"

    def _build_initial_namespace(self, agent: Agent) -> dict[str, Any]:
        """Build the initial execution namespace with tool wrappers and builtins.

        Creates async wrapper functions for each agent tool and injects them
        into a namespace dict. Also adds ``final_answer()`` and a restricted
        ``__import__`` that only allows safe modules.

        Args:
            agent: The agent whose tools to wrap.

        Returns:
            Namespace dict ready for code execution.
        """
        namespace: dict[str, Any] = {}

        # Add restricted builtins
        safe_builtins = {
            k: v
            for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k
            not in {
                "exec",
                "eval",
                "compile",
                "__import__",
                "globals",
                "locals",
                "breakpoint",
                "exit",
                "quit",
            }
        }

        # Add controlled import
        def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            """Import only allowed modules."""
            # Check if the module or its parent is allowed
            parts = name.split(".")
            for i in range(len(parts), 0, -1):
                if ".".join(parts[:i]) in self._allowed_modules:
                    return __builtins__["__import__"](name, *args, **kwargs)  # type: ignore[index]
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(self._allowed_modules))}"
            )

        safe_builtins["__import__"] = _safe_import
        namespace["__builtins__"] = safe_builtins

        # Add asyncio for await support
        namespace["asyncio"] = asyncio

        # Add final_answer function
        def final_answer(result: Any) -> None:
            """Call this when you have the final answer."""
            namespace["__final_answer__"] = result

        namespace["final_answer"] = final_answer

        # Add tool wrappers
        for tool_name in agent.tool_registry.registry:
            namespace[tool_name.replace("-", "_")] = self._make_tool_wrapper(agent, tool_name)

        return namespace

    def _make_tool_wrapper(self, agent: Agent, tool_name: str) -> Any:
        """Create an async wrapper function for an agent tool.

        The wrapper calls ``agent.tool.X()`` when invoked, making the tool
        available as a regular async function in the code execution namespace.

        Args:
            agent: The agent instance.
            tool_name: Name of the tool to wrap.

        Returns:
            An async function that calls the tool.
        """
        # Normalize for Python identifier
        python_name = tool_name.replace("-", "_")

        async def tool_wrapper(**kwargs: Any) -> Any:
            """Async wrapper that calls agent.tool.{tool_name}()."""
            logger.debug("tool_name=<%s> | codeact calling tool", tool_name)
            try:
                caller = getattr(agent.tool, python_name)
                result = caller(record_direct_tool_call=False, **kwargs)
                # Extract text content from ToolResult
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list):
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                texts.append(block["text"])
                        return "\n".join(texts) if texts else str(result)
                return result
            except Exception as e:
                logger.warning("tool_name=<%s>, error=<%s> | codeact tool call failed", tool_name, e)
                raise

        tool_wrapper.__name__ = python_name
        tool_wrapper.__qualname__ = python_name

        # Add docstring from tool spec for model context
        tool_config = agent.tool_registry.get_all_tools_config()
        spec = tool_config.get(tool_name, {})
        description = spec.get("description", f"Call the {tool_name} tool.")
        tool_wrapper.__doc__ = description

        return tool_wrapper

    def _build_tool_signatures(self, agent: Agent) -> str:
        """Generate Python function signatures for all available tools.

        Reads tool specs from the agent's tool registry and formats them
        as function signatures with type hints and docstrings.

        Args:
            agent: The agent whose tools to describe.

        Returns:
            Formatted string of tool function signatures.
        """
        tool_config = agent.tool_registry.get_all_tools_config()
        signatures = []

        for tool_name, spec in tool_config.items():
            python_name = tool_name.replace("-", "_")
            description = spec.get("description", "")
            input_schema = spec.get("inputSchema", {}).get("json", {})
            properties = input_schema.get("properties", {})
            required = set(input_schema.get("required", []))

            # Build parameter list
            params = []
            for param_name, param_spec in properties.items():
                param_type = self._json_type_to_python(param_spec.get("type", "str"))
                param_desc = param_spec.get("description", "")

                if param_name in required:
                    params.append(f"{param_name}: {param_type}")
                else:
                    default = param_spec.get("default", None)
                    params.append(f"{param_name}: {param_type} = {repr(default)}")

            params_str = ", ".join(params)
            sig = f"async def {python_name}({params_str}) -> str"

            # Format with docstring
            entry = f'{sig}:\n    """{description}"""'
            signatures.append(entry)

        return "\n\n".join(signatures)

    @staticmethod
    def _json_type_to_python(json_type: str) -> str:
        """Convert JSON schema type to Python type hint string.

        Args:
            json_type: The JSON schema type string.

        Returns:
            Python type hint string.
        """
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }
        return type_map.get(json_type, "str")

    @staticmethod
    def _extract_response_text(result: Any) -> str:
        """Extract text content from an AgentResult.

        Args:
            result: The AgentResult from the model.

        Returns:
            The text content of the response, or empty string.
        """
        try:
            message = result.message
            if message and "content" in message:
                texts = []
                for block in message["content"]:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                return "\n".join(texts)
        except (AttributeError, KeyError, TypeError):
            pass

        # Fallback: try string conversion
        text = str(result)
        return text if text and text != "None" else ""

    @staticmethod
    def _parse_code_block(text: str) -> str | None:
        """Extract Python code from a markdown code block.

        Supports both ```python and ``` (bare) fenced code blocks.
        Returns the first code block found, or None if no code block exists.

        Args:
            text: The model's response text.

        Returns:
            The extracted code, or None if no code block was found.
        """
        # Match ```python ... ``` or ```py ... ```
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```py\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code:
                    return code

        return None

    def _validate_code(self, code: str) -> str | None:
        """Validate code before execution.

        Performs AST-level validation to catch syntax errors and
        potentially dangerous constructs.

        Args:
            code: The code to validate.

        Returns:
            Error message string if validation fails, None if valid.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"

        # Check for disallowed constructs
        for node in ast.walk(tree):
            # Block eval/exec calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ("exec", "eval", "compile"):
                    return f"Use of '{func.id}()' is not allowed"

            # Block __dunder__ attribute access (except __init__, __name__, etc.)
            if isinstance(node, ast.Attribute):
                if (
                    node.attr.startswith("__")
                    and node.attr.endswith("__")
                    and node.attr not in ("__init__", "__name__", "__doc__", "__class__", "__len__", "__str__")
                ):
                    return f"Access to '{node.attr}' is not allowed"

        return None

    def _execute_code(self, code: str, namespace: dict[str, Any]) -> str:
        """Execute Python code in the given namespace, capturing output.

        Wraps the code in an async function to support ``await`` calls,
        then executes it and captures stdout. Local variables from the code
        are copied back to the namespace for persistence across turns.

        Args:
            code: The Python code to execute.
            namespace: The execution namespace (persistent across turns).

        Returns:
            Captured stdout output, or error message if execution failed.
        """
        # Wrap code in async function to support top-level await.
        # The __ns__ parameter receives the namespace dict so we can
        # copy local variables back after execution (otherwise they'd
        # be lost when the function returns).
        indented_code = textwrap.indent(code, "    ")
        ns_update = '    __ns__.update({k: v for k, v in locals().items() if not k.startswith("_")})'
        wrapped = f"async def __codeact_main__(__ns__):\n{indented_code}\n{ns_update}\n"

        stdout_capture = io.StringIO()

        try:
            # Compile and exec the wrapper function definition
            compiled = compile(wrapped, "<codeact>", "exec")
            exec(compiled, namespace)  # noqa: S102

            # Run the async main function, capturing stdout
            with redirect_stdout(stdout_capture):
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(namespace["__codeact_main__"](namespace))
                finally:
                    loop.close()

            output = stdout_capture.getvalue()
            if not output:
                output = "(No output)"

            logger.debug("output_length=<%d> | codeact execution complete", len(output))
            return output

        except Exception as e:
            captured = stdout_capture.getvalue()
            error_output = f"Error: {type(e).__name__}: {e}"
            if captured:
                error_output = f"{captured}\n{error_output}"

            logger.debug("error=<%s> | codeact execution failed", e)
            return error_output

        finally:
            # Clean up the async function from namespace
            namespace.pop("__codeact_main__", None)

    def _set_state_field(self, agent: Agent, key: str, value: Any) -> None:
        """Set a single field in the plugin's agent state dict.

        Args:
            agent: The agent whose state to update.
            key: The state field key.
            value: The value to set.
        """
        state_data = agent.state.get("codeact")
        if state_data is not None and not isinstance(state_data, dict):
            state_data = {}
        if state_data is None:
            state_data = {}
        state_data[key] = value
        agent.state.set("codeact", state_data)
