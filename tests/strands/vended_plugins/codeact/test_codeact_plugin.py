"""Tests for CodeAct plugin.

Tests cover:
- Plugin initialization and lifecycle
- System prompt injection
- Code block parsing
- Code validation (AST checks)
- Code execution with namespace persistence
- Tool wrapper generation and invocation
- final_answer() termination
- Max iterations safety
- Error handling and self-correction loop
- Import restrictions
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from strands.agent.agent_result import AgentResult
from strands.hooks.events import AfterInvocationEvent, BeforeInvocationEvent
from strands.vended_plugins.codeact.codeact_plugin import (
    CodeActPlugin,
    _CODEACT_OBSERVATION_PREFIX,
    _CODEACT_OBSERVATION_SUFFIX,
    _CODEACT_SYSTEM_PROMPT_PREFIX,
)


@pytest.fixture
def plugin():
    """Create a CodeAct plugin with default settings."""
    return CodeActPlugin()


@pytest.fixture
def plugin_custom():
    """Create a CodeAct plugin with custom settings."""
    return CodeActPlugin(max_iterations=3, allowed_modules={"math", "json"})


@pytest.fixture
def mock_agent():
    """Create a mock agent with tool registry."""
    agent = MagicMock()
    agent.system_prompt = "You are a helpful assistant."
    agent.state = MagicMock()
    agent.state.get.return_value = None

    # Mock tool registry
    mock_tool = MagicMock()
    mock_tool.tool_spec = {
        "name": "calculator",
        "description": "Perform calculations",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    }
                },
                "required": ["expression"],
            }
        },
    }

    mock_shell_tool = MagicMock()
    mock_shell_tool.tool_spec = {
        "name": "shell",
        "description": "Execute shell commands",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                    },
                },
                "required": ["command"],
            }
        },
    }

    agent.tool_registry = MagicMock()
    agent.tool_registry.registry = {"calculator": mock_tool, "shell": mock_shell_tool}
    agent.tool_registry.get_all_tools_config.return_value = {
        "calculator": mock_tool.tool_spec,
        "shell": mock_shell_tool.tool_spec,
    }

    # Mock tool caller
    agent.tool = MagicMock()

    return agent


@pytest.fixture
def mock_result():
    """Create a mock AgentResult with text content."""

    def _make_result(text):
        result = MagicMock()
        result.message = {"content": [{"text": text}]}
        result.__str__ = lambda self: text
        return result

    return _make_result


class TestPluginInitialization:
    """Test plugin setup and configuration."""

    def test_plugin_name(self, plugin):
        """Plugin should have correct name."""
        assert plugin.name == "codeact"

    def test_default_max_iterations(self, plugin):
        """Default max iterations should be 10."""
        assert plugin._max_iterations == 10

    def test_custom_max_iterations(self, plugin_custom):
        """Custom max iterations should be respected."""
        assert plugin_custom._max_iterations == 3

    def test_default_allowed_modules(self, plugin):
        """Default allowed modules should include safe standard library modules."""
        assert "math" in plugin._allowed_modules
        assert "json" in plugin._allowed_modules
        assert "re" in plugin._allowed_modules
        assert "os.path" in plugin._allowed_modules

    def test_custom_allowed_modules(self, plugin_custom):
        """Custom allowed modules should override defaults."""
        assert plugin_custom._allowed_modules == frozenset({"math", "json"})

    def test_hooks_registered(self, plugin):
        """Plugin should have hooks registered."""
        assert len(plugin.hooks) == 2  # before_invocation + after_invocation


class TestSystemPromptInjection:
    """Test BeforeInvocationEvent hook."""

    def test_injects_codeact_instructions(self, plugin, mock_agent):
        """Should inject CodeAct instructions into system prompt."""
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state={})
        plugin._on_before_invocation(event)

        new_prompt = mock_agent.system_prompt
        assert "CodeAct agent" in new_prompt
        assert "```python" in new_prompt

    def test_injects_tool_signatures(self, plugin, mock_agent):
        """Should inject tool function signatures."""
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state={})
        plugin._on_before_invocation(event)

        new_prompt = mock_agent.system_prompt
        assert "calculator" in new_prompt
        assert "expression: str" in new_prompt
        assert "shell" in new_prompt
        assert "command: str" in new_prompt

    def test_initializes_invocation_state(self, plugin, mock_agent):
        """Should initialize codeact_namespace and codeact_iteration."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        assert "codeact_namespace" in invocation_state
        assert "codeact_iteration" in invocation_state
        assert invocation_state["codeact_iteration"] == 0

    def test_namespace_has_tools(self, plugin, mock_agent):
        """Namespace should contain tool wrapper functions."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        assert "calculator" in namespace
        assert "shell" in namespace
        assert callable(namespace["calculator"])
        assert callable(namespace["shell"])

    def test_namespace_has_final_answer(self, plugin, mock_agent):
        """Namespace should contain final_answer function."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        assert "final_answer" in namespace
        assert callable(namespace["final_answer"])

    def test_preserves_existing_system_prompt(self, plugin, mock_agent):
        """Should preserve existing system prompt content."""
        mock_agent.system_prompt = "Original prompt."
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state={})
        plugin._on_before_invocation(event)

        new_prompt = mock_agent.system_prompt
        assert "Original prompt." in new_prompt

    def test_handles_none_system_prompt(self, plugin, mock_agent):
        """Should handle None system prompt gracefully."""
        mock_agent.system_prompt = None
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state={})
        plugin._on_before_invocation(event)

        assert mock_agent.system_prompt is not None
        assert "CodeAct" in mock_agent.system_prompt


class TestCodeBlockParsing:
    """Test code block extraction from model responses."""

    def test_parse_python_code_block(self, plugin):
        """Should parse ```python code blocks."""
        text = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        code = plugin._parse_code_block(text)
        assert code == "print('hello')"

    def test_parse_py_code_block(self, plugin):
        """Should parse ```py code blocks."""
        text = "```py\nx = 1 + 2\nprint(x)\n```"
        code = plugin._parse_code_block(text)
        assert code == "x = 1 + 2\nprint(x)"

    def test_parse_bare_code_block(self, plugin):
        """Should parse bare ``` code blocks."""
        text = "```\nresult = 42\nprint(result)\n```"
        code = plugin._parse_code_block(text)
        assert code == "result = 42\nprint(result)"

    def test_no_code_block(self, plugin):
        """Should return None when no code block present."""
        text = "The answer is 42."
        code = plugin._parse_code_block(text)
        assert code is None

    def test_empty_code_block(self, plugin):
        """Should return None for empty code blocks."""
        text = "```python\n\n```"
        code = plugin._parse_code_block(text)
        assert code is None

    def test_multiline_code(self, plugin):
        """Should handle multi-line code blocks."""
        text = """Here's the solution:
```python
total = 0
for i in range(10):
    total += i
print(f"Sum: {total}")
```
"""
        code = plugin._parse_code_block(text)
        assert "total = 0" in code
        assert "for i in range(10):" in code
        assert "print" in code

    def test_first_code_block_wins(self, plugin):
        """Should return the first code block when multiple exist."""
        text = "```python\nfirst_block()\n```\n\n```python\nsecond_block()\n```"
        code = plugin._parse_code_block(text)
        assert code == "first_block()"

    def test_prefers_python_over_bare(self, plugin):
        """Should prefer ```python over bare ``` blocks."""
        text = "```\nbare_block()\n```\n\n```python\npython_block()\n```"
        code = plugin._parse_code_block(text)
        # The python pattern is checked first
        assert code == "python_block()"


class TestCodeValidation:
    """Test AST-level code validation."""

    def test_valid_code(self, plugin):
        """Should accept valid Python code."""
        assert plugin._validate_code("x = 1 + 2\nprint(x)") is None

    def test_syntax_error(self, plugin):
        """Should catch syntax errors."""
        result = plugin._validate_code("def foo(:\n  pass")
        assert result is not None
        assert "SyntaxError" in result

    def test_blocks_exec(self, plugin):
        """Should block exec() calls."""
        result = plugin._validate_code("exec('print(1)')")
        assert result is not None
        assert "exec" in result

    def test_blocks_eval(self, plugin):
        """Should block eval() calls."""
        result = plugin._validate_code("eval('1+1')")
        assert result is not None
        assert "eval" in result

    def test_blocks_compile(self, plugin):
        """Should block compile() calls."""
        result = plugin._validate_code("compile('x=1', '', 'exec')")
        assert result is not None
        assert "compile" in result

    def test_blocks_dunder_access(self, plugin):
        """Should block dangerous __dunder__ access."""
        result = plugin._validate_code("obj.__subclasses__()")
        assert result is not None
        assert "__subclasses__" in result

    def test_allows_safe_dunders(self, plugin):
        """Should allow safe __dunder__ attributes."""
        assert plugin._validate_code("x.__name__") is None
        assert plugin._validate_code("x.__doc__") is None
        assert plugin._validate_code("x.__class__") is None
        assert plugin._validate_code("x.__len__()") is None

    def test_allows_loops_and_conditionals(self, plugin):
        """Should allow normal control flow."""
        code = """
for i in range(10):
    if i % 2 == 0:
        print(i)
"""
        assert plugin._validate_code(code) is None

    def test_allows_async_await(self, plugin):
        """Should allow async/await constructs."""
        code = """
async def foo():
    result = await bar()
    return result
"""
        assert plugin._validate_code(code) is None


class TestCodeExecution:
    """Test code execution with namespace persistence."""

    def test_simple_execution(self, plugin):
        """Should execute simple code and capture output."""
        namespace = {"__builtins__": __builtins__}
        output = plugin._execute_code("print('hello world')", namespace)
        assert "hello world" in output

    def test_namespace_persistence(self, plugin):
        """Variables should persist across execution calls."""
        namespace = {"__builtins__": __builtins__}
        plugin._execute_code("x = 42", namespace)
        assert namespace.get("x") == 42

        output = plugin._execute_code("print(x * 2)", namespace)
        assert "84" in output

    def test_error_handling(self, plugin):
        """Should capture exceptions as output."""
        namespace = {"__builtins__": __builtins__}
        output = plugin._execute_code("1/0", namespace)
        assert "ZeroDivisionError" in output

    def test_no_output(self, plugin):
        """Should return '(No output)' when nothing is printed."""
        namespace = {"__builtins__": __builtins__}
        output = plugin._execute_code("x = 1 + 1", namespace)
        assert output == "(No output)"

    def test_partial_output_on_error(self, plugin):
        """Should include partial output before error."""
        namespace = {"__builtins__": __builtins__}
        output = plugin._execute_code("print('before')\n1/0", namespace)
        assert "before" in output
        assert "ZeroDivisionError" in output

    def test_async_execution(self, plugin):
        """Should support await expressions."""
        namespace = {"__builtins__": __builtins__, "asyncio": asyncio}
        output = plugin._execute_code("result = await asyncio.sleep(0)\nprint('async done')", namespace)
        assert "async done" in output

    def test_final_answer_in_code(self, plugin, mock_agent):
        """final_answer() should set namespace flag."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        plugin._execute_code('final_answer("the answer is 42")', namespace)
        assert namespace["__final_answer__"] == "the answer is 42"


class TestAfterInvocationHook:
    """Test the after-invocation hook (code execution + resume loop)."""

    def test_no_code_block_stops_loop(self, plugin, mock_agent, mock_result):
        """Should not resume when model responds without code."""
        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 0, "codeact_namespace": {}},
            result=mock_result("The answer is 42."),
        )
        plugin._on_after_invocation(event)
        assert event.resume is None

    def test_code_execution_sets_resume(self, plugin, mock_agent, mock_result):
        """Should resume with execution output when code block found."""
        namespace = {"__builtins__": __builtins__, "asyncio": asyncio}
        namespace["final_answer"] = lambda r: namespace.__setitem__("__final_answer__", r)

        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 0, "codeact_namespace": namespace},
            result=mock_result("```python\nprint('hello from code')\n```"),
        )
        plugin._on_after_invocation(event)

        assert event.resume is not None
        assert "hello from code" in event.resume

    def test_final_answer_stops_loop(self, plugin, mock_agent, mock_result):
        """Should include final_answer content in resume."""
        namespace = {"__builtins__": __builtins__, "asyncio": asyncio}
        namespace["final_answer"] = lambda r: namespace.__setitem__("__final_answer__", r)

        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 0, "codeact_namespace": namespace},
            result=mock_result('```python\nfinal_answer("result is 42")\n```'),
        )
        plugin._on_after_invocation(event)

        assert event.resume is not None
        assert "final_answer was called" in event.resume
        assert "result is 42" in event.resume
        assert "Do NOT write any more code" in event.resume

    def test_max_iterations_stops_loop(self, plugin, mock_agent, mock_result):
        """Should stop when max iterations reached."""
        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 10, "codeact_namespace": {}},
            result=mock_result("```python\nprint('more code')\n```"),
        )
        plugin._on_after_invocation(event)
        assert event.resume is None

    def test_iteration_counter_increments(self, plugin, mock_agent, mock_result):
        """Should increment iteration counter on each execution."""
        namespace = {"__builtins__": __builtins__, "asyncio": asyncio}
        namespace["final_answer"] = lambda r: namespace.__setitem__("__final_answer__", r)
        invocation_state = {"codeact_iteration": 0, "codeact_namespace": namespace}

        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state=invocation_state,
            result=mock_result("```python\nprint('iteration 1')\n```"),
        )
        plugin._on_after_invocation(event)
        assert invocation_state["codeact_iteration"] == 1

    def test_validation_error_feeds_back(self, plugin, mock_agent, mock_result):
        """Should feed validation errors back for self-correction."""
        namespace = {"__builtins__": __builtins__, "asyncio": asyncio}
        namespace["final_answer"] = lambda r: namespace.__setitem__("__final_answer__", r)

        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 0, "codeact_namespace": namespace},
            result=mock_result("```python\nexec('malicious')\n```"),
        )
        plugin._on_after_invocation(event)

        assert event.resume is not None
        assert "validation error" in event.resume.lower()

    def test_none_result_is_noop(self, plugin, mock_agent):
        """Should do nothing when result is None."""
        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 0},
            result=None,
        )
        plugin._on_after_invocation(event)
        assert event.resume is None

    def test_post_final_answer_is_noop(self, plugin, mock_agent, mock_result):
        """Should not execute code after final_answer was delivered."""
        namespace = {"__builtins__": __builtins__, "__final_answer_delivered__": True}

        event = AfterInvocationEvent(
            agent=mock_agent,
            invocation_state={"codeact_iteration": 1, "codeact_namespace": namespace},
            result=mock_result("```python\nprint('should not run')\n```"),
        )
        plugin._on_after_invocation(event)
        assert event.resume is None


class TestToolSignatureGeneration:
    """Test tool signature formatting for system prompt."""

    def test_generates_signatures(self, plugin, mock_agent):
        """Should generate function signatures from tool specs."""
        signatures = plugin._build_tool_signatures(mock_agent)
        assert "async def calculator(expression: str) -> str" in signatures
        assert "async def shell(command: str" in signatures

    def test_optional_parameters(self, plugin, mock_agent):
        """Should format optional params with defaults."""
        signatures = plugin._build_tool_signatures(mock_agent)
        # timeout is optional in shell tool
        assert "timeout" in signatures

    def test_includes_descriptions(self, plugin, mock_agent):
        """Should include tool descriptions as docstrings."""
        signatures = plugin._build_tool_signatures(mock_agent)
        assert "Perform calculations" in signatures
        assert "Execute shell commands" in signatures


class TestImportRestrictions:
    """Test import safety in code execution."""

    def test_allowed_import(self, plugin, mock_agent):
        """Should allow imports of safe modules."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        output = plugin._execute_code("import math\nprint(math.pi)", namespace)
        assert "3.14" in output

    def test_blocked_import(self, plugin, mock_agent):
        """Should block imports of disallowed modules."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        output = plugin._execute_code("import subprocess", namespace)
        assert "ImportError" in output
        assert "not allowed" in output

    def test_blocked_os_import(self, plugin, mock_agent):
        """Should block full os module (only os.path is allowed)."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        output = plugin._execute_code("import os\nos.system('echo pwned')", namespace)
        assert "ImportError" in output

    def test_allowed_submodule(self, plugin, mock_agent):
        """Should allow importing allowed submodules."""
        invocation_state = {}
        event = BeforeInvocationEvent(agent=mock_agent, invocation_state=invocation_state)
        plugin._on_before_invocation(event)

        namespace = invocation_state["codeact_namespace"]
        output = plugin._execute_code("from os.path import join\nprint(join('a', 'b'))", namespace)
        # os.path is allowed
        assert "a" in output


class TestJsonTypeToPython:
    """Test JSON schema type to Python type conversion."""

    def test_string(self):
        assert CodeActPlugin._json_type_to_python("string") == "str"

    def test_integer(self):
        assert CodeActPlugin._json_type_to_python("integer") == "int"

    def test_number(self):
        assert CodeActPlugin._json_type_to_python("number") == "float"

    def test_boolean(self):
        assert CodeActPlugin._json_type_to_python("boolean") == "bool"

    def test_array(self):
        assert CodeActPlugin._json_type_to_python("array") == "list"

    def test_object(self):
        assert CodeActPlugin._json_type_to_python("object") == "dict"

    def test_unknown_defaults_to_str(self):
        assert CodeActPlugin._json_type_to_python("unknown") == "str"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_no_tools_registered(self, plugin):
        """Should handle agent with no tools."""
        agent = MagicMock()
        agent.system_prompt = ""
        agent.state = MagicMock()
        agent.state.get.return_value = None
        agent.tool_registry = MagicMock()
        agent.tool_registry.registry = {}
        agent.tool_registry.get_all_tools_config.return_value = {}

        event = BeforeInvocationEvent(agent=agent, invocation_state={})
        plugin._on_before_invocation(event)

        # Should still work, just no tool signatures
        assert "CodeAct" in agent.system_prompt

    def test_hyphenated_tool_names(self, plugin):
        """Should normalize hyphenated tool names to underscores."""
        agent = MagicMock()
        agent.state = MagicMock()
        agent.state.get.return_value = None

        mock_tool = MagicMock()
        mock_tool.tool_spec = {
            "name": "my-tool",
            "description": "A tool",
            "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
        }
        agent.tool_registry = MagicMock()
        agent.tool_registry.registry = {"my-tool": mock_tool}
        agent.tool_registry.get_all_tools_config.return_value = {"my-tool": mock_tool.tool_spec}
        agent.tool = MagicMock()

        namespace = plugin._build_initial_namespace(agent)
        # Should be accessible as my_tool (underscore)
        assert "my_tool" in namespace

    def test_tool_wrapper_exception_handling(self, plugin, mock_agent):
        """Tool wrapper should propagate exceptions."""
        mock_agent.tool.calculator.side_effect = RuntimeError("Tool failed")

        wrapper = plugin._make_tool_wrapper(mock_agent, "calculator")

        with pytest.raises(RuntimeError, match="Tool failed"):
            asyncio.get_event_loop().run_until_complete(wrapper(expression="1+1"))

    def test_extract_response_text_with_multiple_blocks(self, plugin):
        """Should concatenate text from multiple content blocks."""
        result = MagicMock()
        result.message = {
            "content": [
                {"text": "First part."},
                {"toolUse": {"name": "foo"}},  # Non-text block
                {"text": "Second part."},
            ]
        }
        text = plugin._extract_response_text(result)
        assert "First part." in text
        assert "Second part." in text

    def test_extract_response_text_no_message(self, plugin):
        """Should handle result with no message gracefully."""
        result = MagicMock()
        result.message = None
        result.__str__ = lambda self: "fallback text"
        text = plugin._extract_response_text(result)
        assert text == "fallback text"
