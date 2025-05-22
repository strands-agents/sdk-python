import os
import pathlib
import re
import textwrap
import unittest.mock

import pytest
import strands
from strands.tools.loader import ToolLoader
from strands.tools.tools import FunctionTool, PythonAgentTool


def test_load_function_tool():
    @strands.tools.tool
    def tool_function(a):
        return a

    tool = strands.tools.loader.load_function_tool(tool_function)

    assert isinstance(tool, FunctionTool)


def test_load_function_tool_no_function():
    tool = strands.tools.loader.load_function_tool("no_function")

    assert tool is None


def test_load_function_tool_no_spec():
    def tool_function(a):
        return a

    tool = strands.tools.loader.load_function_tool(tool_function)

    assert tool is None


def test_load_function_tool_invalid():
    def tool_function(a):
        return a

    tool_function.TOOL_SPEC = "invalid"

    tool = strands.tools.loader.load_function_tool(tool_function)

    assert tool is None


def test_scan_module_for_tools():
    @strands.tools.tool
    def tool_function_1(a):
        return a

    @strands.tools.tool
    def tool_function_2(b):
        return b

    def tool_function_3(c):
        return c

    def tool_function_4(d):
        return d

    tool_function_4.TOOL_SPEC = "invalid"

    mock_module = unittest.mock.MagicMock()
    mock_module.tool_function_1 = tool_function_1
    mock_module.tool_function_2 = tool_function_2
    mock_module.tool_function_3 = tool_function_3
    mock_module.tool_function_4 = tool_function_4

    tools = strands.tools.loader.scan_module_for_tools(mock_module)

    assert len(tools) == 2
    assert all(isinstance(tool, FunctionTool) for tool in tools)


def test_scan_directory_for_tools(tmp_path):
    tool_definition_1 = textwrap.dedent("""
        import strands

        @strands.tools.tool
        def tool_function_1(a):
            return a
    """)
    tool_definition_2 = textwrap.dedent("""
        import strands

        @strands.tools.tool
        def tool_function_2(b):
            return b
    """)
    tool_definition_3 = textwrap.dedent("""
        def tool_function_3(c):
            return c
    """)
    tool_definition_4 = textwrap.dedent("""
        def tool_function_4(d):
            return d
    """)
    tool_definition_5 = ""
    tool_definition_6 = "**invalid**"

    tool_path_1 = tmp_path / "tool_1.py"
    tool_path_2 = tmp_path / "tool_2.py"
    tool_path_3 = tmp_path / "tool_3.py"
    tool_path_4 = tmp_path / "tool_4.py"
    tool_path_5 = tmp_path / "_tool_5.py"
    tool_path_6 = tmp_path / "tool_6.py"

    tool_path_1.write_text(tool_definition_1)
    tool_path_2.write_text(tool_definition_2)
    tool_path_3.write_text(tool_definition_3)
    tool_path_4.write_text(tool_definition_4)
    tool_path_5.write_text(tool_definition_5)
    tool_path_6.write_text(tool_definition_6)

    tools = strands.tools.loader.scan_directory_for_tools(tmp_path)

    tru_tool_names = sorted(tools.keys())
    exp_tool_names = ["tool_function_1", "tool_function_2"]

    assert tru_tool_names == exp_tool_names
    assert all(isinstance(tool, FunctionTool) for tool in tools.values())


def test_scan_directory_for_tools_does_not_exist():
    tru_tools = strands.tools.loader.scan_directory_for_tools(pathlib.Path("does_not_exist"))
    exp_tools = {}

    assert tru_tools == exp_tools


@pytest.fixture
def tool_path(request, tmp_path, monkeypatch):
    definition = request.param

    package_dir = tmp_path / f"package_{request.function.__name__}"
    package_dir.mkdir()

    init_path = package_dir / "__init__.py"
    init_path.touch()

    definition_path = package_dir / f"module_{request.function.__name__}.py"
    definition_path.write_text(definition)

    monkeypatch.syspath_prepend(str(tmp_path))

    return str(definition_path)


@pytest.fixture
def tool_module(tool_path):
    return ".".join(os.path.splitext(tool_path)[0].split(os.sep)[-2:])


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            import strands

            @strands.tools.tool
            def identity(a: int):
                return a
        """)
    ],
    indirect=True,
)
def test_load_python_tool_path_function_based(tool_path):
    tool = ToolLoader.load_python_tool(tool_path, "identity")

    assert isinstance(tool, FunctionTool)


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            TOOL_SPEC = {
                "name": "identity",
                "description": "identity tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                        },
                    },
                },
            }

            def identity(a: int):
                return a
        """)
    ],
    indirect=True,
)
def test_load_python_tool_path_module_based(tool_path):
    tool = ToolLoader.load_python_tool(tool_path, "identity")

    assert isinstance(tool, PythonAgentTool)


def test_load_python_tool_path_invalid():
    with pytest.raises(ImportError, match="Could not create spec for identity"):
        ToolLoader.load_python_tool("invalid", "identity")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            def no_spec():
                return
        """)
    ],
    indirect=True,
)
def test_load_python_tool_path_no_spec(tool_path):
    with pytest.raises(AttributeError, match="Tool no_spec missing TOOL_SPEC"):
        ToolLoader.load_python_tool(tool_path, "no_spec")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            TOOL_SPEC = {"name": "no_function"}
        """)
    ],
    indirect=True,
)
def test_load_python_tool_path_no_function(tool_path):
    with pytest.raises(AttributeError, match="Tool no_function missing function"):
        ToolLoader.load_python_tool(tool_path, "no_function")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            TOOL_SPEC = {"name": "no_callable"}

            no_callable = "not callable"
        """)
    ],
    indirect=True,
)
def test_load_python_tool_path_no_callable(tool_path):
    with pytest.raises(TypeError, match="Tool no_callable function is not callable"):
        ToolLoader.load_python_tool(tool_path, "no_callable")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            import strands

            @strands.tools.tool
            def identity(a: int):
                return a
        """)
    ],
    indirect=True,
)
def test_load_python_tool_dot_function_based(tool_path, tool_module):
    _ = tool_path
    tool_module = f"{tool_module}:identity"

    tool = ToolLoader.load_python_tool(tool_module, "identity")

    assert isinstance(tool, FunctionTool)


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            TOOL_SPEC = {"name": "no_function"}
        """)
    ],
    indirect=True,
)
def test_load_python_tool_dot_no_function(tool_path, tool_module):
    _ = tool_path

    with pytest.raises(AttributeError, match=re.escape(f"Module {tool_module} has no function named no_function")):
        ToolLoader.load_python_tool(f"{tool_module}:no_function", "no_function")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            def no_decorator():
                return
        """)
    ],
    indirect=True,
)
def test_load_python_tool_dot_no_decorator(tool_path, tool_module):
    _ = tool_path

    with pytest.raises(ValueError, match=re.escape(f"Function no_decorator in {tool_module} is not a valid tool")):
        ToolLoader.load_python_tool(f"{tool_module}:no_decorator", "no_decorator")


def test_load_python_tool_dot_missing():
    with pytest.raises(ImportError, match="Failed to import module missing"):
        ToolLoader.load_python_tool("missing:function", "function")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            import strands

            @strands.tools.tool
            def identity(a: int):
                return a
        """)
    ],
    indirect=True,
)
def test_load_tool(tool_path):
    tool = ToolLoader.load_tool(tool_path, "identity")

    assert isinstance(tool, FunctionTool)


def test_load_tool_missing():
    with pytest.raises(FileNotFoundError, match="Tool file not found"):
        ToolLoader.load_tool("missing", "function")


def test_load_tool_invalid_ext(tmp_path):
    tool_path = tmp_path / "tool.txt"
    tool_path.touch()

    with pytest.raises(ValueError, match="Unsupported tool file type: .txt"):
        ToolLoader.load_tool(str(tool_path), "function")


@pytest.mark.parametrize(
    "tool_path",
    [
        textwrap.dedent("""
            def no_spec():
                return
        """)
    ],
    indirect=True,
)
def test_load_tool_no_spec(tool_path):
    with pytest.raises(AttributeError, match="Tool no_spec missing TOOL_SPEC"):
        ToolLoader.load_tool(tool_path, "no_spec")
