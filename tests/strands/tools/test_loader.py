import os
import re
import textwrap

import pytest

from strands.tools.decorator import DecoratedFunctionTool
from strands.tools.loader import ToolLoader
from strands.tools.tools import PythonAgentTool


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
        textwrap.dedent(
            """
            import strands

            @strands.tools.tool
            def alpha():
                return "alpha"

            @strands.tools.tool
            def bravo():
                return "bravo"
            """
        )
    ],
    indirect=True,
)
def test_load_python_tool_path_multiple_function_based(tool_path):
    # load_python_tools, load_tools returns a list when multiple decorated tools are present
    loaded_python_tools = ToolLoader.load_python_tools(tool_path, "alpha")

    assert isinstance(loaded_python_tools, list)
    assert len(loaded_python_tools) == 2
    assert all(isinstance(t, DecoratedFunctionTool) for t in loaded_python_tools)
    names = {t.tool_name for t in loaded_python_tools}
    assert names == {"alpha", "bravo"}

    loaded_tools = ToolLoader.load_tools(tool_path, "alpha")

    assert isinstance(loaded_tools, list)
    assert len(loaded_tools) == 2
    assert all(isinstance(t, DecoratedFunctionTool) for t in loaded_tools)
    names = {t.tool_name for t in loaded_tools}
    assert names == {"alpha", "bravo"}


