"""Tests for tool_spec setter on DecoratedFunctionTool and PythonAgentTool."""

from strands.tools.decorator import tool
from strands.tools.tools import PythonAgentTool
from strands.types.tools import ToolSpec


class TestDecoratedFunctionToolSpecSetter:
    """Tests for DecoratedFunctionTool.tool_spec setter."""

    def test_set_tool_spec_replaces_spec(self):
        @tool
        def my_tool(query: str) -> str:
            """A test tool."""
            return query

        new_spec: ToolSpec = {
            "name": "my_tool",
            "description": "Updated tool",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                }
            },
        }
        my_tool.tool_spec = new_spec
        assert my_tool.tool_spec is new_spec
        assert "limit" in my_tool.tool_spec["inputSchema"]["json"]["properties"]

    def test_set_tool_spec_persists_across_reads(self):
        @tool
        def another_tool(x: int) -> int:
            """Another test tool."""
            return x

        new_spec: ToolSpec = {
            "name": "another_tool",
            "description": "Modified",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x"],
                }
            },
        }
        another_tool.tool_spec = new_spec
        assert another_tool.tool_spec["description"] == "Modified"
        assert another_tool.tool_spec["description"] == "Modified"

    def test_add_property_via_setter(self):
        @tool
        def dynamic_tool(base: str) -> str:
            """A dynamic tool."""
            return base

        spec = dynamic_tool.tool_spec.copy()
        spec["inputSchema"] = dynamic_tool.tool_spec["inputSchema"].copy()
        spec["inputSchema"]["json"] = dynamic_tool.tool_spec["inputSchema"]["json"].copy()
        spec["inputSchema"]["json"]["properties"] = dynamic_tool.tool_spec["inputSchema"]["json"]["properties"].copy()
        spec["inputSchema"]["json"]["properties"]["extra"] = {
            "type": "string",
            "description": "Extra param",
        }
        dynamic_tool.tool_spec = spec
        assert "extra" in dynamic_tool.tool_spec["inputSchema"]["json"]["properties"]


class TestPythonAgentToolSpecSetter:
    """Tests for PythonAgentTool.tool_spec setter."""

    def _make_tool(self) -> PythonAgentTool:
        def func(tool_use, **kwargs):
            return {"status": "success", "content": [{"text": "ok"}], "toolUseId": tool_use["toolUseId"]}

        spec: ToolSpec = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                }
            },
        }
        return PythonAgentTool("test_tool", spec, func)

    def test_set_tool_spec(self):
        t = self._make_tool()
        new_spec: ToolSpec = {
            "name": "test_tool",
            "description": "Updated",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "extra": {"type": "integer"},
                    },
                    "required": ["input"],
                }
            },
        }
        t.tool_spec = new_spec
        assert t.tool_spec is new_spec
        assert "extra" in t.tool_spec["inputSchema"]["json"]["properties"]

    def test_set_tool_spec_persists(self):
        t = self._make_tool()
        new_spec: ToolSpec = {
            "name": "test_tool",
            "description": "Persisted",
            "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
        }
        t.tool_spec = new_spec
        assert t.tool_spec["description"] == "Persisted"
        assert t.tool_spec["description"] == "Persisted"
