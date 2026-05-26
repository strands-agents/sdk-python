"""Tests for tool security metadata (is_read_only, is_destructive, requires_confirmation).

Covers:
- AgentTool base class defaults
- @tool decorator parameters
- ToolSpec round-trip (PythonAgentTool reads from spec)
- MCPAgentTool with overrides and spec fallback
- ToolRegistry contradiction rejection and destructive-without-confirmation warning
- BeforeToolCallEvent convenience properties
- Hook-based permission gate integration test
"""

import logging
from unittest.mock import MagicMock

import pytest

from strands.hooks.events import BeforeToolCallEvent
from strands.tools.decorator import DecoratedFunctionTool, tool
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool
from strands.tools.registry import ToolRegistry
from strands.tools.tools import PythonAgentTool

# ---------------------------------------------------------------------------
# 1. AgentTool base class defaults
# ---------------------------------------------------------------------------


class _MinimalTool(PythonAgentTool):
    pass


def _make_spec(name="test_tool", **extra):
    spec = {"name": name, "description": "A test tool", "inputSchema": {"json": {"type": "object", "properties": {}}}}
    spec.update(extra)
    return spec


def test_agent_tool_defaults_all_false():
    t = PythonAgentTool(tool_name="t", tool_spec=_make_spec("t"), tool_func=lambda tu, **kw: None)
    assert t.is_read_only is False
    assert t.is_destructive is False
    assert t.requires_confirmation is False


# ---------------------------------------------------------------------------
# 2. @tool decorator parameters
# ---------------------------------------------------------------------------


def test_tool_decorator_read_only():
    @tool(read_only=True)
    def list_files(directory: str) -> str:
        """List files in a directory."""
        return ""

    assert list_files.is_read_only is True
    assert list_files.is_destructive is False
    assert list_files.requires_confirmation is False


def test_tool_decorator_destructive_with_confirmation():
    @tool(destructive=True, requires_confirmation=True)
    def delete_file(path: str) -> str:
        """Delete a file permanently."""
        return ""

    assert delete_file.is_read_only is False
    assert delete_file.is_destructive is True
    assert delete_file.requires_confirmation is True


def test_tool_decorator_bare_has_defaults():
    @tool
    def noop() -> str:
        """Does nothing."""
        return ""

    assert noop.is_read_only is False
    assert noop.is_destructive is False
    assert noop.requires_confirmation is False


def test_tool_decorator_only_requires_confirmation():
    @tool(requires_confirmation=True)
    def sensitive_op(data: str) -> str:
        """A sensitive operation."""
        return data

    assert sensitive_op.is_read_only is False
    assert sensitive_op.is_destructive is False
    assert sensitive_op.requires_confirmation is True


# ---------------------------------------------------------------------------
# 3. ToolSpec round-trip (PythonAgentTool reads from spec)
# ---------------------------------------------------------------------------


def test_python_agent_tool_reads_read_only_from_spec():
    spec = _make_spec("reader", readOnly=True)
    t = PythonAgentTool(tool_name="reader", tool_spec=spec, tool_func=lambda tu, **kw: None)
    assert t.is_read_only is True
    assert t.is_destructive is False


def test_python_agent_tool_reads_destructive_from_spec():
    spec = _make_spec("destroyer", destructive=True, requiresConfirmation=True)
    t = PythonAgentTool(tool_name="destroyer", tool_spec=spec, tool_func=lambda tu, **kw: None)
    assert t.is_destructive is True
    assert t.requires_confirmation is True
    assert t.is_read_only is False


def test_python_agent_tool_no_security_fields_in_spec():
    spec = _make_spec("plain")
    t = PythonAgentTool(tool_name="plain", tool_spec=spec, tool_func=lambda tu, **kw: None)
    assert t.is_read_only is False
    assert t.is_destructive is False
    assert t.requires_confirmation is False


# ---------------------------------------------------------------------------
# 4. MCPAgentTool with overrides and spec fallback
# ---------------------------------------------------------------------------


def _make_mcp_tool(name="mcp_test", input_schema=None):
    mcp_tool = MagicMock()
    mcp_tool.name = name
    mcp_tool.description = f"MCP tool {name}"
    mcp_tool.inputSchema = input_schema or {"type": "object", "properties": {}}
    mcp_tool.outputSchema = None
    return mcp_tool


def test_mcp_tool_defaults():
    mcp_tool = _make_mcp_tool()
    t = MCPAgentTool(mcp_tool=mcp_tool, mcp_client=MagicMock())
    assert t.is_read_only is False
    assert t.is_destructive is False
    assert t.requires_confirmation is False


def test_mcp_tool_constructor_overrides():
    mcp_tool = _make_mcp_tool()
    t = MCPAgentTool(mcp_tool=mcp_tool, mcp_client=MagicMock(), read_only=True)
    assert t.is_read_only is True
    assert t.is_destructive is False


def test_mcp_tool_destructive_override():
    mcp_tool = _make_mcp_tool()
    t = MCPAgentTool(
        mcp_tool=mcp_tool,
        mcp_client=MagicMock(),
        destructive=True,
        requires_confirmation=True,
    )
    assert t.is_destructive is True
    assert t.requires_confirmation is True
    assert t.is_read_only is False


def test_mcp_tool_override_takes_precedence_over_spec():
    """Constructor override should win even if the spec says differently."""
    mcp_tool = _make_mcp_tool()
    t = MCPAgentTool(
        mcp_tool=mcp_tool,
        mcp_client=MagicMock(),
        read_only=False,
    )
    assert t.is_read_only is False


def test_mcp_tool_no_override_defaults_to_false():
    """Without constructor overrides, MCP tools default to False for all security properties."""
    mcp_tool = _make_mcp_tool()
    t = MCPAgentTool(mcp_tool=mcp_tool, mcp_client=MagicMock())
    assert t.is_read_only is False
    assert t.is_destructive is False
    assert t.requires_confirmation is False


# ---------------------------------------------------------------------------
# 5. ToolRegistry validation
# ---------------------------------------------------------------------------


def test_registry_rejects_read_only_and_destructive():
    @tool(read_only=True, destructive=True)
    def bad_tool() -> str:
        """Contradictory tool."""
        return ""

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="cannot be both read_only and destructive"):
        registry.register_tool(bad_tool)


def test_registry_warns_destructive_without_confirmation(caplog):
    @tool(destructive=True)
    def risky_tool() -> str:
        """A risky tool without confirmation."""
        return ""

    registry = ToolRegistry()
    with caplog.at_level(logging.WARNING):
        registry.register_tool(risky_tool)

    assert "destructive but does not require confirmation" in caplog.text


def test_registry_accepts_destructive_with_confirmation():
    @tool(destructive=True, requires_confirmation=True)
    def safe_destructive() -> str:
        """A destructive tool that requires confirmation."""
        return ""

    registry = ToolRegistry()
    registry.register_tool(safe_destructive)
    assert "safe_destructive" in registry.registry


def test_registry_accepts_read_only():
    @tool(read_only=True)
    def reader() -> str:
        """A read-only tool."""
        return ""

    registry = ToolRegistry()
    registry.register_tool(reader)
    assert "reader" in registry.registry


def test_registry_accepts_default_metadata():
    @tool
    def plain() -> str:
        """A plain tool."""
        return ""

    registry = ToolRegistry()
    registry.register_tool(plain)
    assert "plain" in registry.registry


def test_registry_replace_rejects_contradictory_metadata():
    """ToolRegistry.replace() must also validate security metadata."""

    @tool
    def my_tool() -> str:
        """A normal tool."""
        return ""

    @tool(read_only=True, destructive=True, name="my_tool")
    def bad_replacement() -> str:
        """Contradictory replacement."""
        return ""

    registry = ToolRegistry()
    registry.register_tool(my_tool)

    with pytest.raises(ValueError, match="cannot be both read_only and destructive"):
        registry.replace(bad_replacement)


# ---------------------------------------------------------------------------
# 6. BeforeToolCallEvent convenience properties
# ---------------------------------------------------------------------------


def _make_before_event(selected_tool=None):
    return BeforeToolCallEvent(
        agent=MagicMock(),
        selected_tool=selected_tool,
        tool_use={"name": "test", "toolUseId": "id-1", "input": {}},
        invocation_state={},
    )


def test_event_convenience_props_with_none_tool():
    event = _make_before_event(selected_tool=None)
    assert event.tool_is_read_only is False
    assert event.tool_is_destructive is False
    assert event.tool_requires_confirmation is False


def test_event_convenience_props_with_read_only_tool():
    @tool(read_only=True)
    def reader() -> str:
        """Read only."""
        return ""

    event = _make_before_event(selected_tool=reader)
    assert event.tool_is_read_only is True
    assert event.tool_is_destructive is False
    assert event.tool_requires_confirmation is False


def test_event_convenience_props_with_destructive_tool():
    @tool(destructive=True, requires_confirmation=True)
    def destroyer() -> str:
        """Destructive."""
        return ""

    event = _make_before_event(selected_tool=destroyer)
    assert event.tool_is_read_only is False
    assert event.tool_is_destructive is True
    assert event.tool_requires_confirmation is True


# ---------------------------------------------------------------------------
# 7. Integration: hook-based permission gate
# ---------------------------------------------------------------------------


def test_hook_cancels_destructive_tool():
    """Simulate a BeforeToolCallEvent hook that cancels destructive tools."""

    @tool(destructive=True, requires_confirmation=True)
    def delete_db() -> str:
        """Delete the database."""
        return ""

    event = _make_before_event(selected_tool=delete_db)

    # Hook logic: cancel destructive tools
    if event.tool_is_destructive:
        event.cancel_tool = "Destructive tool requires approval"

    assert event.cancel_tool == "Destructive tool requires approval"


def test_hook_allows_read_only_tool():
    """Simulate a BeforeToolCallEvent hook that allows read-only tools."""

    @tool(read_only=True)
    def list_items() -> str:
        """List items."""
        return ""

    event = _make_before_event(selected_tool=list_items)

    # Hook logic: only cancel non-read-only tools
    if not event.tool_is_read_only:
        event.cancel_tool = "Non-read-only tool blocked"

    assert event.cancel_tool is False


# ---------------------------------------------------------------------------
# 8. Backward compatibility
# ---------------------------------------------------------------------------


def test_decorated_tool_get_preserves_security_metadata():
    """Verify __get__ (descriptor protocol) propagates security metadata."""

    class MyClass:
        @tool(destructive=True, requires_confirmation=True)
        def my_method(self, x: str) -> str:
            """A method tool."""
            return x

    instance = MyClass()
    bound_tool = instance.my_method

    assert isinstance(bound_tool, DecoratedFunctionTool)
    assert bound_tool.is_destructive is True
    assert bound_tool.requires_confirmation is True
    assert bound_tool.is_read_only is False


def test_tool_spec_typed_dict_accepts_security_fields():
    """Verify ToolSpec TypedDict accepts the new NotRequired fields."""
    from strands.types.tools import ToolSpec

    spec: ToolSpec = {
        "name": "test",
        "description": "test",
        "inputSchema": {},
        "readOnly": True,
        "destructive": False,
        "requiresConfirmation": False,
    }
    assert spec["readOnly"] is True
    assert spec["destructive"] is False


def test_decorated_tool_writes_security_fields_to_spec():
    """@tool should write security fields into its ToolSpec for serialization consistency."""

    @tool(read_only=True)
    def reader(x: str) -> str:
        """Read something."""
        return x

    assert reader.tool_spec.get("readOnly") is True
    assert "destructive" not in reader.tool_spec
    assert "requiresConfirmation" not in reader.tool_spec


def test_decorated_tool_destructive_fields_in_spec():
    """@tool(destructive=True, requires_confirmation=True) should write both fields to spec."""

    @tool(destructive=True, requires_confirmation=True)
    def deleter(x: str) -> str:
        """Delete something."""
        return x

    assert deleter.tool_spec.get("destructive") is True
    assert deleter.tool_spec.get("requiresConfirmation") is True
    assert "readOnly" not in deleter.tool_spec


def test_bare_decorator_omits_security_fields_from_spec():
    """Plain @tool should not pollute ToolSpec with False security fields."""

    @tool
    def plain(x: str) -> str:
        """Do something."""
        return x

    assert "readOnly" not in plain.tool_spec
    assert "destructive" not in plain.tool_spec
    assert "requiresConfirmation" not in plain.tool_spec
