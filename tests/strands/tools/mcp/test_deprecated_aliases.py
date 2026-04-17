"""Tests for the backwards-compatible aliases at ``strands.tools.mcp``.

The MCP integration moved to ``strands.mcp`` but the old import paths must
continue to work and emit ``DeprecationWarning`` until a future release
removes them.
"""

import importlib
import sys
import warnings


def _reimport(module_name: str) -> None:
    """Force a fresh import so the module-level warnings fire again."""
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


def test_package_import_emits_deprecation_warning() -> None:
    _reimport("strands.tools.mcp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reimport("strands.tools.mcp")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "strands.mcp" in str(deprecations[0].message)


def test_package_reexports_public_names() -> None:
    from strands import mcp as new_mcp
    from strands.tools import mcp as old_mcp

    for name in ("MCPAgentTool", "MCPClient", "MCPTransport", "TasksConfig", "ToolFilters"):
        assert getattr(old_mcp, name) is getattr(new_mcp, name)


def test_submodule_imports_still_work() -> None:
    """Legacy submodule paths like ``strands.tools.mcp.mcp_client`` must resolve."""
    from strands.mcp import mcp_agent_tool as new_agent_tool
    from strands.mcp import mcp_client as new_client
    from strands.mcp import mcp_instrumentation as new_instrumentation
    from strands.mcp import mcp_tasks as new_tasks
    from strands.mcp import mcp_types as new_types
    from strands.tools.mcp import mcp_agent_tool as old_agent_tool
    from strands.tools.mcp import mcp_client as old_client
    from strands.tools.mcp import mcp_instrumentation as old_instrumentation
    from strands.tools.mcp import mcp_tasks as old_tasks
    from strands.tools.mcp import mcp_types as old_types

    assert old_client.MCPClient is new_client.MCPClient
    assert old_client.ToolFilters is new_client.ToolFilters
    assert old_agent_tool.MCPAgentTool is new_agent_tool.MCPAgentTool
    assert old_types.MCPTransport is new_types.MCPTransport
    assert old_types.MCPToolResult is new_types.MCPToolResult
    assert old_tasks.TasksConfig is new_tasks.TasksConfig
    assert old_tasks.DEFAULT_TASK_POLL_TIMEOUT is new_tasks.DEFAULT_TASK_POLL_TIMEOUT
    assert old_tasks.DEFAULT_TASK_TTL is new_tasks.DEFAULT_TASK_TTL
    assert old_instrumentation.mcp_instrumentation is new_instrumentation.mcp_instrumentation


def test_new_path_does_not_emit_deprecation_warning() -> None:
    for name in (
        "strands.mcp",
        "strands.mcp.mcp_client",
        "strands.mcp.mcp_agent_tool",
        "strands.mcp.mcp_types",
        "strands.mcp.mcp_tasks",
        "strands.mcp.mcp_instrumentation",
    ):
        sys.modules.pop(name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reimport("strands.mcp")
        _reimport("strands.mcp.mcp_client")
        _reimport("strands.mcp.mcp_agent_tool")
        _reimport("strands.mcp.mcp_types")
        _reimport("strands.mcp.mcp_tasks")
        _reimport("strands.mcp.mcp_instrumentation")

    from_strands_mcp = [
        w for w in caught if "strands.mcp" in str(w.message) and "strands.tools.mcp" not in str(w.message)
    ]
    deprecations = [w for w in from_strands_mcp if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []
