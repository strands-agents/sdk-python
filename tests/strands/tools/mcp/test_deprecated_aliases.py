"""Tests for the backwards-compatible aliases at ``strands.tools.mcp``.

After the move to ``strands.mcp``, the legacy package re-exports the
public API, emits a ``DeprecationWarning`` at package-import time, and
registers ``sys.modules`` aliases so that legacy submodule paths such as
``strands.tools.mcp.mcp_client`` continue to resolve.
"""

import importlib
import sys
import warnings


def _reimport(name: str):
    """Force ``name`` to be reimported so import-time side-effects fire again."""

    for mod_name in [n for n in list(sys.modules) if n == name or n.startswith(f"{name}.")]:
        del sys.modules[mod_name]
    return importlib.import_module(name)


def test_package_import_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reimport("strands.tools.mcp")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("strands.tools.mcp has moved to strands.mcp" in str(w.message) for w in deprecations)


def test_public_api_is_identical_to_strands_mcp() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = _reimport("strands.tools.mcp")

    from strands import mcp as canonical

    assert legacy.MCPClient is canonical.MCPClient
    assert legacy.MCPAgentTool is canonical.MCPAgentTool
    assert legacy.MCPTransport is canonical.MCPTransport
    assert legacy.TasksConfig is canonical.TasksConfig
    assert legacy.ToolFilters is canonical.ToolFilters


def test_legacy_submodule_paths_resolve_via_sys_modules() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _reimport("strands.tools.mcp")

        from strands.tools.mcp import mcp_agent_tool as legacy_agent_tool
        from strands.tools.mcp import mcp_client as legacy_client
        from strands.tools.mcp import mcp_instrumentation as legacy_instrumentation
        from strands.tools.mcp import mcp_tasks as legacy_tasks
        from strands.tools.mcp import mcp_types as legacy_types

    from strands.mcp import mcp_agent_tool as canonical_agent_tool
    from strands.mcp import mcp_client as canonical_client
    from strands.mcp import mcp_instrumentation as canonical_instrumentation
    from strands.mcp import mcp_tasks as canonical_tasks
    from strands.mcp import mcp_types as canonical_types

    assert legacy_agent_tool is canonical_agent_tool
    assert legacy_client is canonical_client
    assert legacy_instrumentation is canonical_instrumentation
    assert legacy_tasks is canonical_tasks
    assert legacy_types is canonical_types


def test_deep_submodule_import_resolves() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _reimport("strands.tools.mcp")

        from strands.tools.mcp.mcp_client import MCPClient as LegacyMCPClient

    from strands.mcp.mcp_client import MCPClient as CanonicalMCPClient

    assert LegacyMCPClient is CanonicalMCPClient
