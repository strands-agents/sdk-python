"""Tests that ``strands.mcp`` is the canonical home of the MCP API.

The implementation lives under ``strands.mcp`` and ``strands.tools.mcp``
is a backwards-compatibility alias that re-exports the same classes and
registers ``sys.modules`` aliases for its submodule paths. Object
identity must be preserved for migrating code to remain correct.
"""

import warnings


def test_strands_tools_mcp_aliases_strands_mcp() -> None:
    import strands.mcp as canonical

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import strands.tools.mcp as legacy

    assert legacy.MCPClient is canonical.MCPClient
    assert legacy.MCPAgentTool is canonical.MCPAgentTool
    assert legacy.MCPTransport is canonical.MCPTransport
    assert legacy.TasksConfig is canonical.TasksConfig
    assert legacy.ToolFilters is canonical.ToolFilters


def test_strands_mcp_all_matches_tools_mcp_all() -> None:
    import strands.mcp as canonical

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import strands.tools.mcp as legacy

    assert sorted(canonical.__all__) == sorted(legacy.__all__)
