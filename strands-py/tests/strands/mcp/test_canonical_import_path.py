"""Tests for the canonical ``strands.mcp`` import path.

The implementation currently lives in ``strands.tools.mcp``. This test
locks in the contract that ``strands.mcp`` re-exports the same objects so
that users can migrate imports ahead of the follow-up refactor that
moves the implementation.
"""


def test_strands_mcp_reexports_public_api() -> None:
    import strands.mcp as new
    import strands.tools.mcp as old

    assert new.MCPClient is old.MCPClient
    assert new.MCPAgentTool is old.MCPAgentTool
    assert new.MCPTransport is old.MCPTransport
    assert new.TasksConfig is old.TasksConfig
    assert new.ToolFilters is old.ToolFilters


def test_strands_mcp_all_matches_tools_mcp_all() -> None:
    import strands.mcp as new
    import strands.tools.mcp as old

    assert sorted(new.__all__) == sorted(old.__all__)
