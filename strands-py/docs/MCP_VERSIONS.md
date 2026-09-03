# MCP 1.x and 2.x Support and Migration

This page explains which major versions of the `mcp` package the Strands Python SDK supports, how to install each one, and what to check when moving from 1.x to 2.x. The threading design of the client itself is covered in [MCP_CLIENT_ARCHITECTURE.md](./MCP_CLIENT_ARCHITECTURE.md).

## What changed in mcp 2.0

The official `mcp` package made 2.0 a breaking release. The changes that affect Strands:

- The default protocol revision is `2026-07-28`, which adds multi round-trip requests (a tool call can pause to ask the client for input) and an extension registry.
- Many public names were renamed or relocated. For example, the server class `FastMCP` became `MCPServer`, the transport factory `streamablehttp_client` became `streamable_http_client`, and model attributes moved from camelCase to snake_case.
- Tasks left the core protocol. The experimental 2025-11-25 task API (`ClientSession.experimental`) was removed, and the finalized replacement (SEP-2663) lives in the `io.modelcontextprotocol/tasks` extension.
- The client credentials OAuth provider renamed its `scopes` argument to `scope`.

## What Strands does about it

The SDK ships a compatibility layer (`src/strands/tools/mcp/_compat.py`) that handles the renames and most behavior differences, so `MCPClient` and the tools it produces work the same on both versions. Code that only uses the Strands API needs no changes when the installed `mcp` version changes. The overall adoption work is tracked in #1659.

## Choosing a version

The dependency range is `mcp>=1.23.0,<2.2`. A fresh install resolves to the newest 2.x release in that range.

To stay on 1.x, pin it in your own project:

```
pip install strands-agents "mcp<2"
```

Pinning an exact `mcp` version works too and is the safest way to control upgrades. The upper bound excludes `mcp` releases we have not verified yet, and we raise it as we test new releases. Every release line inside the current range has been verified: 1.29.x and 2.1.x, each exercised end to end against live MCP servers over stdio, streamable HTTP, and SSE.

CI covers both major versions on every PR: the regular test matrix resolves mcp 2.x, and the "MCP 1.x Compat" job force-installs mcp 1.x and runs the MCP client suite against it.

## Migrating your code

If your code only uses the Strands API, nothing changes. `MCPClient`, `Agent(tools=client.list_tools_sync())`, tool calls, prompts, resources, and OAuth client credentials behave the same on both versions.

Three things behave differently on 2.x:

- `read_timeout_seconds` on tool calls bounds each request round instead of the whole call, because a 2.x tool call can involve several round trips.
- 2.x validates `ToolAnnotations` strictly and drops unknown extra keys that 1.x preserved.
- MCP Tasks currently require mcp 1.x. The task client is built on the experimental 2025-11-25 API (`ClientSession.experimental`) that 2.x removed, so passing `tasks_config` to `MCPClient` on a 2.x install raises `ImportError` at construction. Pin `mcp<2` to keep using tasks. Support for the finalized SEP-2663 task extension on 2.x is tracked in #4125.

The compatibility layer only covers the Strands API. If you write your own MCP server with the `mcp` package, or build transports from `mcp` APIs yourself before passing them to `MCPClient`, that code uses `mcp` directly and the renames above apply to it. Follow the official guide at [modelcontextprotocol/python-sdk `docs/migration.md`](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/migration.md) for that part of the migration.
