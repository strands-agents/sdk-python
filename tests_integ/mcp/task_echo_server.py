"""
Task Echo Server for MCP Integration Testing

This module implements an MCP server with task-augmented tool execution support.
It provides tools that demonstrate the full task workflow (call_tool_as_task -> poll_task -> get_task_result)
which is useful for integration testing the MCP client's task support.

The server runs with streamable HTTP transport, which supports task-augmented requests.

Usage:
    Run this file directly to start the task echo server:
    $ python task_echo_server.py --port 8010
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import click
import mcp.types as types
from mcp.server.experimental.task_context import ServerTaskContext
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount


def create_task_server() -> Server:
    """Create and configure the task-supporting MCP server."""
    server = Server("task-echo-server")

    # Enable task support - auto-registers get_task, get_task_result, list_tasks, cancel_task
    server.experimental.enable_tasks()

    # WORKAROUND: The MCP Python SDK's enable_tasks() doesn't properly set tasks.requests.tools.call
    # capability (it creates TasksToolsCapability() but doesn't set call=TasksCallCapability()).
    # We wrap update_capabilities to fix this until the SDK is patched.
    # See: https://github.com/modelcontextprotocol/python-sdk/issues/XXX (TODO: file issue)
    original_update_capabilities = server.experimental.update_capabilities

    def patched_update_capabilities(capabilities: types.ServerCapabilities) -> None:
        original_update_capabilities(capabilities)
        # Fix the missing call capability
        if capabilities.tasks and capabilities.tasks.requests and capabilities.tasks.requests.tools:
            capabilities.tasks.requests.tools.call = types.TasksCallCapability()

    server.experimental.update_capabilities = patched_update_capabilities  # type: ignore[method-assign]

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            # Tool that requires task-augmented execution
            types.Tool(
                name="task_required_echo",
                description="Echo that requires task-augmented execution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"},
                    },
                    "required": ["message"],
                },
                execution=types.ToolExecution(taskSupport=types.TASK_REQUIRED),
            ),
            # Tool that optionally supports task execution (we prefer tasks when server supports)
            types.Tool(
                name="task_optional_echo",
                description="Echo that optionally supports task-augmented execution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"},
                    },
                    "required": ["message"],
                },
                execution=types.ToolExecution(taskSupport=types.TASK_OPTIONAL),
            ),
            # Tool that forbids task execution (must use regular call_tool)
            types.Tool(
                name="task_forbidden_echo",
                description="Echo that does not support task-augmented execution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"},
                    },
                    "required": ["message"],
                },
                execution=types.ToolExecution(taskSupport=types.TASK_FORBIDDEN),
            ),
            # Tool without explicit taskSupport (defaults to forbidden)
            types.Tool(
                name="echo",
                description="Simple echo without task support setting",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"},
                    },
                    "required": ["message"],
                },
            ),
            # Task tool with status updates
            types.Tool(
                name="task_with_status",
                description="Task that provides status updates during execution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "description": "Number of steps to simulate", "default": 3},
                    },
                },
                execution=types.ToolExecution(taskSupport=types.TASK_REQUIRED),
            ),
        ]

    async def handle_task_required_echo(arguments: dict[str, Any]) -> types.CreateTaskResult:
        """Handle task_required_echo tool - must be called as a task."""
        ctx = server.request_context
        ctx.experimental.validate_task_mode(types.TASK_REQUIRED)

        message = arguments.get("message", "")

        async def work(task: ServerTaskContext) -> types.CallToolResult:
            await task.update_status("Processing echo...")
            return types.CallToolResult(content=[types.TextContent(type="text", text=f"Task echo: {message}")])

        return await ctx.experimental.run_task(work)

    async def handle_task_optional_echo(
        arguments: dict[str, Any],
    ) -> types.CallToolResult | types.CreateTaskResult:
        """Handle task_optional_echo tool - can be called either way."""
        ctx = server.request_context
        message = arguments.get("message", "")

        if ctx.experimental.is_task:
            # Called as a task
            async def work(task: ServerTaskContext) -> types.CallToolResult:
                await task.update_status("Processing optional task echo...")
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Task optional echo: {message}")]
                )

            return await ctx.experimental.run_task(work)
        else:
            # Called directly
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Direct optional echo: {message}")]
            )

    async def handle_task_forbidden_echo(arguments: dict[str, Any]) -> types.CallToolResult:
        """Handle task_forbidden_echo tool - regular call_tool only."""
        message = arguments.get("message", "")
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"Forbidden echo: {message}")])

    async def handle_simple_echo(arguments: dict[str, Any]) -> types.CallToolResult:
        """Handle simple echo tool - regular call_tool only."""
        message = arguments.get("message", "")
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"Simple echo: {message}")])

    async def handle_task_with_status(arguments: dict[str, Any]) -> types.CreateTaskResult:
        """Handle task_with_status tool - demonstrates status updates."""
        ctx = server.request_context
        ctx.experimental.validate_task_mode(types.TASK_REQUIRED)

        steps = arguments.get("steps", 3)

        async def work(task: ServerTaskContext) -> types.CallToolResult:
            for i in range(steps):
                await task.update_status(f"Processing step {i + 1}/{steps}...")
                await anyio.sleep(0.1)  # Small delay to simulate work

            return types.CallToolResult(content=[types.TextContent(type="text", text=f"Completed {steps} steps")])

        return await ctx.experimental.run_task(work)

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult | types.CreateTaskResult:
        """Dispatch tool calls to their handlers."""
        if name == "task_required_echo":
            return await handle_task_required_echo(arguments)
        elif name == "task_optional_echo":
            return await handle_task_optional_echo(arguments)
        elif name == "task_forbidden_echo":
            return await handle_task_forbidden_echo(arguments)
        elif name == "echo":
            return await handle_simple_echo(arguments)
        elif name == "task_with_status":
            return await handle_task_with_status(arguments)
        else:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

    return server


def create_starlette_app(port: int) -> tuple[Starlette, StreamableHTTPSessionManager]:
    """Create the Starlette app with MCP session manager."""
    server = create_task_server()
    session_manager = StreamableHTTPSessionManager(app=server)

    @asynccontextmanager
    async def app_lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=session_manager.handle_request)],
        lifespan=app_lifespan,
    )

    return starlette_app, session_manager


@click.command()
@click.option("--port", default=8010, help="Port to listen on")
def main(port: int) -> int:
    """Start the task echo server."""
    import uvicorn

    starlette_app, _ = create_starlette_app(port)

    print(f"Starting task echo server on http://localhost:{port}/mcp")
    uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    return 0


if __name__ == "__main__":
    main()
