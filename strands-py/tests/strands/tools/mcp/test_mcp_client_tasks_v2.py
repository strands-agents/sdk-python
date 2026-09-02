"""Tests for finalized SEP-2663 Tasks support on mcp 2.x."""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from strands.tools.mcp import (
    MCPCancelTaskResult,
    MCPClient,
    MCPCreateTaskResult,
    MCPGetTaskResult,
    MCPUpdateTaskResult,
)
from strands.tools.mcp._compat import MCP_V2

pytestmark = pytest.mark.skipif(not MCP_V2, reason="requires mcp 2.x")

TASK_BASE = {
    "taskId": "task-1",
    "createdAt": "2026-09-01T00:00:00Z",
    "lastUpdatedAt": "2026-09-01T00:00:01Z",
    "ttlMs": None,
    "pollIntervalMs": 0,
}


class FakeTaskSession:
    """Minimal modern session fake for task protocol tests."""

    protocol_version = "2026-07-28"

    def __init__(self, responses: dict[str, list[Any]] | None = None) -> None:
        self.responses = responses or {}
        self.requests: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.input_requests: list[Any] = []

    async def send_request(
        self,
        request: Any,
        result_type: type[Any],
        request_read_timeout_seconds: float | None = None,
    ) -> Any:
        dumped = request.model_dump(by_alias=True, mode="json", exclude_none=True)
        dumped["timeout"] = request_read_timeout_seconds
        dumped["name_param"] = type(request).name_param
        self.requests.append(dumped)
        return result_type.model_validate(self.responses[request.method].pop(0))

    async def call_tool(self, name: str, arguments: dict[str, Any] | None, timeout: float | None, **kwargs: Any) -> Any:
        self.tool_calls.append({"name": name, "arguments": arguments, "timeout": timeout, **kwargs})
        return self.responses["tools/call"].pop(0)

    async def dispatch_input_request(self, context: Any, request: Any) -> Any:
        from mcp.types import ElicitResult

        self.input_requests.append((context, request))
        return ElicitResult.model_validate({"action": "accept", "content": {"approved": True}})


def task_client(session: FakeTaskSession) -> MCPClient:
    """Create a disconnected client wired directly to a fake modern session."""
    client = MCPClient(lambda: None, tasks_config={"poll_interval": timedelta(microseconds=1)})
    client._background_thread_session = session
    client._server_task_capable = True
    return client


def create_task(status: str = "working") -> MCPCreateTaskResult:
    """Create a valid task handle."""
    return MCPCreateTaskResult.model_validate({**TASK_BASE, "resultType": "task", "status": status})


def test_public_task_lifecycle_over_real_mcp_transport() -> None:
    """Test public task APIs, negotiation, decoding, and routing headers end to end."""
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server import MCPServer
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
    from mcp.server.extension import Extension, MethodBinding
    from mcp_types import CallToolRequestParams
    from pydantic import BaseModel, Field

    class GetTaskParams(BaseModel):
        task_id: str = Field(alias="taskId")

    class UpdateTaskParams(GetTaskParams):
        input_responses: dict[str, Any] = Field(alias="inputResponses")

    seen_headers: dict[str, dict[str, str | None]] = {}

    def completed(task_id: str) -> dict[str, Any]:
        return {
            **TASK_BASE,
            "resultType": "complete",
            "taskId": task_id,
            "status": "completed",
            "result": {
                "resultType": "complete",
                "content": [{"type": "text", "text": "completed"}],
                "isError": False,
            },
        }

    class TasksExtension(Extension):
        identifier = "io.modelcontextprotocol/tasks"

        def methods(self) -> tuple[MethodBinding, ...]:
            versions = frozenset({"2026-07-28"})
            return (
                MethodBinding("tasks/get", GetTaskParams, self.get_task, versions),
                MethodBinding("tasks/update", UpdateTaskParams, self.update_task, versions),
                MethodBinding("tasks/cancel", GetTaskParams, self.cancel_task, versions),
            )

        @staticmethod
        def record(context: ServerRequestContext[Any, Any]) -> None:
            seen_headers[context.method] = {
                "mcp-method": context.request.headers.get("mcp-method"),
                "mcp-name": context.request.headers.get("mcp-name"),
            }

        async def get_task(self, context: ServerRequestContext[Any, Any], params: GetTaskParams) -> HandlerResult:
            self.record(context)
            return completed(params.task_id)

        async def update_task(self, context: ServerRequestContext[Any, Any], params: UpdateTaskParams) -> HandlerResult:
            self.record(context)
            assert params.input_responses == {"request-1": {"action": "decline"}}
            return {"resultType": "complete"}

        async def cancel_task(self, context: ServerRequestContext[Any, Any], params: GetTaskParams) -> HandlerResult:
            self.record(context)
            return {"resultType": "complete"}

        async def intercept_tool_call(
            self,
            params: CallToolRequestParams,
            context: ServerRequestContext[Any, Any],
            call_next: CallNext,
        ) -> HandlerResult:
            _ = (context, call_next)
            if params.name == "direct-runtime":
                return {
                    "resultType": "complete",
                    "content": [{"type": "text", "text": "direct"}],
                    "isError": False,
                }
            return {
                **TASK_BASE,
                "resultType": "task",
                "taskId": "task-runtime",
                "status": "working",
            }

    server = MCPServer("sep2663-test", extensions=[TasksExtension()])
    app = server.streamable_http_app(json_response=True, stateless_http=True)

    @asynccontextmanager
    async def transport() -> Any:
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
            ) as http_client:
                async with streamable_http_client(
                    url="http://127.0.0.1:8000/mcp",
                    http_client=http_client,
                ) as streams:
                    yield streams

    client = MCPClient(transport, tasks_config={"poll_interval": timedelta(milliseconds=1)})
    with client:
        task = client.call_tool_with_task_sync("task-runtime")
        assert isinstance(task, MCPCreateTaskResult)
        assert client.get_task_sync(task.task_id).status == "completed"
        assert isinstance(
            client.update_task_sync(task.task_id, {"request-1": {"action": "decline"}}),
            MCPUpdateTaskResult,
        )
        assert isinstance(client.cancel_task_sync(task.task_id), MCPCancelTaskResult)

        auto = client.call_tool_sync(tool_use_id="auto", name="auto-runtime", arguments={})
        direct = client.call_tool_sync(tool_use_id="direct", name="direct-runtime", arguments={})

    assert auto["content"][0]["text"] == "completed"
    assert direct["content"][0]["text"] == "direct"
    assert seen_headers == {
        method: {"mcp-method": method, "mcp-name": "task-runtime"}
        for method in ("tasks/get", "tasks/update", "tasks/cancel")
    }


@pytest.mark.parametrize(
    "model,value",
    [
        (
            MCPGetTaskResult,
            {**TASK_BASE, "resultType": "complete", "status": "completed"},
        ),
        (
            MCPCreateTaskResult,
            {
                **TASK_BASE,
                "resultType": "task",
                "status": "working",
                "lastUpdatedAt": "2025-09-01T00:00:00Z",
            },
        ),
        (
            MCPUpdateTaskResult,
            {"resultType": "complete", "taskId": "unexpected"},
        ),
        (
            MCPCreateTaskResult,
            {**TASK_BASE, "status": "working"},
        ),
        (
            MCPGetTaskResult,
            {**TASK_BASE, "status": "working"},
        ),
        (
            MCPGetTaskResult,
            {**TASK_BASE, "resultType": "complete", "status": "working", "result": None},
        ),
        (
            MCPGetTaskResult,
            {
                **TASK_BASE,
                "resultType": "complete",
                "status": "input_required",
                "inputRequests": {"request-1": {"method": "invalid/request"}},
            },
        ),
        (
            MCPUpdateTaskResult,
            {},
        ),
    ],
)
def test_task_models_reject_malformed_status_shapes(model: type[Any], value: dict[str, Any]) -> None:
    """Test task models reject invalid payloads, chronology, and acknowledgements."""
    with pytest.raises(ValidationError):
        model.model_validate(value)


@pytest.mark.asyncio
async def test_task_lifecycle_rejects_invalid_responses() -> None:
    """Test lifecycle operations reject mismatched task IDs and invalid input."""
    session = FakeTaskSession(
        {
            "tasks/get": [{**TASK_BASE, "taskId": "different-task", "resultType": "complete", "status": "working"}],
        }
    )
    client = task_client(session)

    with pytest.raises(ValueError, match="different taskId"):
        await client._get_task_async("task-1")
    with pytest.raises(ValidationError):
        await client._update_task_async("task-1", {"request-1": "invalid"})  # type: ignore[dict-item]


@pytest.mark.asyncio
async def test_task_aware_call_retries_request_state_with_fresh_call() -> None:
    """Test core input requests are dispatched and echoed before returning a task."""
    from mcp.types import InputRequiredResult

    task = create_task()
    input_request = {
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "Approve?",
            "requestedSchema": {"type": "object"},
        },
    }
    session = FakeTaskSession(
        {
            "tools/call": [
                InputRequiredResult.model_validate(
                    {
                        "inputRequests": {"request-1": input_request},
                        "requestState": "opaque-state",
                    }
                ),
                task,
            ]
        }
    )
    client = task_client(session)

    result = await client._call_tool_with_task_once_async("slow-tool", {"value": 1}, timedelta(seconds=2), None, None)

    assert result == task
    assert len(session.tool_calls) == 2
    assert session.tool_calls[0]["request_state"] is None
    assert session.tool_calls[1]["request_state"] == "opaque-state"
    assert session.tool_calls[1]["input_responses"]["request-1"].action == "accept"
    assert len(session.input_requests) == 1


@pytest.mark.asyncio
async def test_task_completion_handles_input_and_returns_nested_tool_result() -> None:
    """Test input-required task state is updated before completion."""
    input_request = {
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "Approve?",
            "requestedSchema": {"type": "object"},
        },
    }
    completed_result = CallToolResult(content=[TextContent(type="text", text="done")], is_error=False)
    session = FakeTaskSession(
        {
            "tasks/get": [
                {
                    **TASK_BASE,
                    "resultType": "complete",
                    "status": "input_required",
                    "inputRequests": {"request-1": input_request},
                },
                {
                    **TASK_BASE,
                    "resultType": "complete",
                    "status": "completed",
                    "result": completed_result.model_dump(by_alias=True, mode="json"),
                },
            ],
            "tasks/update": [{"resultType": "complete"}],
        }
    )
    client = task_client(session)

    result = await client._complete_modern_task(
        create_task("input_required"),
        cancellation_state={},
    )

    assert result == completed_result
    assert len(session.input_requests) == 1
    update = next(request for request in session.requests if request["method"] == "tasks/update")
    assert update["params"]["inputResponses"]["request-1"] == {
        "action": "accept",
        "content": {"approved": True},
    }


@pytest.mark.asyncio
async def test_task_completion_ignores_stale_detailed_state() -> None:
    """Test polling does not regress after observing a newer detailed state."""
    completed_result = CallToolResult(content=[TextContent(type="text", text="done")], is_error=False)
    session = FakeTaskSession(
        {
            "tasks/get": [
                {
                    **TASK_BASE,
                    "resultType": "complete",
                    "status": "working",
                    "lastUpdatedAt": "2026-09-01T00:00:10Z",
                },
                {
                    **TASK_BASE,
                    "resultType": "complete",
                    "status": "completed",
                    "lastUpdatedAt": "2026-09-01T00:00:05Z",
                    "result": completed_result.model_dump(by_alias=True, mode="json"),
                },
                {
                    **TASK_BASE,
                    "resultType": "complete",
                    "status": "completed",
                    "lastUpdatedAt": "2026-09-01T00:00:11Z",
                    "result": completed_result.model_dump(by_alias=True, mode="json"),
                },
            ]
        }
    )
    client = task_client(session)

    result = await client._complete_modern_task(create_task(), cancellation_state={})

    assert result == completed_result
    assert [request["method"] for request in session.requests] == ["tasks/get", "tasks/get", "tasks/get"]


@pytest.mark.parametrize(
    "state,message",
    [
        ({"status": "failed", "error": {"code": -32000, "message": "boom"}}, "boom"),
        ({"status": "cancelled", "statusMessage": "stopped"}, "stopped"),
    ],
)
@pytest.mark.asyncio
async def test_task_completion_maps_terminal_errors(state: dict[str, Any], message: str) -> None:
    """Test failed and cancelled tasks become tool errors."""
    session = FakeTaskSession({"tasks/get": [{**TASK_BASE, "resultType": "complete", **state}]})
    client = task_client(session)

    result = await client._complete_modern_task(create_task(), None)

    assert result.is_error is True
    assert result.content[0].text == message


@pytest.mark.asyncio
async def test_task_timeout_requests_remote_cancellation() -> None:
    """Test an overall task timeout best-effort cancels the server task."""
    task = create_task()
    session = FakeTaskSession(
        {
            "tools/call": [task],
            "tasks/get": [{**TASK_BASE, "resultType": "complete", "status": "working"}],
            "tasks/cancel": [{"resultType": "complete"}],
        }
    )
    client = task_client(session)
    cancellation_state: dict[str, Any] = {}

    result = await client._call_tool_with_task_and_poll_async(
        "slow-tool",
        poll_timeout=timedelta(milliseconds=1),
        cancellation_state=cancellation_state,
    )

    assert result.is_error is True
    assert "timed out" in result.content[0].text
    assert any(request["method"] == "tasks/cancel" for request in session.requests)
