"""Tests for the tool_registry tool.

The tool provides CRUD access to the agent's own tool registry. Only tools
hosted on a pre-approved MCP client may be registered; developer-registered
tools are never removable or updatable via the tool. These tests exercise the
tool as a plain async callable (bypassing the streaming/error-wrapping path);
validation failures surface as raised ``ToolRegistryError``.
"""

from __future__ import annotations

import asyncio
import unittest.mock
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import Tool as MCPTool

from strands.tools.decorator import tool as tool_decorator
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool
from strands.tools.registry import ToolRegistry
from strands.types import PaginatedList
from strands.types.tools import ToolContext
from strands.vended_tools.tool_registry import (
    ToolRegistryError,
    make_tool_registry,
)
from strands.vended_tools.tool_registry import tool_registry as trmod
from strands.vended_tools.tool_registry.types import MAX_DYNAMIC_TOOLS


@dataclass
class _FakeMCPClient:
    """Minimal stand-in for ``MCPClient``: pretends to host a fixed set of tools.

    Only the surface the tool_registry tool actually uses is exposed:
    ``list_tools_sync`` yielding ``MCPAgentTool`` instances (via the real class
    from the SDK), with the pagination shape the tool expects.
    """

    tools: list[MCPTool]

    def list_tools_sync(self, pagination_token: str | None = None, **kwargs: Any) -> PaginatedList[Any]:
        adapted = [MCPAgentTool(t, self) for t in self.tools]  # type: ignore[arg-type]
        return PaginatedList(adapted, token=None)


def _mcp_tool(name: str, description: str = "a fake tool") -> MCPTool:
    return MCPTool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )


def _tool_context(registry: ToolRegistry) -> ToolContext:
    agent = SimpleNamespace(tool_registry=registry)
    return ToolContext(
        tool_use={"name": "tool_registry", "toolUseId": "id", "input": {}},
        agent=agent,
        invocation_state={},
    )


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()

    # Two developer-registered tools that must remain untouchable.
    @tool_decorator(name="dev_echo", description="developer-registered echo")
    def dev_echo(text: str) -> str:
        return text

    @tool_decorator(name="dev_ping", description="developer-registered ping")
    def dev_ping() -> str:
        return "pong"

    r.register_tool(dev_echo)
    r.register_tool(dev_ping)
    return r


@pytest.fixture
def mcp_client() -> _FakeMCPClient:
    return _FakeMCPClient(
        tools=[
            _mcp_tool("remote_alpha"),
            _mcp_tool("remote_beta"),
            _mcp_tool("remote_gamma"),
        ]
    )


@pytest.fixture
def registry_tool(mcp_client: _FakeMCPClient) -> Any:
    # Cast MCPClient explicitly via typing.Any to sidestep the isinstance check
    # in the tool factory; the factory only requires the ``list_tools_sync`` API.
    return make_tool_registry(mcp_clients={"weather": mcp_client})


class TestListOperation:
    @pytest.mark.asyncio
    async def test_lists_developer_registered_tools_as_not_owned(self, registry_tool, registry: ToolRegistry) -> None:
        tru_result = await registry_tool(operation="list", tool_context=_tool_context(registry))
        exp_result = {
            "tools": [
                {
                    "name": "dev_echo",
                    "description": "developer-registered echo",
                    "input_schema": {
                        "json": {
                            "type": "object",
                            "properties": {"text": {"description": "Parameter text", "type": "string"}},
                            "required": ["text"],
                        }
                    },
                    "registered_by_tool_registry": False,
                },
                {
                    "name": "dev_ping",
                    "description": "developer-registered ping",
                    "input_schema": {"json": {"type": "object", "properties": {}, "required": []}},
                    "registered_by_tool_registry": False,
                },
            ],
            "dynamic_count": 0,
            "dynamic_limit": MAX_DYNAMIC_TOOLS,
        }
        assert tru_result == exp_result


class TestCreateOperation:
    @pytest.mark.asyncio
    async def test_registers_mcp_tool_and_marks_owned(self, registry_tool, registry: ToolRegistry) -> None:
        tru_result = await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=_tool_context(registry),
        )
        exp_result = {"operation": "create", "name": "alpha", "dynamic_count": 1}
        assert tru_result == exp_result
        assert "alpha" in registry.dynamic_tools

    @pytest.mark.asyncio
    async def test_uses_tool_name_as_default_remote_name(self, registry_tool, registry: ToolRegistry) -> None:
        tru_result = await registry_tool(
            operation="create",
            tool_name="remote_beta",
            source="weather",
            tool_context=_tool_context(registry),
        )
        assert tru_result == {"operation": "create", "name": "remote_beta", "dynamic_count": 1}
        assert "remote_beta" in registry.dynamic_tools

    @pytest.mark.asyncio
    async def test_description_override_shadowed_on_spec(self, registry_tool, registry: ToolRegistry) -> None:
        await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            description_override="custom text",
            tool_context=_tool_context(registry),
        )
        spec = registry.dynamic_tools["alpha"].tool_spec
        assert spec["description"] == "custom text"

    @pytest.mark.asyncio
    async def test_rejects_empty_description_override(self, registry_tool, registry: ToolRegistry) -> None:
        for bad_desc in ["", "   "]:
            with pytest.raises(ToolRegistryError, match="non-empty"):
                await registry_tool(
                    operation="create",
                    tool_name="alpha",
                    source="weather",
                    remote_name="remote_alpha",
                    description_override=bad_desc,
                    tool_context=_tool_context(registry),
                )

    @pytest.mark.asyncio
    async def test_rejects_unknown_source(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="unknown source"):
            await registry_tool(
                operation="create",
                tool_name="alpha",
                source="not_a_real_client",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_remote_tool(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="not found on MCP server"):
            await registry_tool(
                operation="create",
                tool_name="alpha",
                source="weather",
                remote_name="does_not_exist",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    async def test_rejects_collision_with_developer_tool(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="already registered"):
            await registry_tool(
                operation="create",
                tool_name="dev_echo",
                source="weather",
                remote_name="remote_alpha",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    async def test_cannot_register_over_itself(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="itself"):
            await registry_tool(
                operation="create",
                tool_name="tool_registry",
                source="weather",
                remote_name="remote_alpha",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad",
        ["", "1_starts_with_digit", "has-dash", "has space", "toolname!", "a" * 65, "weather\n"],
    )
    async def test_rejects_invalid_tool_name(self, registry_tool, registry: ToolRegistry, bad: str) -> None:
        with pytest.raises(ToolRegistryError, match="invalid tool name"):
            await registry_tool(
                operation="create",
                tool_name=bad,
                source="weather",
                remote_name="remote_alpha",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    async def test_rejects_normalized_name_conflict(self, registry_tool) -> None:
        """A tool name that only differs by '-' vs '_' from an existing entry is rejected.

        The tool replicates the check to fail with a clear error.
        """
        reg = ToolRegistry()

        # Insert a developer tool whose name contains a dash. The tool_registry
        # tool would never allow the model to register such a name (dashes are
        # rejected by TOOL_NAME_PATTERN), but developer tools can carry them.
        @tool_decorator(name="my-tool", description="dash-named developer tool")
        def my_tool() -> str:
            return "ok"

        reg.register_tool(my_tool)

        # Now the model tries to add `my_tool` — which normalizes to the same
        # canonical name — and should be rejected before any MCP work happens.
        with pytest.raises(ToolRegistryError, match="differ only by '-' vs '_'"):
            await registry_tool(
                operation="create",
                tool_name="my_tool",
                source="weather",
                remote_name="remote_alpha",
                tool_context=_tool_context(reg),
            )

    @pytest.mark.asyncio
    async def test_wraps_register_dynamic_tool_value_error(
        self,
        mcp_client: _FakeMCPClient,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raw `ValueError` from `register_dynamic_tool` is surfaced as `ToolRegistryError`.

        Another `tool_registry` factory could race a duplicate registration
        between our checks and the write; callers should still see a single
        exception type from the tool.
        """
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
        )

        def raise_duplicate(_tool: Any) -> None:
            raise ValueError("Tool 'alpha' already exists")

        monkeypatch.setattr(registry, "register_dynamic_tool", raise_duplicate)
        with pytest.raises(ToolRegistryError, match="already exists"):
            await registry_tool(
                operation="create",
                tool_name="alpha",
                source="weather",
                remote_name="remote_alpha",
                tool_context=_tool_context(registry),
            )
        # Reservation released on failure.
        tru_list = await registry_tool(operation="list", tool_context=_tool_context(registry))
        assert tru_list["dynamic_count"] == 0

    @pytest.mark.asyncio
    async def test_accepts_maximum_length_tool_name(self, registry_tool, registry: ToolRegistry) -> None:
        # 64 chars: one leading letter + 63 alphanumeric/underscore characters.
        max_name = "a" * 64
        tru_result = await registry_tool(
            operation="create",
            tool_name=max_name,
            source="weather",
            remote_name="remote_alpha",
            tool_context=_tool_context(registry),
        )
        assert tru_result == {"operation": "create", "name": max_name, "dynamic_count": 1}

    @pytest.mark.asyncio
    async def test_enforces_dynamic_tool_cap(self, mcp_client: _FakeMCPClient, registry: ToolRegistry) -> None:
        # Cap at 2 so we can exhaust it quickly.
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
            max_dynamic_tools=2,
        )
        ctx = _tool_context(registry)
        for local_name, remote in [("alpha", "remote_alpha"), ("beta", "remote_beta")]:
            await registry_tool(
                operation="create",
                tool_name=local_name,
                source="weather",
                remote_name=remote,
                tool_context=ctx,
            )
        with pytest.raises(ToolRegistryError, match="dynamic tool cap reached"):
            await registry_tool(
                operation="create",
                tool_name="gamma",
                source="weather",
                remote_name="remote_gamma",
                tool_context=ctx,
            )

    @pytest.mark.asyncio
    async def test_enforces_dynamic_tool_cap_under_concurrent_creates(
        self,
        mcp_client: _FakeMCPClient,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cap = 2
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
            max_dynamic_tools=cap,
        )
        ctx = _tool_context(registry)

        # Replace _resolve_mcp_tool with a version that yields to let concurrent creates overlap.
        real_resolve = trmod._resolve_mcp_tool

        async def yielding_resolve(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)  # force a real suspend point
            return await real_resolve(*args, **kwargs)

        monkeypatch.setattr(trmod, "_resolve_mcp_tool", yielding_resolve)

        # Fire 2x cap concurrent creates against a shared context. Under a
        # check-then-await-then-write ordering all N would race past the cap.
        results = await asyncio.gather(
            *(
                registry_tool(
                    operation="create",
                    tool_name=f"t{i}",
                    source="weather",
                    remote_name="remote_alpha",
                    tool_context=ctx,
                )
                for i in range(cap * 2)
            ),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, ToolRegistryError)]
        assert len(successes) == cap
        assert len(failures) == cap

    @pytest.mark.asyncio
    async def test_cancellation_releases_reservation(
        self,
        mcp_client: _FakeMCPClient,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A task cancelled during the MCP round-trip must not leak its cap slot or name reservation.

        Without the fix (except Exception instead of except BaseException), the
        CancelledError bypasses the cleanup branch, leaving the name in `owned`
        permanently — subsequent creates hit "already registered" and the cap
        error says to delete a tool that doesn't exist.
        """
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
        )
        ctx = _tool_context(registry)

        async def hanging_resolve(*args: Any, **kwargs: Any) -> Any:
            # Park indefinitely so the task can be cancelled mid-await.
            await asyncio.Event().wait()
            raise AssertionError("unreachable")  # pragma: no cover

        monkeypatch.setattr(trmod, "_resolve_mcp_tool", hanging_resolve)

        create_task = asyncio.create_task(
            registry_tool(
                operation="create",
                tool_name="alpha",
                source="weather",
                remote_name="remote_alpha",
                tool_context=ctx,
            )
        )
        await asyncio.sleep(0)  # let the task reach the await
        create_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create_task

        # Reservation must be released — cap and name are available again.
        monkeypatch.undo()
        tru_result = await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=ctx,
        )
        assert tru_result == {"operation": "create", "name": "alpha", "dynamic_count": 1}

    @pytest.mark.asyncio
    async def test_concurrent_create_and_update_same_name(
        self,
        mcp_client: _FakeMCPClient,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concurrent update on a name that is only pending (in-flight create) must be rejected."""
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
        )
        ctx = _tool_context(registry)

        real_resolve = trmod._resolve_mcp_tool
        gate = asyncio.Event()

        async def gated_resolve(*args: Any, **kwargs: Any) -> Any:
            await gate.wait()
            return await real_resolve(*args, **kwargs)

        monkeypatch.setattr(trmod, "_resolve_mcp_tool", gated_resolve)

        # Start create and let it park on the gate (name is now in pending, not owned).
        create_task = asyncio.create_task(
            registry_tool(
                operation="create",
                tool_name="alpha",
                source="weather",
                remote_name="remote_alpha",
                tool_context=ctx,
            )
        )
        await asyncio.sleep(0)

        # Concurrent update on the same name must be rejected — alpha is only pending.
        with pytest.raises(ToolRegistryError, match="developer-registered"):
            await registry_tool(
                operation="update",
                tool_name="alpha",
                source="weather",
                remote_name="remote_beta",
                tool_context=ctx,
            )

        # Release create; it completes normally and alpha is now owned.
        gate.set()
        tru_result = await create_task
        assert tru_result == {"operation": "create", "name": "alpha", "dynamic_count": 1}
        assert "alpha" in registry.dynamic_tools

    @pytest.mark.asyncio
    async def test_does_not_orphan_reservation_on_concurrent_delete(
        self,
        mcp_client: _FakeMCPClient,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `delete` landing between `create`'s reservation and write must not orphan."""
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
        )
        ctx = _tool_context(registry)

        real_resolve = trmod._resolve_mcp_tool
        gate = asyncio.Event()

        async def gated_resolve(*args: Any, **kwargs: Any) -> Any:
            # Suspend until the test releases us so a concurrent `delete` can
            # slip between the synchronous reservation and the post-await
            # registry write.
            await gate.wait()
            return await real_resolve(*args, **kwargs)

        monkeypatch.setattr(trmod, "_resolve_mcp_tool", gated_resolve)

        # Task B: start `create` and let it park on the gate.
        create_task = asyncio.create_task(
            registry_tool(
                operation="create",
                tool_name="alpha",
                source="weather",
                remote_name="remote_alpha",
                tool_context=ctx,
            )
        )
        for _ in range(3):
            await asyncio.sleep(0)

        # Task A: delete the reserved-but-not-yet-registered tool.
        delete_result = await registry_tool(
            operation="delete",
            tool_name="alpha",
            tool_context=ctx,
        )
        assert delete_result == {"operation": "delete", "name": "alpha", "dynamic_count": 0}

        # Release create; it must abort rather than resurrect an orphan.
        gate.set()
        with pytest.raises(ToolRegistryError, match="cancelled by a concurrent delete"):
            await create_task

        # No orphan in the SDK registry; ownership set is empty.
        assert "alpha" not in registry.dynamic_tools
        assert "alpha" not in registry.registry
        assert await registry_tool(operation="list", tool_context=ctx) == {
            "tools": unittest.mock.ANY,
            "dynamic_count": 0,
            "dynamic_limit": MAX_DYNAMIC_TOOLS,
        }


class TestUpdateOperation:
    @pytest.mark.asyncio
    async def test_updates_own_tool(self, registry_tool, registry: ToolRegistry) -> None:
        ctx = _tool_context(registry)
        await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=ctx,
        )
        result = await registry_tool(
            operation="update",
            tool_name="alpha",
            source="weather",
            remote_name="remote_beta",
            tool_context=ctx,
        )
        assert result == {"operation": "update", "name": "alpha", "dynamic_count": 1}
        # The adapter points at remote_beta now.
        assert registry.dynamic_tools["alpha"].mcp_tool.name == "remote_beta"

    @pytest.mark.asyncio
    async def test_update_applies_description_override(self, registry_tool, registry: ToolRegistry) -> None:
        ctx = _tool_context(registry)
        await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=ctx,
        )
        await registry_tool(
            operation="update",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            description_override="overridden on update",
            tool_context=ctx,
        )
        assert registry.dynamic_tools["alpha"].tool_spec["description"] == "overridden on update"

    @pytest.mark.asyncio
    async def test_cannot_update_developer_tool(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="developer-registered"):
            await registry_tool(
                operation="update",
                tool_name="dev_echo",
                source="weather",
                remote_name="remote_alpha",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    async def test_does_not_resurrect_on_concurrent_delete(
        self,
        mcp_client: _FakeMCPClient,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `delete` landing during `update`'s MCP lookup must not be undone.

        Mirrors the create-during-delete guard: if `update` re-checks ownership
        only before the await, a delete landing during the await could resurrect
        a tool the model believed deleted, leaving an orphan this instance can
        no longer track.
        """
        registry_tool = make_tool_registry(
            mcp_clients={"weather": mcp_client},
        )
        ctx = _tool_context(registry)

        # First register so `update` has something to target.
        await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=ctx,
        )

        real_resolve = trmod._resolve_mcp_tool
        gate = asyncio.Event()

        async def gated_resolve(*args: Any, **kwargs: Any) -> Any:
            await gate.wait()
            return await real_resolve(*args, **kwargs)

        monkeypatch.setattr(trmod, "_resolve_mcp_tool", gated_resolve)

        # Task B: start `update` and let it park on the gate after the
        # pre-await ownership check.
        update_task = asyncio.create_task(
            registry_tool(
                operation="update",
                tool_name="alpha",
                source="weather",
                remote_name="remote_beta",
                tool_context=ctx,
            )
        )
        for _ in range(3):
            await asyncio.sleep(0)

        # Task A: delete the tool while the update is still awaiting.
        delete_result = await registry_tool(
            operation="delete",
            tool_name="alpha",
            tool_context=ctx,
        )
        assert delete_result == {"operation": "delete", "name": "alpha", "dynamic_count": 0}

        # Release the update; it must abort rather than resurrect the tool.
        gate.set()
        with pytest.raises(ToolRegistryError, match="cancelled by a concurrent delete"):
            await update_task

        # Deletion stuck; no orphan.
        assert "alpha" not in registry.dynamic_tools
        assert "alpha" not in registry.registry
        assert await registry_tool(operation="list", tool_context=ctx) == {
            "tools": unittest.mock.ANY,
            "dynamic_count": 0,
            "dynamic_limit": MAX_DYNAMIC_TOOLS,
        }


class TestDeleteOperation:
    @pytest.mark.asyncio
    async def test_deletes_own_tool(self, registry_tool, registry: ToolRegistry) -> None:
        ctx = _tool_context(registry)
        await registry_tool(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=ctx,
        )
        result = await registry_tool(operation="delete", tool_name="alpha", tool_context=ctx)
        assert result == {"operation": "delete", "name": "alpha", "dynamic_count": 0}
        assert "alpha" not in registry.dynamic_tools

    @pytest.mark.asyncio
    async def test_rejects_deleting_developer_tool(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="developer-registered"):
            await registry_tool(
                operation="delete",
                tool_name="dev_echo",
                tool_context=_tool_context(registry),
            )
        # Untouched.
        assert "dev_echo" in registry.registry

    @pytest.mark.asyncio
    async def test_does_not_leak_ownership_between_registries(self, mcp_client: _FakeMCPClient) -> None:
        """Two agents sharing a factory bind their own ownership sets."""
        registry_tool_shared = make_tool_registry(mcp_clients={"weather": mcp_client})
        reg_a = ToolRegistry()
        reg_b = ToolRegistry()

        await registry_tool_shared(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=_tool_context(reg_a),
        )

        # Registry B never registered "alpha", so deletion must be rejected there
        # even though registry A has it. Ownership is per-registry, not global.
        with pytest.raises(ToolRegistryError, match="developer-registered"):
            await registry_tool_shared(
                operation="delete",
                tool_name="alpha",
                tool_context=_tool_context(reg_b),
            )

    @pytest.mark.asyncio
    async def test_two_factories_on_one_agent_do_not_collide(self, mcp_client: _FakeMCPClient) -> None:
        """Two separate factories on one agent track ownership independently."""
        factory_a = make_tool_registry(mcp_clients={"weather": mcp_client})
        factory_b = make_tool_registry(
            mcp_clients={"weather": mcp_client},
            name="tool_registry_b",
        )
        reg = ToolRegistry()
        ctx = _tool_context(reg)

        await factory_a(
            operation="create",
            tool_name="alpha",
            source="weather",
            remote_name="remote_alpha",
            tool_context=ctx,
        )
        # Factory B never registered "alpha"; it must refuse to delete it,
        # matching the TS `WeakMap<LocalAgent, Set<string>>` semantics.
        with pytest.raises(ToolRegistryError, match="developer-registered"):
            await factory_b(
                operation="delete",
                tool_name="alpha",
                tool_context=ctx,
            )
        # Factory A still owns it.
        assert "alpha" in reg.dynamic_tools


class TestOperationDispatch:
    @pytest.mark.asyncio
    async def test_rejects_unknown_operation(self, registry_tool, registry: ToolRegistry) -> None:
        with pytest.raises(ToolRegistryError, match="invalid operation"):
            await registry_tool(
                operation="wipe_registry",
                tool_context=_tool_context(registry),
            )

    @pytest.mark.asyncio
    async def test_read_only_when_no_mcp_clients_configured(self, registry: ToolRegistry) -> None:
        registry_tool = make_tool_registry()  # empty
        # list still works
        tru_list = await registry_tool(operation="list", tool_context=_tool_context(registry))
        assert tru_list == {"tools": unittest.mock.ANY, "dynamic_count": 0, "dynamic_limit": MAX_DYNAMIC_TOOLS}
        # create fails
        with pytest.raises(ToolRegistryError, match="unknown source"):
            await registry_tool(
                operation="create",
                tool_name="alpha",
                source="anything",
                tool_context=_tool_context(registry),
            )


class TestFactoryValidation:
    def test_rejects_zero_max_dynamic_tools(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            make_tool_registry(max_dynamic_tools=0)

    def test_uses_custom_name_and_description(self) -> None:
        t = make_tool_registry(name="registry_ctl", description="custom desc")
        assert t.tool_name == "registry_ctl"
        assert t.tool_spec["description"] == "custom desc"
