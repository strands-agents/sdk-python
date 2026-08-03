"""Tool for CRUD over the agent's own tool registry.

Provides :func:`make_tool_registry` (a factory that binds the tool to a set of
already-connected MCP clients from which new tools may be pulled). The tool
exposes four operations to the model:

- ``list``: enumerate all tools currently on the agent's registry, marking the
  ones this tool has registered itself.
- ``create``: register a new tool that is a thin binding to a specific tool
  hosted on one of the pre-approved MCP clients.
- ``update``: re-bind a previously registered tool (e.g. to point at a
  different remote tool, or to change its description). The local name is the
  lookup key and is not modifiable; delete and re-create to rename.
- ``delete``: unregister a tool that this same tool_registry instance
  registered. Developer-registered tools are never removable.

Design decision (see PR body): ``create`` only accepts references to already-
connected MCP servers pre-provided by the developer via ``mcp_clients``.
Accepting a filesystem path (loading arbitrary Python from disk) or inline
source code (executing arbitrary Python or JS) are explicitly out of scope --
both introduce a filesystem/code-execution attack surface that is
disproportionate to the value of runtime tool registration for a general-purpose
agent tool. Consumers that need those flows should build a separate, more
locked-down tool. This one is a thin shim onto MCPClient, so its blast radius
is exactly whatever the developer-approved MCP servers already expose.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import TYPE_CHECKING

from ...tools.decorator import tool
from ...tools.mcp import MCPAgentTool, MCPClient
from ...types.tools import ToolContext
from .types import (
    MAX_DYNAMIC_TOOLS,
    TOOL_NAME_PATTERN,
    TOOL_REGISTRY_DESCRIPTION,
    ListResult,
    MutationResult,
    RegisteredTool,
    ToolRegistryError,
)

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool
    from ...tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _get_owned_names(
    ownership: weakref.WeakKeyDictionary[ToolRegistry, set[str]],
    registry: ToolRegistry,
) -> set[str]:
    owned = ownership.get(registry)
    if owned is None:
        owned = set()
        ownership[registry] = owned
    return owned


def _validate_tool_name(name: str) -> None:
    if not isinstance(name, str) or not TOOL_NAME_PATTERN.match(name):
        raise ToolRegistryError(f"invalid tool name '{name}': must match {TOOL_NAME_PATTERN.pattern}")


def _find_normalized_conflict(registry: ToolRegistry, name: str) -> str | None:
    """Return an existing registry name that collides with ``name``.

    The SDK's normalized-name rule treats ``-`` and ``_`` as equivalent.
    ``ToolRegistry.register_tool`` enforces this rule but
    ``register_dynamic_tool`` does not, so we replicate the check here to keep
    the tool_registry surface consistent with ``register_tool`` and to fail
    with a clear ToolRegistryError rather than a bare ValueError from the SDK.
    """
    normalized = name.replace("-", "_")
    for existing in list(registry.registry.keys()) + list(registry.dynamic_tools.keys()):
        if existing == name:
            continue
        if existing.replace("-", "_") == normalized:
            return existing
    return None


async def _resolve_mcp_tool(
    client: MCPClient,
    remote_name: str,
    local_name: str,
    description_override: str | None,
) -> MCPAgentTool:
    """Look up a specific tool on the MCP client and adapt it to the local name.

    ``list_tools_sync`` is a blocking network call, so it is dispatched to a
    worker thread with :func:`asyncio.to_thread`; without that the whole event
    loop would stall on the round-trip. The suspension point is real, which is
    what makes the reservation and concurrent-delete/update guards reachable in
    production the same way the tests exercise them.

    Raises:
        ToolRegistryError: If ``remote_name`` is not exposed by the MCP server.
    """
    pagination_token: str | None = None
    while True:
        page = await asyncio.to_thread(client.list_tools_sync, pagination_token=pagination_token)
        for mcp_tool in page:
            # ``list_tools_sync`` returns MCPAgentTool objects; the underlying
            # ``mcp_tool.name`` is the remote-side name, unaffected by any
            # client-level prefix. Match on that.
            if mcp_tool.mcp_tool.name == remote_name:
                adapted = MCPAgentTool(
                    mcp_tool.mcp_tool,
                    client,
                    name_override=local_name,
                )
                adapted.mark_dynamic()
                if description_override is not None:
                    # ``MCPAgentTool.tool_spec`` reads from ``mcp_tool.description``
                    # each call; override by copying the underlying pydantic model.
                    adapted.mcp_tool = adapted.mcp_tool.model_copy(update={"description": description_override})
                return adapted
        pagination_token = page.pagination_token
        if pagination_token is None:
            break
    raise ToolRegistryError(f"tool '{remote_name}' not found on MCP server")


def make_tool_registry(
    *,
    mcp_clients: dict[str, MCPClient] | None = None,
    max_dynamic_tools: int = MAX_DYNAMIC_TOOLS,
    name: str = "tool_registry",
    description: str = TOOL_REGISTRY_DESCRIPTION,
) -> DecoratedFunctionTool:
    """Create a runtime tool-registry-management tool bound to a set of MCP sources.

    Args:
        mcp_clients: Mapping from a stable, developer-chosen alias to an
            already-connected :class:`~strands.tools.mcp.MCPClient`. Only tools
            hosted on these clients may be registered dynamically. When ``None``
            or empty, ``create`` and ``update`` always fail; the tool degrades to
            a read-only view.
        max_dynamic_tools: Upper bound on the number of tools this instance may
            add to the agent's registry. Defaults to :data:`.types.MAX_DYNAMIC_TOOLS`.
        name: Tool name. Defaults to ``"tool_registry"``.
        description: Tool description shown to the model.

    Returns:
        A decorated tool that performs the four CRUD operations on the agent's
        tool registry.

    Raises:
        ValueError: If ``max_dynamic_tools`` is less than one.
    """
    clients: dict[str, MCPClient] = dict(mcp_clients or {})

    if max_dynamic_tools < 1:
        raise ValueError("max_dynamic_tools must be at least 1")

    # Per-factory ownership map: keyed on the agent's ToolRegistry, closed over
    # by this factory instance. Two `make_tool_registry` factories on one agent
    # therefore track their own tools independently. `WeakKeyDictionary` lets
    # the entry disappear when the registry (and its agent) is garbage-collected.
    ownership: weakref.WeakKeyDictionary[ToolRegistry, set[str]] = weakref.WeakKeyDictionary()

    @tool(name=name, description=description, context="tool_context")
    async def tool_registry_tool(
        operation: str,
        tool_context: ToolContext,
        tool_name: str | None = None,
        source: str | None = None,
        remote_name: str | None = None,
        description_override: str | None = None,
    ) -> ListResult | MutationResult:
        """Manage the agent's tool registry at runtime (MCP-backed).

        Args:
            operation: One of ``"list"``, ``"create"``, ``"update"``, ``"delete"``.
            tool_context: Injected by the framework. Not user-facing.
            tool_name: Local name for the tool on the agent's registry. Required
                for ``create``, ``update``, and ``delete``. Must match
                ``^[a-zA-Z_][a-zA-Z0-9_]{0,63}$``.
            source: For ``create``/``update``, the alias of an MCP client
                previously registered with this tool via ``mcp_clients``. The
                set of valid aliases is fixed at tool construction time.
            remote_name: For ``create``/``update``, the name of the tool on the
                MCP server pointed at by ``source``. Defaults to ``tool_name``
                when omitted.
            description_override: Optional description to expose to the model in
                place of the MCP server's advertised description. Useful when the
                remote description is empty or too generic.

        Raises:
            ToolRegistryError: For any validation failure surfaced by the tool
                itself: unknown operation, missing or invalid ``tool_name``,
                unknown ``source``, unknown remote tool, dynamic-tool cap
                reached, duplicate registration, self-mutation attempt,
                deletion or update of a tool this instance did not register,
                or a concurrent-delete race that cancels an in-flight
                ``create``/``update``.
        """
        agent_registry: ToolRegistry = tool_context.agent.tool_registry
        owned = _get_owned_names(ownership, agent_registry)

        if operation == "list":
            tools_out: list[RegisteredTool] = []
            for spec in agent_registry.get_all_tool_specs():
                entry: RegisteredTool = {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "registered_by_tool_registry": spec["name"] in owned,
                }
                # Omit input_schema only when the underlying tool doesn't
                # declare the key at all, matching the TypeScript SDK's
                # optional-field convention. An explicit empty dict is a
                # valid schema and must be preserved.
                if "inputSchema" in spec:
                    entry["input_schema"] = spec["inputSchema"]
                tools_out.append(entry)
            return ListResult(
                tools=tools_out,
                dynamic_count=len(owned),
                dynamic_limit=max_dynamic_tools,
            )

        if operation not in ("create", "update", "delete"):
            raise ToolRegistryError(
                f"invalid operation '{operation}': must be one of 'list', 'create', 'update', 'delete'"
            )

        if tool_name is None:
            raise ToolRegistryError(f"'tool_name' is required for operation '{operation}'")
        _validate_tool_name(tool_name)

        # Never allow this tool to remove or replace itself.
        if tool_name == name:
            raise ToolRegistryError(f"cannot {operation} the tool_registry tool ('{tool_name}') itself")

        if operation == "delete":
            if tool_name not in owned:
                raise ToolRegistryError(
                    f"tool '{tool_name}' was not registered via tool_registry; "
                    "developer-registered tools cannot be removed"
                )
            # Owned ⇒ present in dynamic_tools; belt-and-braces guard against
            # external tampering with the registry between calls.
            agent_registry.dynamic_tools.pop(tool_name, None)
            agent_registry.registry.pop(tool_name, None)
            owned.discard(tool_name)
            return MutationResult(operation="delete", name=tool_name, dynamic_count=len(owned))

        # create / update below share the source-resolution logic.
        if not source:
            raise ToolRegistryError(f"'source' is required for operation '{operation}'")
        if source not in clients:
            allowed = ", ".join(sorted(clients)) or "<none>"
            raise ToolRegistryError(f"unknown source '{source}': allowed sources are: {allowed}")
        effective_remote = remote_name or tool_name

        if operation == "create":
            if (
                tool_name in agent_registry.registry
                or tool_name in agent_registry.dynamic_tools
                or tool_name in owned
            ):
                raise ToolRegistryError(f"a tool named '{tool_name}' is already registered")
            conflict = _find_normalized_conflict(agent_registry, tool_name)
            if conflict is not None:
                raise ToolRegistryError(
                    f"a tool named '{conflict}' is already registered; "
                    f"tool names cannot differ only by '-' vs '_'"
                )
            if len(owned) >= max_dynamic_tools:
                raise ToolRegistryError(
                    f"dynamic tool cap reached ({max_dynamic_tools}); "
                    "delete an existing dynamically-registered tool first"
                )
            # Reserve the slot in the owned set before the MCP lookup so
            # concurrent `create` calls in the same turn can't all pass the
            # cap check before any of them registers. On failure, release.
            owned.add(tool_name)
            try:
                new_tool = await _resolve_mcp_tool(
                    clients[source], effective_remote, tool_name, description_override
                )
                # A concurrent `delete` could have observed our reservation,
                # treated it as if the tool were already registered, and cleared
                # it while we were awaiting. If so, abort rather than write a
                # tool this instance can no longer track.
                if tool_name not in owned:
                    raise ToolRegistryError(
                        f"create of '{tool_name}' was cancelled by a concurrent delete "
                        "before the tool could be registered"
                    )
                try:
                    agent_registry.register_dynamic_tool(new_tool)
                except ValueError as err:
                    # A concurrent registration (e.g. another tool_registry
                    # instance) can still land a duplicate between our checks
                    # and the write. Surface as ToolRegistryError so callers
                    # get a single exception type from the tool.
                    raise ToolRegistryError(str(err)) from err
            except Exception:
                owned.discard(tool_name)
                raise
            return MutationResult(operation="create", name=tool_name, dynamic_count=len(owned))

        # operation == "update"
        if tool_name not in owned:
            raise ToolRegistryError(
                f"tool '{tool_name}' was not registered via tool_registry; developer-registered tools cannot be updated"
            )
        new_tool = await _resolve_mcp_tool(clients[source], effective_remote, tool_name, description_override)
        # A concurrent `delete` could have run during the await above and
        # cleared this instance's ownership of the tool. If so, abort rather
        # than resurrect a tool the model believed deleted (mirrors the
        # create-during-delete guard).
        if tool_name not in owned:
            raise ToolRegistryError(
                f"update of '{tool_name}' was cancelled by a concurrent delete "
                "before the tool could be re-bound"
            )
        # Replace under the same name, keeping ownership.
        agent_registry.dynamic_tools[tool_name] = new_tool
        if tool_name in agent_registry.registry:
            agent_registry.registry[tool_name] = new_tool
        return MutationResult(operation="update", name=tool_name, dynamic_count=len(owned))

    return tool_registry_tool
