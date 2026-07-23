"""Built-in middleware stages and their context/result types."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..interrupt import Interrupt, InterruptException
from .types import MiddlewareStage

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..experimental.bidi import BidiAgent
    from ..interrupt import _InterruptState
    from ..types._events import ModelStopReason, ToolResultEvent, TypedEvent
    from ..types.content import Messages, SystemPrompt
    from ..types.tools import AgentTool, ToolChoice, ToolSpec, ToolUse


@dataclass
class InvokeModelContext:
    """Context passed to InvokeModelStage middleware.

    All collection fields (messages, system_prompt, tool_specs, tool_choice) are
    defensive copies — middleware cannot accidentally mutate agent state.
    invocation_state is shared by reference (hooks and tools write to it during streaming).
    """

    agent: Agent
    messages: Messages
    system_prompt: SystemPrompt
    tool_specs: list[ToolSpec]
    tool_choice: ToolChoice | None
    invocation_state: dict[str, Any]
    projected_input_tokens: int | None = None


InvokeModelStage: MiddlewareStage[InvokeModelContext, ModelStopReason, TypedEvent] = MiddlewareStage(name="invokeModel")
"""Built-in stage wrapping core model invocation.

Middleware registered for this stage can rate-limit, cache, or transform model inputs/outputs.
"""


@dataclass
class MiddlewareInterruptResult:
    """Value returned by ``ExecuteToolContext.interrupt()`` when the agent resumes.

    Wrapping the response (rather than returning it bare) mirrors the TypeScript SDK and
    leaves room to add fields later without breaking callers.

    Attributes:
        response: The human-provided response the agent resumed with.
    """

    response: Any


@dataclass
class ExecuteToolContext:
    """Context passed to ExecuteToolStage middleware.

    ``tool_use`` is a shallow copy of the executor's dict, so reassigning its top-level
    keys (e.g. ``name``, ``toolUseId``) cannot corrupt executor state. Its ``input`` value
    is shared by reference — it can hold arbitrary, non-copyable objects (e.g. the agent
    injected on direct tool calls), so a deep copy is not possible; mutating ``input`` in
    place still leaks. ``invocation_state`` is likewise shared by reference (matching how
    hooks receive it). Middleware that needs a fully isolated ``tool_use`` should build a
    new one and pass a modified context via ``dataclasses.replace()``.

    Supports middleware-initiated interrupts via ``interrupt()`` for human-in-the-loop
    approval flows.
    """

    agent: Agent | BidiAgent
    tool: AgentTool | None
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    # Interrupt state is threaded in from the agent so interrupt() can register/resolve
    # interrupts. Required (the executor is the sole constructor and always supplies it);
    # excluded from repr to avoid dumping unrelated interrupt bookkeeping.
    _interrupt_state: _InterruptState = field(repr=False)

    def interrupt(self, name: str, *, reason: Any = None, response: Any = None) -> MiddlewareInterruptResult:
        """Request a human-in-the-loop interrupt.

        On first execution (no prior response) this raises ``InterruptException`` to halt
        the agent. After the user resumes with a response, the second call returns that
        response. Providing ``response`` preemptively skips the interrupt entirely.

        This method is read-only with respect to interrupt state: it inspects prior
        responses but does not register the interrupt itself. The tool executor registers
        it (in its ``InterruptException`` handler) as the single source of truth, matching
        the TypeScript SDK where middleware interrupts never write to interrupt state.

        Args:
            name: User-defined name for the interrupt. The interrupt id is scoped to the tool
                call (``v1:middleware_execute_tool:<toolUseId>:<uuid5(name)>``) but not to the
                individual middleware, so the name must be unique across all middleware that
                interrupt this tool call — two middleware using the same name on the same tool
                call collide and share one response. (This matches the hook/tool interrupt
                contract, which is likewise unique per tool call, not per callback.)
            reason: Optional reason for the interrupt (surfaced to the user).
            response: Optional preemptive response — when set, no interrupt is raised.

        Returns:
            The user's response wrapped in a ``MiddlewareInterruptResult``.

        Raises:
            InterruptException: When no response is available yet and none was provided.
        """
        interrupt_id = self._interrupt_id(name)

        existing = self._interrupt_state.interrupts.get(interrupt_id)
        if existing is not None and existing.response is not None:
            return MiddlewareInterruptResult(response=existing.response)

        if response is not None:
            return MiddlewareInterruptResult(response=response)

        raise InterruptException(Interrupt(id=interrupt_id, name=name, reason=reason))

    def _interrupt_id(self, name: str) -> str:
        """Derive the interrupt id for ``name``, namespaced by the tool call.

        Follows the SDK's ``v1:`` interrupt-id scheme (see ``types/interrupt.py``), hashing
        the user-provided name so ids stay stable across resumes for the same tool call.
        """
        return f"v1:middleware_execute_tool:{self.tool_use['toolUseId']}:{uuid.uuid5(uuid.NAMESPACE_OID, name)}"


ExecuteToolStage: MiddlewareStage[ExecuteToolContext, ToolResultEvent, TypedEvent] = MiddlewareStage(name="executeTool")
"""Built-in stage wrapping individual tool execution.

Middleware registered for this stage can add telemetry, validate inputs, mock responses,
or gate execution behind a human-in-the-loop interrupt. The result event is the
``ToolResultEvent`` produced by the tool (matching the "last event is the result"
convention used across the SDK).
"""
