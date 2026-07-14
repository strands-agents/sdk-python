"""Built-in middleware stages and their context/result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import MiddlewareStage

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..types._events import ModelStopReason, TypedEvent
    from ..types.content import Messages, SystemPrompt
    from ..types.tools import ToolChoice, ToolSpec


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
