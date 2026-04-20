"""Checkpoint-based agent execution for durable workflows.

⚠️ Experimental — APIs may change in future releases.

Design:
- Three checkpoint positions: after_model, after_tool, after_tools
- Per-tool granularity: each tool executes in its own step
- Model-driven: tool errors become tool results the model sees and decides on
- Provider-agnostic: any durability provider can wrap invoke_with_checkpoint
- Returns Checkpoint (keep going) or AgentResult (done) — no wrapper type

V1 simplifications (documented for future extension):
- _consume_model_stream: bypasses hooks and ModelRetryStrategy.
- _execute_single_tool: bypasses tool executor, hooks, tracing, and interrupts.
- No max_cycles guard — the caller/orchestrator is responsible for stopping.

See types.py module docstring for the full list of known limitations.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from ...agent.agent_result import AgentResult
from ...event_loop.streaming import stream_messages
from ...types._events import ToolResultEvent
from ...types._snapshot import Snapshot
from ...types.content import Message
from ...types.streaming import Metrics, StopReason, Usage
from ...types.tools import ToolResult, ToolUse
from .types import Checkpoint

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...agent.agent import Agent


async def invoke_with_checkpoint(
    agent: Agent,
    prompt: str | None = None,
    *,
    checkpoint: Checkpoint | None = None,
) -> Checkpoint | AgentResult:
    """Run one step of the agent loop.

    Returns a Checkpoint if there's more work to do, or an AgentResult if done.

    Each call does exactly one unit of I/O:
    - A model call, OR
    - A single tool execution

    This per-tool granularity ensures that on crash recovery, only the
    incomplete tool re-executes. Completed tools are cached by the
    durability provider (e.g. Temporal Activity results in Event History).

    Model and tool errors propagate as exceptions; the caller maps them
    to their orchestrator's retry policy.

    Usage::

        result = await invoke_with_checkpoint(agent, "Hello")
        while isinstance(result, Checkpoint):
            result = await invoke_with_checkpoint(agent, checkpoint=result)
        # result is AgentResult
        print(result)

    Args:
        agent: The agent to execute.
        prompt: User prompt for the first call.
        checkpoint: Checkpoint from a previous call to continue from.

    Returns:
        Checkpoint if the loop should continue, AgentResult if done.

    Raises:
        ValueError: If both prompt and checkpoint are provided, or if the
            checkpoint has an invalid position/stop_reason combination.
        Exception: Model or infrastructure errors propagate directly.
    """
    if prompt is not None and checkpoint is not None:
        raise ValueError(
            "Cannot provide both 'prompt' and 'checkpoint'. "
            "Use 'prompt' for a fresh invocation or 'checkpoint' to resume."
        )

    # Fresh invocation
    if checkpoint is None:
        logger.debug("has_prompt=<%s> | starting fresh checkpoint invocation", prompt is not None)
        agent.event_loop_metrics.reset_usage_metrics()
        if prompt is not None:
            messages = await agent._convert_prompt_to_messages(prompt)
            await agent._append_messages(*messages)
        return await _run_model_and_checkpoint(agent, cycle_index=0)

    # Resuming — restore agent state, then dispatch on position
    logger.debug(
        "position=<%s>, cycle_index=<%s>, tool_index=<%s> | resuming from checkpoint",
        checkpoint.position,
        checkpoint.cycle_index,
        checkpoint.tool_index,
    )
    if checkpoint.snapshot:
        agent.load_snapshot(Snapshot.from_dict(checkpoint.snapshot))

    if checkpoint.position in ("after_model", "after_tool") and checkpoint.stop_reason == "tool_use":
        return await _resume_tool_execution(agent, checkpoint)

    if checkpoint.position == "after_tools":
        return await _run_model_and_checkpoint(agent, checkpoint.cycle_index + 1)

    if checkpoint.position in ("after_model", "after_tool"):
        raise ValueError(
            f"Checkpoint at position={checkpoint.position!r} requires stop_reason='tool_use', "
            f"got {checkpoint.stop_reason!r}"
        )

    raise ValueError(f"Unknown checkpoint position: {checkpoint.position!r}")


async def _run_model_and_checkpoint(agent: Agent, cycle_index: int) -> Checkpoint | AgentResult:
    """Run a model call and return a Checkpoint or AgentResult.

    V1: Calls model directly via stream_messages. Does not fire
    BeforeModelCallEvent/AfterModelCallEvent or use ModelRetryStrategy.
    """
    # On crash recovery, a fresh agent may have empty agent_invocations.
    # start_cycle requires at least one invocation entry.
    if not agent.event_loop_metrics.agent_invocations:
        agent.event_loop_metrics.reset_usage_metrics()

    agent.event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": str(uuid.uuid4())})
    logger.debug("cycle_index=<%s> | calling model", cycle_index)
    stop_reason, message, usage, metrics = await _consume_model_stream(agent)
    logger.debug("cycle_index=<%s>, stop_reason=<%s> | model call completed", cycle_index, stop_reason)

    agent.event_loop_metrics.update_usage(usage)
    agent.event_loop_metrics.update_metrics(metrics)

    await agent._append_messages(message)

    if stop_reason == "tool_use":
        return Checkpoint(
            position="after_model",
            stop_reason=stop_reason,
            cycle_index=cycle_index,
            tool_index=0,
            completed_tool_results=[],
            snapshot=agent.take_snapshot(preset="session").to_dict(),
        )

    return AgentResult(
        stop_reason=stop_reason,
        message=message,
        metrics=agent.event_loop_metrics,
        state={},
    )


async def _resume_tool_execution(agent: Agent, checkpoint: Checkpoint) -> Checkpoint:
    """Execute ONE tool and return a checkpoint.

    If more tools remain, returns after_tool checkpoint.
    If this was the last tool, appends the combined tool result message
    and returns after_tools checkpoint.

    The model message is already in agent.messages (restored from snapshot).
    Tool use blocks are extracted from the last assistant message.

    V1: Executes tool via tool_func.stream() directly. Does not use the
    agent's tool_executor, fire before/after tool hooks, create tracing
    spans, or handle interrupts.
    Tool errors are captured as tool results with status="error" — the model
    will see them on the next cycle and decide what to do.
    """
    model_message = agent.messages[-1]
    tool_use_blocks: list[ToolUse] = [
        content["toolUse"] for content in model_message.get("content", []) if "toolUse" in content
    ]

    if not tool_use_blocks:
        raise RuntimeError(
            f"Checkpoint at position={checkpoint.position} has stop_reason='tool_use' "
            f"but no toolUse blocks in the model message"
        )

    tool_index = checkpoint.tool_index
    if tool_index >= len(tool_use_blocks):
        raise ValueError(
            f"Checkpoint tool_index={tool_index} is out of range for {len(tool_use_blocks)} tool use blocks"
        )
    completed = list(checkpoint.completed_tool_results)

    tool_use = tool_use_blocks[tool_index]
    tool_name = tool_use.get("name", "")
    logger.debug(
        "cycle_index=<%s>, tool_index=<%s>, total_tools=<%s>, tool_name=<%s> | executing tool",
        checkpoint.cycle_index,
        tool_index,
        len(tool_use_blocks),
        tool_name,
    )
    result = await _execute_single_tool(agent, tool_use)
    completed.append(result)

    next_index = tool_index + 1

    if next_index < len(tool_use_blocks):
        # More tools remaining — checkpoint between them
        return Checkpoint(
            position="after_tool",
            stop_reason=checkpoint.stop_reason,
            cycle_index=checkpoint.cycle_index,
            tool_index=next_index,
            completed_tool_results=completed,
            snapshot=agent.take_snapshot(preset="session").to_dict(),
        )

    # Last tool — build combined tool result message, append, return after_tools
    # The API contract requires one user message with all toolResult blocks together.
    tool_result_message: Message = {
        "role": "user",
        "content": [{"toolResult": r} for r in completed],
    }
    await agent._append_messages(tool_result_message)

    return Checkpoint(
        position="after_tools",
        stop_reason=checkpoint.stop_reason,
        cycle_index=checkpoint.cycle_index,
        snapshot=agent.take_snapshot(preset="session").to_dict(),
    )


# ── Internal helpers (V1 simplified implementations) ──


async def _consume_model_stream(agent: Agent) -> tuple[StopReason, Message, Usage, Metrics]:
    """Call model, drain stream, return (stop_reason, message, usage, metrics).

    V1: Direct call to stream_messages. No hooks, no retry strategy.
    """
    tool_specs = agent.tool_registry.get_all_tool_specs()
    result: tuple[StopReason, Message, Usage, Metrics] | None = None

    async for event in stream_messages(
        agent.model,
        agent.system_prompt,
        agent.messages,
        tool_specs,
        system_prompt_content=agent._system_prompt_content,
        model_state=agent._model_state,
        invocation_state={"agent": agent},
    ):
        if "stop" in event:
            result = event["stop"]
            break

    if result is None:
        raise RuntimeError("Model stream ended without a stop event")

    return result


async def _execute_single_tool(agent: Agent, tool_use: ToolUse) -> ToolResult:
    """Execute one tool. Errors are captured as error results, not raised.

    Handles both SDK tools (which yield ToolResultEvent) and custom AgentTool
    subclasses (which may yield raw dicts with toolUseId).
    """
    tool_name = tool_use.get("name", "")
    # NOTE: ToolRegistry does not expose a public get_tool(name) method.
    # This reaches into internals (registry + dynamic_tools dicts).
    tool_func = agent.tool_registry.registry.get(tool_name) or agent.tool_registry.dynamic_tools.get(tool_name)

    if tool_func is None:
        return ToolResult(
            toolUseId=tool_use["toolUseId"],
            status="error",
            content=[{"text": f"Tool not found: {tool_name}"}],
        )

    try:
        result: ToolResult | None = None
        async for event in tool_func.stream(tool_use, {"agent": agent}):
            if isinstance(event, ToolResultEvent):
                result = event.tool_result
                break
            # Fallback for custom AgentTool subclasses that yield raw dicts
            if isinstance(event, dict) and "toolUseId" in event:
                result = ToolResult(
                    toolUseId=event["toolUseId"],
                    status=event.get("status", "success"),
                    content=event.get("content", []),
                )
                break

        if result is None:
            return ToolResult(
                toolUseId=tool_use["toolUseId"],
                status="error",
                content=[{"text": f"Tool {tool_name} did not return a result"}],
            )
        return result

    except Exception as e:
        logger.warning("tool_name=<%s> | tool execution failed: %s", tool_name, e, exc_info=True)
        return ToolResult(
            toolUseId=tool_use["toolUseId"],
            status="error",
            content=[{"text": f"Tool error: {e}"}],
        )
