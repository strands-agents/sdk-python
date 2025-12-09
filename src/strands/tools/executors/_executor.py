"""Abstract base class for tool executors.

Tool executors are responsible for determining how tools are executed (e.g., concurrently, sequentially, with custom
thread pools, etc.).
"""

import abc
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast

from opentelemetry import trace as trace_api

from ...experimental.hooks.events import BidiAfterToolCallEvent, BidiBeforeToolCallEvent
from ...hooks import AfterToolCallEvent, BeforeToolCallEvent
from ...telemetry.metrics import Trace
from ...telemetry.tracer import get_tracer, serialize
from ...types._events import ToolCancelEvent, ToolInterruptEvent, ToolResultEvent, ToolStreamEvent, TypedEvent
from ...types.content import Message
from ...types.interrupt import Interrupt
from ...types.tools import ToolChoice, ToolChoiceAuto, ToolConfig, ToolResult, ToolUse
from ..structured_output._structured_output_context import StructuredOutputContext

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent
    from ...experimental.bidi import BidiAgent

logger = logging.getLogger(__name__)


class ToolExecutor(abc.ABC):
    """Abstract base class for tool executors."""

    @staticmethod
    def _is_agent(agent: "Agent | BidiAgent") -> bool:
        """Check if the agent is an Agent instance, otherwise we assume BidiAgent.

        Note, we use a runtime import to avoid a circular dependency error.
        """
        from ...agent import Agent

        return isinstance(agent, Agent)

    @staticmethod
    async def _invoke_before_tool_call_hook(
        agent: "Agent | BidiAgent",
        tool_func: Any,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
    ) -> tuple[BeforeToolCallEvent | BidiBeforeToolCallEvent, list[Interrupt]]:
        """Invoke the appropriate before tool call hook based on agent type."""
        kwargs = {
            "selected_tool": tool_func,
            "tool_use": tool_use,
            "invocation_state": invocation_state,
        }
        event = (
            BeforeToolCallEvent(agent=cast("Agent", agent), **kwargs)
            if ToolExecutor._is_agent(agent)
            else BidiBeforeToolCallEvent(agent=cast("BidiAgent", agent), **kwargs)
        )

        return await agent.hooks.invoke_callbacks_async(event)

    @staticmethod
    async def _invoke_after_tool_call_hook(
        agent: "Agent | BidiAgent",
        selected_tool: Any,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        result: ToolResult,
        exception: Exception | None = None,
        cancel_message: str | None = None,
    ) -> tuple[AfterToolCallEvent | BidiAfterToolCallEvent, list[Interrupt]]:
        """Invoke the appropriate after tool call hook based on agent type."""
        kwargs = {
            "selected_tool": selected_tool,
            "tool_use": tool_use,
            "invocation_state": invocation_state,
            "result": result,
            "exception": exception,
            "cancel_message": cancel_message,
        }
        event = (
            AfterToolCallEvent(agent=cast("Agent", agent), **kwargs)
            if ToolExecutor._is_agent(agent)
            else BidiAfterToolCallEvent(agent=cast("BidiAgent", agent), **kwargs)
        )

        return await agent.hooks.invoke_callbacks_async(event)

    @staticmethod
    async def _stream(
        agent: "Agent | BidiAgent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        invocation_state: dict[str, Any],
        structured_output_context: StructuredOutputContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Stream tool events.

        This method adds additional logic to the stream invocation including:

        - Tool lookup and validation
        - Before/after hook execution
        - Tracing and metrics collection
        - Error handling and recovery
        - Interrupt handling for human-in-the-loop workflows

        Args:
            agent: The agent (Agent or BidiAgent) for which the tool is being executed.
            tool_use: Metadata and inputs for the tool to be executed.
            tool_results: List of tool results from each tool execution.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output management.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        logger.debug("tool_use=<%s> | streaming", tool_use)
        tool_name = tool_use["name"]
        structured_output_context = structured_output_context or StructuredOutputContext()

        tool_info = agent.tool_registry.dynamic_tools.get(tool_name)
        tool_func = tool_info if tool_info is not None else agent.tool_registry.registry.get(tool_name)
        tool_spec = tool_func.tool_spec if tool_func is not None else None

        current_span = trace_api.get_current_span()
        if current_span and tool_spec is not None:
            current_span.set_attribute("gen_ai.tool.description", tool_spec["description"])
            input_schema = tool_spec["inputSchema"]
            if "json" in input_schema:
                current_span.set_attribute("gen_ai.tool.json_schema", serialize(input_schema["json"]))

        invocation_state.update(
            {
                "agent": agent,
                "model": agent.model,
                "messages": agent.messages,
                "system_prompt": agent.system_prompt,
                "tool_config": ToolConfig(  # for backwards compatibility
                    tools=[{"toolSpec": tool_spec} for tool_spec in agent.tool_registry.get_all_tool_specs()],
                    toolChoice=cast(ToolChoice, {"auto": ToolChoiceAuto()}),
                ),
            }
        )

        before_event, interrupts = await ToolExecutor._invoke_before_tool_call_hook(
            agent, tool_func, tool_use, invocation_state
        )

        if interrupts:
            yield ToolInterruptEvent(tool_use, interrupts)
            return

        if before_event.cancel_tool:
            cancel_message = (
                before_event.cancel_tool if isinstance(before_event.cancel_tool, str) else "tool cancelled by user"
            )
            yield ToolCancelEvent(tool_use, cancel_message)

            cancel_result: ToolResult = {
                "toolUseId": str(tool_use.get("toolUseId")),
                "status": "error",
                "content": [{"text": cancel_message}],
            }

            after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                agent, None, tool_use, invocation_state, cancel_result, cancel_message=cancel_message
            )
            yield ToolResultEvent(after_event.result)
            tool_results.append(after_event.result)
            return

        try:
            selected_tool = before_event.selected_tool
            tool_use = before_event.tool_use
            invocation_state = before_event.invocation_state

            if not selected_tool:
                if tool_func == selected_tool:
                    logger.error(
                        "tool_name=<%s>, available_tools=<%s> | tool not found in registry",
                        tool_name,
                        list(agent.tool_registry.registry.keys()),
                    )
                else:
                    logger.debug(
                        "tool_name=<%s>, tool_use_id=<%s> | a hook resulted in a non-existing tool call",
                        tool_name,
                        str(tool_use.get("toolUseId")),
                    )

                result: ToolResult = {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Unknown tool: {tool_name}"}],
                }

                after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                    agent, selected_tool, tool_use, invocation_state, result
                )
                yield ToolResultEvent(after_event.result)
                tool_results.append(after_event.result)
                return
            if structured_output_context.is_enabled:
                kwargs["structured_output_context"] = structured_output_context
            async for event in selected_tool.stream(tool_use, invocation_state, **kwargs):
                # Internal optimization; for built-in AgentTools, we yield TypedEvents out of .stream()
                # so that we don't needlessly yield ToolStreamEvents for non-generator callbacks.
                # In which case, as soon as we get a ToolResultEvent we're done and for ToolStreamEvent
                # we yield it directly; all other cases (non-sdk AgentTools), we wrap events in
                # ToolStreamEvent and the last event is just the result.

                if isinstance(event, ToolInterruptEvent):
                    yield event
                    return

                if isinstance(event, ToolResultEvent):
                    # below the last "event" must point to the tool_result
                    event = event.tool_result
                    break

                if isinstance(event, ToolStreamEvent):
                    yield event
                else:
                    yield ToolStreamEvent(tool_use, event)

            result = cast(ToolResult, event)

            after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                agent, selected_tool, tool_use, invocation_state, result
            )

            yield ToolResultEvent(after_event.result)
            tool_results.append(after_event.result)

        except Exception as e:
            logger.exception("tool_name=<%s> | failed to process tool", tool_name)
            error_result: ToolResult = {
                "toolUseId": str(tool_use.get("toolUseId")),
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }

            after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                agent, selected_tool, tool_use, invocation_state, error_result, exception=e
            )
            yield ToolResultEvent(after_event.result)
            tool_results.append(after_event.result)

    @staticmethod
    async def _stream_with_trace(
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: StructuredOutputContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tool with tracing and metrics collection.

        Args:
            agent: The agent for which the tool is being executed.
            tool_use: Metadata and inputs for the tool to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output management.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        tool_name = tool_use["name"]
        structured_output_context = structured_output_context or StructuredOutputContext()

        tracer = get_tracer()

        tool_call_span = tracer.start_tool_call_span(
            tool_use, cycle_span, custom_trace_attributes=agent.trace_attributes
        )
        tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
        tool_start_time = time.time()

        with trace_api.use_span(tool_call_span):
            async for event in ToolExecutor._stream(
                agent, tool_use, tool_results, invocation_state, structured_output_context, **kwargs
            ):
                yield event

            if isinstance(event, ToolInterruptEvent):
                tracer.end_tool_call_span(tool_call_span, tool_result=None)
                return

            result_event = cast(ToolResultEvent, event)
            result = result_event.tool_result

            tool_success = result.get("status") == "success"
            tool_duration = time.time() - tool_start_time
            message = Message(role="user", content=[{"toolResult": result}])
            if ToolExecutor._is_agent(agent):
                agent.event_loop_metrics.add_tool_usage(tool_use, tool_duration, tool_trace, tool_success, message)
            cycle_trace.add_child(tool_trace)

            tracer.end_tool_call_span(tool_call_span, result)

    @abc.abstractmethod
    # pragma: no cover
    def _execute(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the given tools according to this executor's strategy.

        Args:
            agent: The agent for which tools are being executed.
            tool_uses: Metadata and inputs for the tools to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output management.

        Yields:
            Events from the tool execution stream.
        """
        pass
