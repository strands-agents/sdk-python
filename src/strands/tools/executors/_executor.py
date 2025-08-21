"""Abstract base class for tool executors.

Tool executors are responsible for determining how tools are executed (e.g., concurrently, sequentially, with custom
thread pools, etc.).
"""

import abc
import logging
import time
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import trace as trace_api

from ...experimental.hooks import AfterToolInvocationEvent, BeforeToolInvocationEvent
from ...telemetry.metrics import Trace
from ...telemetry.tracer import get_tracer
from ...types.content import Message
from ...types.tools import ToolChoice, ToolChoiceAuto, ToolConfig, ToolGenerator, ToolResult, ToolUse

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent

logger = logging.getLogger(__name__)


class ToolExecutor(abc.ABC):
    """Abstract base class for tool executors."""

    @staticmethod
    async def _stream(
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolGenerator:
        """Stream tool events.

        This method adds additional logic to the stream invocation including:

        - Tool lookup and validation
        - Before/after hook execution
        - Tracing and metrics collection
        - Error handling and recovery

        Args:
            agent: The agent for which the tool is being executed.
            tool_use: Metadata and inputs for the tool to be executed.
            tool_results: List of tool results from each tool execution.
            invocation_state: Context for the tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        logger.debug("tool_use=<%s> | streaming", tool_use)
        tool_name = tool_use["name"]

        tool_info = agent.tool_registry.dynamic_tools.get(tool_name)
        tool_func = tool_info if tool_info is not None else agent.tool_registry.registry.get(tool_name)

        invocation_state.update(
            {
                "model": agent.model,
                "messages": agent.messages,
                "system_prompt": agent.system_prompt,
                "tool_config": ToolConfig(  # for backwards compatibility
                    tools=[{"toolSpec": tool_spec} for tool_spec in agent.tool_registry.get_all_tool_specs()],
                    toolChoice=cast(ToolChoice, {"auto": ToolChoiceAuto()}),
                ),
            }
        )

        before_event = agent.hooks.invoke_callbacks(
            BeforeToolInvocationEvent(
                agent=agent,
                selected_tool=tool_func,
                tool_use=tool_use,
                invocation_state=invocation_state,
            )
        )

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
                after_event = agent.hooks.invoke_callbacks(
                    AfterToolInvocationEvent(
                        agent=agent,
                        selected_tool=selected_tool,
                        tool_use=tool_use,
                        invocation_state=invocation_state,
                        result=result,
                    )
                )
                yield after_event.result
                tool_results.append(after_event.result)
                return

            async for event in selected_tool.stream(tool_use, invocation_state, **kwargs):
                yield event

            result = cast(ToolResult, event)

            after_event = agent.hooks.invoke_callbacks(
                AfterToolInvocationEvent(
                    agent=agent,
                    selected_tool=selected_tool,
                    tool_use=tool_use,
                    invocation_state=invocation_state,
                    result=result,
                )
            )
            yield after_event.result
            tool_results.append(after_event.result)

        except Exception as e:
            logger.exception("tool_name=<%s> | failed to process tool", tool_name)
            error_result: ToolResult = {
                "toolUseId": str(tool_use.get("toolUseId")),
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }
            after_event = agent.hooks.invoke_callbacks(
                AfterToolInvocationEvent(
                    agent=agent,
                    selected_tool=selected_tool,
                    tool_use=tool_use,
                    invocation_state=invocation_state,
                    result=error_result,
                    exception=e,
                )
            )
            yield after_event.result
            tool_results.append(after_event.result)

    @staticmethod
    async def _stream_with_trace(
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolGenerator:
        """Execute tool with tracing and metrics collection.

        Args:
            agent: The agent for which the tool is being executed.
            tool_use: Metadata and inputs for the tool to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        tool_name = tool_use["name"]

        tracer = get_tracer()

        tool_call_span = tracer.start_tool_call_span(tool_use, cycle_span)
        tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
        tool_start_time = time.time()

        with trace_api.use_span(tool_call_span):
            async for event in ToolExecutor._stream(agent, tool_use, tool_results, invocation_state, **kwargs):
                yield event

            result = cast(ToolResult, event)

            tool_success = result.get("status") == "success"
            tool_duration = time.time() - tool_start_time
            message = Message(role="user", content=[{"toolResult": result}])
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
    ) -> ToolGenerator:
        """Execute the given tools according to this executor's strategy.

        Args:
            agent: The agent for which tools are being executed.
            tool_uses: Metadata and inputs for the tools to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.

        Yields:
            Events from the tool execution stream.
        """
        pass
