"""Concurrent tool executor implementation."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from typing_extensions import override

from ...telemetry.metrics import Trace
from ...types._events import TypedEvent
from ...types.exceptions import AgentDelegationException
from ...types.tools import ToolResult, ToolUse
from ._executor import ToolExecutor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent


class ConcurrentToolExecutor(ToolExecutor):
    """Concurrent tool executor."""

    @override
    async def _execute(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tools concurrently.

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
        task_queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        task_events = [asyncio.Event() for _ in tool_uses]
        stop_event = object()

        tasks = [
            asyncio.create_task(
                self._task(
                    agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    task_id,
                    task_queue,
                    task_events[task_id],
                    stop_event,
                )
            )
            for task_id, tool_use in enumerate(tool_uses)
        ]

        task_count = len(tasks)
        collected_exceptions = []
        while task_count:
            task_id, event = await task_queue.get()
            if event is stop_event:
                task_count -= 1
                continue

            # Check if event is an exception that needs to be raised
            if isinstance(event, Exception):
                logger.debug("Concurrent executor main thread got exception: %s: %s", type(event).__name__, event)
                collected_exceptions.append(event)
                task_events[task_id].set()
                continue

            yield event
            task_events[task_id].set()

        # After all tasks complete, check if we collected any exceptions
        if collected_exceptions:
            # Prioritize delegation exceptions if present
            delegation_exceptions = [e for e in collected_exceptions if isinstance(e, AgentDelegationException)]
            if delegation_exceptions:
                # If there are delegation exceptions, raise the first one
                total_exceptions = len(collected_exceptions)
                logger.debug(
                    "Raising AgentDelegationException from concurrent executor (collected %s exceptions total)",
                    total_exceptions,
                )
                raise delegation_exceptions[0]
            else:
                # For non-delegation exceptions, raise a combined exception with all details
                if len(collected_exceptions) == 1:
                    raise collected_exceptions[0]
                else:
                    # Create a combined exception to report all concurrent errors
                    error_summary = "; ".join([f"{type(e).__name__}: {str(e)}" for e in collected_exceptions])
                    combined_exception = RuntimeError(f"Multiple tool execution errors occurred: {error_summary}")
                    combined_exception.__cause__ = collected_exceptions[0]  # Keep the first as primary cause
                    raise combined_exception

        asyncio.gather(*tasks)

    async def _task(
        self,
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        task_id: int,
        task_queue: asyncio.Queue,
        task_event: asyncio.Event,
        stop_event: object,
    ) -> None:
        """Execute a single tool and put results in the task queue.

        Args:
            agent: The agent executing the tool.
            tool_use: Tool use metadata and inputs.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for tool execution.
            task_id: Unique identifier for this task.
            task_queue: Queue to put tool events into.
            task_event: Event to signal when task can continue.
            stop_event: Sentinel object to signal task completion.
        """
        from ...types.exceptions import AgentDelegationException

        try:
            events = ToolExecutor._stream_with_trace(
                agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state
            )
            async for event in events:
                task_queue.put_nowait((task_id, event))
                await task_event.wait()
                task_event.clear()

        except AgentDelegationException as e:
            logger.debug("Concurrent executor caught AgentDelegationException for %s", e.target_agent)
            # Put delegation exception in the queue to be handled by main thread
            task_queue.put_nowait((task_id, e))
        except Exception as e:
            logger.debug("Concurrent executor caught generic exception: %s: %s", type(e).__name__, e)
            # Put other exceptions in the queue as well
            task_queue.put_nowait((task_id, e))
        finally:
            task_queue.put_nowait((task_id, stop_event))
