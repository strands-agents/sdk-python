"""Strands Agent executor for the A2A protocol.

This module provides the StrandsA2AExecutor class, which adapts a Strands Agent
to be used as an executor in the A2A protocol. It handles the execution of agent
requests and the conversion of Strands Agent streamed responses to A2A events.

The A2A AgentExecutor ensures clients receive responses for synchronous and
streamed requests to the A2AServer.
"""

import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from ...agent.agent import Agent as SAAgent
from ...agent.agent import AgentResult as SAAgentResult

logger = logging.getLogger(__name__)


class StrandsA2AExecutor(AgentExecutor):
    """Executor that adapts a Strands Agent to the A2A protocol.

    This executor uses streaming mode to handle the execution of agent requests
    and converts Strands Agent responses to A2A protocol events.
    """

    def __init__(self, agent: SAAgent):
        """Initialize a StrandsA2AExecutor.

        Args:
            agent: The Strands Agent instance to adapt to the A2A protocol.
        """
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a request using the Strands Agent and send the response as A2A events.

        This method executes the user's input using the Strands Agent in streaming mode
        and converts the agent's response to A2A events.

        Args:
            context: The A2A request context, containing the user's input and task metadata.
            event_queue: The A2A event queue used to send response events back to the client.

        Raises:
            ServerError: If an error occurs during agent execution
        """
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            await self._execute_streaming(context, updater)
        except Exception as e:
            raise ServerError(error=InternalError()) from e

    async def _execute_streaming(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Execute request in streaming mode.

        Streams the agent's response in real-time, sending incremental updates
        as they become available from the agent.

        Args:
            context: The A2A request context, containing the user's input and other metadata.
            updater: The task updater for managing task state and sending updates.
        """
        logger.info("Executing request in streaming mode")
        user_input = context.get_user_input()
        try:
            async for event in self.agent.stream_async(user_input):
                await self._handle_streaming_event(event, updater)
        except Exception:
            logger.exception("Error in streaming execution")
            raise

    async def _handle_streaming_event(self, event: dict[str, Any], updater: TaskUpdater) -> None:
        """Handle a single streaming event from the Strands Agent.

        Processes streaming events from the agent, converting data chunks to A2A
        task updates and handling the final result when streaming is complete.

        Args:
            event: The streaming event from the agent, containing either 'data' for
                incremental content or 'result' for the final response.
            updater: The task updater for managing task state and sending updates.
        """
        logger.debug("Streaming event: %s", event)
        if "data" in event:
            if text_content := event["data"]:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        text_content,
                        updater.context_id,
                        updater.task_id,
                    ),
                )
        elif "result" in event:
            await self._handle_agent_result(event["result"], updater)

    async def _handle_agent_result(self, result: SAAgentResult | None, updater: TaskUpdater) -> None:
        """Handle the final result from the Strands Agent.

        Processes the agent's final result, extracts text content from the response,
        and adds it as an artifact to the task before marking the task as complete.

        Args:
            result: The agent result object containing the final response, or None if no result.
            updater: The task updater for managing task state and adding the final artifact.
        """
        if final_content := str(result):
            await updater.add_artifact(
                [Part(root=TextPart(text=final_content))],
                name="agent_response",
            )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing execution.

        This method is called when a request cancellation is requested. Currently,
        cancellation is not supported by the Strands Agent executor, so this method
        always raises an UnsupportedOperationError.

        Args:
            context: The A2A request context.
            event_queue: The A2A event queue.

        Raises:
            ServerError: Always raised with an UnsupportedOperationError, as cancellation
                is not currently supported.
        """
        logger.warning("Cancellation requested but not supported")
        raise ServerError(error=UnsupportedOperationError())
