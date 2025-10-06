"""FunctionNode implementation for executing deterministic Python functions as graph nodes.

This module provides the FunctionNode class that extends MultiAgentBase to execute
regular Python functions while maintaining compatibility with the existing graph
execution framework, proper error handling, metrics collection, and result formatting.
"""

import logging
import time
from typing import Any, Protocol, Union

from opentelemetry import trace as trace_api

from ..agent import AgentResult
from ..telemetry import get_tracer
from ..telemetry.metrics import EventLoopMetrics
from ..types.content import ContentBlock, Message
from ..types.event_loop import Metrics, Usage
from .base import MultiAgentBase, MultiAgentResult, NodeResult, Status

logger = logging.getLogger(__name__)


class FunctionNodeCallable(Protocol):
    """Protocol defining the required signature for functions used in FunctionNode.

    Functions must accept:
    - task: The input task (string or ContentBlock list)
    - invocation_state: Additional state/context from the calling environment
    - **kwargs: Additional keyword arguments for future extensibility

    Functions must return:
    - A string result that will be converted to a Message
    """

    def __call__(
        self, task: Union[str, list[ContentBlock]], invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> str:
        """Execute the node with the given task."""
        ...


class FunctionNode(MultiAgentBase):
    """Execute deterministic Python functions as graph nodes.

    FunctionNode wraps any callable Python function and executes it within the
    established multiagent framework, handling input conversion, error management,
    metrics collection, and result formatting automatically.

    Args:
        func: The callable function to wrap and execute
        name: Required name for the node
    """

    def __init__(self, func: FunctionNodeCallable, name: str):
        """Initialize FunctionNode with a callable function and required name.

        Args:
            func: The callable function to wrap and execute
            name: Required name for the node
        """
        self.func = func
        self.name = name
        self.tracer = get_tracer()

    async def invoke_async(
        self, task: Union[str, list[ContentBlock]], invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        """Execute the wrapped function and return formatted results.

        Args:
            task: The task input (string or ContentBlock list) to pass to the function
            invocation_state: Additional state/context (preserved for interface compatibility)
            **kwargs: Additional keyword arguments (preserved for future extensibility)

        Returns:
            MultiAgentResult containing the function execution results and metadata
        """
        if invocation_state is None:
            invocation_state = {}

        logger.debug("task=<%s> | starting function node execution", task)
        logger.debug("function_name=<%s> | executing function", self.name)

        start_time = time.time()
        span = self.tracer.start_multiagent_span(task, "function_node")
        with trace_api.use_span(span, end_on_exit=True):
            try:
                # Execute the wrapped function with proper parameters
                function_result = self.func(task, invocation_state, **kwargs)
                logger.debug("function_result=<%s> | function executed successfully", function_result)

                # Calculate execution time
                execution_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

                # Convert function result to Message
                message = Message(role="assistant", content=[ContentBlock(text=str(function_result))])
                agent_result = AgentResult(
                    stop_reason="end_turn",  # "Normal completion of the response" - function executed successfully
                    message=message,
                    metrics=EventLoopMetrics(),
                    state={},
                )

                # Create NodeResult for this function execution
                node_result = NodeResult(
                    result=agent_result,  # type is AgentResult
                    execution_time=execution_time,
                    status=Status.COMPLETED,
                    execution_count=1,
                )

                # Create MultiAgentResult with the NodeResult
                multi_agent_result = MultiAgentResult(
                    status=Status.COMPLETED,
                    results={self.name: node_result},
                    execution_count=1,
                    execution_time=execution_time,
                )

                logger.debug(
                    "function_name=<%s>, execution_time=<%dms> | function node completed successfully",
                    self.name,
                    execution_time,
                )

                return multi_agent_result

            except Exception as e:
                # Calculate execution time even for failed executions
                execution_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

                logger.error("function_name=<%s>, error=<%s> | function node failed", self.name, e)

                # Create failed NodeResult with exception
                node_result = NodeResult(
                    result=e,
                    execution_time=execution_time,
                    status=Status.FAILED,
                    accumulated_usage=Usage(inputTokens=0, outputTokens=0, totalTokens=0),
                    accumulated_metrics=Metrics(latencyMs=execution_time),
                    execution_count=1,
                )

                # Create failed MultiAgentResult
                multi_agent_result = MultiAgentResult(
                    status=Status.FAILED,
                    results={self.name: node_result},
                    accumulated_usage=Usage(inputTokens=0, outputTokens=0, totalTokens=0),
                    accumulated_metrics=Metrics(latencyMs=execution_time),
                    execution_count=1,
                    execution_time=execution_time,
                )

                return multi_agent_result
