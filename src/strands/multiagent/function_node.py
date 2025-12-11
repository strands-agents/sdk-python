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
    """Protocol for functions that can be executed within FunctionNode."""

    def __call__(
        self, task: str | list[ContentBlock], invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> str | list[ContentBlock] | Message:
        """Execute deterministic logic within the multiagent system."""
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

        span = self.tracer.start_multiagent_span(task, "function_node")
        with trace_api.use_span(span, end_on_exit=True):
            try:
                start_time = time.time()
                # Execute the wrapped function with proper parameters
                function_result = self.func(task, invocation_state, **kwargs)
                # Calculate execution time
                execution_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
                logger.debug(
                    "function_result=<%s>, execution_time=<%dms> | function executed successfully",
                    function_result,
                    execution_time,
                )

                # Convert function result to Message based on type
                if isinstance(function_result, dict) and "role" in function_result and "content" in function_result:
                    # Already a Message
                    message = function_result
                elif isinstance(function_result, list):
                    # List of ContentBlocks
                    message = Message(role="assistant", content=function_result)
                else:
                    # String or other type - convert to string
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
