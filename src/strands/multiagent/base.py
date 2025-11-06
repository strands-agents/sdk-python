"""Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Mapping, Union

from .._async import run_async
from ..agent import AgentResult
from ..interrupt import Interrupt
from ..types.event_loop import Metrics, Usage
from ..types.multiagent import MultiAgentInput
from ..types.traces import AttributeValue

logger = logging.getLogger(__name__)


class Status(Enum):
    """Execution status for both graphs and nodes.

    Attributes:
        PENDING: Task has not started execution yet.
        EXECUTING: Task is currently running.
        COMPLETED: Task finished successfully.
        FAILED: Task encountered an error and could not complete.
        INTERRUPTED: Task was interrupted by user.
    """

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class NodeResult:
    """Unified result from node execution - handles both Agent and nested MultiAgentBase results."""

    # Core result data - single AgentResult, nested MultiAgentResult, or Exception
    result: Union[AgentResult, "MultiAgentResult", Exception]

    # Execution metadata
    execution_time: int = 0
    status: Status = Status.PENDING

    # Accumulated metrics from this node and all children
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    interrupts: list[Interrupt] = field(default_factory=list)

    def get_agent_results(self) -> list[AgentResult]:
        """Get all AgentResult objects from this node, flattened if nested."""
        if isinstance(self.result, Exception):
            return []  # No agent results for exceptions
        elif isinstance(self.result, AgentResult):
            return [self.result]
        else:
            # Flatten nested results from MultiAgentResult
            flattened = []
            for nested_node_result in self.result.results.values():
                flattened.extend(nested_node_result.get_agent_results())
            return flattened

    def to_dict(self) -> dict[str, Any]:
        """Convert NodeResult to JSON-serializable dict, ignoring state field."""
        if isinstance(self.result, Exception):
            result_data: dict[str, Any] = {"type": "exception", "message": str(self.result)}
        elif isinstance(self.result, AgentResult):
            result_data = self.result.to_dict()
        else:
            # MultiAgentResult case
            result_data = self.result.to_dict()

        return {
            "result": result_data,
            "execution_time": self.execution_time,
            "status": self.status.value,
            "accumulated_usage": self.accumulated_usage,
            "accumulated_metrics": self.accumulated_metrics,
            "execution_count": self.execution_count,
            "interrupts": [interrupt.to_dict() for interrupt in self.interrupts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeResult":
        """Rehydrate a NodeResult from persisted JSON."""
        if "result" not in data:
            raise TypeError("NodeResult.from_dict: missing 'result'")
        raw = data["result"]

        result: Union[AgentResult, "MultiAgentResult", Exception]
        if isinstance(raw, dict) and raw.get("type") == "agent_result":
            result = AgentResult.from_dict(raw)
        elif isinstance(raw, dict) and raw.get("type") == "exception":
            result = Exception(str(raw.get("message", "node failed")))
        elif isinstance(raw, dict) and raw.get("type") == "multiagent_result":
            result = MultiAgentResult.from_dict(raw)
        else:
            raise TypeError(f"NodeResult.from_dict: unsupported result payload: {raw!r}")

        usage = _parse_usage(data.get("accumulated_usage", {}))
        metrics = _parse_metrics(data.get("accumulated_metrics", {}))

        interrupts = []
        for interrupt_data in data.get("interrupts", []):
            interrupts.append(Interrupt(**interrupt_data))

        return cls(
            result=result,
            execution_time=int(data.get("execution_time", 0)),
            status=Status(data.get("status", "pending")),
            accumulated_usage=usage,
            accumulated_metrics=metrics,
            execution_count=int(data.get("execution_count", 0)),
            interrupts=interrupts,
        )


@dataclass
class MultiAgentResult:
    """Result from multi-agent execution with accumulated metrics."""

    status: Status = Status.PENDING
    results: dict[str, NodeResult] = field(default_factory=lambda: {})
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0
    interrupts: list[Interrupt] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiAgentResult":
        """Rehydrate a MultiAgentResult from persisted JSON."""
        if data.get("type") != "multiagent_result":
            raise TypeError(f"MultiAgentResult.from_dict: unexpected type {data.get('type')!r}")

        results = {k: NodeResult.from_dict(v) for k, v in data.get("results", {}).items()}
        usage = _parse_usage(data.get("accumulated_usage", {}))
        metrics = _parse_metrics(data.get("accumulated_metrics", {}))

        interrupts = []
        for interrupt_data in data.get("interrupts", []):
            interrupts.append(Interrupt(**interrupt_data))

        multiagent_result = cls(
            status=Status(data["status"]),
            results=results,
            accumulated_usage=usage,
            accumulated_metrics=metrics,
            execution_count=int(data.get("execution_count", 0)),
            execution_time=int(data.get("execution_time", 0)),
            interrupts=interrupts,
        )
        return multiagent_result

    def to_dict(self) -> dict[str, Any]:
        """Convert MultiAgentResult to JSON-serializable dict."""
        return {
            "type": "multiagent_result",
            "status": self.status.value,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "accumulated_usage": self.accumulated_usage,
            "accumulated_metrics": self.accumulated_metrics,
            "execution_count": self.execution_count,
            "execution_time": self.execution_time,
            "interrupts": [interrupt.to_dict() for interrupt in self.interrupts],
        }


class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.

    Attributes:
        id: Unique MultiAgent id for session management,etc.
    """

    id: str

    @abstractmethod
    async def invoke_async(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        """Invoke asynchronously.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Additional keyword arguments passed to underlying agents.
        """
        raise NotImplementedError("invoke_async not implemented")

    async def stream_async(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events during multi-agent execution.

        Default implementation executes invoke_async and yields the result as a single event.
        Subclasses can override this method to provide true streaming capabilities.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Additional keyword arguments passed to underlying agents.

        Yields:
            Dictionary events containing multi-agent execution information including:
            - Multi-agent coordination events (node start/complete, handoffs)
            - Forwarded single-agent events with node context
            - Final result event
        """
        # Default implementation for backward compatibility
        # Execute invoke_async and yield the result as a single event
        result = await self.invoke_async(task, invocation_state, **kwargs)
        yield {"result": result}

    def __call__(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        """Invoke synchronously.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Additional keyword arguments passed to underlying agents.
        """
        if invocation_state is None:
            invocation_state = {}

        if kwargs:
            invocation_state.update(kwargs)
            warnings.warn("`**kwargs` parameter is deprecating, use `invocation_state` instead.", stacklevel=2)

        return run_async(lambda: self.invoke_async(task, invocation_state))

    def serialize_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the orchestrator state."""
        raise NotImplementedError

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        """Restore orchestrator state from a session dict."""
        raise NotImplementedError

    def _parse_trace_attributes(
        self, attributes: Mapping[str, AttributeValue] | None = None
    ) -> dict[str, AttributeValue]:
        trace_attributes: dict[str, AttributeValue] = {}
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
                ):
                    trace_attributes[k] = v
        return trace_attributes


# Private helper function to avoid duplicate code


def _parse_usage(usage_data: dict[str, Any]) -> Usage:
    """Parse Usage from dict data."""
    usage = Usage(
        inputTokens=usage_data.get("inputTokens", 0),
        outputTokens=usage_data.get("outputTokens", 0),
        totalTokens=usage_data.get("totalTokens", 0),
    )
    # Add optional fields if they exist
    if "cacheReadInputTokens" in usage_data:
        usage["cacheReadInputTokens"] = usage_data["cacheReadInputTokens"]
    if "cacheWriteInputTokens" in usage_data:
        usage["cacheWriteInputTokens"] = usage_data["cacheWriteInputTokens"]
    return usage


def _parse_metrics(metrics_data: dict[str, Any]) -> Metrics:
    """Parse Metrics from dict data."""
    return Metrics(latencyMs=metrics_data.get("latencyMs", 0))
