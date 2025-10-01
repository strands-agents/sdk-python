"""Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

from ..agent import AgentResult
from ..types.content import ContentBlock
from ..types.event_loop import Metrics, Usage

logger = logging.getLogger(__name__)


class Status(Enum):
    """Execution status for both graphs and nodes."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NodeResult:
    """Unified result from node execution - handles both Agent and nested MultiAgentBase results.

    The status field represents the semantic outcome of the node's work:
    - COMPLETED: The node's task was successfully accomplished
    - FAILED: The node's task failed or produced an error
    """

    # Core result data - single AgentResult, nested MultiAgentResult, or Exception
    result: Union[AgentResult, "MultiAgentResult", Exception]

    # Execution metadata
    execution_time: int = 0
    status: Status = Status.PENDING

    # Accumulated metrics from this node and all children
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0

    def get_agent_results(self) -> list[AgentResult]:
        """Get all AgentResult objects from this node, flattened if nested."""
        if isinstance(self.result, Exception):
            return []  # No agent results for exceptions
        if isinstance(self.result, AgentResult):
            return [self.result]
        if getattr(self.result, "__class__", None) and self.result.__class__.__name__ == "AgentResult":
            return [self.result]  # type: ignore[list-item]

        # If this is a nested MultiAgentResult, flatten children
        if hasattr(self.result, "results") and isinstance(self.result.results, dict):
            flattened: list[AgentResult] = []
            for nested in self.result.results.values():
                if isinstance(nested, NodeResult):
                    flattened.extend(nested.get_agent_results())
            return flattened

        return []

    def to_dict(self) -> dict[str, Any]:
        """Convert NodeResult to JSON-serializable dict, ignoring state field."""
        result_data: Any = None
        if isinstance(self.result, Exception):
            result_data = {"type": "exception", "message": str(self.result)}
        elif isinstance(self.result, AgentResult):
            # Serialize AgentResult without state field
            result_data = {
                "type": "agent_result",
                "stop_reason": self.result.stop_reason,
                "message": self.result.message,  # Message type is JSON serializable
                # Skip metrics and state - not JSON serializable
            }
        elif hasattr(self.result, "to_dict"):
            result_data = self.result.to_dict()
        else:
            result_data = str(self.result)

        return {
            "result": result_data,
            "execution_time": self.execution_time,
            "status": self.status.value,
            "accumulated_usage": dict(self.accumulated_usage),
            "accumulated_metrics": dict(self.accumulated_metrics),
            "execution_count": self.execution_count,
        }


@dataclass
class MultiAgentResult:
    """Result from multi-agent execution with accumulated metrics.

    The status field represents the outcome of the MultiAgentBase execution:
    - COMPLETED: The execution was successfully accomplished
    - FAILED: The execution failed or produced an error
    """

    status: Status = Status.PENDING
    results: dict[str, NodeResult] = field(default_factory=lambda: {})
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert MultiAgentResult to JSON-serializable dict."""
        return {
            "status": self.status.value,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "accumulated_usage": dict(self.accumulated_usage),
            "accumulated_metrics": dict(self.accumulated_metrics),
            "execution_count": self.execution_count,
            "execution_time": self.execution_time,
        }


class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.
    """

    @abstractmethod
    async def invoke_async(
        self, task: str | list[ContentBlock], invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        """Invoke asynchronously.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Additional keyword arguments passed to underlying agents.
        """
        raise NotImplementedError("invoke_async not implemented")

    def __call__(
        self, task: str | list[ContentBlock], invocation_state: dict[str, Any] | None = None, **kwargs: Any
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

        def execute() -> MultiAgentResult:
            return asyncio.run(self.invoke_async(task, invocation_state, **kwargs))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    @abstractmethod
    def serialize_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the orchestrator state."""
        raise NotImplementedError

    @abstractmethod
    def deserialize_state(self, payload: dict) -> None:
        """Restore orchestrator state from a session dict."""
        raise NotImplementedError

    def serialize_node_result_for_persist(self, raw: NodeResult) -> dict[str, Any]:
        """Serialize node result for persistence.

        Args:
            raw: Raw node result to serialize

        Returns:
            JSON-serializable dict representation
        """
        if isinstance(raw, dict):
            return raw

        if hasattr(raw, "to_dict") and callable(raw.to_dict):
            return raw.to_dict()

        # Fallback for strings and other types
        return {"agent_outputs": [str(raw)]}
