"""Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).
"""

import asyncio
import logging
import warnings
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
            # Serialize AgentResult without state field
            result_data = AgentResult.to_dict(self.result)
        elif isinstance(self.result, MultiAgentResult):
            result_data = self.result.to_dict()
        else:
            raise TypeError(f"Unsupported NodeResult.result type for serialization: {type(self.result).__name__}")

        return {
            "result": result_data,
            "execution_time": self.execution_time,
            "status": self.status.value,
            "accumulated_usage": self.accumulated_usage,
            "accumulated_metrics": self.accumulated_metrics,
            "execution_count": self.execution_count,
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

        usage_data = data.get("accumulated_usage", {})
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

        metrics = Metrics(latencyMs=data.get("accumulated_metrics", {}).get("latencyMs", 0))

        return cls(
            result=result,
            execution_time=int(data.get("execution_time", 0)),
            status=Status(data.get("status", "pending")),
            accumulated_usage=usage,
            accumulated_metrics=metrics,
            execution_count=int(data.get("execution_count", 0)),
        )


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
            "type": "multiagent_result",
            "status": self.status.value,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "accumulated_usage": dict(self.accumulated_usage),
            "accumulated_metrics": dict(self.accumulated_metrics),
            "execution_count": self.execution_count,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiAgentResult":
        """Rehydrate a MultiAgentResult from persisted JSON."""
        if data.get("type") != "multiagent_result":
            raise TypeError(f"MultiAgentResult.from_dict: unexpected type {data.get('type')!r}")

        results = {k: NodeResult.from_dict(v) for k, v in data.get("results", {}).items()}
        usage_data = data.get("accumulated_usage", {})
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

        metrics = Metrics(latencyMs=data.get("accumulated_metrics", {}).get("latencyMs", 0))

        multiagent_result = cls(
            status=Status(data.get("status", Status.PENDING.value)),
            results=results,
            accumulated_usage=usage,
            accumulated_metrics=metrics,
            execution_count=int(data.get("execution_count", 0)),
            execution_time=int(data.get("execution_time", 0)),
        )
        return multiagent_result


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

        if kwargs:
            invocation_state.update(kwargs)
            warnings.warn("`**kwargs` parameter is deprecating, use `invocation_state` instead.", stacklevel=2)

        def execute() -> MultiAgentResult:
            return asyncio.run(self.invoke_async(task, invocation_state))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    def serialize_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the orchestrator state."""
        raise NotImplementedError

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        """Restore orchestrator state from a session dict."""
        raise NotImplementedError
