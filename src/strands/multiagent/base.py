"""Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from ..agent import AgentResult
from ..types.event_loop import Metrics, Usage


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


@dataclass
class MultiAgentResult:
    """Result from multi-agent execution with accumulated metrics."""

    results: dict[str, NodeResult]
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0


class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.
    """

    @abstractmethod
    # TODO: for task - multi-modal input (Message), list of messages
    async def execute_async(self, task: str) -> MultiAgentResult:
        """Execute task asynchronously."""
        raise NotImplementedError("execute_async not implemented")

    @abstractmethod
    # TODO: for task - multi-modal input (Message), list of messages
    def execute(self, task: str) -> MultiAgentResult:
        """Execute task synchronously."""
        raise NotImplementedError("execute not implemented")
