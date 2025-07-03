"""Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from ..agent import AgentResult
from ..types.event_loop import Metrics, Usage

logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    """Unified result from node execution - handles both Agent and nested MultiAgentBase results."""

    # Core result data - single AgentResult or nested MultiAgentResult
    results: Union[AgentResult, "MultiAgentResult"]

    # Execution metadata
    execution_time: float = 0.0
    status: Any = None

    # Accumulated metrics from this node and all children
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0

    def get_agent_results(self) -> List[AgentResult]:
        """Get all AgentResult objects from this node, flattened if nested."""
        if isinstance(self.results, AgentResult):
            return [self.results]
        else:
            # Flatten nested results from MultiAgentResult
            flattened = []
            for nested_node_result in self.results.results.values():
                flattened.extend(nested_node_result.get_agent_results())
            return flattened


@dataclass
class MultiAgentResult:
    """Result from multi-agent execution with accumulated metrics."""

    results: Dict[str, NodeResult]
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: float = 0.0


class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.
    """

    @abstractmethod
    # TODO: for task - multi-modal input (Message), list of messages
    async def execute(self, task: str) -> MultiAgentResult:
        """Execute task."""
        raise NotImplementedError("execute not implemented")

    @abstractmethod
    # TODO: for task - multi-modal input (Message), list of messages
    async def resume(self, task: str, state: Any) -> MultiAgentResult:
        """Resume task from previous state."""
        raise NotImplementedError("resume not implemented")
