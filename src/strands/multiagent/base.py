"""Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).
"""

import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

from ..agent import AgentResult
from ..types.content import ContentBlock
from ..types.event_loop import Metrics, Usage


class Status(Enum):
    """Execution status for both graphs and nodes."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SharedContext:
    """Shared context between multi-agent nodes.

    This class provides a key-value store for sharing information across nodes
    in multi-agent systems like Graph and Swarm. It validates that all values
    are JSON serializable to ensure compatibility.
    """

    context: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_context(self, node_id: str, key: str, value: Any) -> None:
        """Add context for a specific node.

        Args:
            node_id: The ID of the node adding the context
            key: The key to store the value under
            value: The value to store (must be JSON serializable)

        Raises:
            ValueError: If key is invalid or value is not JSON serializable
        """
        self._validate_key(key)
        self._validate_json_serializable(value)

        if node_id not in self.context:
            self.context[node_id] = {}
        self.context[node_id][key] = value

    def get_context(self, node_id: str, key: str | None = None) -> Any:
        """Get context for a specific node.

        Args:
            node_id: The ID of the node to get context for
            key: The specific key to retrieve (if None, returns all context for the node)

        Returns:
            The stored value, entire context dict for the node, or None if not found
        """
        if node_id not in self.context:
            return None if key else {}

        if key is None:
            return copy.deepcopy(self.context[node_id])
        else:
            value = self.context[node_id].get(key)
            return copy.deepcopy(value) if value is not None else None

    def _validate_key(self, key: str) -> None:
        """Validate that a key is valid.

        Args:
            key: The key to validate

        Raises:
            ValueError: If key is invalid
        """
        if key is None:
            raise ValueError("Key cannot be None")
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        if not key.strip():
            raise ValueError("Key cannot be empty")

    def _validate_json_serializable(self, value: Any) -> None:
        """Validate that a value is JSON serializable.

        Args:
            value: The value to validate

        Raises:
            ValueError: If value is not JSON serializable
        """
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Value is not JSON serializable: {type(value).__name__}. "
                f"Only JSON-compatible types (str, int, float, bool, list, dict, None) are allowed."
            ) from e


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


class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.
    """

    @abstractmethod
    async def invoke_async(self, task: str | list[ContentBlock], **kwargs: Any) -> MultiAgentResult:
        """Invoke asynchronously."""
        raise NotImplementedError("invoke_async not implemented")

    @abstractmethod
    def __call__(self, task: str | list[ContentBlock], **kwargs: Any) -> MultiAgentResult:
        """Invoke synchronously."""
        raise NotImplementedError("__call__ not implemented")
