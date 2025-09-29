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
    result: Union["AgentResult", "MultiAgentResult", Exception]

    # Execution metadata
    execution_time: int = 0
    status: Status = Status.PENDING

    # Accumulated metrics from this node and all children
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0

    def get_agent_results(self) -> list["AgentResult"]:
        """Get all AgentResult objects from this node, flattened if nested."""
        if isinstance(self.result, Exception):
            return []  # No agent results for exceptions
        elif isinstance(self.result, AgentResult):
            return [self.result]
        # else:
        #     # Flatten nested results from MultiAgentResult
        #     flattened = []
        #     for nested_node_result in self.result.results.values():
        #         if isinstance(nested_node_result, NodeResult):
        #             flattened.extend(nested_node_result.get_agent_results())
        #     return flattened

        if getattr(self.result, "__class__", None) and self.result.__class__.__name__ == "AgentResult":
            return [self.result]  # type: ignore[list-item]

        # If this is a nested MultiAgentResult, flatten children
        if hasattr(self.result, "results") and isinstance(self.result.results, dict):
            flattened: list["AgentResult"] = []
            for nested in self.result.results.values():
                if isinstance(nested, NodeResult):
                    flattened.extend(nested.get_agent_results())
            return flattened

        return []


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

    def _call_hook_safely(self, event_object: object) -> None:
        """Invoke hook callbacks and swallow hook errors.

        Args:
            event_object: The event to dispatch to registered callbacks.
        """
        try:
            self.hooks.invoke_callbacks(event_object)  # type: ignore
        except Exception as e:
            logger.exception("Hook invocation failed for %s: %s", type(event_object).__name__, e)

    @abstractmethod
    def get_state_from_orchestrator(self) -> dict:
        """Return a JSON-serializable snapshot of the orchestrator state."""
        raise NotImplementedError

    @abstractmethod
    def apply_state_from_dict(self, payload: dict) -> None:
        """Restore orchestrator state from a session dict."""
        raise NotImplementedError

    def summarize_node_result_for_persist(self, raw: NodeResult) -> dict[str, Any]:
        """Summarize node result for efficient persistence.

        Args:
            raw: Raw node result to summarize

        Returns:
            Normalized dict with 'agent_outputs' key containing string results
        """

        def _extract_text_from_agent_result(ar: Any) -> str:
            try:
                msg = getattr(ar, "message", None)
                if isinstance(msg, dict):
                    blocks = msg.get("content") or []
                    texts = []
                    for b in blocks:
                        t = b.get("text")
                        if t:
                            texts.append(str(t))
                    if texts:
                        return "\n".join(texts)
                return str(ar)
            except Exception:
                return str(ar)

        # If it's a NodeResult with AgentResults, flatten
        if hasattr(raw, "get_agent_results") and callable(raw.get_agent_results):
            try:
                ars = raw.get_agent_results()  # list[AgentResult]
                if ars:
                    return {"agent_outputs": [_extract_text_from_agent_result(a) for a in ars]}
            except Exception:
                pass

        # If already normalized
        if isinstance(raw, dict) and isinstance(raw.get("agent_outputs"), list):
            return {"agent_outputs": [str(x) for x in raw["agent_outputs"]]}

        # Fallback
        return {"agent_outputs": [str(raw)]}
