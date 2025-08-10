"""Tool usage analytics and performance tracking."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolUsageStats:
    """Statistics for a single tool's usage."""

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    last_used: Optional[float] = None
    avg_relevance_score: float = 0.0
    relevance_scores: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of tool calls."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time per call."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_execution_time / self.successful_calls

    def record_usage(self, success: bool, execution_time: float, relevance_score: Optional[float] = None) -> None:
        """Record a tool usage event.

        Args:
            success: Whether the tool call was successful
            execution_time: Time taken for execution in seconds
            relevance_score: Relevance score if available
        """
        self.total_calls += 1
        self.last_used = time.time()

        if success:
            self.successful_calls += 1
            self.total_execution_time += execution_time
        else:
            self.failed_calls += 1

        if relevance_score is not None:
            self.relevance_scores.append(relevance_score)
            # Update running average
            self.avg_relevance_score = sum(self.relevance_scores) / len(self.relevance_scores)


@dataclass
class ContextPerformanceStats:
    """Performance statistics for context management."""

    total_context_builds: int = 0
    total_pruning_operations: int = 0
    avg_context_size: float = 0.0
    avg_pruning_ratio: float = 0.0
    context_sizes: List[int] = field(default_factory=list)
    pruning_ratios: List[float] = field(default_factory=list)

    def record_context_build(self, context_size: int, original_size: int) -> None:
        """Record a context building operation.

        Args:
            context_size: Final context size
            original_size: Original size before optimization
        """
        self.total_context_builds += 1
        self.context_sizes.append(context_size)

        if original_size > 0:
            pruning_ratio = 1.0 - (context_size / original_size)
            self.pruning_ratios.append(pruning_ratio)
            self.total_pruning_operations += 1

        # Update averages
        self.avg_context_size = sum(self.context_sizes) / len(self.context_sizes)
        if self.pruning_ratios:
            self.avg_pruning_ratio = sum(self.pruning_ratios) / len(self.pruning_ratios)


class ToolUsageAnalytics:
    """Tracks and analyzes tool usage patterns for optimization."""

    def __init__(self) -> None:
        """Initialize tool usage analytics."""
        self.tool_stats: Dict[str, ToolUsageStats] = {}
        self.context_stats = ContextPerformanceStats()
        self._start_time = time.time()

    def get_tool_stats(self, tool_name: str) -> ToolUsageStats:
        """Get or create stats for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool usage statistics
        """
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolUsageStats(tool_name=tool_name)
        return self.tool_stats[tool_name]

    def record_tool_usage(
        self, tool_name: str, success: bool, execution_time: float, relevance_score: Optional[float] = None
    ) -> None:
        """Record a tool usage event.

        Args:
            tool_name: Name of the tool used
            success: Whether the execution was successful
            execution_time: Time taken for execution
            relevance_score: Relevance score if available
        """
        stats = self.get_tool_stats(tool_name)
        stats.record_usage(success, execution_time, relevance_score)

    def record_context_build(self, context_size: int, original_size: int) -> None:
        """Record context building statistics.

        Args:
            context_size: Final optimized context size
            original_size: Original context size before optimization
        """
        self.context_stats.record_context_build(context_size, original_size)

    def get_tool_rankings(self, min_calls: int = 5) -> List[Tuple[str, float]]:
        """Get tools ranked by performance score.

        Args:
            min_calls: Minimum calls required for ranking

        Returns:
            List of (tool_name, score) tuples sorted by score
        """
        rankings = []

        for tool_name, stats in self.tool_stats.items():
            if stats.total_calls >= min_calls:
                # Composite score based on success rate, relevance, and recency
                recency_factor = self._calculate_recency_factor(stats.last_used)
                performance_score = 0.4 * stats.success_rate + 0.4 * stats.avg_relevance_score + 0.2 * recency_factor
                rankings.append((tool_name, performance_score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _calculate_recency_factor(self, last_used: Optional[float]) -> float:
        """Calculate recency factor for tool usage.

        Args:
            last_used: Timestamp of last usage

        Returns:
            Recency factor between 0.0 and 1.0
        """
        if last_used is None:
            return 0.0

        # Decay over 24 hours
        time_since_use = time.time() - last_used
        decay_period = 24 * 3600  # 24 hours in seconds

        return max(0.0, 1.0 - (time_since_use / decay_period))

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics summary.

        Returns:
            Dictionary containing analytics summary
        """
        total_tools = len(self.tool_stats)
        total_calls = sum(stats.total_calls for stats in self.tool_stats.values())

        if total_calls > 0:
            overall_success_rate = sum(stats.successful_calls for stats in self.tool_stats.values()) / total_calls
        else:
            overall_success_rate = 0.0

        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_tools_used": total_tools,
            "total_tool_calls": total_calls,
            "overall_success_rate": overall_success_rate,
            "context_optimization": {
                "total_builds": self.context_stats.total_context_builds,
                "avg_context_size": self.context_stats.avg_context_size,
                "avg_pruning_ratio": self.context_stats.avg_pruning_ratio,
            },
            "top_tools": self.get_tool_rankings()[:5],
        }

    def reset_stats(self) -> None:
        """Reset all analytics data."""
        self.tool_stats.clear()
        self.context_stats = ContextPerformanceStats()
        self._start_time = time.time()
