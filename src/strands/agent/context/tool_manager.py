"""Dynamic tool management with intelligent selection and filtering."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ...types.tools import AgentTool as Tool
from .analytics import ToolUsageAnalytics
from .relevance_scoring import ToolRelevanceScorer


@dataclass
class ToolSelectionCriteria:
    """Criteria for tool selection."""

    task_description: str
    required_capabilities: Optional[List[str]] = None
    excluded_tools: Optional[Set[str]] = None
    max_tools: int = 20
    min_relevance_score: float = 0.2
    prefer_recent: bool = True
    context_hints: Optional[Dict[str, Any]] = None


@dataclass
class ToolSelectionResult:
    """Result of tool selection process."""

    selected_tools: List[Tool]
    relevance_scores: Dict[str, float]
    selection_reasoning: Dict[str, str]
    total_candidates: int
    selection_time: float


class DynamicToolManager:
    """Manages dynamic tool selection based on context and performance."""

    def __init__(self, analytics: Optional[ToolUsageAnalytics] = None, scorer: Optional[ToolRelevanceScorer] = None):
        """Initialize dynamic tool manager.

        Args:
            analytics: Tool usage analytics instance
            scorer: Tool relevance scorer
        """
        self.analytics = analytics or ToolUsageAnalytics()
        self.scorer = scorer or ToolRelevanceScorer()
        self._tool_cache: Dict[str, Tool] = {}

    def select_tools(self, available_tools: List[Tool], criteria: ToolSelectionCriteria) -> ToolSelectionResult:
        """Select optimal tools based on criteria.

        Args:
            available_tools: List of all available tools
            criteria: Selection criteria

        Returns:
            Tool selection result with selected tools and metadata
        """
        start_time = time.time()

        # Update tool cache
        self._update_tool_cache(available_tools)

        # Filter excluded tools
        candidate_tools = [
            tool
            for tool in available_tools
            if not criteria.excluded_tools or tool.tool_name not in criteria.excluded_tools
        ]

        # Score tools based on relevance
        scored_tools = self._score_tools(candidate_tools, criteria)

        # Apply performance-based adjustments
        if criteria.prefer_recent:
            scored_tools = self._adjust_scores_by_performance(scored_tools)

        # Filter by minimum score and capability requirements
        filtered_tools = self._filter_tools(scored_tools, criteria)

        # Select top tools up to max_tools limit
        selected = filtered_tools[: criteria.max_tools]

        # Generate selection reasoning
        reasoning = self._generate_selection_reasoning(selected, scored_tools, criteria)

        # Record analytics
        selection_time = time.time() - start_time

        return ToolSelectionResult(
            selected_tools=[tool for tool, _ in selected],
            relevance_scores={tool.tool_name: score for tool, score in selected},
            selection_reasoning=reasoning,
            total_candidates=len(available_tools),
            selection_time=selection_time,
        )

    def _update_tool_cache(self, tools: List[Tool]) -> None:
        """Update internal tool cache."""
        for tool in tools:
            self._tool_cache[tool.tool_name] = tool

    def _score_tools(self, tools: List[Tool], criteria: ToolSelectionCriteria) -> List[Tuple[Tool, float]]:
        """Score tools based on relevance to criteria.

        Args:
            tools: List of tools to score
            criteria: Selection criteria

        Returns:
            List of (tool, score) tuples
        """
        scored = []

        # Prepare context for scoring
        context = {
            "task": criteria.task_description,
            "requirements": criteria.task_description,
        }

        if criteria.required_capabilities:
            context["required_capabilities"] = ", ".join(criteria.required_capabilities)

        if criteria.context_hints:
            context.update(criteria.context_hints)

        for tool in tools:
            # Create tool info for scoring
            tool_info = {
                "name": tool.tool_name,
                "description": tool.tool_spec.get("description", ""),
            }

            # Calculate relevance score
            score = self.scorer.score(tool_info, context)

            # Boost score for tools with required capabilities
            if criteria.required_capabilities:
                if self._has_required_capabilities(tool, criteria.required_capabilities):
                    score = min(1.0, score * 1.5)

            scored.append((tool, score))

            # Record relevance score in analytics
            self.analytics.record_tool_usage(
                tool.tool_name,
                success=True,  # Just recording relevance, not actual usage
                execution_time=0.0,
                relevance_score=score,
            )

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _adjust_scores_by_performance(self, scored_tools: List[Tuple[Tool, float]]) -> List[Tuple[Tool, float]]:
        """Adjust scores based on historical performance.

        Args:
            scored_tools: List of (tool, score) tuples

        Returns:
            Adjusted list of (tool, score) tuples
        """
        adjusted = []

        for tool, base_score in scored_tools:
            stats = self.analytics.get_tool_stats(tool.tool_name)

            # Calculate performance multiplier
            if stats.total_calls >= 5:  # Minimum calls for reliable stats
                performance_factor = 0.7 * stats.success_rate + 0.3 * stats.avg_relevance_score

                # Adjust score with performance factor
                # Limit adjustment to ±30%
                adjustment = 0.7 + (0.6 * performance_factor)
                adjusted_score = base_score * adjustment
            else:
                # No adjustment for tools with insufficient data
                adjusted_score = base_score

            adjusted.append((tool, min(1.0, adjusted_score)))

        # Re-sort by adjusted scores
        adjusted.sort(key=lambda x: x[1], reverse=True)

        return adjusted

    def _filter_tools(
        self, scored_tools: List[Tuple[Tool, float]], criteria: ToolSelectionCriteria
    ) -> List[Tuple[Tool, float]]:
        """Filter tools based on criteria.

        Args:
            scored_tools: List of (tool, score) tuples
            criteria: Selection criteria

        Returns:
            Filtered list of (tool, score) tuples
        """
        filtered = []

        for tool, score in scored_tools:
            # Check minimum score
            if score < criteria.min_relevance_score:
                continue

            # Check required capabilities
            if criteria.required_capabilities:
                if not self._has_required_capabilities(tool, criteria.required_capabilities):
                    continue

            filtered.append((tool, score))

        return filtered

    def _has_required_capabilities(self, tool: Tool, required_capabilities: List[str]) -> bool:
        """Check if tool has required capabilities.

        Args:
            tool: Tool to check
            required_capabilities: List of required capability keywords

        Returns:
            True if tool has all required capabilities
        """
        # Check tool name and description for capability keywords
        tool_text = f"{tool.tool_name} {tool.tool_spec.get('description', '')}".lower()

        for capability in required_capabilities:
            if capability.lower() not in tool_text:
                return False

        return True

    def _generate_selection_reasoning(
        self, selected: List[Tuple[Tool, float]], all_scored: List[Tuple[Tool, float]], criteria: ToolSelectionCriteria
    ) -> Dict[str, str]:
        """Generate reasoning for tool selection.

        Args:
            selected: Selected tools with scores
            all_scored: All scored tools
            criteria: Selection criteria

        Returns:
            Dictionary of reasoning by tool name
        """
        reasoning = {}

        for tool, score in selected:
            reasons = []

            # Relevance reasoning
            if score >= 0.8:
                reasons.append(f"High relevance to task ({score:.2f})")
            elif score >= 0.5:
                reasons.append(f"Good relevance to task ({score:.2f})")
            else:
                reasons.append(f"Moderate relevance to task ({score:.2f})")

            # Performance reasoning
            stats = self.analytics.get_tool_stats(tool.tool_name)
            if stats.total_calls >= 5:
                if stats.success_rate >= 0.9:
                    reasons.append(f"Excellent success rate ({stats.success_rate:.1%})")
                elif stats.success_rate >= 0.7:
                    reasons.append(f"Good success rate ({stats.success_rate:.1%})")

            # Capability reasoning
            if criteria.required_capabilities:
                matching_caps = [
                    cap
                    for cap in criteria.required_capabilities
                    if cap.lower() in f"{tool.tool_name} {tool.tool_spec.get('description', '')}".lower()
                ]
                if matching_caps:
                    reasons.append(f"Matches capabilities: {', '.join(matching_caps)}")

            reasoning[tool.tool_name] = "; ".join(reasons)

        return reasoning

    def get_tool_recommendations(
        self, task_description: str, recent_tools: Optional[List[str]] = None, max_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """Get tool recommendations based on task and history.

        Args:
            task_description: Description of the task
            recent_tools: Recently used tool names
            max_recommendations: Maximum recommendations to return

        Returns:
            List of (tool_name, confidence) tuples
        """
        recommendations = []

        # Get performance-based recommendations
        tool_rankings = self.analytics.get_tool_rankings(min_calls=3)

        for tool_name, performance_score in tool_rankings[: max_recommendations * 2]:
            if tool_name in self._tool_cache:
                tool = self._tool_cache[tool_name]

                # Calculate relevance to current task
                relevance = self.scorer.score(
                    {"name": tool.tool_name, "description": tool.tool_spec.get("description", "")},
                    {"task": task_description},
                )

                # Combine performance and relevance
                confidence = 0.6 * performance_score + 0.4 * relevance

                # Boost if recently used successfully
                if recent_tools and tool_name in recent_tools:
                    confidence = min(1.0, confidence * 1.2)

                recommendations.append((tool_name, confidence))

        # Sort by confidence and limit
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:max_recommendations]

    def update_tool_performance(self, tool_name: str, success: bool, execution_time: float) -> None:
        """Update tool performance metrics after usage.

        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
            execution_time: Time taken for execution
        """
        self.analytics.record_tool_usage(tool_name, success, execution_time)
