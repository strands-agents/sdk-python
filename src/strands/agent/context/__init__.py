"""Intelligent context management for optimized agent interactions.

This module provides advanced context management capabilities including:
- Dynamic tool selection based on context and task requirements
- Context window optimization and intelligent pruning
- Tool usage analytics and performance tracking
- Relevance-based filtering and scoring

The context management system works alongside the memory management
to provide efficient and intelligent agent interactions.
"""

from .analytics import ToolUsageAnalytics, ToolUsageStats
from .context_optimizer import ContextOptimizer, ContextWindow
from .relevance_scoring import RelevanceScorer, SimilarityMetric
from .tool_manager import DynamicToolManager, ToolSelectionCriteria

__all__ = [
    "DynamicToolManager",
    "ToolSelectionCriteria",
    "ContextOptimizer",
    "ContextWindow",
    "RelevanceScorer",
    "SimilarityMetric",
    "ToolUsageAnalytics",
    "ToolUsageStats",
]
