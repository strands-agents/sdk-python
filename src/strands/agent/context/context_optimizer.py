"""Context window optimization and intelligent pruning."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .relevance_scoring import ContextRelevanceFilter, RelevanceScorer, TextRelevanceScorer


@dataclass
class ContextItem:
    """A single item in the context window."""

    key: str
    value: Any
    size: int  # Estimated size in tokens/characters
    relevance_score: float = 0.0
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContextWindow:
    """Represents an optimized context window."""

    items: List[ContextItem]
    total_size: int
    max_size: int
    optimization_stats: Dict[str, Any]

    @property
    def utilization(self) -> float:
        """Calculate context window utilization."""
        if self.max_size == 0:
            return 0.0
        return self.total_size / self.max_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert context window to dictionary format."""
        return {item.key: item.value for item in self.items}


class ContextOptimizer:
    """Optimizes context windows for efficient agent interactions."""

    def __init__(
        self, max_context_size: int = 8192, relevance_threshold: float = 0.3, scorer: Optional[RelevanceScorer] = None
    ):
        """Initialize context optimizer.

        Args:
            max_context_size: Maximum context size in tokens/characters
            relevance_threshold: Minimum relevance score to include
            scorer: Relevance scorer to use
        """
        self.max_context_size = max_context_size
        self.relevance_threshold = relevance_threshold
        self.scorer = scorer or TextRelevanceScorer()
        self.relevance_filter = ContextRelevanceFilter(self.scorer)

    def optimize_context(
        self,
        context_items: Dict[str, Any],
        task_description: str,
        required_keys: Optional[List[str]] = None,
        size_estimator: Optional[Callable[[Any], int]] = None,
    ) -> ContextWindow:
        """Optimize context window for a specific task.

        Args:
            context_items: All available context items
            task_description: Description of the task/query
            required_keys: Keys that must be included
            size_estimator: Optional function to estimate item size

        Returns:
            Optimized context window
        """
        # Initialize size estimator
        if size_estimator is None:
            size_estimator = self._estimate_size

        # Score all items
        scored_items = []
        for key, value in context_items.items():
            relevance = self.scorer.score(value, task_description)
            size = size_estimator(value)

            item = ContextItem(key=key, value=value, size=size, relevance_score=relevance)
            scored_items.append(item)

        # Separate required and optional items
        required_items = []
        optional_items = []

        for item in scored_items:
            if required_keys and item.key in required_keys:
                required_items.append(item)
            else:
                optional_items.append(item)

        # Build optimized context
        optimized_items = self._build_optimized_context(required_items, optional_items, self.max_context_size)

        # Calculate total size
        total_size = sum(item.size for item in optimized_items)

        # Generate optimization stats
        stats = {
            "original_items": len(context_items),
            "optimized_items": len(optimized_items),
            "original_size": sum(item.size for item in scored_items),
            "optimized_size": total_size,
            "pruning_ratio": 1.0 - (len(optimized_items) / len(context_items)) if context_items else 0.0,
            "avg_relevance": (
                sum(item.relevance_score for item in optimized_items) / len(optimized_items) if optimized_items else 0.0
            ),
        }

        return ContextWindow(
            items=optimized_items, total_size=total_size, max_size=self.max_context_size, optimization_stats=stats
        )

    def _build_optimized_context(
        self, required_items: List[ContextItem], optional_items: List[ContextItem], max_size: int
    ) -> List[ContextItem]:
        """Build optimized context with required and optional items.

        Args:
            required_items: Items that must be included
            optional_items: Items that may be included based on relevance
            max_size: Maximum total size

        Returns:
            List of items for optimized context
        """
        # Start with required items, compressing if needed
        context_items = []
        current_size = 0

        # First add required items, compressing if they're too large
        for item in required_items:
            if item.size <= max_size:
                context_items.append(item)
                current_size += item.size
            else:
                # Required item is too large, must compress
                compressed_item = self._try_compress_item(item, max_size)
                if compressed_item:
                    context_items.append(compressed_item)
                    current_size += compressed_item.size
                else:
                    # Can't compress enough, still add it (will exceed limit)
                    context_items.append(item)
                    current_size += item.size

        # Sort optional items by relevance
        optional_items.sort(key=lambda x: x.relevance_score, reverse=True)

        # Add optional items that fit and meet threshold
        for item in optional_items:
            if item.relevance_score >= self.relevance_threshold:
                if current_size + item.size <= max_size:
                    context_items.append(item)
                    current_size += item.size
                else:
                    # Try compression strategies
                    compressed_item = self._try_compress_item(item, max_size - current_size)
                    if compressed_item and compressed_item.size <= max_size - current_size:
                        context_items.append(compressed_item)
                        current_size += compressed_item.size

        return context_items

    def _try_compress_item(self, item: ContextItem, target_size: int) -> Optional[ContextItem]:
        """Try to compress an item to fit within target size.

        Args:
            item: Item to compress
            target_size: Target size limit

        Returns:
            Compressed item or None if compression not possible
        """
        if item.size <= target_size:
            return item

        # Simple truncation strategy for strings
        if isinstance(item.value, str):
            # Estimate characters per token (rough approximation)
            chars_per_token = 4
            target_chars = target_size * chars_per_token

            if len(item.value) > target_chars:
                truncated_value = item.value[:target_chars] + "..."
                return ContextItem(
                    key=item.key,
                    value=truncated_value,
                    size=self._estimate_size(truncated_value),
                    relevance_score=item.relevance_score * 0.8,  # Reduce score for truncated
                    metadata={"truncated": True, "original_size": item.size},
                )

        # For other types, we can't easily compress
        return None

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of a value in tokens.

        Args:
            value: Value to estimate

        Returns:
            Estimated size in tokens
        """
        # Simple character-based estimation
        # Rough approximation: 1 token ≈ 4 characters
        if isinstance(value, str):
            return max(1, len(value) // 4) if value else 0
        elif isinstance(value, dict):
            json_str = json.dumps(value)
            return max(1, len(json_str) // 4) if json_str else 0
        elif isinstance(value, list):
            json_str = json.dumps(value)
            return max(1, len(json_str) // 4) if json_str else 0
        else:
            str_value = str(value)
            return max(1, len(str_value) // 4) if str_value else 0

    def merge_contexts(self, contexts: List[ContextWindow], task_description: str) -> ContextWindow:
        """Merge multiple context windows into one optimized window.

        Args:
            contexts: List of context windows to merge
            task_description: Task description for relevance scoring

        Returns:
            Merged and optimized context window
        """
        # Collect all items
        all_items: Dict[str, Any] = {}
        item_relevance: Dict[str, float] = {}
        for context in contexts:
            for item in context.items:
                # Use highest relevance score if duplicate keys
                if item.key in all_items:
                    if item.relevance_score > item_relevance[item.key]:
                        all_items[item.key] = item.value
                        item_relevance[item.key] = item.relevance_score
                else:
                    all_items[item.key] = item.value
                    item_relevance[item.key] = item.relevance_score

        # Re-optimize merged context
        return self.optimize_context(all_items, task_description)

    def get_pruning_recommendations(self, context_window: ContextWindow) -> List[Tuple[str, str]]:
        """Get recommendations for further context pruning.

        Args:
            context_window: Current context window

        Returns:
            List of (item_key, recommendation) tuples
        """
        recommendations = []

        for item in context_window.items:
            if item.relevance_score < 0.5:
                recommendations.append((item.key, f"Low relevance ({item.relevance_score:.2f}), consider removing"))
            elif item.size > context_window.max_size * 0.2:
                recommendations.append((item.key, f"Large item ({item.size} tokens), consider summarizing"))
            elif item.metadata and item.metadata.get("truncated"):
                recommendations.append((item.key, "Item was truncated, consider using summary instead"))

        return recommendations
