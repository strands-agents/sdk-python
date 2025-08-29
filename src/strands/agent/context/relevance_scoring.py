"""Relevance scoring and similarity calculations for context management."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class SimilarityMetric(Enum):
    """Available similarity metrics for relevance scoring."""

    JACCARD = "jaccard"
    COSINE = "cosine"
    LEVENSHTEIN = "levenshtein"
    SEMANTIC = "semantic"  # Future: requires embedding model


@dataclass
class ScoredItem:
    """An item with its relevance score."""

    key: str
    value: Any
    score: float
    metadata: Optional[Dict[str, Any]] = None


class RelevanceScorer(ABC):
    """Base class for relevance scoring implementations."""

    @abstractmethod
    def score(self, item: Any, context: Any) -> float:
        """Calculate relevance score between item and context.

        Args:
            item: The item to score
            context: The context to score against

        Returns:
            Relevance score between 0.0 and 1.0
        """
        pass


class TextRelevanceScorer(RelevanceScorer):
    """Relevance scorer for text-based content using string similarity."""

    def __init__(self, metric: SimilarityMetric = SimilarityMetric.JACCARD):
        """Initialize text relevance scorer.

        Args:
            metric: The similarity metric to use
        """
        self.metric = metric

    def score(self, item: Any, context: Any) -> float:
        """Calculate text relevance score.

        Args:
            item: Text item to score
            context: Context text to score against

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Convert to strings
        item_text = self._to_text(item)
        context_text = self._to_text(context)

        if self.metric == SimilarityMetric.JACCARD:
            return self._jaccard_similarity(item_text, context_text)
        elif self.metric == SimilarityMetric.LEVENSHTEIN:
            return self._levenshtein_similarity(item_text, context_text)
        else:
            # Default to Jaccard
            return self._jaccard_similarity(item_text, context_text)

    def _to_text(self, value: Any) -> str:
        """Convert any value to text representation."""
        if isinstance(value, str):
            return value
        elif isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        else:
            return str(value)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        # Tokenize by words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        # Direct intersection
        intersection = words1.intersection(words2)

        # Also count partial matches (e.g., "read" and "reads", "file" and "files")
        partial_matches = 0.0
        for w1 in words1:
            for w2 in words2:
                if w1 != w2 and (w1 in w2 or w2 in w1) and min(len(w1), len(w2)) >= 3:
                    partial_matches += 0.5
                    break

        union = words1.union(words2)

        # Calculate score with partial matches
        score = (len(intersection) + partial_matches) / len(union)
        return min(1.0, score)

    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        distance = self._levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row.copy()

        return previous_row[-1]


class ToolRelevanceScorer(RelevanceScorer):
    """Relevance scorer specifically for tool selection."""

    def __init__(self, text_scorer: Optional[TextRelevanceScorer] = None):
        """Initialize tool relevance scorer.

        Args:
            text_scorer: Text scorer to use for description matching
        """
        self.text_scorer = text_scorer or TextRelevanceScorer()

    def score(self, item: Any, context: Any) -> float:
        """Score tool relevance based on tool metadata and context.

        Args:
            item: Tool or tool metadata
            context: Task context or requirements

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Extract tool information
        tool_info = self._extract_tool_info(item)
        context_info = self._extract_context_info(context)

        # Calculate component scores
        tool_full_text = f"{tool_info.get('name', '')} {tool_info.get('description', '')}"
        context_full_text = f"{context_info.get('task', '')} {context_info.get('requirements', '')}"

        # Score tool against full context
        full_score = self.text_scorer.score(tool_full_text, context_full_text)

        # Also calculate individual component scores for fine-tuning
        name_score = self.text_scorer.score(tool_info.get("name", ""), context_info.get("task", ""))
        description_score = self.text_scorer.score(
            tool_info.get("description", ""), context_info.get("requirements", "")
        )

        # Check for explicit tool mentions
        if "required_tools" in context_info:
            required = context_info["required_tools"]
            if tool_info.get("name") in required:
                return 1.0  # Maximum relevance for required tools

        # Weighted combination with emphasis on full text match
        return 0.5 * full_score + 0.2 * name_score + 0.3 * description_score

    def _extract_tool_info(self, item: Any) -> Dict[str, Any]:
        """Extract relevant information from tool object."""
        if isinstance(item, dict):
            return item

        # Handle tool objects
        info = {}
        if hasattr(item, "tool_name"):
            info["name"] = item.tool_name
        elif hasattr(item, "name"):
            info["name"] = item.name

        if hasattr(item, "tool_spec"):
            info["description"] = item.tool_spec.get("description", "")
        elif hasattr(item, "description"):
            info["description"] = item.description

        if hasattr(item, "parameters"):
            info["parameters"] = item.parameters

        return info

    def _extract_context_info(self, context: Any) -> Dict[str, Any]:
        """Extract relevant information from context."""
        if isinstance(context, dict):
            return context
        elif isinstance(context, str):
            return {"task": context, "requirements": context}
        else:
            return {"task": str(context)}


class ContextRelevanceFilter:
    """Filters and ranks items based on relevance to context."""

    def __init__(self, scorer: RelevanceScorer):
        """Initialize relevance filter.

        Args:
            scorer: The relevance scorer to use
        """
        self.scorer = scorer

    def filter_relevant(
        self, items: Dict[str, Any], context: Any, min_score: float = 0.3, max_items: Optional[int] = None
    ) -> List[ScoredItem]:
        """Filter items by relevance score.

        Args:
            items: Dictionary of items to filter
            context: Context to score against
            min_score: Minimum relevance score threshold
            max_items: Maximum number of items to return

        Returns:
            List of scored items sorted by relevance
        """
        scored_items = []

        for key, value in items.items():
            score = self.scorer.score(value, context)
            if score >= min_score:
                scored_items.append(ScoredItem(key=key, value=value, score=score))

        # Sort by score descending
        scored_items.sort(key=lambda x: x.score, reverse=True)

        if max_items is not None:
            scored_items = scored_items[:max_items]

        return scored_items

    def get_top_k(self, items: Dict[str, Any], context: Any, k: int = 5) -> List[ScoredItem]:
        """Get top-k most relevant items.

        Args:
            items: Dictionary of items to rank
            context: Context to score against
            k: Number of top items to return

        Returns:
            Top k items by relevance score
        """
        return self.filter_relevant(items, context, min_score=0.0, max_items=k)
