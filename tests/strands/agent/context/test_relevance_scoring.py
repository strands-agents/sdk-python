"""Tests for relevance scoring functionality."""

from strands.agent.context.relevance_scoring import (
    ContextRelevanceFilter,
    ScoredItem,
    SimilarityMetric,
    TextRelevanceScorer,
    ToolRelevanceScorer,
)


class TestTextRelevanceScorer:
    """Tests for TextRelevanceScorer."""

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical texts."""
        scorer = TextRelevanceScorer(metric=SimilarityMetric.JACCARD)
        score = scorer.score("hello world", "hello world")
        assert score == 1.0

    def test_jaccard_similarity_partial(self):
        """Test Jaccard similarity for partially overlapping texts."""
        scorer = TextRelevanceScorer(metric=SimilarityMetric.JACCARD)
        score = scorer.score("hello world", "hello universe")
        # Intersection: {hello}, Union: {hello, world, universe}
        # Score: 1/3 ≈ 0.333
        assert 0.3 < score < 0.4

    def test_jaccard_similarity_no_overlap(self):
        """Test Jaccard similarity for non-overlapping texts."""
        scorer = TextRelevanceScorer(metric=SimilarityMetric.JACCARD)
        score = scorer.score("hello world", "foo bar")
        assert score == 0.0

    def test_jaccard_similarity_empty(self):
        """Test Jaccard similarity with empty strings."""
        scorer = TextRelevanceScorer(metric=SimilarityMetric.JACCARD)
        assert scorer.score("", "") == 1.0
        assert scorer.score("hello", "") == 0.0
        assert scorer.score("", "world") == 0.0

    def test_levenshtein_similarity_identical(self):
        """Test Levenshtein similarity for identical texts."""
        scorer = TextRelevanceScorer(metric=SimilarityMetric.LEVENSHTEIN)
        score = scorer.score("hello", "hello")
        assert score == 1.0

    def test_levenshtein_similarity_one_edit(self):
        """Test Levenshtein similarity with one character difference."""
        scorer = TextRelevanceScorer(metric=SimilarityMetric.LEVENSHTEIN)
        score = scorer.score("hello", "hallo")
        # 1 edit in 5 characters = 0.8 similarity
        assert 0.79 < score < 0.81

    def test_text_conversion(self):
        """Test conversion of different types to text."""
        scorer = TextRelevanceScorer()

        # Dict conversion
        dict_score = scorer.score({"key": "value", "number": 42}, '{"key": "value"}')
        assert dict_score > 0

        # List conversion
        list_score = scorer.score([1, 2, 3], "[1, 2, 3]")
        assert list_score > 0

        # Number conversion
        number_score = scorer.score(42, "42")
        assert number_score > 0


class TestToolRelevanceScorer:
    """Tests for ToolRelevanceScorer."""

    def test_tool_scoring_by_name(self):
        """Test tool scoring based on name matching."""
        scorer = ToolRelevanceScorer()

        tool_info = {"name": "file_reader", "description": "Reads files from the filesystem"}

        context = {"task": "I need to read a file", "requirements": "file reading capability"}

        score = scorer.score(tool_info, context)
        assert score > 0.1  # Should have some relevance due to partial matches

    def test_tool_scoring_by_description(self):
        """Test tool scoring based on description matching."""
        scorer = ToolRelevanceScorer()

        tool_info = {"name": "tool_x", "description": "Performs database queries and data analysis"}

        context = {"task": "analyze data from database", "requirements": "need to query and analyze data"}

        score = scorer.score(tool_info, context)
        assert score > 0.1  # Should have some relevance due to partial matches

    def test_required_tools_max_score(self):
        """Test that required tools get maximum score."""
        scorer = ToolRelevanceScorer()

        tool_info = {"name": "calculator", "description": "Basic math operations"}

        context = {"task": "perform calculations", "required_tools": ["calculator", "converter"]}

        score = scorer.score(tool_info, context)
        assert score == 1.0  # Required tool should get max score

    def test_tool_object_extraction(self):
        """Test extraction from tool objects."""
        scorer = ToolRelevanceScorer()

        # Mock tool object
        class MockTool:
            name = "test_tool"
            description = "A test tool"
            parameters = {"param1": "string"}

        context = {"task": "test something"}
        score = scorer.score(MockTool(), context)
        assert score > 0  # Should be able to score tool object


class TestContextRelevanceFilter:
    """Tests for ContextRelevanceFilter."""

    def test_filter_by_min_score(self):
        """Test filtering items by minimum score."""
        scorer = TextRelevanceScorer()
        filter = ContextRelevanceFilter(scorer)

        items = {"item1": "hello world", "item2": "foo bar", "item3": "hello universe"}

        context = "hello world programming"

        filtered = filter.filter_relevant(items, context, min_score=0.2)

        # Should include item1 and item3 (both contain "hello")
        assert len(filtered) >= 2
        assert any(item.key == "item1" for item in filtered)
        assert any(item.key == "item3" for item in filtered)

        # item2 should have low/zero score
        item2_scores = [item.score for item in filtered if item.key == "item2"]
        if item2_scores:
            assert item2_scores[0] < 0.2

    def test_filter_max_items(self):
        """Test limiting number of returned items."""
        scorer = TextRelevanceScorer()
        filter = ContextRelevanceFilter(scorer)

        items = {f"item{i}": f"test content {i}" for i in range(10)}
        context = "test content"

        filtered = filter.filter_relevant(items, context, min_score=0.0, max_items=3)

        assert len(filtered) == 3

    def test_filter_sorting(self):
        """Test that items are sorted by relevance score."""
        scorer = TextRelevanceScorer()
        filter = ContextRelevanceFilter(scorer)

        items = {"exact": "hello world", "partial": "hello", "none": "foo bar"}

        context = "hello world"

        filtered = filter.filter_relevant(items, context, min_score=0.0)

        # Should be sorted by score descending
        assert filtered[0].key == "exact"  # Highest score
        assert filtered[0].score > filtered[1].score
        if len(filtered) > 2:
            assert filtered[1].score >= filtered[2].score

    def test_get_top_k(self):
        """Test getting top-k items."""
        scorer = TextRelevanceScorer()
        filter = ContextRelevanceFilter(scorer)

        items = {f"item{i}": f"content {i % 3}" for i in range(10)}
        context = "content 1"

        top_3 = filter.get_top_k(items, context, k=3)

        assert len(top_3) == 3
        # All should have scores (even if 0)
        assert all(hasattr(item, "score") for item in top_3)
        # Should be sorted by score
        assert top_3[0].score >= top_3[1].score >= top_3[2].score


class TestScoredItem:
    """Tests for ScoredItem dataclass."""

    def test_scored_item_creation(self):
        """Test creating a scored item."""
        item = ScoredItem(key="test_key", value={"data": "test"}, score=0.75, metadata={"source": "test"})

        assert item.key == "test_key"
        assert item.value == {"data": "test"}
        assert item.score == 0.75
        assert item.metadata == {"source": "test"}

    def test_scored_item_defaults(self):
        """Test scored item with default metadata."""
        item = ScoredItem(key="test", value="value", score=0.5)
        assert item.metadata is None
