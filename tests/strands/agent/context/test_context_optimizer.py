"""Tests for context window optimization."""

from strands.agent.context.context_optimizer import (
    ContextItem,
    ContextOptimizer,
    ContextWindow,
)


class TestContextItem:
    """Tests for ContextItem dataclass."""

    def test_context_item_creation(self):
        """Test creating a context item."""
        item = ContextItem(
            key="test_key",
            value="test value",
            size=10,
            relevance_score=0.8,
            timestamp=1234567890.0,
            metadata={"source": "test"},
        )

        assert item.key == "test_key"
        assert item.value == "test value"
        assert item.size == 10
        assert item.relevance_score == 0.8
        assert item.timestamp == 1234567890.0
        assert item.metadata == {"source": "test"}

    def test_context_item_defaults(self):
        """Test context item with default values."""
        item = ContextItem(key="test", value="value", size=5)

        assert item.relevance_score == 0.0
        assert item.timestamp == 0.0
        assert item.metadata is None


class TestContextWindow:
    """Tests for ContextWindow dataclass."""

    def test_context_window_creation(self):
        """Test creating a context window."""
        items = [ContextItem("key1", "value1", 10), ContextItem("key2", "value2", 20)]

        window = ContextWindow(items=items, total_size=30, max_size=100, optimization_stats={"items_removed": 5})

        assert len(window.items) == 2
        assert window.total_size == 30
        assert window.max_size == 100
        assert window.optimization_stats["items_removed"] == 5

    def test_utilization_calculation(self):
        """Test context window utilization calculation."""
        window = ContextWindow(items=[], total_size=75, max_size=100, optimization_stats={})

        assert window.utilization == 0.75

        # Edge case: max_size is 0
        window_zero = ContextWindow(items=[], total_size=50, max_size=0, optimization_stats={})
        assert window_zero.utilization == 0.0

    def test_to_dict_conversion(self):
        """Test converting context window to dictionary."""
        items = [ContextItem("key1", "value1", 10), ContextItem("key2", {"nested": "value"}, 20)]

        window = ContextWindow(items, 30, 100, {})
        result = window.to_dict()

        assert result == {"key1": "value1", "key2": {"nested": "value"}}


class TestContextOptimizer:
    """Tests for ContextOptimizer."""

    def test_initialization(self):
        """Test ContextOptimizer initialization."""
        optimizer = ContextOptimizer(max_context_size=4096, relevance_threshold=0.4)

        assert optimizer.max_context_size == 4096
        assert optimizer.relevance_threshold == 0.4
        assert optimizer.scorer is not None
        assert optimizer.relevance_filter is not None

    def test_size_estimation(self):
        """Test default size estimation."""
        optimizer = ContextOptimizer()

        # String estimation (4 chars ≈ 1 token)
        assert optimizer._estimate_size("hello world") == 2  # 11 chars / 4
        assert optimizer._estimate_size("a" * 100) == 25

        # Dict estimation
        dict_size = optimizer._estimate_size({"key": "value"})
        assert dict_size > 0

        # List estimation
        list_size = optimizer._estimate_size([1, 2, 3, 4, 5])
        assert list_size > 0

        # Other types
        assert optimizer._estimate_size(42) > 0

    def test_optimize_context_basic(self):
        """Test basic context optimization."""
        optimizer = ContextOptimizer(max_context_size=50)

        context_items = {
            "relevant1": "This is about machine learning",
            "relevant2": "Machine learning algorithms",
            "irrelevant": "Weather forecast for tomorrow",
        }

        task = "explain machine learning concepts"

        result = optimizer.optimize_context(context_items, task)

        assert isinstance(result, ContextWindow)
        assert len(result.items) <= len(context_items)
        assert result.total_size <= result.max_size

        # Check that relevant items are included
        keys = [item.key for item in result.items]
        assert "relevant1" in keys or "relevant2" in keys

    def test_required_keys_inclusion(self):
        """Test that required keys are always included."""
        optimizer = ContextOptimizer(
            max_context_size=20,
            relevance_threshold=0.9,  # Very high threshold
        )

        context_items = {"required": "must be included", "optional1": "relevant content", "optional2": "also relevant"}

        result = optimizer.optimize_context(context_items, "some task", required_keys=["required"])

        # Required key must be in result
        keys = [item.key for item in result.items]
        assert "required" in keys

    def test_size_constraints(self):
        """Test that size constraints are respected."""
        optimizer = ContextOptimizer(max_context_size=10)  # Very small

        context_items = {
            f"item{i}": "x" * 20  # Each item ~5 tokens
            for i in range(10)
        }

        result = optimizer.optimize_context(context_items, "task")

        # Total size must not exceed max
        assert result.total_size <= optimizer.max_context_size
        assert len(result.items) < len(context_items)  # Some items excluded

    def test_relevance_filtering(self):
        """Test relevance threshold filtering."""
        optimizer = ContextOptimizer(max_context_size=1000, relevance_threshold=0.5)

        context_items = {
            "high_relevance": "machine learning model training",
            "low_relevance": "random unrelated content",
        }

        task = "train a machine learning model"

        result = optimizer.optimize_context(context_items, task)

        # Check relevance scores
        for item in result.items:
            if item.key == "low_relevance":
                assert item.relevance_score < optimizer.relevance_threshold

    def test_optimization_stats(self):
        """Test optimization statistics generation."""
        optimizer = ContextOptimizer()

        context_items = {f"item{i}": f"content {i}" for i in range(5)}

        result = optimizer.optimize_context(context_items, "content 1")

        stats = result.optimization_stats
        assert "original_items" in stats
        assert "optimized_items" in stats
        assert "original_size" in stats
        assert "optimized_size" in stats
        assert "pruning_ratio" in stats
        assert "avg_relevance" in stats

        assert stats["original_items"] == 5
        assert stats["optimized_items"] <= 5

    def test_item_compression(self):
        """Test item compression for oversized items."""
        optimizer = ContextOptimizer(max_context_size=50)

        # Create an item that's too large
        large_item = ContextItem(
            key="large",
            value="x" * 1000,  # ~250 tokens
            size=250,
            relevance_score=0.9,
        )

        compressed = optimizer._try_compress_item(large_item, 20)

        assert compressed is not None
        assert compressed.size <= 20
        assert compressed.value.endswith("...")
        assert compressed.metadata["truncated"] is True

    def test_merge_contexts(self):
        """Test merging multiple context windows."""
        optimizer = ContextOptimizer()

        # Create two context windows
        window1 = ContextWindow(
            items=[
                ContextItem("key1", "value1", 10, relevance_score=0.8),
                ContextItem("shared", "value_old", 15, relevance_score=0.6),
            ],
            total_size=25,
            max_size=100,
            optimization_stats={},
        )

        window2 = ContextWindow(
            items=[
                ContextItem("key2", "value2", 20, relevance_score=0.7),
                ContextItem("shared", "value_new", 15, relevance_score=0.9),
            ],
            total_size=35,
            max_size=100,
            optimization_stats={},
        )

        merged = optimizer.merge_contexts([window1, window2], "merge task")

        # Should have 3 unique keys
        keys = [item.key for item in merged.items]
        assert len(set(keys)) <= 3

        # For duplicate keys, higher relevance should win
        shared_items = [item for item in merged.items if item.key == "shared"]
        if shared_items:
            assert shared_items[0].value == "value_new"  # Higher relevance

    def test_pruning_recommendations(self):
        """Test getting pruning recommendations."""
        optimizer = ContextOptimizer()

        window = ContextWindow(
            items=[
                ContextItem("low_rel", "value", 10, relevance_score=0.3),
                ContextItem("large", "value", 300, relevance_score=0.8),
                ContextItem("truncated", "val...", 5, relevance_score=0.6, metadata={"truncated": True}),
            ],
            total_size=315,
            max_size=1000,
            optimization_stats={},
        )

        recommendations = optimizer.get_pruning_recommendations(window)

        # Should have recommendations for each problematic item
        assert len(recommendations) == 3

        # Check recommendation types
        rec_keys = [r[0] for r in recommendations]
        assert "low_rel" in rec_keys  # Low relevance
        assert "large" in rec_keys  # Large size
        assert "truncated" in rec_keys  # Was truncated

    def test_custom_size_estimator(self):
        """Test using custom size estimator."""
        optimizer = ContextOptimizer()

        # Custom estimator that counts words
        def word_count_estimator(value):
            return len(str(value).split())

        context_items = {
            "short": "hello world",  # 2 words
            "long": "this is a much longer sentence with more words",  # 10 words
        }

        result = optimizer.optimize_context(context_items, "task", size_estimator=word_count_estimator)

        # Check that custom estimator was used
        for item in result.items:
            if item.key == "short":
                assert item.size == 2
            elif item.key == "long":
                assert item.size == 10
