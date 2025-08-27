"""Tests for memory metrics and monitoring."""

import time

from strands.agent.memory.config import MemoryCategory
from strands.agent.memory.metrics import MemoryMetrics, MemoryUsageStats


def test_memory_usage_stats_defaults():
    """Test MemoryUsageStats default values."""
    stats = MemoryUsageStats()

    assert stats.total_size == 0
    assert stats.active_size == 0
    assert stats.cached_size == 0
    assert stats.archived_size == 0
    assert stats.metadata_size == 0
    assert stats.total_items == 0
    assert stats.hit_rate == 0.0
    assert stats.cleanup_count == 0
    assert stats.promotions == 0


def test_memory_usage_stats_utilization_ratio():
    """Test utilization ratio calculation."""
    stats = MemoryUsageStats()
    stats.total_size = 5000

    # Normal case
    assert stats.utilization_ratio(10000) == 0.5

    # Over limit case
    assert stats.utilization_ratio(2500) == 1.0

    # Zero limit case
    assert stats.utilization_ratio(0) == 0.0


def test_memory_usage_stats_category_distribution():
    """Test category distribution calculation."""
    stats = MemoryUsageStats()
    stats.total_size = 1000
    stats.active_size = 400
    stats.cached_size = 300
    stats.archived_size = 200
    stats.metadata_size = 100

    distribution = stats.category_distribution()

    assert distribution[MemoryCategory.ACTIVE.value] == 0.4
    assert distribution[MemoryCategory.CACHED.value] == 0.3
    assert distribution[MemoryCategory.ARCHIVED.value] == 0.2
    assert distribution[MemoryCategory.METADATA.value] == 0.1


def test_memory_usage_stats_category_distribution_empty():
    """Test category distribution with zero total size."""
    stats = MemoryUsageStats()
    # total_size = 0 by default

    distribution = stats.category_distribution()

    for category in MemoryCategory:
        assert distribution[category.value] == 0.0


def test_memory_metrics_initialization():
    """Test MemoryMetrics initialization."""
    metrics = MemoryMetrics()

    assert isinstance(metrics.stats, MemoryUsageStats)
    assert metrics.history == []
    assert metrics.max_history_size == 100
    assert metrics.access_count == 0
    assert metrics.hit_count == 0
    assert metrics.miss_count == 0
    assert metrics.last_access_time is None
    assert metrics.creation_time <= time.time()


def test_memory_metrics_record_access():
    """Test recording memory access (hits and misses)."""
    metrics = MemoryMetrics()

    # Record hits
    metrics.record_access(hit=True)
    metrics.record_access(hit=True)

    assert metrics.access_count == 2
    assert metrics.hit_count == 2
    assert metrics.miss_count == 0
    assert metrics.stats.hit_rate == 1.0
    assert metrics.stats.miss_rate == 0.0
    assert metrics.last_access_time is not None

    # Record miss
    metrics.record_access(hit=False)

    assert metrics.access_count == 3
    assert metrics.hit_count == 2
    assert metrics.miss_count == 1
    assert metrics.stats.hit_rate == 2 / 3
    assert metrics.stats.miss_rate == 1 / 3


def test_memory_metrics_record_operations():
    """Test recording various memory operations."""
    metrics = MemoryMetrics()

    # Test cleanup recording
    metrics.record_cleanup()
    assert metrics.stats.cleanup_count == 1
    assert metrics.stats.last_cleanup is not None

    # Test promotion recording
    metrics.record_promotion()
    assert metrics.stats.promotions == 1

    # Test demotion recording
    metrics.record_demotion()
    assert metrics.stats.demotions == 1

    # Test archival recording
    metrics.record_archival()
    assert metrics.stats.archival_count == 1


def test_memory_metrics_update_stats():
    """Test updating statistics with history tracking."""
    metrics = MemoryMetrics()

    # Create new stats
    new_stats = MemoryUsageStats()
    new_stats.total_size = 1000
    new_stats.total_items = 10

    # Update stats
    metrics.update_stats(new_stats)

    # Check that old stats were saved to history
    assert len(metrics.history) == 1
    assert metrics.history[0].total_size == 0  # Old default stats

    # Check that new stats are current
    assert metrics.stats.total_size == 1000
    assert metrics.stats.total_items == 10


def test_memory_metrics_history_limit():
    """Test that history is limited to max_history_size."""
    metrics = MemoryMetrics()
    metrics.max_history_size = 3

    # Add more stats than the limit
    for i in range(5):
        new_stats = MemoryUsageStats()
        new_stats.total_size = i * 100
        metrics.update_stats(new_stats)

    # Check that history is limited
    assert len(metrics.history) == 3

    # Check that oldest entries were removed (should have sizes 100, 200, 300)
    assert metrics.history[0].total_size == 100
    assert metrics.history[1].total_size == 200
    assert metrics.history[2].total_size == 300


def test_memory_metrics_estimate_item_size():
    """Test item size estimation."""
    metrics = MemoryMetrics()

    # Test with JSON-serializable objects
    assert metrics.estimate_item_size("hello") > 0
    assert metrics.estimate_item_size({"key": "value"}) > 0
    assert metrics.estimate_item_size([1, 2, 3]) > 0

    # Test that larger objects have larger estimated sizes
    small_obj = "hi"
    large_obj = "hello world" * 100
    assert metrics.estimate_item_size(large_obj) > metrics.estimate_item_size(small_obj)

    # Test with non-serializable object
    class CustomObject:
        def __str__(self):
            return "custom object"

    custom_obj = CustomObject()
    size = metrics.estimate_item_size(custom_obj)
    assert size > 0  # Should fallback to string representation


def test_memory_metrics_trend_analysis():
    """Test trend analysis functionality."""
    metrics = MemoryMetrics()

    # Test with insufficient history
    trend = metrics.get_trend_analysis()
    assert trend["trend"] == 0.0
    assert trend["volatility"] == 0.0

    # Add some history with increasing sizes
    sizes = [100, 200, 300, 400, 500]
    for size in sizes:
        stats = MemoryUsageStats()
        stats.total_size = size
        metrics.update_stats(stats)

    trend = metrics.get_trend_analysis()
    assert trend["trend"] > 0  # Should show upward trend
    assert trend["avg_size"] == 300  # Average of 100, 200, 300, 400, 500
    assert trend["min_size"] == 100
    assert trend["max_size"] == 500
    assert trend["volatility"] > 0


def test_memory_metrics_trend_analysis_with_window():
    """Test trend analysis with custom window size."""
    metrics = MemoryMetrics()

    # Add history
    sizes = [100, 200, 300, 400, 500, 600]
    for size in sizes:
        stats = MemoryUsageStats()
        stats.total_size = size
        metrics.update_stats(stats)

    # Analyze with smaller window
    trend = metrics.get_trend_analysis(window=3)
    # Should only consider last 3 values: 400, 500, 600
    assert trend["avg_size"] == 500
    assert trend["min_size"] == 400
    assert trend["max_size"] == 600


def test_memory_metrics_should_cleanup():
    """Test cleanup decision logic."""
    metrics = MemoryMetrics()

    # Setup stats
    metrics.stats.total_size = 8000

    # Should cleanup when over threshold
    assert metrics.should_cleanup(threshold=0.8, limit=10000) is True

    # Should not cleanup when under threshold
    assert metrics.should_cleanup(threshold=0.9, limit=10000) is False

    # Edge case: zero limit
    assert metrics.should_cleanup(threshold=0.5, limit=0) is False


def test_memory_metrics_get_summary():
    """Test comprehensive summary generation."""
    metrics = MemoryMetrics()

    # Setup some data
    metrics.record_access(hit=True)
    metrics.record_access(hit=False)
    metrics.record_cleanup()
    metrics.record_promotion()

    # Add some history for trend analysis
    for i in range(3):
        stats = MemoryUsageStats()
        stats.total_size = (i + 1) * 100
        metrics.update_stats(stats)

    summary = metrics.get_summary()

    # Check structure
    assert "current_stats" in summary
    assert "performance" in summary
    assert "trends" in summary
    assert "lifecycle" in summary
    assert "timestamps" in summary

    # Check current stats
    current_stats = summary["current_stats"]
    assert "total_size" in current_stats
    assert "distribution" in current_stats
    assert "hit_rate" in current_stats

    # Check performance metrics
    performance = summary["performance"]
    assert performance["access_count"] == 2
    assert performance["hit_rate"] == 0.5
    assert performance["miss_rate"] == 0.5

    # Check lifecycle metrics
    lifecycle = summary["lifecycle"]
    assert lifecycle["promotions"] == 1

    # Check timestamps
    timestamps = summary["timestamps"]
    assert "creation_time" in timestamps
    assert "last_access" in timestamps
