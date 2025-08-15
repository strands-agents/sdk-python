"""Tests for tool usage analytics."""

import time

import pytest

from strands.agent.context.analytics import (
    ContextPerformanceStats,
    ToolUsageAnalytics,
    ToolUsageStats,
)


class TestToolUsageStats:
    """Tests for ToolUsageStats."""

    def test_initialization(self):
        """Test ToolUsageStats initialization."""
        stats = ToolUsageStats(tool_name="test_tool")

        assert stats.tool_name == "test_tool"
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.total_execution_time == 0.0
        assert stats.last_used is None
        assert stats.avg_relevance_score == 0.0
        assert stats.relevance_scores == []

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = ToolUsageStats(tool_name="test")

        # No calls
        assert stats.success_rate == 0.0

        # Some successful calls
        stats.total_calls = 10
        stats.successful_calls = 7
        assert stats.success_rate == 0.7

        # All successful
        stats.successful_calls = 10
        assert stats.success_rate == 1.0

    def test_avg_execution_time(self):
        """Test average execution time calculation."""
        stats = ToolUsageStats(tool_name="test")

        # No successful calls
        assert stats.avg_execution_time == 0.0

        # With successful calls
        stats.successful_calls = 5
        stats.total_execution_time = 10.0
        assert stats.avg_execution_time == 2.0

    def test_record_usage_success(self):
        """Test recording successful usage."""
        stats = ToolUsageStats(tool_name="test")

        stats.record_usage(success=True, execution_time=1.5, relevance_score=0.8)

        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert stats.total_execution_time == 1.5
        assert stats.last_used is not None
        assert stats.avg_relevance_score == 0.8
        assert stats.relevance_scores == [0.8]

    def test_record_usage_failure(self):
        """Test recording failed usage."""
        stats = ToolUsageStats(tool_name="test")

        stats.record_usage(success=False, execution_time=0.5)

        assert stats.total_calls == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1
        assert stats.total_execution_time == 0.0  # Failures don't count
        assert stats.last_used is not None

    def test_relevance_score_averaging(self):
        """Test relevance score averaging."""
        stats = ToolUsageStats(tool_name="test")

        stats.record_usage(True, 1.0, relevance_score=0.8)
        stats.record_usage(True, 1.0, relevance_score=0.6)
        stats.record_usage(True, 1.0, relevance_score=0.7)

        assert stats.avg_relevance_score == pytest.approx(0.7, 0.01)
        assert len(stats.relevance_scores) == 3


class TestContextPerformanceStats:
    """Tests for ContextPerformanceStats."""

    def test_initialization(self):
        """Test ContextPerformanceStats initialization."""
        stats = ContextPerformanceStats()

        assert stats.total_context_builds == 0
        assert stats.total_pruning_operations == 0
        assert stats.avg_context_size == 0.0
        assert stats.avg_pruning_ratio == 0.0
        assert stats.context_sizes == []
        assert stats.pruning_ratios == []

    def test_record_context_build(self):
        """Test recording context build operations."""
        stats = ContextPerformanceStats()

        # First build with pruning
        stats.record_context_build(context_size=500, original_size=1000)

        assert stats.total_context_builds == 1
        assert stats.total_pruning_operations == 1
        assert stats.avg_context_size == 500.0
        assert stats.avg_pruning_ratio == 0.5

        # Second build without pruning (same size)
        stats.record_context_build(context_size=300, original_size=300)

        assert stats.total_context_builds == 2
        assert stats.total_pruning_operations == 2
        assert stats.avg_context_size == 400.0  # (500 + 300) / 2
        assert stats.avg_pruning_ratio == 0.25  # (0.5 + 0) / 2

    def test_pruning_ratio_calculation(self):
        """Test pruning ratio calculation."""
        stats = ContextPerformanceStats()

        # 75% reduction
        stats.record_context_build(context_size=250, original_size=1000)
        assert stats.pruning_ratios[-1] == 0.75

        # No reduction
        stats.record_context_build(context_size=500, original_size=500)
        assert stats.pruning_ratios[-1] == 0.0

        # Edge case: original size is 0
        stats.record_context_build(context_size=100, original_size=0)
        # Should not add a pruning ratio for this case
        assert len(stats.pruning_ratios) == 2


class TestToolUsageAnalytics:
    """Tests for ToolUsageAnalytics."""

    def test_initialization(self):
        """Test ToolUsageAnalytics initialization."""
        analytics = ToolUsageAnalytics()

        assert analytics.tool_stats == {}
        assert isinstance(analytics.context_stats, ContextPerformanceStats)
        assert analytics._start_time <= time.time()

    def test_get_tool_stats(self):
        """Test getting or creating tool stats."""
        analytics = ToolUsageAnalytics()

        # First access creates new stats
        stats1 = analytics.get_tool_stats("tool1")
        assert stats1.tool_name == "tool1"
        assert "tool1" in analytics.tool_stats

        # Second access returns same stats
        stats2 = analytics.get_tool_stats("tool1")
        assert stats1 is stats2

    def test_record_tool_usage(self):
        """Test recording tool usage."""
        analytics = ToolUsageAnalytics()

        analytics.record_tool_usage(tool_name="calculator", success=True, execution_time=0.5, relevance_score=0.9)

        stats = analytics.tool_stats["calculator"]
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.total_execution_time == 0.5
        assert stats.avg_relevance_score == 0.9

    def test_record_context_build(self):
        """Test recording context build stats."""
        analytics = ToolUsageAnalytics()

        analytics.record_context_build(context_size=500, original_size=1000)

        assert analytics.context_stats.total_context_builds == 1
        assert analytics.context_stats.avg_pruning_ratio == 0.5

    def test_tool_rankings(self):
        """Test tool performance rankings."""
        analytics = ToolUsageAnalytics()

        # Add usage data for multiple tools
        for i in range(10):
            analytics.record_tool_usage("tool_a", True, 1.0, 0.9)
            analytics.record_tool_usage("tool_b", True, 1.0, 0.7)
            analytics.record_tool_usage("tool_c", i < 5, 1.0, 0.8)  # 50% success

        rankings = analytics.get_tool_rankings(min_calls=5)

        # Should be sorted by performance score
        assert len(rankings) == 3
        assert rankings[0][0] == "tool_a"  # Highest score
        assert rankings[0][1] > rankings[1][1]  # Decreasing scores
        assert rankings[1][1] > rankings[2][1]

    def test_tool_rankings_min_calls_filter(self):
        """Test that min_calls filter works correctly."""
        analytics = ToolUsageAnalytics()

        # tool_a: 10 calls, tool_b: 3 calls
        for _ in range(10):
            analytics.record_tool_usage("tool_a", True, 1.0, 0.9)
        for _ in range(3):
            analytics.record_tool_usage("tool_b", True, 1.0, 0.9)

        # With min_calls=5, only tool_a should be ranked
        rankings = analytics.get_tool_rankings(min_calls=5)
        assert len(rankings) == 1
        assert rankings[0][0] == "tool_a"

    def test_recency_factor_calculation(self):
        """Test recency factor calculation."""
        analytics = ToolUsageAnalytics()

        # Recent usage
        current_time = time.time()
        factor_recent = analytics._calculate_recency_factor(current_time - 3600)  # 1 hour ago
        assert 0.9 < factor_recent <= 1.0

        # Old usage
        factor_old = analytics._calculate_recency_factor(current_time - 86400)  # 24 hours ago
        assert factor_old == pytest.approx(0.0, 0.1)

        # No usage
        factor_none = analytics._calculate_recency_factor(None)
        assert factor_none == 0.0

    def test_summary_report(self):
        """Test comprehensive summary report generation."""
        analytics = ToolUsageAnalytics()

        # Add some usage data
        analytics.record_tool_usage("tool1", True, 1.0, 0.8)
        analytics.record_tool_usage("tool2", False, 0.5, 0.6)
        analytics.record_context_build(500, 1000)

        report = analytics.get_summary_report()

        # Check report structure
        assert "uptime_seconds" in report
        assert report["total_tools_used"] == 2
        assert report["total_tool_calls"] == 2
        assert report["overall_success_rate"] == 0.5

        # Check context optimization
        ctx_opt = report["context_optimization"]
        assert ctx_opt["total_builds"] == 1
        assert ctx_opt["avg_context_size"] == 500.0
        assert ctx_opt["avg_pruning_ratio"] == 0.5

        # Check top tools (may be empty due to min_calls filter)
        assert "top_tools" in report
        assert isinstance(report["top_tools"], list)

    def test_reset_stats(self):
        """Test resetting analytics data."""
        analytics = ToolUsageAnalytics()

        # Add some data
        analytics.record_tool_usage("tool1", True, 1.0)
        analytics.record_context_build(500, 1000)
        original_start_time = analytics._start_time

        # Reset
        analytics.reset_stats()

        assert len(analytics.tool_stats) == 0
        assert analytics.context_stats.total_context_builds == 0
        assert analytics._start_time >= original_start_time
