"""Tests for dynamic tool management."""

from typing import Any, Dict

from strands.agent.context.analytics import ToolUsageAnalytics
from strands.agent.context.tool_manager import (
    DynamicToolManager,
    ToolSelectionCriteria,
    ToolSelectionResult,
)
from strands.types.tools import AgentTool, ToolSpec, ToolUse


class MockTool(AgentTool):
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = None):
        super().__init__()
        self._name = name
        self._description = description or f"A tool that {name}"

    @property
    def tool_name(self) -> str:
        return self._name

    @property
    def tool_spec(self) -> ToolSpec:
        return {
            "name": self._name,
            "description": self._description,
            "inputSchema": {"type": "object", "properties": {}},
        }

    @property
    def tool_type(self) -> str:
        return "mock"

    async def stream(self, tool_use: ToolUse, invocation_state: Dict[str, Any], **kwargs):
        """Mock stream implementation."""
        yield {"type": "result", "result": "mock"}


class TestToolSelectionCriteria:
    """Tests for ToolSelectionCriteria dataclass."""

    def test_criteria_creation(self):
        """Test creating selection criteria."""
        criteria = ToolSelectionCriteria(
            task_description="analyze data",
            required_capabilities=["data", "analysis"],
            excluded_tools={"dangerous_tool"},
            max_tools=15,
            min_relevance_score=0.4,
            prefer_recent=False,
            context_hints={"domain": "finance"},
        )

        assert criteria.task_description == "analyze data"
        assert criteria.required_capabilities == ["data", "analysis"]
        assert "dangerous_tool" in criteria.excluded_tools
        assert criteria.max_tools == 15
        assert criteria.min_relevance_score == 0.4
        assert criteria.prefer_recent is False
        assert criteria.context_hints["domain"] == "finance"

    def test_criteria_defaults(self):
        """Test default values for criteria."""
        criteria = ToolSelectionCriteria(task_description="test task")

        assert criteria.required_capabilities is None
        assert criteria.excluded_tools is None
        assert criteria.max_tools == 20
        assert criteria.min_relevance_score == 0.2
        assert criteria.prefer_recent is True
        assert criteria.context_hints is None


class TestDynamicToolManager:
    """Tests for DynamicToolManager."""

    def test_initialization(self):
        """Test DynamicToolManager initialization."""
        manager = DynamicToolManager()

        assert manager.analytics is not None
        assert isinstance(manager.analytics, ToolUsageAnalytics)
        assert manager.scorer is not None
        assert manager._tool_cache == {}

    def test_tool_selection_basic(self):
        """Test basic tool selection."""
        manager = DynamicToolManager()

        tools = [
            MockTool("file_reader", "Reads files from disk and file system for data processing"),
            MockTool("file_writer", "Writes files to disk"),
            MockTool("calculator", "Performs calculations and computes statistics on data"),
            MockTool("web_scraper", "Scrapes web pages"),
        ]

        criteria = ToolSelectionCriteria(
            task_description="I need to read a file and calculate some statistics",
            min_relevance_score=0.1,  # Lower threshold for testing
        )

        result = manager.select_tools(tools, criteria)

        assert isinstance(result, ToolSelectionResult)
        assert len(result.selected_tools) > 0
        assert len(result.selected_tools) <= criteria.max_tools

        # Should include relevant tools
        tool_names = [t.tool_name for t in result.selected_tools]
        assert "file_reader" in tool_names or "calculator" in tool_names

    def test_excluded_tools_filtering(self):
        """Test that excluded tools are filtered out."""
        manager = DynamicToolManager()

        tools = [MockTool("safe_tool"), MockTool("dangerous_tool"), MockTool("another_tool")]

        criteria = ToolSelectionCriteria(task_description="any task", excluded_tools={"dangerous_tool"})

        result = manager.select_tools(tools, criteria)

        tool_names = [t.tool_name for t in result.selected_tools]
        assert "dangerous_tool" not in tool_names

    def test_max_tools_limit(self):
        """Test that max_tools limit is respected."""
        manager = DynamicToolManager()

        # Create many tools
        tools = [MockTool(f"tool_{i}") for i in range(50)]

        criteria = ToolSelectionCriteria(task_description="use all tools", max_tools=5)

        result = manager.select_tools(tools, criteria)

        assert len(result.selected_tools) <= 5

    def test_minimum_relevance_filtering(self):
        """Test filtering by minimum relevance score."""
        manager = DynamicToolManager()

        tools = [
            MockTool("very_relevant_tool", "exactly what the task needs"),
            MockTool("unrelated_tool", "something completely different"),
        ]

        criteria = ToolSelectionCriteria(
            task_description="exactly what the task needs",
            min_relevance_score=0.7,  # High threshold
        )

        result = manager.select_tools(tools, criteria)

        # Only highly relevant tools should be selected
        for tool in result.selected_tools:
            assert result.relevance_scores[tool.tool_name] >= 0.7

    def test_required_capabilities_filtering(self):
        """Test filtering by required capabilities."""
        manager = DynamicToolManager()

        tools = [
            MockTool("data_analyzer", "analyzes data and generates reports"),
            MockTool("file_reader", "reads files"),
            MockTool("data_visualizer", "creates data visualizations"),
        ]

        criteria = ToolSelectionCriteria(
            task_description="analyze some data",
            required_capabilities=["data", "analyz"],  # Partial match
        )

        result = manager.select_tools(tools, criteria)

        # Should include tools with "data" and "analyz" in name/description
        tool_names = [t.tool_name for t in result.selected_tools]
        assert "data_analyzer" in tool_names

    def test_performance_based_adjustment(self):
        """Test performance-based score adjustment."""
        analytics = ToolUsageAnalytics()
        manager = DynamicToolManager(analytics=analytics)

        # Record good performance for tool1
        for _ in range(10):
            analytics.record_tool_usage("tool1", True, 0.5, 0.8)

        # Record poor performance for tool2
        for _ in range(10):
            analytics.record_tool_usage("tool2", False, 1.0, 0.8)

        tools = [
            MockTool("tool1", "description"),
            MockTool("tool2", "description"),  # Same description
        ]

        criteria = ToolSelectionCriteria(task_description="description", prefer_recent=True)

        result = manager.select_tools(tools, criteria)

        # tool1 should have higher adjusted score due to better performance
        if len(result.selected_tools) > 0:
            assert result.selected_tools[0].tool_name == "tool1"

    def test_selection_reasoning_generation(self):
        """Test that selection reasoning is generated."""
        manager = DynamicToolManager()

        tools = [
            MockTool("high_relevance_tool", "exactly matches the task"),
            MockTool("low_relevance_tool", "unrelated functionality"),
        ]

        criteria = ToolSelectionCriteria(task_description="exactly matches the task")

        result = manager.select_tools(tools, criteria)

        # Should have reasoning for selected tools
        for tool in result.selected_tools:
            assert tool.tool_name in result.selection_reasoning
            reasoning = result.selection_reasoning[tool.tool_name]
            assert len(reasoning) > 0
            assert "relevance" in reasoning.lower()

    def test_tool_recommendations(self):
        """Test getting tool recommendations."""
        analytics = ToolUsageAnalytics()
        manager = DynamicToolManager(analytics=analytics)

        # Add some tools to cache
        tools = [MockTool("frequently_used", "common tool"), MockTool("rarely_used", "uncommon tool")]
        manager._update_tool_cache(tools)

        # Record usage history
        for _ in range(10):
            analytics.record_tool_usage("frequently_used", True, 0.5, 0.9)

        recommendations = manager.get_tool_recommendations(task_description="common task", max_recommendations=3)

        assert len(recommendations) <= 3
        # Recommendations should be (tool_name, confidence) tuples
        if recommendations:
            assert len(recommendations[0]) == 2
            assert isinstance(recommendations[0][0], str)
            assert isinstance(recommendations[0][1], float)

    def test_recent_tools_boost(self):
        """Test that recently used tools get a confidence boost."""
        manager = DynamicToolManager()

        # Add tools to cache
        tools = [MockTool("tool1"), MockTool("tool2")]
        manager._update_tool_cache(tools)

        # Record some usage
        manager.analytics.record_tool_usage("tool1", True, 0.5, 0.5)
        manager.analytics.record_tool_usage("tool2", True, 0.5, 0.5)

        # Get recommendations with recent tools
        recommendations = manager.get_tool_recommendations(task_description="any task", recent_tools=["tool1"])

        # tool1 should have higher confidence due to recency boost
        rec_dict = dict(recommendations)
        if "tool1" in rec_dict and "tool2" in rec_dict:
            assert rec_dict["tool1"] > rec_dict["tool2"]

    def test_update_tool_performance(self):
        """Test updating tool performance metrics."""
        manager = DynamicToolManager()

        # Update performance
        manager.update_tool_performance("test_tool", success=True, execution_time=1.5)

        # Check that it was recorded
        stats = manager.analytics.get_tool_stats("test_tool")
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.total_execution_time == 1.5

    def test_selection_result_metadata(self):
        """Test that selection result contains proper metadata."""
        manager = DynamicToolManager()

        tools = [MockTool(f"tool_{i}") for i in range(10)]
        criteria = ToolSelectionCriteria(task_description="test")

        result = manager.select_tools(tools, criteria)

        assert result.total_candidates == 10
        assert result.selection_time > 0
        assert isinstance(result.relevance_scores, dict)
        assert isinstance(result.selection_reasoning, dict)

    def test_no_tools_scenario(self):
        """Test behavior when no tools are available."""
        manager = DynamicToolManager()

        criteria = ToolSelectionCriteria(task_description="any task")
        result = manager.select_tools([], criteria)

        assert len(result.selected_tools) == 0
        assert result.total_candidates == 0
        assert result.relevance_scores == {}

    def test_all_tools_below_threshold(self):
        """Test when all tools are below relevance threshold."""
        manager = DynamicToolManager()

        tools = [MockTool("unrelated1", "completely unrelated"), MockTool("unrelated2", "also unrelated")]

        criteria = ToolSelectionCriteria(
            task_description="specific database query optimization",
            min_relevance_score=0.9,  # Very high threshold
        )

        result = manager.select_tools(tools, criteria)

        # No tools should be selected
        assert len(result.selected_tools) == 0
