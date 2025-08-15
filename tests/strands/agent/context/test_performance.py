"""Performance tests for context management."""

import time
from typing import Any, Dict, List

import pytest

from strands.agent.context import (
    ContextOptimizer,
    DynamicToolManager,
    ToolSelectionCriteria,
)
from strands.agent.context.relevance_scoring import TextRelevanceScorer
from strands.types.tools import AgentTool, ToolSpec, ToolUse


class MockTool(AgentTool):
    """Mock tool for performance testing."""

    def __init__(self, name: str, description: str):
        super().__init__()
        self._name = name
        self._description = description

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


class TestContextOptimizerPerformance:
    """Performance tests for ContextOptimizer."""

    @pytest.mark.parametrize("num_items", [100, 500, 1000])
    def test_optimization_speed(self, num_items: int):
        """Test context optimization speed with varying sizes."""
        optimizer = ContextOptimizer(max_context_size=1000)

        # Create large context
        context_items = {
            f"item_{i}": f"This is context item {i} with some content about {i % 10}" for i in range(num_items)
        }

        task = "Find items about content 5"

        start_time = time.time()
        result = optimizer.optimize_context(context_items, task)
        optimization_time = time.time() - start_time

        # Performance assertions
        assert optimization_time < 1.0  # Should complete within 1 second
        assert result.total_size <= optimizer.max_context_size

        # Verify optimization worked
        pruning_ratio = result.optimization_stats["pruning_ratio"]
        if num_items > 100:
            assert pruning_ratio > 0  # Should prune something for large contexts

        print(f"Optimized {num_items} items in {optimization_time:.3f}s (pruned {pruning_ratio:.1%})")

    def test_relevance_scoring_performance(self):
        """Test relevance scoring performance."""
        scorer = TextRelevanceScorer()

        # Create test data
        items = [f"Item {i} with content about topic {i % 20}" for i in range(1000)]
        context = "Looking for items about topic 15"

        start_time = time.time()
        scores = [scorer.score(item, context) for item in items]
        scoring_time = time.time() - start_time

        assert scoring_time < 0.5  # 1000 items in 0.5s
        assert len(scores) == 1000
        assert all(0 <= s <= 1 for s in scores)

        print(f"Scored 1000 items in {scoring_time:.3f}s ({1000 / scoring_time:.0f} items/sec)")

    def test_large_context_compression(self):
        """Test performance of context compression for large items."""
        optimizer = ContextOptimizer(max_context_size=500)

        # Create context with large items
        large_text = "x" * 10000  # Very large item
        context_items = {"large_item": large_text, "normal_item": "Normal sized content", "another_large": "y" * 5000}

        start_time = time.time()
        result = optimizer.optimize_context(
            context_items,
            "task requiring all items",
            required_keys=["large_item"],  # Force inclusion of large item
        )
        compression_time = time.time() - start_time

        assert compression_time < 0.1  # Should be fast
        assert result.total_size <= optimizer.max_context_size

        # Check that large item was compressed
        large_items = [item for item in result.items if item.key == "large_item"]
        if large_items:
            assert large_items[0].value != large_text  # Should be compressed
            assert large_items[0].value.endswith("...")


class TestToolManagerPerformance:
    """Performance tests for DynamicToolManager."""

    def create_mock_tools(self, count: int) -> List[AgentTool]:
        """Create mock tools for testing."""
        categories = ["file", "data", "web", "system", "analysis", "compute"]
        actions = ["read", "write", "process", "analyze", "fetch", "transform"]

        tools = []
        for i in range(count):
            category = categories[i % len(categories)]
            action = actions[i % len(actions)]
            name = f"{category}_{action}_tool_{i}"
            description = f"A tool that {action}s {category} data and performs operations"
            tools.append(MockTool(name, description))

        return tools

    @pytest.mark.parametrize("num_tools", [50, 100, 500])
    def test_tool_selection_speed(self, num_tools: int):
        """Test tool selection speed with varying numbers of tools."""
        manager = DynamicToolManager()
        tools = self.create_mock_tools(num_tools)

        criteria = ToolSelectionCriteria(task_description="analyze data from files and web sources", max_tools=20)

        start_time = time.time()
        result = manager.select_tools(tools, criteria)
        selection_time = time.time() - start_time

        # Performance assertions
        assert selection_time < 0.5  # Should complete quickly
        assert len(result.selected_tools) <= criteria.max_tools
        assert result.selection_time > 0

        print(f"Selected from {num_tools} tools in {selection_time:.3f}s")

    def test_tool_scoring_with_history(self):
        """Test tool selection performance with usage history."""
        manager = DynamicToolManager()
        tools = self.create_mock_tools(100)

        # Simulate usage history
        for i in range(50):
            tool_name = tools[i].tool_name
            success = i % 3 != 0  # 2/3 success rate
            manager.update_tool_performance(tool_name, success, 0.1 * (i % 5))

        criteria = ToolSelectionCriteria(task_description="process and analyze data", prefer_recent=True)

        start_time = time.time()
        result = manager.select_tools(tools, criteria)
        selection_time = time.time() - start_time

        assert selection_time < 0.5
        assert len(result.selected_tools) > 0

        # Tools with history should be considered
        selected_names = [t.tool_name for t in result.selected_tools]
        tools_with_history = [tools[i].tool_name for i in range(50)]
        overlap = set(selected_names) & set(tools_with_history)
        assert len(overlap) > 0  # Some tools with history should be selected

    def test_recommendations_performance(self):
        """Test performance of tool recommendations."""
        manager = DynamicToolManager()
        tools = self.create_mock_tools(200)
        manager._update_tool_cache(tools)

        # Add usage history for subset of tools
        for i in range(100):
            tool = tools[i]
            for _ in range(5):  # Minimum calls for ranking
                manager.update_tool_performance(tool.tool_name, success=True, execution_time=0.1)

        start_time = time.time()
        recommendations = manager.get_tool_recommendations(task_description="analyze web data", max_recommendations=10)
        rec_time = time.time() - start_time

        assert rec_time < 0.1  # Should be very fast
        assert len(recommendations) <= 10

        print(f"Generated {len(recommendations)} recommendations in {rec_time:.3f}s")


class TestIntegratedPerformance:
    """Integration performance tests."""

    def test_full_context_optimization_pipeline(self):
        """Test full pipeline from tool selection to context optimization."""
        # Setup
        tool_manager = DynamicToolManager()
        context_optimizer = ContextOptimizer(max_context_size=2000)

        # Create tools and context
        tools = []
        for i in range(100):
            # Create tools with more relevant descriptions
            category = ["analyze", "process", "extract", "transform", "filter"][i % 5]
            topic = i % 10
            tools.append(MockTool(f"{category}_tool_{i}", f"Tool that {category}s data related to topic {topic}"))

        context_items = {f"ctx_{i}": f"Context data {i} related to {i % 10}" for i in range(500)}

        task = "Analyze data items related to topic 5"

        # Measure full pipeline
        start_time = time.time()

        # Step 1: Select tools
        tool_criteria = ToolSelectionCriteria(task_description=task, max_tools=10)
        tool_result = tool_manager.select_tools(tools, tool_criteria)

        # Step 2: Optimize context
        context_result = context_optimizer.optimize_context(
            context_items,
            task,
            required_keys=["ctx_5", "ctx_15"],  # Require some specific items
        )

        total_time = time.time() - start_time

        # Performance assertions
        assert total_time < 1.0  # Full pipeline under 1 second
        assert len(tool_result.selected_tools) > 0
        assert context_result.total_size <= context_optimizer.max_context_size

        print(f"Full pipeline completed in {total_time:.3f}s:")
        print(f"  - Selected {len(tool_result.selected_tools)} tools")
        print(f"  - Optimized context from {len(context_items)} to {len(context_result.items)} items")
        print(f"  - Context reduction: {context_result.optimization_stats['pruning_ratio']:.1%}")
