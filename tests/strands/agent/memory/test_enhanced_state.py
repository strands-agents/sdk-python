"""Tests for enhanced agent state with memory management."""

import pytest

from strands.agent.memory.config import MemoryCategory, MemoryConfig
from strands.agent.memory.enhanced_state import EnhancedAgentState


def test_enhanced_agent_state_initialization():
    """Test EnhancedAgentState initialization."""
    state = EnhancedAgentState()

    assert state.memory_config is not None
    assert state.memory_manager is not None
    assert isinstance(state.memory_config, MemoryConfig)


def test_enhanced_agent_state_with_initial_state():
    """Test EnhancedAgentState initialization with initial state."""
    initial_state = {"key1": "value1", "key2": "value2"}
    state = EnhancedAgentState(initial_state=initial_state)

    # Should be available through both interfaces
    assert state.get("key1") == "value1"
    assert state.get("key2") == "value2"

    # Should be available through parent interface (backward compatibility)
    parent_state = super(EnhancedAgentState, state).get()
    assert parent_state["key1"] == "value1"
    assert parent_state["key2"] == "value2"


def test_enhanced_agent_state_with_memory_config():
    """Test EnhancedAgentState with custom memory configuration."""
    config = MemoryConfig.conservative()
    state = EnhancedAgentState(memory_config=config)

    assert state.memory_config is config
    assert state.memory_manager.config is config


def test_enhanced_agent_state_set_get_basic():
    """Test basic set and get operations."""
    state = EnhancedAgentState()

    state.set("test_key", "test_value")
    assert state.get("test_key") == "test_value"


def test_enhanced_agent_state_set_with_category():
    """Test set operation with memory category."""
    state = EnhancedAgentState()

    state.set("active_key", "active_value", MemoryCategory.ACTIVE)
    state.set("cached_key", "cached_value", MemoryCategory.CACHED)
    state.set("metadata_key", "metadata_value", MemoryCategory.METADATA)

    assert state.get("active_key") == "active_value"
    assert state.get("cached_key") == "cached_value"
    assert state.get("metadata_key") == "metadata_value"


def test_enhanced_agent_state_get_by_category():
    """Test retrieving items by category."""
    state = EnhancedAgentState()

    state.set("active1", "value1", MemoryCategory.ACTIVE)
    state.set("active2", "value2", MemoryCategory.ACTIVE)
    state.set("cached1", "value3", MemoryCategory.CACHED)

    active_items = state.get_by_category(MemoryCategory.ACTIVE)
    cached_items = state.get_by_category(MemoryCategory.CACHED)
    archived_items = state.get_by_category(MemoryCategory.ARCHIVED)

    assert len(active_items) == 2
    assert len(cached_items) == 1
    assert len(archived_items) == 0
    assert active_items["active1"] == "value1"
    assert cached_items["cached1"] == "value3"


def test_enhanced_agent_state_get_by_category_disabled():
    """Test get_by_category when categorization is disabled."""
    config = MemoryConfig(enable_categorization=False)
    state = EnhancedAgentState(memory_config=config)

    state.set("key1", "value1")
    state.set("key2", "value2")

    # Should return all items regardless of category when disabled
    items = state.get_by_category(MemoryCategory.ACTIVE)
    assert len(items) == 2
    assert items["key1"] == "value1"
    assert items["key2"] == "value2"


def test_enhanced_agent_state_get_entire_state():
    """Test getting entire state."""
    state = EnhancedAgentState()

    state.set("key1", "value1", MemoryCategory.ACTIVE)
    state.set("key2", "value2", MemoryCategory.CACHED)

    # Get entire state
    all_items = state.get()
    assert len(all_items) == 2
    assert all_items["key1"] == "value1"
    assert all_items["key2"] == "value2"


def test_enhanced_agent_state_get_filtered_by_category():
    """Test getting state filtered by category."""
    state = EnhancedAgentState()

    state.set("active1", "value1", MemoryCategory.ACTIVE)
    state.set("cached1", "value2", MemoryCategory.CACHED)

    # Get only active items
    active_items = state.get(category=MemoryCategory.ACTIVE)
    assert len(active_items) == 1
    assert active_items["active1"] == "value1"

    # Get only cached items
    cached_items = state.get(category=MemoryCategory.CACHED)
    assert len(cached_items) == 1
    assert cached_items["cached1"] == "value2"


def test_enhanced_agent_state_delete():
    """Test deleting items."""
    state = EnhancedAgentState()

    state.set("key1", "value1")
    state.set("key2", "value2")

    state.delete("key1")

    assert state.get("key1") is None
    assert state.get("key2") == "value2"

    # Should also be deleted from parent state
    parent_state = super(EnhancedAgentState, state).get()
    assert "key1" not in parent_state
    assert parent_state["key2"] == "value2"


def test_enhanced_agent_state_convenience_methods():
    """Test convenience methods for memory categories."""
    state = EnhancedAgentState()

    # Test set_metadata
    state.set_metadata("meta_key", "meta_value")
    metadata_items = state.get_by_category(MemoryCategory.METADATA)
    assert metadata_items["meta_key"] == "meta_value"

    # Test get_active_memory
    state.set("active_key", "active_value", MemoryCategory.ACTIVE)
    active_items = state.get_active_memory()
    assert active_items["active_key"] == "active_value"

    # Test get_cached_memory
    state.set("cached_key", "cached_value", MemoryCategory.CACHED)
    cached_items = state.get_cached_memory()
    assert cached_items["cached_key"] == "cached_value"


def test_enhanced_agent_state_cleanup_memory():
    """Test memory cleanup functionality."""
    state = EnhancedAgentState()

    state.set("key1", "value1")
    state.set("key2", "value2")

    # Test cleanup (might not remove anything without time passage)
    removed_count = state.cleanup_memory()
    assert removed_count >= 0

    # Test forced cleanup
    removed_count = state.cleanup_memory(force=True)
    assert removed_count >= 0


def test_enhanced_agent_state_cleanup_memory_disabled():
    """Test cleanup when lifecycle management is disabled."""
    config = MemoryConfig(enable_lifecycle=False)
    state = EnhancedAgentState(memory_config=config)

    state.set("key1", "value1")

    # Should return 0 when lifecycle disabled
    removed_count = state.cleanup_memory()
    assert removed_count == 0

    # Should work with force=True
    removed_count = state.cleanup_memory(force=True)
    assert removed_count >= 0


def test_enhanced_agent_state_get_memory_stats():
    """Test getting memory statistics."""
    state = EnhancedAgentState()

    state.set("key1", "value1", MemoryCategory.ACTIVE)
    state.set("key2", "value2", MemoryCategory.CACHED)

    stats = state.get_memory_stats()

    assert "current_stats" in stats or "total_items" in stats

    # Test with metrics disabled
    config = MemoryConfig(enable_metrics=False)
    state_no_metrics = EnhancedAgentState(memory_config=config)
    stats = state_no_metrics.get_memory_stats()

    assert stats["metrics_enabled"] is False
    assert "total_items" in stats


def test_enhanced_agent_state_optimize_memory():
    """Test memory optimization."""
    state = EnhancedAgentState()

    state.set("key1", "value1")
    state.set("key2", "value2")

    optimization_results = state.optimize_memory()

    # Should return optimization results or skip info
    assert isinstance(optimization_results, dict)

    # Test with lifecycle disabled
    config = MemoryConfig(enable_lifecycle=False)
    state_no_lifecycle = EnhancedAgentState(memory_config=config)
    results = state_no_lifecycle.optimize_memory()

    assert results.get("optimization_skipped") is True
    assert results.get("reason") == "lifecycle_disabled"


def test_enhanced_agent_state_configure_memory():
    """Test memory configuration updates."""
    state = EnhancedAgentState()
    original_config = state.get_memory_config()

    new_config = MemoryConfig.aggressive()
    state.configure_memory(new_config)

    assert state.get_memory_config() is new_config
    assert state.memory_manager.config is new_config
    assert state.get_memory_config() is not original_config


def test_enhanced_agent_state_backward_compatibility():
    """Test backward compatibility with base AgentState interface."""
    state = EnhancedAgentState()

    # All base AgentState operations should work
    state.set("key1", "value1")
    assert state.get("key1") == "value1"

    # Get entire state should work
    all_state = state.get()
    assert all_state["key1"] == "value1"

    # Delete should work
    state.delete("key1")
    assert state.get("key1") is None

    # Validation should still work
    with pytest.raises(ValueError, match="Key cannot be None"):
        state.set(None, "value")

    with pytest.raises(ValueError, match="not JSON serializable"):
        state.set("key", lambda x: x)


def test_enhanced_agent_state_with_categorization_disabled():
    """Test behavior when categorization is disabled."""
    config = MemoryConfig(enable_categorization=False)
    state = EnhancedAgentState(memory_config=config)

    # Set operations should still work
    state.set("key1", "value1")
    state.set("key2", "value2", MemoryCategory.CACHED)  # Category ignored

    # Get should work normally
    assert state.get("key1") == "value1"
    assert state.get("key2") == "value2"

    # Should use parent implementation when categorization disabled
    all_items = state.get()
    assert all_items["key1"] == "value1"
    assert all_items["key2"] == "value2"


def test_enhanced_agent_state_json_validation():
    """Test that JSON validation is maintained."""
    state = EnhancedAgentState()

    # Valid JSON types should work
    state.set("string", "test")
    state.set("int", 42)
    state.set("bool", True)
    state.set("list", [1, 2, 3])
    state.set("dict", {"nested": "value"})
    state.set("null", None)

    # Invalid JSON types should raise ValueError
    with pytest.raises(ValueError, match="not JSON serializable"):
        state.set("function", lambda x: x)

    with pytest.raises(ValueError, match="not JSON serializable"):
        state.set("object", object())


def test_enhanced_agent_state_key_validation():
    """Test that key validation is maintained."""
    state = EnhancedAgentState()

    # Invalid keys should raise ValueError
    with pytest.raises(ValueError, match="Key cannot be None"):
        state.set(None, "value")

    with pytest.raises(ValueError, match="Key cannot be empty"):
        state.set("", "value")

    with pytest.raises(ValueError, match="Key must be a string"):
        state.set(123, "value")
