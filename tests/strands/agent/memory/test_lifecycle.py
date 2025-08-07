"""Tests for memory lifecycle management."""

import time
from unittest.mock import patch

from strands.agent.memory.config import MemoryCategory, MemoryConfig, MemoryThresholds
from strands.agent.memory.lifecycle import CategorizedMemoryItem, MemoryLifecycleManager


def test_categorized_memory_item_creation():
    """Test CategorizedMemoryItem creation and properties."""
    item = CategorizedMemoryItem("test_key", "test_value")

    assert item.key == "test_key"
    assert item.value == "test_value"
    assert item.category == MemoryCategory.ACTIVE
    assert item.access_count == 0
    assert item.size > 0
    assert item.created_at <= time.time()
    assert item.last_accessed <= time.time()


def test_categorized_memory_item_with_category():
    """Test CategorizedMemoryItem with specific category."""
    item = CategorizedMemoryItem("test_key", "test_value", MemoryCategory.CACHED)

    assert item.category == MemoryCategory.CACHED


def test_categorized_memory_item_access():
    """Test memory item access tracking."""
    item = CategorizedMemoryItem("test_key", "test_value")
    initial_access_time = item.last_accessed
    initial_access_count = item.access_count

    time.sleep(0.01)  # Small delay to ensure time difference
    item.access()

    assert item.access_count == initial_access_count + 1
    assert item.last_accessed > initial_access_time


def test_categorized_memory_item_age():
    """Test memory item age calculation."""
    with patch("time.time") as mock_time:
        mock_time.return_value = 1000.0
        item = CategorizedMemoryItem("test_key", "test_value")

        mock_time.return_value = 1010.0
        assert item.age() == 10.0


def test_categorized_memory_item_idle_time():
    """Test memory item idle time calculation."""
    with patch("time.time") as mock_time:
        mock_time.return_value = 1000.0
        item = CategorizedMemoryItem("test_key", "test_value")

        mock_time.return_value = 1005.0
        item.access()

        mock_time.return_value = 1015.0
        assert item.idle_time() == 10.0


def test_categorized_memory_item_should_demote():
    """Test demotion logic for memory items."""
    item = CategorizedMemoryItem("test_key", "test_value", MemoryCategory.ACTIVE)

    with patch.object(item, "idle_time", return_value=3700):  # > 1 hour
        assert item.should_demote(3600) is True

    with patch.object(item, "idle_time", return_value=1800):  # < 1 hour
        assert item.should_demote(3600) is False

    # Non-active items should not be demoted
    item.category = MemoryCategory.CACHED
    with patch.object(item, "idle_time", return_value=3700):
        assert item.should_demote(3600) is False


def test_categorized_memory_item_should_archive():
    """Test archival logic for memory items."""
    item = CategorizedMemoryItem("test_key", "test_value", MemoryCategory.CACHED)

    with patch.object(item, "age", return_value=90000):  # > 24 hours
        assert item.should_archive(86400) is True

    with patch.object(item, "age", return_value=43200):  # < 24 hours
        assert item.should_archive(86400) is False

    # Non-cached items should not be archived
    item.category = MemoryCategory.ACTIVE
    with patch.object(item, "age", return_value=90000):
        assert item.should_archive(86400) is False


def test_categorized_memory_item_size_estimation():
    """Test size estimation for different value types."""
    # Test with different value types
    string_item = CategorizedMemoryItem("key", "hello")
    dict_item = CategorizedMemoryItem("key", {"nested": "data"})
    list_item = CategorizedMemoryItem("key", [1, 2, 3, 4, 5])

    assert string_item.size > 0
    assert dict_item.size > 0
    assert list_item.size > 0

    # Larger objects should have larger estimated sizes
    large_string = "x" * 1000
    large_item = CategorizedMemoryItem("key", large_string)
    assert large_item.size > string_item.size


def test_memory_lifecycle_manager_initialization():
    """Test MemoryLifecycleManager initialization."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    assert manager.config is config
    assert manager._items == {}
    assert manager._last_cleanup <= time.time()


def test_memory_lifecycle_manager_add_item():
    """Test adding items to memory manager."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    manager.add_item("key1", "value1")
    manager.add_item("key2", "value2", MemoryCategory.CACHED)

    assert "key1" in manager._items
    assert "key2" in manager._items
    assert manager._items["key1"].category == MemoryCategory.ACTIVE
    assert manager._items["key2"].category == MemoryCategory.CACHED
    assert manager._items["key1"].value == "value1"
    assert manager._items["key2"].value == "value2"


def test_memory_lifecycle_manager_get_item():
    """Test retrieving items from memory manager."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    manager.add_item("existing_key", "existing_value")

    # Test existing key
    value = manager.get_item("existing_key")
    assert value == "existing_value"

    # Test non-existing key
    value = manager.get_item("non_existing_key")
    assert value is None

    # Check that access was recorded
    item = manager._items["existing_key"]
    assert item.access_count == 1


def test_memory_lifecycle_manager_remove_item():
    """Test removing items from memory manager."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    manager.add_item("key1", "value1")

    # Remove existing item
    assert manager.remove_item("key1") is True
    assert "key1" not in manager._items

    # Remove non-existing item
    assert manager.remove_item("non_existing") is False


def test_memory_lifecycle_manager_get_items_by_category():
    """Test retrieving items by category."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    manager.add_item("active1", "value1", MemoryCategory.ACTIVE)
    manager.add_item("active2", "value2", MemoryCategory.ACTIVE)
    manager.add_item("cached1", "value3", MemoryCategory.CACHED)

    active_items = manager.get_items_by_category(MemoryCategory.ACTIVE)
    cached_items = manager.get_items_by_category(MemoryCategory.CACHED)
    archived_items = manager.get_items_by_category(MemoryCategory.ARCHIVED)

    assert len(active_items) == 2
    assert len(cached_items) == 1
    assert len(archived_items) == 0
    assert active_items["active1"] == "value1"
    assert cached_items["cached1"] == "value3"


def test_memory_lifecycle_manager_get_all_items():
    """Test retrieving all items."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    manager.add_item("key1", "value1", MemoryCategory.ACTIVE)
    manager.add_item("key2", "value2", MemoryCategory.CACHED)

    all_items = manager.get_all_items()

    assert len(all_items) == 2
    assert all_items["key1"] == "value1"
    assert all_items["key2"] == "value2"


def test_memory_lifecycle_manager_promotion():
    """Test automatic promotion of frequently accessed items."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    # Add cached item
    manager.add_item("key1", "value1", MemoryCategory.CACHED)

    # Mock frequent access pattern
    with patch.object(manager, "_should_promote", return_value=True):
        # Access should trigger promotion
        manager.get_item("key1")

        # Check that item was promoted
        assert manager._items["key1"].category == MemoryCategory.ACTIVE


def test_memory_lifecycle_manager_should_promote():
    """Test promotion logic."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    # Create item with access pattern
    item = CategorizedMemoryItem("key", "value", MemoryCategory.CACHED)

    # Test promotion logic
    with patch.object(item, "age", return_value=7200):  # 2 hours
        with patch.object(item, "access_count", 3):  # 3 accesses in 2 hours > 1/hour
            assert manager._should_promote(item) is True

        with patch.object(item, "access_count", 1):  # 1 access in 2 hours < 1/hour
            assert manager._should_promote(item) is False

    # Test with new item (age = 0)
    with patch.object(item, "age", return_value=0):
        assert manager._should_promote(item) is False


def test_memory_lifecycle_manager_cleanup_memory():
    """Test memory cleanup functionality."""
    thresholds = MemoryThresholds(
        cache_ttl=1800,  # 30 minutes
        archive_after=3600,  # 1 hour
    )
    config = MemoryConfig(thresholds=thresholds, enable_lifecycle=True)
    manager = MemoryLifecycleManager(config)

    # Add items with different ages
    manager.add_item("active_old", "value1", MemoryCategory.ACTIVE)
    manager.add_item("cached_old", "value2", MemoryCategory.CACHED)

    # Mock ages for demotion/archival
    active_item = manager._items["active_old"]
    cached_item = manager._items["cached_old"]

    with patch.object(active_item, "should_demote", return_value=True):
        with patch.object(cached_item, "should_archive", return_value=True):
            # Run cleanup
            manager.cleanup_memory()

            # Check changes
            assert active_item.category == MemoryCategory.CACHED
            assert cached_item.category == MemoryCategory.ARCHIVED


def test_memory_lifecycle_manager_emergency_cleanup():
    """Test emergency cleanup when memory limits exceeded."""
    thresholds = MemoryThresholds(total_memory_limit=100)  # Very small limit
    config = MemoryConfig(thresholds=thresholds, enable_lifecycle=True)
    manager = MemoryLifecycleManager(config)

    # Add items that exceed limit
    large_value = "x" * 50
    manager.add_item("key1", large_value, MemoryCategory.CACHED)
    manager.add_item("key2", large_value, MemoryCategory.CACHED)
    manager.add_item("key3", large_value, MemoryCategory.ACTIVE)

    initial_count = len(manager._items)

    # Force cleanup
    removed_count = manager.cleanup_memory(force=True)

    # Should have removed some items
    assert removed_count > 0
    assert len(manager._items) < initial_count


def test_memory_lifecycle_manager_disabled_lifecycle():
    """Test behavior when lifecycle management is disabled."""
    config = MemoryConfig(enable_lifecycle=False)
    manager = MemoryLifecycleManager(config)

    manager.add_item("key1", "value1")

    # Cleanup should do nothing when lifecycle disabled
    removed_count = manager.cleanup_memory(force=False)
    assert removed_count == 0

    # But should work with force=True
    removed_count = manager.cleanup_memory(force=True)
    assert removed_count >= 0  # May or may not remove items


def test_memory_lifecycle_manager_get_memory_report():
    """Test memory usage report generation."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    manager.add_item("key1", "value1", MemoryCategory.ACTIVE)
    manager.add_item("key2", "value2", MemoryCategory.CACHED)

    report = manager.get_memory_report()

    assert "current_stats" in report
    assert "performance" in report
    assert "trends" in report
    assert "lifecycle" in report
    assert "timestamps" in report


def test_memory_lifecycle_manager_optimize_memory():
    """Test memory optimization."""
    config = MemoryConfig()
    manager = MemoryLifecycleManager(config)

    # Add some items
    manager.add_item("key1", "value1")
    manager.add_item("key2", "value2")

    optimization_results = manager.optimize_memory()

    assert "initial_size" in optimization_results
    assert "final_size" in optimization_results
    assert "size_reduction" in optimization_results
    assert "initial_count" in optimization_results
    assert "final_count" in optimization_results
    assert "items_removed" in optimization_results
    assert "size_reduction_pct" in optimization_results
