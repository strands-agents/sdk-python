"""Memory lifecycle management for automatic cleanup and archival."""

import time
from typing import Any, Dict, Optional, Protocol

from .config import MemoryCategory, MemoryConfig, MemoryThresholds
from .metrics import MemoryMetrics


class MemoryItem(Protocol):
    """Protocol for memory items that can be managed by lifecycle manager."""

    category: MemoryCategory
    created_at: float
    last_accessed: float
    access_count: int
    size: int


class CategorizedMemoryItem:
    """A memory item with categorization and lifecycle metadata."""

    def __init__(self, key: str, value: Any, category: MemoryCategory = MemoryCategory.ACTIVE):
        """Initialize a categorized memory item.

        Args:
            key: The key for this memory item
            value: The value to store
            category: Memory category for this item
        """
        self.key = key
        self.value = value
        self.category = category
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.size = self._estimate_size(value)

    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of the value."""
        import json

        try:
            return len(json.dumps(value).encode("utf-8"))
        except (TypeError, ValueError):
            return len(str(value).encode("utf-8"))

    def access(self) -> None:
        """Record an access to this memory item."""
        self.last_accessed = time.time()
        self.access_count += 1

    def age(self) -> float:
        """Get the age of this memory item in seconds."""
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Get the idle time since last access in seconds."""
        return time.time() - self.last_accessed

    def should_demote(self, inactive_threshold: float) -> bool:
        """Check if item should be demoted from active to cached."""
        return self.category == MemoryCategory.ACTIVE and self.idle_time() > inactive_threshold

    def should_archive(self, archive_threshold: float) -> bool:
        """Check if item should be archived."""
        return self.category == MemoryCategory.CACHED and self.age() > archive_threshold


class MemoryLifecycleManager:
    """Manages the lifecycle of memory items with automatic cleanup and archival."""

    def __init__(self, config: MemoryConfig):
        """Initialize the memory lifecycle manager.

        Args:
            config: Memory management configuration
        """
        self.config = config
        # Ensure thresholds are initialized
        assert config.thresholds is not None, "MemoryConfig must have initialized thresholds"
        self.metrics = MemoryMetrics()
        self._items: Dict[str, CategorizedMemoryItem] = {}
        self._last_cleanup = time.time()

    @property
    def thresholds(self) -> MemoryThresholds:
        """Get memory thresholds, ensuring they are never None."""
        assert self.config.thresholds is not None
        return self.config.thresholds

    def add_item(self, key: str, value: Any, category: MemoryCategory = MemoryCategory.ACTIVE) -> None:
        """Add or update a memory item."""
        item = CategorizedMemoryItem(key, value, category)
        self._items[key] = item

        if category == MemoryCategory.ACTIVE:
            self.metrics.record_promotion()

        # Trigger cleanup if needed
        if self.config.enable_lifecycle:
            self._check_and_cleanup()

    def get_item(self, key: str) -> Optional[Any]:
        """Get a memory item by key, updating access statistics."""
        if key in self._items:
            item = self._items[key]
            item.access()
            self.metrics.record_access(hit=True)

            # Promote to active if accessed frequently
            if item.category == MemoryCategory.CACHED and self._should_promote(item):
                item.category = MemoryCategory.ACTIVE
                self.metrics.record_promotion()

            return item.value
        else:
            self.metrics.record_access(hit=False)
            return None

    def remove_item(self, key: str) -> bool:
        """Remove a memory item."""
        if key in self._items:
            del self._items[key]
            return True
        return False

    def get_items_by_category(self, category: MemoryCategory) -> Dict[str, Any]:
        """Get all items in a specific category."""
        return {key: item.value for key, item in self._items.items() if item.category == category}

    def get_all_items(self) -> Dict[str, Any]:
        """Get all memory items (backward compatibility with AgentState)."""
        return {key: item.value for key, item in self._items.items()}

    def cleanup_memory(self, force: bool = False) -> int:
        """Perform memory cleanup and return number of items removed."""
        if not self.config.enable_lifecycle and not force:
            return 0

        removed_count = 0
        current_time = time.time()

        # First, demote old active items to cached
        for item in list(self._items.values()):
            if item.should_demote(self.thresholds.cache_ttl):
                item.category = MemoryCategory.CACHED
                self.metrics.record_demotion()

        # Archive old cached items
        if self.config.enable_archival:
            for _key, item in list(self._items.items()):
                if item.should_archive(self.thresholds.archive_after):
                    item.category = MemoryCategory.ARCHIVED
                    self.metrics.record_archival()

        # Remove items if over memory limits
        total_size = sum(item.size for item in self._items.values())
        if force or total_size > self.thresholds.total_memory_limit:
            removed_count += self._emergency_cleanup()

        self.metrics.record_cleanup()
        self._last_cleanup = current_time
        self._update_metrics()

        return removed_count

    def _check_and_cleanup(self) -> None:
        """Check if cleanup is needed and perform it."""
        total_size = sum(item.size for item in self._items.values())
        utilization = total_size / self.thresholds.total_memory_limit

        if utilization >= self.thresholds.cleanup_threshold:
            self.cleanup_memory()

        # Update metrics periodically
        current_time = time.time()
        if current_time - self._last_cleanup > 300:  # Update every 5 minutes
            self._update_metrics()
            self._last_cleanup = current_time

    def _should_promote(self, item: CategorizedMemoryItem) -> bool:
        """Determine if a cached item should be promoted to active."""
        # Promote if accessed frequently in recent time
        recent_accesses = item.access_count
        age_hours = item.age() / 3600

        # Simple heuristic: if accessed more than once per hour on average
        return age_hours > 0 and (recent_accesses / age_hours) > 1.0

    def _emergency_cleanup(self) -> int:
        """Perform emergency cleanup when memory limits are exceeded."""
        removed_count = 0

        # Sort items by priority (LRU-like: least recently used first)
        items_by_priority = sorted(self._items.items(), key=lambda x: (x[1].category.value, x[1].last_accessed))

        # Calculate how many items to remove
        total_items = len(self._items)
        target_removal = max(1, int(total_items * self.thresholds.emergency_cleanup_ratio))

        # Remove least important items first
        for key, item in items_by_priority[:target_removal]:
            # Don't remove active metadata
            if item.category != MemoryCategory.METADATA:
                del self._items[key]
                removed_count += 1

        return removed_count

    def _update_metrics(self) -> None:
        """Update memory usage statistics."""
        from .metrics import MemoryUsageStats

        stats = MemoryUsageStats()

        for item in self._items.values():
            stats.total_size += item.size
            stats.total_items += 1

            if item.category == MemoryCategory.ACTIVE:
                stats.active_size += item.size
                stats.active_items += 1
            elif item.category == MemoryCategory.CACHED:
                stats.cached_size += item.size
                stats.cached_items += 1
            elif item.category == MemoryCategory.ARCHIVED:
                stats.archived_size += item.size
                stats.archived_items += 1
            elif item.category == MemoryCategory.METADATA:
                stats.metadata_size += item.size
                stats.metadata_items += 1

        self.metrics.update_stats(stats)

    def get_memory_report(self) -> Dict[str, Any]:
        """Get a comprehensive memory usage report."""
        self._update_metrics()
        return self.metrics.get_summary()

    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization and return optimization results."""
        initial_size = sum(item.size for item in self._items.values())
        initial_count = len(self._items)

        # Perform cleanup
        removed_count = self.cleanup_memory(force=True)

        final_size = sum(item.size for item in self._items.values())
        final_count = len(self._items)

        return {
            "initial_size": initial_size,
            "final_size": final_size,
            "size_reduction": initial_size - final_size,
            "initial_count": initial_count,
            "final_count": final_count,
            "items_removed": removed_count,
            "size_reduction_pct": int(((initial_size - final_size) / initial_size * 100)) if initial_size > 0 else 0,
        }
