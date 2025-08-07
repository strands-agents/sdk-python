"""Enhanced agent state with memory management capabilities."""

import copy
from typing import Any, Dict, Optional

from ..state import AgentState
from .config import MemoryCategory, MemoryConfig
from .lifecycle import MemoryLifecycleManager


class EnhancedAgentState(AgentState):
    """Enhanced AgentState with memory categorization, lifecycle management, and metrics.

    This class extends the base AgentState to provide:
    - Memory categorization (active, cached, archived, metadata)
    - Automatic memory lifecycle management
    - Memory usage monitoring and metrics
    - Configurable memory thresholds and cleanup policies

    The enhanced state maintains full backward compatibility with the base AgentState
    interface while adding advanced memory management capabilities.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None, memory_config: Optional[MemoryConfig] = None):
        """Initialize EnhancedAgentState.

        Args:
            initial_state: Initial state dictionary (backward compatibility)
            memory_config: Memory management configuration
        """
        # Initialize base AgentState for backward compatibility
        super().__init__(initial_state)

        # Initialize memory management
        self.memory_config = memory_config or MemoryConfig()
        self.memory_manager = MemoryLifecycleManager(self.memory_config)

        # Migrate existing state to memory manager if provided
        if initial_state:
            for key, value in initial_state.items():
                self.memory_manager.add_item(key, value, MemoryCategory.ACTIVE)

    def set(self, key: str, value: Any, category: MemoryCategory = MemoryCategory.ACTIVE) -> None:
        """Set a value in the state with optional memory category.

        Args:
            key: The key to store the value under
            value: The value to store (must be JSON serializable)
            category: Memory category for the value

        Raises:
            ValueError: If key is invalid, or if value is not JSON serializable
        """
        # Validate using parent class methods
        self._validate_key(key)
        self._validate_json_serializable(value)

        # Store in both base state (for backward compatibility) and memory manager
        super().set(key, value)

        if self.memory_config.enable_categorization:
            self.memory_manager.add_item(key, value, category)

    def get(self, key: Optional[str] = None, category: Optional[MemoryCategory] = None) -> Any:
        """Get a value or entire state, optionally filtered by category.

        Args:
            key: The key to retrieve (if None, returns entire state object)
            category: Optional memory category filter

        Returns:
            The stored value, filtered state dict, or None if not found
        """
        if key is None:
            # Return entire state, optionally filtered by category
            if category is not None and self.memory_config.enable_categorization:
                return copy.deepcopy(self.memory_manager.get_items_by_category(category))
            else:
                # Backward compatibility: return base state
                return super().get()
        else:
            # Return specific key
            if self.memory_config.enable_categorization:
                return self.memory_manager.get_item(key)
            else:
                return super().get(key)

    def delete(self, key: str) -> None:
        """Delete a specific key from the state.

        Args:
            key: The key to delete
        """
        self._validate_key(key)

        # Delete from both base state and memory manager
        super().delete(key)

        if self.memory_config.enable_categorization:
            self.memory_manager.remove_item(key)

    def get_by_category(self, category: MemoryCategory) -> Dict[str, Any]:
        """Get all items in a specific memory category.

        Args:
            category: The memory category to retrieve

        Returns:
            Dictionary of all items in the specified category
        """
        if not self.memory_config.enable_categorization:
            # If categorization disabled, return all items for any category
            return super().get() or {}

        return self.memory_manager.get_items_by_category(category)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value (convenience method).

        Args:
            key: The key to store the metadata under
            value: The metadata value
        """
        self.set(key, value, MemoryCategory.METADATA)

    def get_active_memory(self) -> Dict[str, Any]:
        """Get all active memory items (convenience method).

        Returns:
            Dictionary of all active memory items
        """
        return self.get_by_category(MemoryCategory.ACTIVE)

    def get_cached_memory(self) -> Dict[str, Any]:
        """Get all cached memory items (convenience method).

        Returns:
            Dictionary of all cached memory items
        """
        return self.get_by_category(MemoryCategory.CACHED)

    def cleanup_memory(self, force: bool = False) -> int:
        """Perform memory cleanup and return number of items removed.

        Args:
            force: Force cleanup even if lifecycle management is disabled

        Returns:
            Number of items removed during cleanup
        """
        if not self.memory_config.enable_lifecycle and not force:
            return 0

        removed_count = self.memory_manager.cleanup_memory(force)

        # Sync base state with memory manager after cleanup
        self._sync_base_state()

        return removed_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics.

        Returns:
            Dictionary containing memory usage statistics and metrics
        """
        if not self.memory_config.enable_metrics:
            # Return basic stats if metrics disabled
            all_items = super().get() or {}
            return {
                "total_items": len(all_items),
                "categories_enabled": False,
                "lifecycle_enabled": self.memory_config.enable_lifecycle,
                "metrics_enabled": False,
            }

        return self.memory_manager.get_memory_report()

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage and return optimization results.

        Returns:
            Dictionary containing optimization results and statistics
        """
        if not self.memory_config.enable_lifecycle:
            return {"optimization_skipped": True, "reason": "lifecycle_disabled"}

        optimization_results = self.memory_manager.optimize_memory()

        # Sync base state after optimization
        self._sync_base_state()

        return optimization_results

    def _sync_base_state(self) -> None:
        """Synchronize base state with memory manager state."""
        if self.memory_config.enable_categorization:
            # Update base state to match memory manager
            all_items = self.memory_manager.get_all_items()
            self._state = copy.deepcopy(all_items)

    def configure_memory(self, config: MemoryConfig) -> None:
        """Update memory configuration.

        Args:
            config: New memory configuration
        """
        self.memory_config = config
        self.memory_manager.config = config

    def get_memory_config(self) -> MemoryConfig:
        """Get current memory configuration.

        Returns:
            Current memory configuration
        """
        return self.memory_config
