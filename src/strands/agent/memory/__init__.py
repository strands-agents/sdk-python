"""Memory management system for agents.

This package provides enhanced memory management capabilities including:
- Memory categorization (active, cached, archived)
- Memory lifecycle management with automatic cleanup
- Memory usage monitoring and metrics
- Configurable memory thresholds and policies

The memory management system is designed to be backward compatible with
existing AgentState usage while providing advanced memory optimization
capabilities for complex multi-agent scenarios.
"""

from .config import MemoryCategory, MemoryConfig, MemoryThresholds
from .enhanced_state import EnhancedAgentState
from .lifecycle import MemoryLifecycleManager
from .metrics import MemoryMetrics, MemoryUsageStats

__all__ = [
    "MemoryConfig",
    "MemoryCategory",
    "MemoryThresholds",
    "EnhancedAgentState",
    "MemoryLifecycleManager",
    "MemoryMetrics",
    "MemoryUsageStats",
]
