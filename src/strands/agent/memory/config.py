"""Memory management configuration and types."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MemoryCategory(Enum):
    """Categories for memory classification."""
    
    ACTIVE = "active"           # Currently active/working memory
    CACHED = "cached"           # Recently used but not active
    ARCHIVED = "archived"       # Historical data for potential retrieval
    METADATA = "metadata"       # System metadata and statistics


@dataclass
class MemoryThresholds:
    """Memory management thresholds and limits."""
    
    # Size thresholds (in estimated tokens/bytes)
    active_memory_limit: int = 8192        # ~8K tokens for active memory
    cached_memory_limit: int = 32768       # ~32K tokens for cached memory
    total_memory_limit: int = 131072       # ~128K tokens total limit
    
    # Cleanup thresholds (percentages)
    cleanup_threshold: float = 0.8         # Start cleanup at 80% of limit
    emergency_threshold: float = 0.95      # Emergency cleanup at 95%
    
    # Time-based thresholds (seconds)
    cache_ttl: int = 3600                  # Cache TTL: 1 hour
    archive_after: int = 86400             # Archive after: 24 hours
    
    # Cleanup ratios (how much to remove during cleanup)
    cleanup_ratio: float = 0.3             # Remove 30% during cleanup
    emergency_cleanup_ratio: float = 0.5   # Remove 50% during emergency


@dataclass
class MemoryConfig:
    """Configuration for memory management system."""
    
    # Feature toggles
    enable_categorization: bool = True      # Enable memory categorization
    enable_lifecycle: bool = True           # Enable automatic lifecycle management
    enable_metrics: bool = True             # Enable memory metrics collection
    enable_archival: bool = True            # Enable memory archival
    
    # Thresholds configuration
    thresholds: MemoryThresholds = None
    
    # Cleanup strategy
    cleanup_strategy: str = "lru"           # LRU, FIFO, or custom
    
    # Validation settings
    strict_validation: bool = True          # Strict JSON validation
    
    def __post_init__(self):
        """Initialize default thresholds if not provided."""
        if self.thresholds is None:
            self.thresholds = MemoryThresholds()
    
    @classmethod
    def conservative(cls) -> "MemoryConfig":
        """Create conservative memory configuration with lower limits."""
        return cls(
            thresholds=MemoryThresholds(
                active_memory_limit=4096,
                cached_memory_limit=16384,
                total_memory_limit=65536,
                cleanup_threshold=0.7,
                cleanup_ratio=0.4
            )
        )
    
    @classmethod
    def aggressive(cls) -> "MemoryConfig":
        """Create aggressive memory configuration with higher limits."""
        return cls(
            thresholds=MemoryThresholds(
                active_memory_limit=16384,
                cached_memory_limit=65536,
                total_memory_limit=262144,
                cleanup_threshold=0.9,
                cleanup_ratio=0.2
            )
        )
    
    @classmethod
    def minimal(cls) -> "MemoryConfig":
        """Create minimal memory configuration with basic features only."""
        return cls(
            enable_lifecycle=False,
            enable_metrics=False,
            enable_archival=False,
            thresholds=MemoryThresholds(
                active_memory_limit=2048,
                cached_memory_limit=8192,
                total_memory_limit=32768
            )
        )