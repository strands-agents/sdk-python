"""Memory usage metrics and monitoring."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import MemoryCategory


@dataclass
class MemoryUsageStats:
    """Statistics for memory usage tracking."""
    
    # Size metrics (estimated tokens/bytes)
    total_size: int = 0
    active_size: int = 0
    cached_size: int = 0
    archived_size: int = 0
    metadata_size: int = 0
    
    # Count metrics
    total_items: int = 0
    active_items: int = 0
    cached_items: int = 0
    archived_items: int = 0
    metadata_items: int = 0
    
    # Performance metrics
    hit_rate: float = 0.0           # Cache hit rate
    miss_rate: float = 0.0          # Cache miss rate
    cleanup_count: int = 0          # Number of cleanups performed
    last_cleanup: Optional[float] = None  # Timestamp of last cleanup
    
    # Lifecycle metrics
    promotions: int = 0             # Items promoted to active
    demotions: int = 0              # Items demoted from active
    archival_count: int = 0         # Items archived
    
    def utilization_ratio(self, limit: int) -> float:
        """Calculate memory utilization ratio."""
        if limit == 0:
            return 0.0
        return min(1.0, self.total_size / limit)
    
    def category_distribution(self) -> Dict[str, float]:
        """Get distribution of memory across categories."""
        if self.total_size == 0:
            return {cat.value: 0.0 for cat in MemoryCategory}
        
        return {
            MemoryCategory.ACTIVE.value: self.active_size / self.total_size,
            MemoryCategory.CACHED.value: self.cached_size / self.total_size,
            MemoryCategory.ARCHIVED.value: self.archived_size / self.total_size,
            MemoryCategory.METADATA.value: self.metadata_size / self.total_size,
        }


@dataclass  
class MemoryMetrics:
    """Memory metrics collection and analysis."""
    
    # Current statistics
    stats: MemoryUsageStats = field(default_factory=MemoryUsageStats)
    
    # Historical data (last N measurements)
    history: List[MemoryUsageStats] = field(default_factory=list)
    max_history_size: int = 100
    
    # Access tracking
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    # Timing metrics
    last_access_time: Optional[float] = None
    creation_time: float = field(default_factory=time.time)
    
    def record_access(self, hit: bool = True) -> None:
        """Record a memory access (hit or miss)."""
        self.access_count += 1
        self.last_access_time = time.time()
        
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1
        
        # Update hit/miss rates
        if self.access_count > 0:
            self.stats.hit_rate = self.hit_count / self.access_count
            self.stats.miss_rate = self.miss_count / self.access_count
    
    def record_cleanup(self) -> None:
        """Record a memory cleanup operation."""
        self.stats.cleanup_count += 1
        self.stats.last_cleanup = time.time()
    
    def record_promotion(self) -> None:
        """Record a memory item promotion."""
        self.stats.promotions += 1
    
    def record_demotion(self) -> None:
        """Record a memory item demotion."""
        self.stats.demotions += 1
    
    def record_archival(self) -> None:
        """Record a memory item archival."""
        self.stats.archival_count += 1
    
    def update_stats(self, new_stats: MemoryUsageStats) -> None:
        """Update current statistics and save to history."""
        # Save current stats to history
        if len(self.history) >= self.max_history_size:
            self.history.pop(0)  # Remove oldest entry
        
        self.history.append(self.stats)
        
        # Preserve accumulated metrics from current stats
        new_stats.hit_rate = self.stats.hit_rate
        new_stats.miss_rate = self.stats.miss_rate
        new_stats.cleanup_count = self.stats.cleanup_count
        new_stats.promotions = self.stats.promotions
        new_stats.demotions = self.stats.demotions
        new_stats.archival_count = self.stats.archival_count
        new_stats.last_cleanup = self.stats.last_cleanup
        
        self.stats = new_stats
    
    def estimate_item_size(self, value: Any) -> int:
        """Estimate the size of a memory item in bytes/tokens."""
        try:
            # Use JSON serialization size as rough estimate
            json_str = json.dumps(value)
            # Rough token estimate: ~4 characters per token
            return len(json_str.encode('utf-8'))
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            return len(str(value).encode('utf-8'))
    
    def get_trend_analysis(self, window: int = 10) -> Dict[str, float]:
        """Analyze trends in memory usage over the last N measurements."""
        if len(self.history) < 1:
            return {"trend": 0.0, "volatility": 0.0}
        
        # Get all stats (history + current), excluding initial empty stats
        all_history = self.history + [self.stats]
        # Skip the first item if it's the initial empty stats (total_size == 0)
        if all_history and all_history[0].total_size == 0 and len(all_history) > 1:
            all_history = all_history[1:]
        
        # Apply window to the complete set
        all_stats = all_history[-window:] if len(all_history) >= window else all_history
        
        # Calculate trend (simple linear regression slope)
        if len(all_stats) < 2:
            return {"trend": 0.0, "volatility": 0.0}
        
        sizes = [stats.total_size for stats in all_stats]
        n = len(sizes)
        x_mean = (n - 1) / 2
        y_mean = sum(sizes) / n
        
        numerator = sum((i - x_mean) * (sizes[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        trend = numerator / denominator if denominator != 0 else 0.0
        
        # Calculate volatility (standard deviation of sizes)
        variance = sum((size - y_mean) ** 2 for size in sizes) / n
        volatility = variance ** 0.5
        
        return {
            "trend": trend,
            "volatility": volatility,
            "avg_size": y_mean,
            "min_size": min(sizes),
            "max_size": max(sizes)
        }
    
    def should_cleanup(self, threshold: float, limit: int) -> bool:
        """Determine if memory cleanup should be triggered."""
        utilization = self.stats.utilization_ratio(limit)
        return utilization >= threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of memory metrics."""
        trend_analysis = self.get_trend_analysis()
        
        return {
            "current_stats": {
                "total_size": self.stats.total_size,
                "total_items": self.stats.total_items,
                "distribution": self.stats.category_distribution(),
                "hit_rate": self.stats.hit_rate,
                "cleanup_count": self.stats.cleanup_count
            },
            "performance": {
                "access_count": self.access_count,
                "hit_rate": self.hit_count / self.access_count if self.access_count > 0 else 0.0,
                "miss_rate": self.miss_count / self.access_count if self.access_count > 0 else 0.0,
                "avg_response_time": self._calculate_avg_response_time()
            },
            "trends": trend_analysis,
            "lifecycle": {
                "promotions": self.stats.promotions,
                "demotions": self.stats.demotions,
                "archival_count": self.stats.archival_count
            },
            "timestamps": {
                "creation_time": self.creation_time,
                "last_access": self.last_access_time,
                "last_cleanup": self.stats.last_cleanup
            }
        }
    
    def _calculate_avg_response_time(self) -> Optional[float]:
        """Calculate average response time based on access patterns."""
        # Placeholder for future implementation
        # Could track timing data for memory operations
        return None