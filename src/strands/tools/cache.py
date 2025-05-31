"""Tool result caching implementation."""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from ..types.tools import ToolResult

logger = logging.getLogger(__name__)


class ToolResultCache:
    """Cache for tool execution results.

    This class provides caching functionality for tool execution results to avoid
    redundant tool executions with the same parameters.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize the tool result cache.

        Args:
            max_size: Maximum number of entries to store in the cache.
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self.cache: Dict[str, Tuple[ToolResult, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()

    def get(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[ToolResult]:
        """Get a cached tool result.

        Args:
            tool_name: Name of the tool.
            tool_input: Input parameters for the tool.

        Returns:
            The cached tool result, or None if not found or expired.
        """
        with self.lock:
            key = self._make_key(tool_name, tool_input)
            if key in self.cache:
                result, timestamp = self.cache[key]
                # Check TTL
                if time.time() - timestamp <= self.ttl_seconds:
                    self.hits += 1
                    logger.debug("tool_name=<%s> | cache hit", tool_name)
                    return result
                else:
                    del self.cache[key]
                    logger.debug("tool_name=<%s> | cache entry expired", tool_name)

            self.misses += 1
            return None

    def set(self, tool_name: str, tool_input: Dict[str, Any], result: ToolResult) -> None:
        """Store a tool result in the cache.

        Args:
            tool_name: Name of the tool.
            tool_input: Input parameters for the tool.
            result: The tool execution result to cache.
        """
        with self.lock:
            key = self._make_key(tool_name, tool_input)

            # Check the size of cache
            if len(self.cache) >= self.max_size:
                self._evict_cache()

            self.cache[key] = (result.copy(), time.time())
            logger.debug("tool_name=<%s> | cached result", tool_name)

    def _make_key(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Generate a cache key from tool name and input parameters.

        Args:
            tool_name: Name of the tool.
            tool_input: Input parameters for the tool.

        Returns:
            A unique cache key.
        """
        input_str = json.dumps(tool_input, sort_keys=True)
        return f"{tool_name}:{hashlib.md5(input_str.encode()).hexdigest()}"

    def _evict_cache(self) -> None:
        """Evict the oldest entry from the cache when it reaches max size."""
        if not self.cache:
            return

        # LRU
        oldest_key = None
        oldest_time = float("inf")

        for key, (_, timestamp) in self.cache.items():
            if timestamp < oldest_time:
                oldest_time = timestamp
                oldest_key = key

        if oldest_key:
            del self.cache[oldest_key]
            logger.debug("cache_key=<%s> | evicted from cache", oldest_key)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self.lock:
            self.cache.clear()
            logger.debug("cleared tool result cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with self.lock:
            total = self.hits + self.misses
            hit_ratio = self.hits / total if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_ratio": hit_ratio,
                "ttl_seconds": self.ttl_seconds,
            }
