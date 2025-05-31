import threading
import time

import pytest

from strands.tools.cache import ToolResultCache


@pytest.fixture
def cache():
    """Fixture providing a basic cache instance"""
    return ToolResultCache(max_size=3, ttl_seconds=1)


@pytest.fixture
def tool_result():
    """Fixture providing a test tool result"""
    return {"toolUseId": "test_id", "status": "success", "content": [{"text": "Test result"}]}


def test_cache_get_set(cache, tool_result):
    """Test basic get/set operations"""
    # Should return None when cache is empty
    assert cache.get("test_tool", {"param": "value"}) is None

    # Set a value in the cache
    cache.set("test_tool", {"param": "value"}, tool_result)

    # Get the value from cache
    cached_result = cache.get("test_tool", {"param": "value"})
    assert cached_result == tool_result

    # Should be cache miss for different parameters
    assert cache.get("test_tool", {"param": "different"}) is None
    assert cache.get("different_tool", {"param": "value"}) is None


def test_cache_ttl(cache, tool_result):
    """Test TTL (time-to-live) functionality"""
    # Set a value in the cache
    cache.set("test_tool", {"param": "value"}, tool_result)

    # Value should be retrievable before TTL expires
    assert cache.get("test_tool", {"param": "value"}) == tool_result

    # Wait for TTL to expire (1 second + a bit more)
    time.sleep(1.1)

    # Value should not be retrievable after TTL expires (returns None)
    assert cache.get("test_tool", {"param": "value"}) is None


def test_cache_max_size(cache, tool_result):
    """Test maximum size limit"""
    # Add items to cache up to maximum size (3)
    cache.set("tool1", {"param": "1"}, tool_result)
    cache.set("tool2", {"param": "2"}, tool_result)
    cache.set("tool3", {"param": "3"}, tool_result)

    # All items should be cache hits
    assert cache.get("tool1", {"param": "1"}) == tool_result
    assert cache.get("tool2", {"param": "2"}) == tool_result
    assert cache.get("tool3", {"param": "3"}) == tool_result

    # Adding a 4th item should evict the oldest item (tool1)
    cache.set("tool4", {"param": "4"}, tool_result)

    # tool1 should be evicted from cache
    assert cache.get("tool1", {"param": "1"}) is None

    # Other items should still be in cache
    assert cache.get("tool2", {"param": "2"}) == tool_result
    assert cache.get("tool3", {"param": "3"}) == tool_result
    assert cache.get("tool4", {"param": "4"}) == tool_result


def test_cache_clear(cache, tool_result):
    """Test cache clear functionality"""
    # Set some values in the cache
    cache.set("tool1", {"param": "1"}, tool_result)
    cache.set("tool2", {"param": "2"}, tool_result)

    # Clear the cache
    cache.clear()

    # All items should be removed from cache
    assert cache.get("tool1", {"param": "1"}) is None
    assert cache.get("tool2", {"param": "2"}) is None


def test_cache_stats(cache, tool_result):
    """Test cache statistics"""
    # Initially, hits and misses should be 0
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 0

    # Cache miss should be recorded
    cache.get("tool1", {"param": "1"})
    stats = cache.get_stats()
    assert stats["misses"] == 1

    # Set a value in the cache
    cache.set("tool1", {"param": "1"}, tool_result)

    # Cache hit should be recorded
    cache.get("tool1", {"param": "1"})
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["hit_ratio"] == 0.5  # 1 hit / (1 hit + 1 miss)


def test_cache_thread_safety():
    """Test thread safety"""
    cache = ToolResultCache(max_size=100, ttl_seconds=10)
    tool_result = {"toolUseId": "test_id", "status": "success", "content": [{"text": "Test result"}]}

    # Access cache concurrently from multiple threads
    def worker(worker_id: int):
        for i in range(50):
            # Alternate between read and write operations
            if i % 2 == 0:
                cache.set(f"tool{worker_id}", {"param": str(i)}, tool_result)
            else:
                cache.get(f"tool{worker_id}", {"param": str(i - 1)})

    # Create and run 10 threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Test passes if no exceptions were raised
    # Additionally, verify cache size is correct
    stats = cache.get_stats()
    assert 0 <= stats["size"] <= 100  # Should not exceed max size


def test_cache_key_generation(cache, tool_result):
    """Test edge cases for cache key generation"""
    # Verify that different parameter order generates the same key
    cache.set("tool", {"a": 1, "b": 2}, tool_result)
    assert cache.get("tool", {"b": 2, "a": 1}) == tool_result

    # Test numeric values separately - don't expect 1 and 1.0 to be equivalent
    # since JSON serialization treats them differently
    cache.set("tool", {"num": 1}, tool_result)
    assert cache.get("tool", {"num": 1}) == tool_result

    cache.set("tool", {"num": 1.0}, tool_result)
    assert cache.get("tool", {"num": 1.0}) == tool_result

    # Verify that nested parameters are handled correctly
    cache.set("tool", {"nested": {"a": 1, "b": 2}}, tool_result)
    assert cache.get("tool", {"nested": {"b": 2, "a": 1}}) == tool_result


def test_cache_with_different_tool_results(cache):
    """Test caching different tool results"""
    result1 = {"toolUseId": "id1", "status": "success", "content": [{"text": "Result 1"}]}

    result2 = {"toolUseId": "id2", "status": "success", "content": [{"text": "Result 2"}]}

    # Cache different results for different tools
    cache.set("tool1", {"param": "value"}, result1)
    cache.set("tool2", {"param": "value"}, result2)

    # Verify correct results are retrieved
    assert cache.get("tool1", {"param": "value"}) == result1
    assert cache.get("tool2", {"param": "value"}) == result2
