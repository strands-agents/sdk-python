"""Integration tests for rate limiting functionality."""

import os
import threading
import time
from typing import List

import pytest

from strands import Agent
from strands.models import BedrockModel
from strands.models.rate_limiter import rate_limit_model, reset_rate_limits_for_testing


@pytest.fixture(autouse=True)
def clean_rate_limits():
    """Reset rate limits before each test to ensure isolation."""
    reset_rate_limits_for_testing()
    yield
    reset_rate_limits_for_testing()


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_basic_rate_limiting():
    """Test basic rate limiting with real API calls."""
    # Use very low RPM for quick test
    # Using Sonnet model that's verified to work in test_model_bedrock.py
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    # Use explicit bucket key to ensure consistency
    bucket_key = "test-basic-limit"
    limited_model = rate_limit_model(model, rpm=2, window=10, timeout=0.5, bucket_key=bucket_key)

    agent = Agent(model=limited_model, load_tools_from_directory=False, callback_handler=None)

    # Should succeed for first 2 calls
    result1 = agent("Reply with just the word 'one'")
    assert "one" in str(result1).lower()

    result2 = agent("Reply with just the word 'two'")
    assert "two" in str(result2).lower()

    # Third call should timeout
    with pytest.raises(TimeoutError, match="Rate limit token acquisition timed out"):
        agent("Reply with just the word 'three'")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_shared_rate_limits():
    """Test multiple agents sharing rate limits."""
    # Create two agents with shared bucket
    bucket_key = "shared-integ-test"

    model1 = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    # Use lower RPM and shorter timeout to ensure we hit the limit
    limited_model1 = rate_limit_model(model1, rpm=2, window=10, bucket_key=bucket_key, timeout=0.1)

    model2 = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    limited_model2 = rate_limit_model(model2, rpm=2, window=10, bucket_key=bucket_key, timeout=0.1)

    agent1 = Agent(model=limited_model1, load_tools_from_directory=False, callback_handler=None)
    agent2 = Agent(model=limited_model2, load_tools_from_directory=False, callback_handler=None)

    # Use up rate limit with both agents (2 RPM = 2 requests)
    agent1("Reply with just 'one'")
    agent2("Reply with just 'two'")

    # Both should now be rate limited
    with pytest.raises(TimeoutError):
        agent1("Reply with just 'three'")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_with_streaming():
    """Test rate limiting works correctly with streaming models."""
    # Use streaming model
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", streaming=True)
    limited_model = rate_limit_model(model, rpm=2, window=10, timeout=1.0)

    agent = Agent(model=limited_model, load_tools_from_directory=False, callback_handler=None)

    # First two calls should work
    result1 = agent("Reply with 'streaming one'")
    assert "one" in str(result1).lower()

    result2 = agent("Reply with 'streaming two'")
    assert "two" in str(result2).lower()

    # Third should fail
    with pytest.raises(TimeoutError):
        agent("Reply with 'streaming three'")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_concurrent_rate_limiting():
    """Test rate limiting under concurrent load."""
    # Create rate-limited model with slightly higher RPM
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    limited_model = rate_limit_model(model, rpm=5, window=10, timeout=0.1)

    results: List[bool] = []
    errors: List[Exception] = []
    lock = threading.Lock()

    def make_request(idx: int):
        """Make a request and record result."""
        try:
            agent = Agent(model=limited_model, load_tools_from_directory=False, callback_handler=None)
            agent(f"Reply with just the number '{idx}'")
            with lock:
                results.append(True)
        except TimeoutError as e:
            with lock:
                errors.append(e)

    # Create threads for concurrent requests
    threads = []
    for i in range(8):  # More than RPM limit
        thread = threading.Thread(target=make_request, args=(i,))
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Should have exactly 5 successes (RPM limit) and 3 timeouts
    assert len(results) == 5, f"Expected 5 successes, got {len(results)}"
    assert len(errors) == 3, f"Expected 3 timeouts, got {len(errors)}"
    assert all(isinstance(e, TimeoutError) for e in errors)


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_with_tools():
    """Test rate limiting works when agent uses tools."""
    import strands

    @strands.tool
    def get_time() -> str:
        """Get the current time."""
        return "12:00 PM"

    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    # Tool use requires 2 API calls per request (initial + tool result)
    # Use 2 RPM to allow exactly 1 tool-using request
    bucket_key = "test-tools-unique"
    limited_model = rate_limit_model(model, rpm=2, window=60, bucket_key=bucket_key, timeout=0.5)

    agent = Agent(model=limited_model, tools=[get_time], load_tools_from_directory=False, callback_handler=None)

    # First call should work (uses 2 tokens: one for initial request, one for tool result)
    result1 = agent("What time is it?")
    assert "12:00" in str(result1)

    # Second call should fail immediately because we've used both tokens
    with pytest.raises(TimeoutError):
        agent("Tell me the current time")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_without_timeout_timing():
    """Test rate limiting without timeout - the typical use case."""
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    # 4 RPM = 4 tokens in bucket
    bucket_key = "test-no-timeout"
    limited_model = rate_limit_model(model, rpm=4, window=60, bucket_key=bucket_key)  # No timeout!

    # Make 6 parallel requests with 4 RPM capacity
    # 4 should complete quickly, 2 should wait
    results = []
    lock = threading.Lock()

    def make_request(idx: int):
        """Make a request and record completion time."""
        start = time.time()
        agent = Agent(model=limited_model, load_tools_from_directory=False, callback_handler=None)
        agent(f"Say '{idx}'")
        elapsed = time.time() - start
        with lock:
            results.append(elapsed)

    # Start all threads
    threads = []
    for i in range(6):
        thread = threading.Thread(target=make_request, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all to complete
    for thread in threads:
        thread.join()

    # Sort results by completion time
    results.sort()

    # Check that there's a clear timing difference between the first 4 and last 2 requests
    # The first 4 should have similar timing (all getting tokens immediately)
    # The last 2 should be noticeably slower (waiting for token refill)

    avg_first_four = sum(results[:4]) / 4
    avg_last_two = sum(results[4:]) / 2

    # The last 2 requests should take at least 50% longer than the first 4 on average
    # This accounts for the fact they need to wait for token refill
    ratio = avg_last_two / avg_first_four
    assert ratio > 1.5, (
        f"Expected last 2 requests to be at least 50% slower than first 4. "
        f"First 4 avg: {avg_first_four:.2f}s, Last 2 avg: {avg_last_two:.2f}s, "
        f"Ratio: {ratio:.2f}. All times: {[f'{t:.2f}' for t in results]}"
    )


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_error_handling():
    """Test that rate limiter counts failed requests towards the limit."""
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    limited_model = rate_limit_model(model, rpm=2, window=10, timeout=0.5)

    agent = Agent(model=limited_model, load_tools_from_directory=False, callback_handler=None)

    # Make a request that might fail (e.g., with a very complex prompt)
    # Even if it fails, it should count towards rate limit
    try:
        agent("Reply with 'one'")
    except Exception:
        pass  # Ignore any errors

    try:
        agent("Reply with 'two'")
    except Exception:
        pass

    # Third request should be rate limited regardless of previous failures
    with pytest.raises(TimeoutError):
        agent("Reply with 'three'")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_class_wrapping():
    """Test creating rate-limited model classes."""
    # Create a rate-limited class
    LimitedBedrockModel = rate_limit_model(BedrockModel, rpm=2, window=10, timeout=0.5)

    # Should be able to instantiate multiple times
    model1 = LimitedBedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    model2 = LimitedBedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")

    # Both should share the same rate limit
    agent1 = Agent(model=model1, load_tools_from_directory=False, callback_handler=None)
    agent2 = Agent(model=model2, load_tools_from_directory=False, callback_handler=None)

    # Use up the shared limit
    agent1("Say 'one'")
    agent2("Say 'two'")

    # Both should be rate limited
    with pytest.raises(TimeoutError):
        agent1("Say 'three'")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_automatic_bucket_key():
    """Test automatic bucket key generation for rate limiters."""
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")

    # Create a rate limiter without specifying bucket_key
    limited = rate_limit_model(model, rpm=2, window=10, timeout=0.5)

    agent = Agent(model=limited, load_tools_from_directory=False, callback_handler=None)

    # Should be able to make 2 requests with first limiter
    agent("Say 'one'")
    agent("Say 'two'")

    # Third request should timeout
    with pytest.raises(TimeoutError):
        agent("Say 'three'")


@pytest.mark.skipif("AWS_REGION" not in os.environ, reason="AWS credentials not configured")
def test_rate_limiter_steady_rate():
    """Test that rate limiter correctly limits parallel requests."""
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    # 4 RPM = 4 tokens in the bucket initially
    bucket_key = "test-parallel-steady"
    limited_model = rate_limit_model(model, rpm=4, window=60, timeout=0.5, bucket_key=bucket_key)

    # Make 6 parallel requests - more than the 4 token capacity
    results: List[bool] = []
    errors: List[Exception] = []
    lock = threading.Lock()

    def make_request(idx: int):
        """Make a request and record result."""
        try:
            agent = Agent(model=limited_model, load_tools_from_directory=False, callback_handler=None)
            agent(f"Say '{idx}'")
            with lock:
                results.append(True)
        except TimeoutError as e:
            with lock:
                errors.append(e)

    # Create threads for parallel requests
    threads = []
    for i in range(6):
        thread = threading.Thread(target=make_request, args=(i,))
        threads.append(thread)

    # Start all threads at once
    start_time = time.time()
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    elapsed = time.time() - start_time

    # Should have 4 successes and 2 timeouts
    assert len(results) == 4, f"Expected 4 successes, got {len(results)}"
    assert len(errors) == 2, f"Expected 2 timeouts, got {len(errors)}"
    assert all(isinstance(e, TimeoutError) for e in errors)

    # Test should complete quickly
    assert elapsed < 5.0, f"Test took too long: {elapsed:.2f}s"
