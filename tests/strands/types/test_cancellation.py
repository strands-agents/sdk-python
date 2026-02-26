"""Tests for CancellationToken."""

import threading
import time

from strands.types import CancellationToken


def test_cancellation_token_initial_state():
    """Test that token starts in non-cancelled state.

    Why: We need to ensure the default state is False so agents
    don't immediately stop when created with a token.
    """
    token = CancellationToken()
    assert not token.is_cancelled()


def test_cancellation_token_cancel():
    """Test that cancel() sets cancelled state.

    Why: This is the core functionality - when cancel() is called,
    the token must transition to cancelled state.
    """
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled()


def test_cancellation_token_idempotent():
    """Test that multiple cancel() calls are safe.

    Why: In distributed systems, multiple cancel requests might arrive.
    The token must handle this gracefully without errors or side effects.
    """
    token = CancellationToken()
    token.cancel()
    token.cancel()
    token.cancel()
    assert token.is_cancelled()


def test_cancellation_token_thread_safety():
    """Test that token is thread-safe.

    Why: The token will be accessed from multiple threads:
    - Main thread: agent checking is_cancelled()
    - Background thread: poller calling cancel()

    This test ensures no race conditions occur.
    """
    token = CancellationToken()
    results = []

    def cancel_from_thread():
        time.sleep(0.01)  # Small delay to ensure check_from_thread starts first
        token.cancel()

    def check_from_thread():
        for _ in range(100):
            results.append(token.is_cancelled())
            time.sleep(0.001)

    t1 = threading.Thread(target=cancel_from_thread)
    t2 = threading.Thread(target=check_from_thread)

    t2.start()
    t1.start()

    t1.join()
    t2.join()

    # Should have some False and some True values
    # This proves the state transition was visible across threads
    assert False in results
    assert True in results


def test_cancellation_token_multiple_threads_checking():
    """Test that multiple threads can check cancellation simultaneously.

    Why: In complex agents, multiple components might check cancellation
    at the same time. This ensures thread-safe reads.
    """
    token = CancellationToken()
    results = []

    def check_repeatedly():
        for _ in range(50):
            results.append(token.is_cancelled())
            time.sleep(0.001)

    # Start multiple checker threads
    threads = [threading.Thread(target=check_repeatedly) for _ in range(3)]
    for t in threads:
        t.start()

    # Cancel while threads are checking
    time.sleep(0.025)
    token.cancel()

    for t in threads:
        t.join()

    # All threads should have seen both states
    assert False in results
    assert True in results
    # No exceptions should have occurred


def test_cancellation_token_shared_reference():
    """Test that token works when shared across objects.

    Why: This simulates the real use case where the same token
    is passed to both the agent and the poller. Changes made
    through one reference must be visible through the other.
    """
    token = CancellationToken()

    # Simulate agent holding reference
    class FakeAgent:
        def __init__(self, cancellation_token):
            self.cancellation_token = cancellation_token

    # Simulate poller holding reference
    class FakePoller:
        def __init__(self, cancellation_token):
            self.cancellation_token = cancellation_token

    agent = FakeAgent(cancellation_token=token)
    poller = FakePoller(cancellation_token=token)

    # Verify they're the same object
    assert agent.cancellation_token is token
    assert poller.cancellation_token is token
    assert agent.cancellation_token is poller.cancellation_token

    # Cancel through poller
    poller.cancellation_token.cancel()

    # Agent should see the change
    assert agent.cancellation_token.is_cancelled()
    assert token.is_cancelled()
