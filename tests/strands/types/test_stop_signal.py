"""Unit tests for StopSignal (internal cancellation mechanism)."""

import threading
import time

from strands.types.stop_signal import StopSignal


def test_stop_signal_initial_state():
    """Test that signal starts in non-cancelled state.

    Verifies that a newly created StopSignal is not in a cancelled state.
    """
    signal = StopSignal()
    assert not signal.is_cancelled()


def test_stop_signal_cancel():
    """Test that cancel() sets cancelled state.

    Verifies that calling cancel() transitions the signal to cancelled state.
    """
    signal = StopSignal()
    signal.cancel()
    assert signal.is_cancelled()


def test_stop_signal_idempotent():
    """Test that multiple cancel() calls are safe.

    Verifies that calling cancel() multiple times is idempotent and doesn't
    cause any issues.
    """
    signal = StopSignal()
    signal.cancel()
    signal.cancel()
    signal.cancel()
    assert signal.is_cancelled()


def test_stop_signal_thread_safety():
    """Test that signal is thread-safe.

    Verifies that the signal can be safely cancelled from multiple threads
    simultaneously without race conditions.
    """
    signal = StopSignal()
    results = []

    def cancel_from_thread():
        signal.cancel()
        results.append(signal.is_cancelled())

    threads = [threading.Thread(target=cancel_from_thread) for _ in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # All threads should see cancelled state
    assert all(results)
    assert signal.is_cancelled()


def test_stop_signal_multiple_threads_checking():
    """Test that multiple threads can check cancellation simultaneously.

    Verifies that multiple threads can safely check the cancellation state
    while another thread is cancelling the signal.
    """
    signal = StopSignal()
    check_results = []
    cancel_done = threading.Event()

    def check_cancellation():
        # Wait a bit to let cancel thread start
        time.sleep(0.01)
        # Check multiple times
        for _ in range(100):
            check_results.append(signal.is_cancelled())
            time.sleep(0.001)

    def cancel_signal():
        time.sleep(0.05)
        signal.cancel()
        cancel_done.set()

    # Start multiple checker threads
    checker_threads = [threading.Thread(target=check_cancellation) for _ in range(5)]
    cancel_thread = threading.Thread(target=cancel_signal)

    for thread in checker_threads:
        thread.start()
    cancel_thread.start()

    for thread in checker_threads:
        thread.join()
    cancel_thread.join()

    # Should have mix of False and True results
    assert False in check_results  # Some checks before cancel
    assert True in check_results  # Some checks after cancel
    assert signal.is_cancelled()  # Final state is cancelled


def test_stop_signal_shared_reference():
    """Test that signal works when shared across objects.

    This test simulates the real-world scenario where a StopSignal is shared
    between an agent and an external poller (e.g., DynamoDB poller in
    MaxDomeSageMindAgent). When the poller cancels the signal, the agent
    should see the change immediately since they share the same object reference.
    """
    signal = StopSignal()

    # Simulate agent holding reference
    class FakeAgent:
        def __init__(self, stop_signal):
            self.stop_signal = stop_signal

    # Simulate poller holding reference
    class FakePoller:
        def __init__(self, stop_signal):
            self.stop_signal = stop_signal

    agent = FakeAgent(stop_signal=signal)
    poller = FakePoller(stop_signal=signal)

    # Verify they're the same object
    assert agent.stop_signal is signal
    assert poller.stop_signal is signal
    assert agent.stop_signal is poller.stop_signal

    # Cancel through poller
    poller.stop_signal.cancel()

    # Agent should see the change
    assert agent.stop_signal.is_cancelled()
    assert signal.is_cancelled()
