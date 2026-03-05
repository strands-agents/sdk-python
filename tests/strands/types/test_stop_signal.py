"""Unit tests for threading.Event usage in agent cancellation."""

import threading
import time


def test_threading_event_initial_state():
    """Test that event starts in non-set state.

    Verifies that a newly created threading.Event is not in a set state.
    """
    event = threading.Event()
    assert not event.is_set()


def test_threading_event_set():
    """Test that set() transitions the event to set state.

    Verifies that calling set() transitions the event to set state.
    """
    event = threading.Event()
    event.set()
    assert event.is_set()


def test_threading_event_idempotent():
    """Test that multiple set() calls are safe.

    Verifies that calling set() multiple times doesn't cause any issues.
    """
    event = threading.Event()
    event.set()
    event.set()
    assert event.is_set()


def test_threading_event_thread_safety():
    """Test that event is thread-safe.

    Verifies that multiple threads can set the event simultaneously without race conditions.
    """
    event = threading.Event()
    results = []

    def set_from_thread():
        event.set()
        results.append(event.is_set())

    threads = [threading.Thread(target=set_from_thread) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # All threads should see set state
    assert all(results)
    assert event.is_set()


def test_threading_event_multiple_threads_checking():
    """Test that multiple threads can check event state simultaneously.

    Verifies that one thread can set the event while another thread is checking the event state.
    """
    event = threading.Event()
    check_results = []
    set_done = threading.Event()

    def check_repeatedly():
        # Check multiple times
        for _ in range(100):
            check_results.append(event.is_set())
            time.sleep(0.001)

    def set_after_delay():
        time.sleep(0.05)  # Let checker run for a bit
        event.set()
        set_done.set()

    checker_thread = threading.Thread(target=check_repeatedly)
    setter_thread = threading.Thread(target=set_after_delay)

    checker_thread.start()
    setter_thread.start()

    checker_thread.join()
    setter_thread.join()

    # Should have both False and True results
    assert False in check_results  # Some checks before set
    assert True in check_results  # Some checks after set
    assert event.is_set()  # Final state is set
