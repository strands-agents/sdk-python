"""Tests for the thread-pinning worker.

``_ShellWorker`` confines a thread-unsendable object to a single OS thread. These
tests use a plain sentinel object (no native dependency) to verify the pinning
guarantees: the factory, every submitted call, and the drop all run on the same
thread, factory errors propagate to the constructor, and shutdown is safe.
"""

import threading

import pytest

from strands.experimental.sandbox._worker import _ShellWorker


def test_factory_runs_on_worker_thread():
    main_thread = threading.get_ident()
    worker = _ShellWorker(lambda: threading.get_ident())
    factory_thread = worker.submit(lambda obj: obj).result()
    call_thread = worker.submit(lambda obj: threading.get_ident()).result()
    # The object the factory built is the worker thread's id; the call runs there too.
    assert factory_thread != main_thread
    assert call_thread == factory_thread
    worker.shutdown()


def test_submitted_calls_receive_the_pinned_object():
    worker = _ShellWorker(lambda: ["state"])
    worker.submit(lambda obj: obj.append("more")).result()
    assert worker.submit(lambda obj: list(obj)).result() == ["state", "more"]
    worker.shutdown()


def test_factory_error_propagates_to_constructor():
    with pytest.raises(RuntimeError, match="boom"):
        _ShellWorker(lambda: (_ for _ in ()).throw(RuntimeError("boom")))


def test_call_exception_propagates_through_future():
    worker = _ShellWorker(lambda: object())
    with pytest.raises(ValueError, match="bad"):
        worker.submit(lambda obj: (_ for _ in ()).throw(ValueError("bad"))).result()
    # Worker survives a failed call and keeps serving.
    assert worker.submit(lambda obj: 42).result() == 42
    worker.shutdown()


def test_shutdown_is_idempotent():
    worker = _ShellWorker(lambda: object())
    worker.shutdown()
    worker.shutdown()  # second call must not raise


def test_calls_are_serialized_on_one_thread():
    # Every call observes the same thread id, proving single-threaded execution.
    worker = _ShellWorker(lambda: object())
    thread_ids = {worker.submit(lambda obj: threading.get_ident()).result() for _ in range(50)}
    assert len(thread_ids) == 1
    worker.shutdown()
