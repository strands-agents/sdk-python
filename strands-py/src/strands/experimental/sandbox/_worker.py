"""Single-thread worker that pins a thread-unsendable object to one OS thread.

The native ``strands_shell.Shell`` is built with PyO3 and is *unsendable*: it
panics the interpreter if created, used, or dropped on any thread other than the
one that created it. Strands executes tools on a pool of threads and from
asyncio, so the shell must be confined to a thread of its own.

:class:`_ShellWorker` owns a dedicated daemon thread. The shell is constructed on
that thread and kept as a local variable in the thread's run loop — never stored
anywhere reachable from another thread — so when the loop exits, the shell is
dropped on the same thread that created it. Callers submit functions that run on
the worker thread and receive a :class:`concurrent.futures.Future`.
"""

import concurrent.futures
import queue
import threading
from collections.abc import Callable
from typing import Any


class _ShellWorker:
    """Run callables against a thread-pinned object on a single dedicated thread.

    Args:
        factory: Builds the pinned object on the worker thread. Any exception it
            raises is re-raised from the constructor.

    Raises:
        BaseException: Whatever ``factory`` raised, propagated to the caller.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._queue: queue.SimpleQueue[tuple[Callable[[Any], Any], concurrent.futures.Future[Any]] | None] = (
            queue.SimpleQueue()
        )
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(factory,), name="strands-shell", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            raise self._init_error

    def _run(self, factory: Callable[[], Any]) -> None:
        """Build the pinned object, then service the queue until shut down.

        The object lives only as a local here, so it is created and dropped on
        this thread — satisfying the unsendable contract end to end.
        """
        try:
            obj = factory()
        except BaseException as e:  # noqa: BLE001 - surfaced to constructor caller
            self._init_error = e
            self._ready.set()
            return
        # Drop the factory immediately: it may close over caller state, and this
        # frame stays alive for the worker's whole lifetime (it is a GC root), so
        # holding the factory could keep that state — including the owner — alive
        # and prevent the finalizer-driven shutdown.
        del factory
        self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(obj))
            except BaseException as e:  # noqa: BLE001 - propagated via the future
                future.set_exception(e)

    def submit(self, fn: Callable[[Any], Any]) -> "concurrent.futures.Future[Any]":
        """Schedule ``fn(obj)`` on the worker thread and return its future."""
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._queue.put((fn, future))
        return future

    def shutdown(self) -> None:
        """Signal the worker to stop; the pinned object is dropped on its thread.

        Idempotent and safe to call from any thread (it only enqueues a
        sentinel). The thread is a daemon, so a missed shutdown never blocks
        interpreter exit.
        """
        self._queue.put(None)
