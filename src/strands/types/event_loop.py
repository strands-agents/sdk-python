"""Event loop-related type definitions for the SDK."""

from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Protocol

from typing_extensions import TypedDict, runtime_checkable


class Usage(TypedDict):
    """Token usage information for model interactions.

    Attributes:
        inputTokens: Number of tokens sent in the request to the model..
        outputTokens: Number of tokens that the model generated for the request.
        totalTokens: Total number of tokens (input + output).
    """

    inputTokens: int
    outputTokens: int
    totalTokens: int


class Metrics(TypedDict):
    """Performance metrics for model interactions.

    Attributes:
        latencyMs (int): Latency of the model request in milliseconds.
    """

    latencyMs: int


StopReason = Literal[
    "content_filtered",
    "end_turn",
    "guardrail_intervened",
    "max_tokens",
    "stop_sequence",
    "tool_use",
]
"""Reason for the model ending its response generation.

- "content_filtered": Content was filtered due to policy violation
- "end_turn": Normal completion of the response
- "guardrail_intervened": Guardrail system intervened
- "max_tokens": Maximum token limit reached
- "stop_sequence": Stop sequence encountered
- "tool_use": Model requested to use a tool
"""


@runtime_checkable
class Future(Protocol):
    """Interface representing the result of an asynchronous computation."""

    def result(self, timeout: Optional[int] = None) -> Any:
        """Return the result of the call that the future represents.

        This method will block until the asynchronous operation completes or until the specified timeout is reached.

        Args:
            timeout: The number of seconds to wait for the result.
                If None, then there is no limit on the wait time.

        Returns:
            Any: The result of the asynchronous operation.
        """


@runtime_checkable
class ParallelToolExecutorInterface(Protocol):
    """Interface for parallel tool execution.

    Attributes:
        timeout: Default timeout in seconds for futures.
    """

    timeout: int = 900  # default 15 minute timeout for futures

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        """Submit a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns a Future instance representing the
        execution of the callable.

        Args:
            fn: The callable to execute.
            *args: Positional arguments to pass to the callable.
            **kwargs: Keyword arguments to pass to the callable.

        Returns:
            Future: A Future representing the given call.
        """

    def as_completed(self, futures: Iterable[Future], timeout: Optional[int] = timeout) -> Iterator[Future]:
        """Iterate over the given futures, yielding each as it completes.

        Args:
            futures: The sequence of Futures to iterate over.
            timeout: The maximum number of seconds to wait.
                If None, then there is no limit on the wait time.

        Returns:
            An iterator that yields the given Futures as they complete (finished or cancelled).
        """

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor and free associated resources.

        Args:
            wait: If True, shutdown will not return until all running futures have finished executing.
        """
