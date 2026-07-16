"""Shared types and constants for the code_execution tool."""

from typing import TypedDict


class CodeExecutionOutput(TypedDict):
    """Result of a code_execution call.

    Attributes:
        stdout: Standard output captured from the interpreter. May be truncated
            with a trailing marker if the sandbox produced more than
            :data:`DEFAULT_MAX_OUTPUT_BYTES` bytes.
        stderr: Standard error captured from the interpreter. Truncated on the
            same terms as ``stdout``.
        exit_code: Exit code from the interpreter. ``0`` indicates success.
        elapsed_ms: Wall-clock time in milliseconds from the call entering the
            sandbox to the sandbox returning a result.
    """

    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: int


#: Default interpreter used when the factory is not passed a ``language``.
#: The tool executes the SDK's own language; ``python3`` is Python's convention.
DEFAULT_LANGUAGE = "python3"

#: Default upper bound on the source-code size accepted from the model (bytes,
#: UTF-8). Keeps a runaway prompt from stuffing the sandbox with megabytes of
#: source before the interpreter ever runs.
DEFAULT_MAX_CODE_BYTES = 100_000

#: Default upper bound on the stdout/stderr the tool returns to the model
#: (bytes, UTF-8). Anything past this is dropped and a truncation marker is
#: appended so the model knows it happened.
DEFAULT_MAX_OUTPUT_BYTES = 100_000

#: Default execution timeout in seconds; passed through to the sandbox, which
#: owns the actual kill.
DEFAULT_TIMEOUT_SECONDS = 60

#: Marker appended when stdout/stderr is trimmed to :data:`DEFAULT_MAX_OUTPUT_BYTES`.
TRUNCATION_MARKER = "\n... [truncated]"

CODE_EXECUTION_DESCRIPTION = (
    "Executes source code through a configured sandbox and returns stdout, stderr, "
    "the exit code, and wall-clock elapsed milliseconds. Each call is a fresh "
    "interpreter invocation; state does not persist across calls. Requires an "
    "isolating sandbox to be configured on the agent."
)
"""Description shown to the model."""
