"""Code execution tool for running source code through a configured sandbox.

Provides :func:`make_code_execution` (a factory for a sandbox-routed code
execution tool) and :data:`code_execution` (the default instance that reads the
sandbox from the agent at call time). Each call runs a fresh interpreter through
the sandbox; state does not persist across calls.

The tool is a thin shim over :meth:`~strands.sandbox.base.Sandbox.execute_code`.
The sandbox is the security boundary; the tool refuses to execute if the agent
has no isolating sandbox configured (i.e. it falls back to
:class:`~strands.sandbox.not_a_sandbox_local_environment.NotASandboxLocalEnvironment`,
whose deliberately blunt name signals "no isolation").
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from ...sandbox.errors import SandboxTimeoutError
from ...sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import (
    CODE_EXECUTION_DESCRIPTION,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CODE_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    TRUNCATION_MARKER,
    CodeExecutionOutput,
)

if TYPE_CHECKING:
    from ...sandbox.base import Sandbox
    from ...tools.decorator import DecoratedFunctionTool


def _truncate(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to ``max_bytes`` UTF-8 bytes and append a marker if trimmed.

    Truncation is intentionally byte-oriented (not character-oriented) because
    the cap protects downstream consumers that may be counting tokens or
    bytes. Decoding with ``errors="ignore"`` drops any incomplete multi-byte
    sequence at the cut point rather than raising.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    trimmed = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return trimmed + TRUNCATION_MARKER


def make_code_execution(
    *,
    sandbox: Sandbox | None = None,
    language: str = DEFAULT_LANGUAGE,
    name: str = "code_execution",
    description: str = CODE_EXECUTION_DESCRIPTION,
    max_code_bytes: int = DEFAULT_MAX_CODE_BYTES,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> DecoratedFunctionTool:
    """Create a sandbox-routed code execution tool.

    If a ``sandbox`` is passed, it is bound at creation time. Otherwise the tool
    reads the sandbox from ``tool_context.agent.sandbox`` at call time and
    refuses to run when that sandbox is the host default
    (:class:`NotASandboxLocalEnvironment`).

    ``language`` is fixed at factory time -- the model cannot select an
    interpreter. This keeps the tool aligned with the SDK's own runtime (Python
    here) and prevents the tool from becoming a polyglot execution surface.

    Args:
        sandbox: Sandbox to bind at creation. When ``None``, the agent's
            configured sandbox is used at call time. When it is a
            :class:`NotASandboxLocalEnvironment`, the tool refuses to execute.
        language: Interpreter to run. Passed through to the sandbox, which
            validates it against :data:`~strands.sandbox.LANGUAGE_PATTERN`.
        name: Tool name. Defaults to ``"code_execution"``.
        description: Tool description shown to the model.
        max_code_bytes: Upper bound on ``code`` size (bytes, UTF-8). Rejects
            larger inputs before touching the sandbox.
        max_output_bytes: Upper bound on stdout/stderr size returned to the
            model (bytes, UTF-8). Excess is dropped and a truncation marker is
            appended.
        default_timeout: Timeout in seconds passed to the sandbox when the
            caller does not supply one.

    Returns:
        A decorated tool that executes code through the sandbox.
    """
    # Reject NaN and Infinity explicitly: ``nan <= 0`` is false, so a bare
    # ``<= 0`` check would silently disable the cap (``code_bytes > nan`` is
    # always false).
    if not isinstance(max_code_bytes, int) or isinstance(max_code_bytes, bool) or max_code_bytes <= 0:
        raise ValueError(f"max_code_bytes must be a positive integer, got {max_code_bytes!r}")
    if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
        raise ValueError(f"max_output_bytes must be a positive integer, got {max_output_bytes!r}")
    if (
        isinstance(default_timeout, bool)
        or not isinstance(default_timeout, (int, float))
        or not math.isfinite(default_timeout)
        or default_timeout <= 0
    ):
        raise ValueError(f"default_timeout must be a positive, finite number, got {default_timeout!r}")

    async def code_execution_tool(
        code: str,
        tool_context: ToolContext,
        timeout: float = default_timeout,
    ) -> CodeExecutionOutput:
        # Docstring is set below so the ``timeout`` default in the model-facing
        # schema reflects the factory's ``default_timeout`` rather than a
        # hardcoded literal.
        # Input validation at the tool boundary.
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        code_bytes = len(code.encode("utf-8"))
        if code_bytes > max_code_bytes:
            raise ValueError(f"code size ({code_bytes} bytes) exceeds maximum allowed size ({max_code_bytes} bytes)")
        if timeout is not None:
            # Reject non-finite (nan, +/-inf) and booleans explicitly. ``nan <= 0``
            # is false, so a bare comparison would let a non-finite timeout past
            # the tool boundary and into the sandbox.
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError(f"timeout must be a positive, finite number, got {timeout!r}")

        active = sandbox if sandbox is not None else tool_context.agent.sandbox

        # The sandbox is the security boundary. Refuse execution when the agent
        # is running against the host default -- its name says "no isolation"
        # and executing model-authored code there would be a footgun.
        if isinstance(active, NotASandboxLocalEnvironment):
            raise RuntimeError(
                "code_execution requires an isolating sandbox (e.g. DockerSandbox, SshSandbox) "
                "to be configured on the agent. Refusing to execute against "
                "NotASandboxLocalEnvironment, which provides no isolation."
            )

        started = time.monotonic()
        try:
            result = await active.execute_code(code, language, timeout=timeout)
        except SandboxTimeoutError:
            # Preserve the sandbox's timeout type so callers can branch on it.
            raise
        except Exception as e:
            raise RuntimeError(str(e)) from e
        elapsed_ms = int((time.monotonic() - started) * 1000)

        return {
            "stdout": _truncate(result.stdout, max_output_bytes),
            "stderr": _truncate(result.stderr, max_output_bytes),
            "exit_code": result.exit_code,
            "elapsed_ms": elapsed_ms,
        }

    # Assign the docstring after the definition so ``default_timeout`` is
    # interpolated dynamically; the tool decorator parses this for the
    # model-facing parameter descriptions.
    code_execution_tool.__doc__ = (
        "Executes source code and returns stdout, stderr, exit_code, elapsed_ms.\n"
        "\n"
        "Args:\n"
        "    code: Source code to execute in the configured language.\n"
        "    tool_context: Injected by the framework. Not user-facing.\n"
        f"    timeout: Timeout in seconds (default: {default_timeout}).\n"
    )

    return tool(name=name, description=description, context="tool_context")(code_execution_tool)


code_execution = make_code_execution()
"""Default code execution tool. Reads the sandbox from the agent's context at call time."""
