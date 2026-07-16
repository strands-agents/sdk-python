"""Code execution tool for running source code through a configured sandbox.

Each call runs a fresh interpreter invocation; state does not persist across
calls. The tool refuses to execute when the agent has no isolating sandbox
configured -- the sandbox module owns the security boundary.

Example Usage:
    ```python
    from strands import Agent
    from strands.sandbox.docker import DockerSandbox
    from strands.vended_tools import code_execution

    agent = Agent(sandbox=DockerSandbox(container="my-container"), tools=[code_execution])
    ```
"""

from .code_execution import code_execution, make_code_execution
from .types import (
    CODE_EXECUTION_DESCRIPTION,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CODE_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    CodeExecutionOutput,
)

__all__ = [
    "CODE_EXECUTION_DESCRIPTION",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MAX_CODE_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "CodeExecutionOutput",
    "code_execution",
    "make_code_execution",
]
