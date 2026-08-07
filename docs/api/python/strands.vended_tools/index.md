Built-in tools for commands, files, HTTP, and pausing.

The :func:`make_shell` and :func:`make_file_editor` factories produce sandbox-routed tools that either bind to a :class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH sandboxes do when vending tools) or read the sandbox from the agent at call time. Each :data:`shell` call runs in a fresh shell, so state does not persist between calls. The :data:`sleep` tool pauses execution for a bounded, cancellable duration.

The :data:`http_request` tool makes raw HTTP calls; use :func:`make_http_request` to supply a pre-configured `httpx.AsyncClient` with custom timeouts, redirects, authentication, or proxies.

Example Usage:

```python
from strands import Agent
from strands.vended_tools import file_editor, http_request, shell, sleep

agent = Agent(tools=[file_editor, http_request, shell, sleep])
```

#### make\_bash

```python
@deprecated(
    f"make_bash is deprecated and will be removed in v2.0.0. Use make_shell instead. \{_RENAME_RATIONALE}"
)
def make_bash(**kwargs: Any) -> "DecoratedFunctionTool"
```

Defined in: [src/strands/vended\_tools/**init**.py:43](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/__init__.py#L43)

Deprecated alias for :func:`make_shell`.