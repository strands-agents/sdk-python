Shared types and constants for the shell tool.

## ShellOutput

```python
class ShellOutput(TypedDict)
```

Defined in: [src/strands/vended\_tools/shell/types.py:6](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/shell/types.py#L6)

Output of a shell command execution.

**Attributes**:

-   `output` - Standard output captured from the command.
-   `error` - Standard error captured from the command. Empty when there was none.

## ShellExecutionError

```python
class ShellExecutionError(RuntimeError)
```

Defined in: [src/strands/vended\_tools/shell/types.py:18](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/shell/types.py#L18)

Raised when a sandbox-routed shell command fails.

Subclasses :class:`RuntimeError` so existing `except RuntimeError` handlers keep working, while giving callers a shell-specific type to branch on. Mirrors `ShellExecutionError` in `strands-ts/src/vended-tools/bash/types.ts`.

#### SANDBOX\_SHELL\_DESCRIPTION

Description for the shell tool.