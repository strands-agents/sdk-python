Looking for something beyond the built-in implementations? Build a custom sandbox when your execution environment is not a local Docker container or an SSH host: a microVM, a cloud code-execution API, or a managed runtime. The agent loop, model, and vended tools all stay the same; you only implement the methods that run commands and access files in your backend.

## Extending PosixShellSandbox

`PosixShellSandbox` is a base class that reduces the implementation burden to a single method. If you can implement `execute_streaming``executeStreaming` (run a shell command via your backend and stream the output), you get everything else for free:

-   Code execution via base64-encoded heredoc piped to the interpreter
-   File read/write via base64 encoding over the shell
-   Directory listing via `ls`

Both `DockerSandbox` and `SshSandbox` extend `PosixShellSandbox`.

(( tab "TypeScript" ))
```typescript
import { spawn } from 'node:child_process'
import { PosixShellSandbox } from '@strands-agents/sdk/sandbox'
import type { ExecuteOptions, StreamChunk, ExecutionResult } from '@strands-agents/sdk/sandbox'

class FirecrackerSandbox extends PosixShellSandbox {
  constructor(private readonly vmId: string) {
    super()
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    const proc = spawn('fc-exec', [this.vmId, 'sh', '-c', command])

    let stdout = ''
    let stderr = ''
    for await (const data of proc.stdout) {
      const text = data.toString()
      stdout += text
      yield { type: 'streamChunk', data: text, streamType: 'stdout' }
    }
    for await (const data of proc.stderr) {
      const text = data.toString()
      stderr += text
      yield { type: 'streamChunk', data: text, streamType: 'stderr' }
    }
    const exitCode: number = await new Promise((resolve) =>
      proc.on('close', (code) => resolve(code ?? 0))
    )
    yield { type: 'executionResult', exitCode, stdout, stderr, outputFiles: [] }
  }
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from strands.sandbox import PosixShellSandbox
from strands.sandbox.types import ExecutionResult, StreamChunk


class FirecrackerSandbox(PosixShellSandbox):
    """Run commands in a Firecracker microVM addressed by id."""

    def __init__(self, vm_id: str) -> None:
        self.vm_id = vm_id

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        proc = await asyncio.create_subprocess_exec(
            "fc-exec", self.vm_id, "sh", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if stdout:
            yield StreamChunk(data=stdout.decode(), stream_type="stdout")
        if stderr:
            yield StreamChunk(data=stderr.decode(), stream_type="stderr")
        yield ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )
```
(( /tab "Python" ))

## Vending tools from a custom sandbox

To give your sandbox the same `sandbox_bash` and `sandbox_file_editor` tools the built-in sandboxes provide, override `getTools()` / `get_tools()` and return tools bound to it:

(( tab "TypeScript" ))
```typescript
import type { Tool } from '@strands-agents/sdk'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
import { makeFileEditor } from '@strands-agents/sdk/vended-tools/file-editor'

override getTools(): Tool[] {
  return [
    makeFileEditor(this, { name: 'sandbox_file_editor' }),
    makeBash(this, { name: 'sandbox_bash' }),
  ]
}
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands.types.tools import AgentTool
from strands.vended_tools import make_bash, make_file_editor


# Inside your custom sandbox class:
def get_tools(self) -> list[AgentTool]:
    return [
        make_file_editor(sandbox=self, name="sandbox_file_editor"),
        make_bash(sandbox=self, name="sandbox_bash"),
    ]
```
(( /tab "Python" ))

## Extend the base interface directly

For environments where you have native API access (no shell), extend `Sandbox` directly and implement all six abstract methods: `execute_streaming``executeStreaming`, `execute_code_streaming``executeCodeStreaming`, `read_file``readFile`, `write_file``writeFile`, `remove_file``removeFile`, and `list_files``listFiles`.

Prefer the shell base whenever your backend can run `sh -c`. Reach for the raw interface only when shaping every operation as a shell command would be a worse fit than calling your backend’s native API.

## Security

A custom sandbox is a boundary only when the environment behind it is isolated. The interface routes operations; it does not confine them. Whatever the agent can reach through your `execute_streaming` implementation (the method that runs commands in your environment), it can reach.

A container running as root with the host filesystem mounted is not a boundary, even though it uses the same `Sandbox` interface as a locked-down container. The security comes from the environment you provision, not from the interface itself. Scope the environment to the least privilege the task needs, and treat that configuration as the actual control.

## Next steps

-   [Sandbox Overview](/docs/user-guide/concepts/sandbox/index.md) — what a sandbox is, the tools it vends, and plugin compatibility
-   [Available Sandboxes](/docs/user-guide/concepts/sandbox/available-sandboxes/index.md) — the built-in Docker and SSH backends
-   [Vended Tools](/docs/user-guide/concepts/tools/vended-tools/index.md) — the `sandbox_bash` and `sandbox_file_editor` tool factories

## Related pages

- [Available Sandboxes](/docs/user-guide/concepts/sandbox/available-sandboxes/index.md) (2 shared tags)
- [Sandbox](/docs/user-guide/concepts/sandbox/index.md) (2 shared tags)
- [Creating a Custom Model Provider](/docs/user-guide/concepts/model-providers/custom_model_provider/index.md) (1 shared tag)
- [Tool Executors](/docs/user-guide/concepts/tools/executors/index.md) (1 shared tag)
- [Human in the Loop](/docs/user-guide/concepts/agents/interventions/human-in-the-loop/index.md) (1 shared tag)
- [Cedar Authorization](/docs/user-guide/concepts/agents/interventions/cedar-authorization/index.md) (1 shared tag)
- [Agent Loop](/docs/user-guide/concepts/agents/agent-loop/index.md) (1 shared tag)
- [Hooks](/docs/user-guide/concepts/agents/hooks/index.md) (1 shared tag)
- [Steering (Interventions)](/docs/user-guide/concepts/agents/interventions/steering/index.md) (1 shared tag)
- [Agents as Tools with Strands Agents SDK](/docs/user-guide/concepts/multi-agent/agents-as-tools/index.md) (1 shared tag)
