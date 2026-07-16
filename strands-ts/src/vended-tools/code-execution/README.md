# Code Execution Tool

Runs source code through a configured sandbox and returns stdout, stderr, the interpreter exit code, and wall-clock elapsed milliseconds.

The tool is a thin shim over Sandbox.executeCode. The sandbox is the security boundary; the tool refuses to execute when the agent falls back to NotASandboxLocalEnvironment, whose name signals no isolation. Language selection is fixed at factory time and is not exposed to the model. Every input the model controls is capped: code size, stdout and stderr size returned to the model, and execution time. Timeout and agent cancellation both surface as an Error whose name is AbortError, so callers can branch on error.name.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { codeExecution } from '@strands-agents/sdk/vended-tools/code-execution'

const agent = new Agent({
  sandbox: new DockerSandbox({ container: 'my-node-container' }),
  tools: [codeExecution],
})
await agent.invoke('Compute the sum of primes below one hundred and print the result.')
```

Custom factory:

```typescript
import { makeCodeExecution } from '@strands-agents/sdk/vended-tools/code-execution'

const tool = makeCodeExecution(sandbox, {
  name: 'sandbox_code',
  maxCodeBytes: 50_000,
  maxOutputBytes: 200_000,
  defaultTimeout: 30,
})
```

## API

### `codeExecution`

The default tool, produced by `makeCodeExecution()` with the built-in caps.

### `makeCodeExecution(sandbox?, options?)`

| Option           | Type     | Default          | Description                                                                    |
| ---------------- | -------- | ---------------- | ------------------------------------------------------------------------------ |
| `name`           | `string` | `code_execution` | Tool name.                                                                     |
| `description`    | `string` | (built-in)       | Description shown to the model.                                                |
| `language`       | `string` | `node`           | Interpreter to run. Fixed at factory time; not exposed to the model.           |
| `maxCodeBytes`   | `number` | `100000`         | Upper bound on the `code` input in UTF-8 bytes.                                |
| `maxOutputBytes` | `number` | `100000`         | Upper bound on returned stdout and stderr in UTF-8 bytes. Excess is truncated. |
| `defaultTimeout` | `number` | `60`             | Timeout in seconds when the caller does not supply one.                        |

Throws if any cap is not a positive number.

### Input

| Property  | Type     | Required | Description                                                       |
| --------- | -------- | -------- | ----------------------------------------------------------------- |
| `code`    | `string` | Yes      | Source code to execute in the configured language.                |
| `timeout` | `number` | No       | Timeout in seconds. Falls back to the factory's `defaultTimeout`. |

### Output

| Field       | Type     | Description                                                  |
| ----------- | -------- | ------------------------------------------------------------ |
| `stdout`    | `string` | Standard output. May end with a truncation marker if capped. |
| `stderr`    | `string` | Standard error. Same truncation rules as stdout.             |
| `exitCode`  | `number` | Interpreter exit code. Zero indicates success.               |
| `elapsedMs` | `number` | Wall-clock time from tool entry to sandbox return.           |
