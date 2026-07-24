Vended tools are pre-built tools included directly in the Strands SDK for common agent tasks like file operations, shell commands, HTTP requests, and persistent notes.

They ship as part of the SDK package and are updated alongside it — see [Versioning & Maintenance](#versioning--maintenance) for details on how changes are communicated and what level of backwards compatibility they maintain.

## Quick Start

Each tool is imported from its own subpath under `@strands-agents/sdk/vended-tools` — no additional packages required:

```typescript
import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

const agent = new Agent({
  tools: [bash, fileEditor, httpRequest, notebook],
})
```

## Available Tools

| Tool | Description | Supported in |
| --- | --- | --- |
| [File Editor](#file-editor) | View, create, and edit files | Python, TypeScript (Node.js) |
| [HTTP Request](#http-request) | Make HTTP requests to external APIs | Python, TypeScript (Node.js 20+, browsers) |
| [Notebook](#notebook) | Manage persistent text notebooks | TypeScript (Node.js, browsers) |
| [Bash](#bash) | Execute shell commands with persistent sessions | Python, TypeScript (Node.js, Unix/Linux/macOS) |
| [Sleep](#sleep) | Pause execution for a bounded, cancellable duration | Python, TypeScript (Node.js, browsers) |
| [Stop](#stop-experimental) | Gracefully end the agent loop when the task is complete | Python, TypeScript (Node.js, browsers) |

### File Editor

Gives your agent the ability to read and modify files on disk — useful for coding agents, config management, or any workflow where the agent needs to inspect output and make targeted edits.

Security Warning

This tool reads and writes files at arbitrary absolute paths with the full permissions of the process. Only use with trusted input and consider running in a [sandboxed environment](/docs/user-guide/concepts/sandbox/index.md) for production.

**Example:**

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'

const agent = new Agent({
  tools: [fileEditor],
})

// Create, view, and edit files
await agent.invoke('Create a file /tmp/config.json with {"debug": false}')
await agent.invoke('Replace "debug": false with "debug": true in /tmp/config.json')
await agent.invoke('View lines 1-10 of /tmp/config.json')
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_tools import file_editor

agent = Agent(tools=[file_editor])
agent("Create a file at /tmp/hello.txt with the contents 'Hello, world!'")
```
(( /tab "Python" ))

📖 [Full API Reference](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/file-editor/README.md)

---

### HTTP Request

Lets your agent call external APIs and fetch web content. Supports all HTTP methods, custom headers, and request bodies. Default timeout is 30 seconds.

*Supported in: Python; Node.js 20+, modern browsers (TypeScript).*

The Python tool ships with a strict default posture that rejects non-public destinations (loopback, RFC1918, link-local, multicast, reserved, and known cloud-metadata endpoints), refuses non-http and non-https schemes, caps redirect chains, response bodies, and response headers, and rejects model-supplied Authorization, Cookie, and Proxy-Authorization headers unless the tool operator has opted the target host in. Redirects are re-validated at every hop and cross-origin hops strip credentials. Use the make\_http\_request factory to relax individual controls when the deployment requires it.

Security Warning

Even with the default posture, this tool can call any public URL. Only use with trusted input and consider running in a [sandboxed environment](/docs/user-guide/concepts/sandbox/index.md) for production.

**Example:**

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'

const agent = new Agent({
  tools: [httpRequest],
})

// Make API requests
await agent.invoke('Get data from https://api.example.com/users')
await agent.invoke('Post {"name": "John"} to https://api.example.com/users')
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_tools import http_request

agent = Agent(tools=[http_request])
agent("Get data from https://api.example.com/data")
```

Custom posture:

```python
from strands import Agent
from strands.vended_tools import make_http_request

tool = make_http_request(
    allow_private_hosts=["metrics.internal.example.com"],
    allow_auth_for_hosts=["api.example.com"],
)
agent = Agent(tools=[tool])
```
(( /tab "Python" ))

📖 Full API Reference: [TypeScript](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/http-request/README.md) · [Python](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/README.md)

---

### Notebook

A scratchpad the agent can read and write across invocations. The most effective use is giving the agent a notebook at the start of a task and instructing it to plan its work there — it can break the task into steps, check things off as it goes, and always have a clear picture of what’s left. Notebook state is part of the agent’s state, so it persists automatically with [Session Management](/docs/user-guide/concepts/agents/session-management/index.md).

*Supported in: Node.js, browsers.*

**Example - Task Management:**

```typescript
import { Agent } from '@strands-agents/sdk'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

const agent = new Agent({
  tools: [notebook],
  systemPrompt:
    'Before starting any multi-step task, create a notebook with a checklist of steps. ' +
    'Check off each step as you complete it.',
})

// The agent uses the notebook to plan and track its work
await agent.invoke('Write a project plan for building a personal budget tracker app')
```

**Example - State Persistence:**

```typescript
import { Agent, SessionManager, FileStorage } from '@strands-agents/sdk'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

const session = new SessionManager({
  sessionId: 'my-session',
  storage: { snapshot: new FileStorage('./sessions') },
})

const agent = new Agent({ tools: [notebook], sessionManager: session })

// Notebooks are automatically persisted as part of the session
await agent.invoke('Create a notebook called "ideas" with "# Project Ideas"')
await agent.invoke('Add "- Build a web scraper" to the ideas notebook')

// ...

// Later, a new agent with the same session restores notebooks automatically
const restoredAgent = new Agent({ tools: [notebook], sessionManager: session })
await restoredAgent.invoke('Read the ideas notebook')
```

📖 [Full API Reference](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/notebook/README.md)

---

### Bash

Lets your agent run shell commands and act on the output. Shell state — variables, working directory, exported functions — persists across invocations within the same session, so the agent can build up context incrementally. Sessions can be restarted to clear state.

*Supported in: Node.js on Unix/Linux/macOS (TypeScript), all platforms (Python).*

Security Warning

This tool executes arbitrary bash commands. Without a [Sandbox](/docs/user-guide/concepts/sandbox/index.md), commands run with the full permissions of the process. Only use with trusted input and consider running in a [sandboxed environment](/docs/user-guide/concepts/sandbox/index.md) for production.

**Example - File Operations:**

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'

const agent = new Agent({
  tools: [bash],
})

// List files and create a new file
await agent.invoke('List all files in the current directory')
await agent.invoke('Create a new file called notes.txt with "Hello World"')
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_tools import bash

agent = Agent(tools=[bash])
agent("List all Python files in the current directory and count them")
```
(( /tab "Python" ))

**Example - Session Persistence (TypeScript):**

```typescript
import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'

const agent = new Agent({
  tools: [bash],
})

// Variables persist across invocations within the same session
await agent.invoke('Run: export MY_VAR="hello"')
await agent.invoke('Run: echo $MY_VAR') // Will show "hello"

// Restart session to clear state
await agent.invoke('Restart the bash session')
await agent.invoke('Run: echo $MY_VAR') // Variable will be empty
```

📖 [Full API Reference](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/bash/README.md)

---

### Sleep

Pauses the agent for a bounded number of seconds. Cancelling the enclosing invocation aborts the sleep immediately rather than waiting for the full duration, so a long timer never ties up a session the caller has moved on from.

*Supported in: Node.js, modern browsers (TypeScript); all platforms (Python).*

The maximum duration is configurable at construction (default: 60 seconds) and cannot be raised by the model. Negative, `NaN`, infinite, non-numeric, and boolean durations are rejected at the tool boundary.

**Example:**

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { sleep } from '@strands-agents/sdk/vended-tools/sleep'

const agent = new Agent({
  tools: [sleep],
})
await agent.invoke('Pause for two seconds, then continue.')
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_tools import sleep

agent = Agent(tools=[sleep])
agent("Pause for two seconds, then continue.")
```
(( /tab "Python" ))

**Custom maximum:**

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { makeSleep } from '@strands-agents/sdk/vended-tools/sleep'

const shortSleep = makeSleep({ maxDuration: 5 })
const agent = new Agent({ tools: [shortSleep] })
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_tools import make_sleep

short_sleep = make_sleep(max_duration=5)
agent = Agent(tools=[short_sleep])
```
(( /tab "Python" ))

📖 [Full API Reference](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/sleep/README.md)

---

### Stop (Experimental)

> This tool is experimental and subject to change in future revisions without notice.

Lets the model gracefully end the agent loop with an optional final message. The default loop already terminates when the model returns without any tool call; the stop tool is useful when you want an explicit “I am done” affordance, when a workflow enforces that termination is a deliberate model decision, or when a sub-agent needs to signal completion back to a coordinator via the loop’s last assistant message.

*Supported in: Node.js, modern browsers (TypeScript); all platforms (Python).*

This is a cooperative stop, not an abort. Any other tools the model requested in the same turn still run to completion; the loop halts after that batch without calling the model again. The final message defaults to a 4096-character cap; pass `max_message_length` / `maxMessageLength` to `make_stop` / `makeStop` when a longer summary is legitimate.

The two SDKs shim onto different loop-termination primitives, which produces a small difference in the final `AgentResult`. TypeScript halts via `AfterToolsEvent.endTurn` and returns `stopReason: "endTurn"` with the stop text as the last assistant message. Python halts via `invocation_state["request_state"]["stop_event_loop"]` and returns `stop_reason: "tool_use"` with the model’s tool-use message as the final message; the stop text lives in history as the tool result, not as a new assistant turn.

**Example:**

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { stop } from '@strands-agents/sdk/experimental/vended-tools/stop'

const agent = new Agent({
  tools: [stop],
  systemPrompt: 'Complete the task. Call stop with a short summary when you are done.',
})
await agent.invoke('Summarize the changes in ./CHANGELOG.md')
```
(( /tab "TypeScript" ))

(( tab "Python" ))
```python
from strands import Agent
from strands.experimental.tools import stop

agent = Agent(
    tools=[stop],
    system_prompt="Complete the task. Call stop with a short summary when you are done.",
)
result = agent("Summarize the changes in ./CHANGELOG.md")
```
(( /tab "Python" ))

📖 [Full API Reference](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/experimental/vended-tools/stop/README.md)

---

## Using Multiple Tools Together

Combine vended tools to build powerful agent workflows:

```typescript
import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

const agent = new Agent({
  tools: [bash, fileEditor, notebook],
  systemPrompt: [
    'You are a software development assistant.',
    'When given a feature to implement:',
    '1. Use the notebook tool to create a plan with a checklist of steps',
    '2. Work through each step, checking them off as you go',
    '3. Use the bash tool to run tests and verify your changes',
  ].join('\n'),
})

// Agent plans the work, implements it, and tracks progress
await agent.invoke(
  'Add input validation to the createUser function in src/users.ts. ' +
    'It should reject empty names and invalid email formats.'
)
```

## Versioning & Maintenance

Vended tools ship as part of the SDK and are updated alongside it. Report bugs and feature requests in the [GitHub repository](https://github.com/strands-agents/harness-sdk/issues).

Tool names are stable and will not change. In minor versions, a tool’s description, spec, or parameters may be updated to improve effectiveness — these changes are noted in SDK release notes. Pin your SDK version and test after upgrades if your workflows depend on specific tool behavior.

## See also

-   [Custom Tools](/docs/user-guide/concepts/tools/custom-tools/index.md) — Build your own tools
-   [Community Tools Package](/docs/user-guide/concepts/tools/community-tools-package/index.md) — Python tools package with 30+ tools
-   [Session Management](/docs/user-guide/concepts/agents/session-management/index.md) — Persist agent state including notebooks
-   [Interrupts](/docs/user-guide/concepts/interrupts/index.md) — Implement approval workflows for sensitive operations
-   [Hooks](/docs/user-guide/concepts/agents/hooks/index.md) — Intercept and customize tool execution

## Related pages

- [Community Built Tools](/docs/user-guide/concepts/tools/community-tools-package/index.md) (1 shared tag)
- [Creating Custom Tools](/docs/user-guide/concepts/tools/custom-tools/index.md) (1 shared tag)
- [Build with AI](/docs/user-guide/build-with-ai/index.md) (1 shared tag)
- [Model Context Protocol (MCP) Tools](/docs/user-guide/concepts/tools/mcp-tools/index.md) (1 shared tag)
- [Tools Overview](/docs/user-guide/concepts/tools/index.md) (1 shared tag)
- [Agents as Tools with Strands Agents SDK](/docs/user-guide/concepts/multi-agent/agents-as-tools/index.md) (1 shared tag)
- [Agent Configuration](/docs/user-guide/concepts/experimental/agent-config/index.md) (1 shared tag)


## Implementation

### TypeScript

- [harness-sdk/strands-ts/src/vended-tools/file-editor/file-editor.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/file-editor/file-editor.ts)
- [harness-sdk/strands-ts/src/vended-tools/bash/bash.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/bash/bash.ts)
- [harness-sdk/strands-ts/src/vended-tools/http-request/http-request.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/http-request/http-request.ts)
- [harness-sdk/strands-ts/src/vended-tools/notebook/notebook.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/notebook/notebook.ts)
- [harness-sdk/strands-ts/src/vended-tools/sleep/sleep.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-tools/sleep/sleep.ts)
- [harness-sdk/strands-ts/src/experimental/vended-tools/stop/stop.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/experimental/vended-tools/stop/stop.ts)

### Python

- [harness-sdk/strands-py/src/strands/vended_tools/http_request/http_request.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/http_request.py)
- [harness-sdk/strands-py/src/strands/vended_tools/sleep/sleep.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/sleep/sleep.py)
- [harness-sdk/strands-py/src/strands/experimental/tools/stop/stop.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/tools/stop/stop.py)
