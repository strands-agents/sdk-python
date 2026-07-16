# use_agent

Delegates a single task to a nested agent that the calling agent constructs at runtime.

The calling agent supplies a system prompt, an exact-name allowlist of parent-registered tool names to expose to the child, and the task. The child inherits the parent's model instance, runs on a fresh conversation, and returns its final assistant text to the parent. It is the runtime equivalent of writing `child.asTool()` in application code, but with the system prompt and tool surface chosen by the model at call time.

Delegation is a privilege-escalation surface. Registering this tool gives the calling model permission to choose which parent tools the child sees and to spend inference tokens on the parent's account. The model cannot choose a different model provider or supply provider-specific settings, because those are credential-injection vectors. The exact-name allowlist rejects wildcards, empty strings, and the multi-agent tool names use_agent, swarm, graph, and a2a_client. Nested delegation is capped at depth three, shared across the multi-agent vended tools via `invocationState.multiagentDepth`, and is not model-controllable. The system prompt is capped at 8 KiB and the task at 32 KiB. The child gets a fresh conversation; parent messages are never forwarded. On cancellation the tool re-raises an AbortError so callers can distinguish cancellation from other failures via `error.name === 'AbortError'`, matching the sibling http-request tool. Do not register this tool for untrusted models without an additional guardrail such as a Cedar intervention or an audit hook.

## Usage

```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'
import { useAgent } from '@strands-agents/sdk/vended-tools/use-agent'

const agent = new Agent({
  model: new BedrockModel({ region: 'us-west-2' }),
  tools: [httpRequest, useAgent],
})

await agent.invoke('Delegate a sub-task to a researcher agent that has access to http_request.')
```

## Input

| Field          | Type       | Required | Description                                                                |
| -------------- | ---------- | -------- | -------------------------------------------------------------------------- |
| `systemPrompt` | `string`   | Yes      | Child system prompt. Non-empty, capped at 8 KiB UTF-8.                     |
| `task`         | `string`   | Yes      | Task to hand the child. Non-empty, capped at 32 KiB UTF-8.                 |
| `tools`        | `string[]` | No       | Exact-name allowlist of parent tools to expose to the child. No wildcards. |

## Output

Returns an object matching the shared multi-agent result shape:

```typescript
{
  status: 'completed' | 'failed' | 'interrupted',
  output: string,             // child agent's final assistant text
  executionTimeMs: number,    // total wall-clock in milliseconds
}
```

Only a child that finishes with `stopReason === 'endTurn'` maps to `completed`. `stopReason === 'interrupt'` maps to `interrupted`. Any other non-cancelled stop reason (`limitTurns`, `contentFiltered`, `maxTokens`, `guardrailIntervened`, ...) maps to `failed` so the parent can distinguish a completed delegation from one that hit a policy or limit. Cancellation raises an `AbortError` rather than returning.

## Related

The shared model dialect is documented in the sibling `_multiagent-conventions.md`. The child agent is constructed via the standard `Agent` class; see `agent.asTool()` for the underlying shim primitive.
