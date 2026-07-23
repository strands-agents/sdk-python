# Stop Tool

Lets the model gracefully end the agent loop when it decides its work is complete.

The model calls stop with an optional final message. The current tool batch runs to completion, then the loop ends without calling the model again. The final `AgentResult.stopReason` is `endTurn`, and the loop's last assistant `Message` carries a `TextBlock` whose text is the string the model passed to stop (or a default if it passed none).

This is a cooperative stop, not an abort. Any other tools the model requested in the same turn still run; the loop halts after the batch finishes, not mid-batch. For hard cancellation, use `agent.cancel()`.

## SDK behavioral difference

The Python and TypeScript SDKs terminate on different primitives, which produces a small but observable difference in the final `AgentResult`:

- **TypeScript** halts via `AfterToolsEvent.endTurn`, which synthesizes a new assistant message with the stop text. `result.stopReason` is `"endTurn"` and `result.lastMessage` is a `Message` whose text content carries the stop text (extract via the message's `TextBlock` content).
- **Python** halts via `invocation_state["request_state"]["stop_event_loop"] = True`, which short-circuits inside the current tool-use cycle. `result.stop_reason` is `"tool_use"` and `result.message` is the model's tool-use assistant message from that batch; the stop text appears in history as the tool's `toolResult`, not as a new final assistant turn.

If a caller needs the stop string as the last assistant message on Python, read it from the tool result on the final message or append it explicitly.

## When to use it

The default agent loop already terminates when the model returns without any tool use. The stop tool is useful when:

- The model tends to keep calling tools past the point of usefulness and needs an explicit "I'm done" affordance.
- A workflow enforces that termination is a deliberate model decision (e.g. structured multi-step tasks) rather than an accident of not calling a tool.
- Sub-agents need to signal completion back to a coordinator via the text content of `AgentResult.lastMessage`.

If none of the above applies, you probably don't need to install this tool.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { stop } from '@strands-agents/sdk/vended-tools/stop'

const agent = new Agent({
  model,
  tools: [stop],
  systemPrompt: 'Complete the task. Call stop with a short summary when you are done.',
})

const result = await agent.invoke('Summarize the changes in ./CHANGELOG.md')
console.log(result.stopReason) // 'endTurn'
// lastMessage is a Message; pull the text out of its content blocks:
const stopText = result.lastMessage.content
  .filter((block) => block.type === 'text')
  .map((block) => block.text)
  .join('')
console.log(stopText) // The model's summary passed to stop()
```

## Input schema

```typescript
interface StopInput {
  /** Optional final assistant-facing message. Capped at 4096 characters. */
  message?: string | null
}
```

## How it works

The tool shims onto the SDK's existing loop-termination primitive rather than introducing a new one. When the model calls stop, the tool writes a marker onto `context.invocationState` and returns the message. A lazily-installed `AfterToolsEvent` hook (once per agent, tracked in a `WeakSet`) reads that marker on the terminal event and sets `event.endTurn = message`, which the agent loop already interprets as "halt after this batch and emit an `endTurn` result with the given text." The marker is consumed each time the hook fires, so it does not leak across invocations.

## Limitations

- Cooperative only. If the model requests stop alongside a very long-running tool call, the loop still waits for that call to finish before ending. Use `agent.cancel()` if you need to bail out immediately.
- Removing the tool from `agent.tools` mid-conversation does not uninstall the hook, but with no more calls to write the marker the hook is a no-op.
- Ending the loop does not drop the tool-result message or the surrounding turn; those remain in `agent.messages` as normal history.
