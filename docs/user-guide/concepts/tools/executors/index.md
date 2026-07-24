Tool executors control whether tools from a single assistant turn run concurrently or sequentially. Both SDKs default to concurrent execution.

## Concurrent Executor

Concurrent execution runs all tool calls from a single turn in parallel. This is the default in both SDKs — you get it without any extra configuration.

(( tab "Python" ))
```python
from strands import Agent
from strands.tools.executors import ConcurrentToolExecutor

agent = Agent(
    tool_executor=ConcurrentToolExecutor(),
    tools=[weather_tool, time_tool]
)
# or simply Agent(tools=[weather_tool, time_tool])

agent("What is the weather and time in New York?")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({
  tools: [weatherTool, timeTool],
  toolExecutor: 'concurrent',
})
// Omit toolExecutor to use concurrent execution by default.

await agent.invoke('What is the weather and time in New York?')
```

The `'concurrent'` string shorthand keeps your imports minimal. Passing `new ConcurrentToolExecutor()` is equivalent if you prefer to be explicit.
(( /tab "TypeScript" ))

Assuming the model returns `weather_tool` and `time_tool` use requests, the concurrent executor runs both at the same time. End-to-end latency scales with the slowest tool rather than their sum.

### Sequential Behavior

On certain prompts, the model may decide to return one tool use request at a time. Under these circumstances, the tools will execute sequentially. Concurrency is only achieved if the model returns multiple tool use requests in a single response. Certain models however offer additional abilities to coerce a desired behavior. For example, Anthropic exposes an explicit parallel tool use setting ([docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#parallel-tool-use)).

## Sequential Executor

Use sequential execution when tool order matters — for example, when a later tool depends on a side effect of an earlier one:

(( tab "Python" ))
```python
from strands import Agent
from strands.tools.executors import SequentialToolExecutor

agent = Agent(
    tool_executor=SequentialToolExecutor(),
    tools=[screenshot_tool, email_tool]
)

agent("Please take a screenshot and then email the screenshot to my friend")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({
  tools: [screenshotTool, emailTool],
  toolExecutor: 'sequential',
})

await agent.invoke('Take a screenshot and email it to my friend')
```

The `'sequential'` string shorthand keeps your imports minimal. Passing `new SequentialToolExecutor()` is equivalent if you prefer to be explicit.
(( /tab "TypeScript" ))

Assuming the model returns `screenshot_tool` and `email_tool` use requests, the sequential executor runs both in the order given.

## Event Ordering

Both modes preserve per-tool event order. In concurrent mode, events from different tools may interleave across that per-tool sequence.

(( tab "Python" ))
Per-tool event order: `BeforeToolCallEvent` → `ToolStreamEvent*` → `AfterToolCallEvent` → `ToolResultEvent`.
(( /tab "Python" ))

(( tab "TypeScript" ))
Per-tool event order: `BeforeToolCallEvent` → `ToolStreamUpdateEvent*` → `AfterToolCallEvent` → `ToolResultEvent`.
(( /tab "TypeScript" ))

## Cancellation

Cancellation works identically in both modes. Call `agent.cancel()` to request cooperative cancellation. In TypeScript, this flips `agent.cancelSignal`.

(( tab "Python" ))
-   **Pre-launch cancel**: set `BeforeToolCallEvent.cancel_tool` on a per-tool hook to produce an error result for that tool.
-   **Mid-flight cancel** in sequential mode short-circuits not-yet-started tools. In concurrent mode, all tools have already launched, so each in-flight tool must cooperatively check for cancellation to stop early.
(( /tab "Python" ))

(( tab "TypeScript" ))
-   **Pre-launch cancel**: set `BeforeToolsEvent.cancel` on the batch-level hook, or call `agent.cancel()` before tools start, to produce error results for every tool in the batch.
-   **Mid-flight cancel** in sequential mode short-circuits not-yet-started tools. In concurrent mode, all tools have already launched, so each in-flight tool must cooperatively observe `agent.cancelSignal` to stop early.
(( /tab "TypeScript" ))

## Custom Executors

Custom tool executors are not currently supported but are planned for a future release. You can track progress on this feature at [GitHub Issue #762](https://github.com/strands-agents/harness-sdk/issues/762).

## Related pages

- [Agent Loop](/docs/user-guide/concepts/agents/agent-loop/index.md) (2 shared tags)
- [Hooks](/docs/user-guide/concepts/agents/hooks/index.md) (2 shared tags)
- [Steering (Interventions)](/docs/user-guide/concepts/agents/interventions/steering/index.md) (2 shared tags)
- [Interrupts](/docs/user-guide/concepts/interrupts/index.md) (2 shared tags)
- [Creating a Custom Model Provider](/docs/user-guide/concepts/model-providers/custom_model_provider/index.md) (1 shared tag)
- [Interventions](/docs/user-guide/concepts/agents/interventions/index.md) (2 shared tags)
- [Retry Strategies](/docs/user-guide/concepts/agents/retry-strategies/index.md) (1 shared tag)
- [Plugins](/docs/user-guide/concepts/plugins/index.md) (1 shared tag)
- [Available Sandboxes](/docs/user-guide/concepts/sandbox/available-sandboxes/index.md) (1 shared tag)
- [Building a Custom Sandbox](/docs/user-guide/concepts/sandbox/custom-sandbox/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/tools/executors/concurrent.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/executors/concurrent.py)
- [harness-sdk/strands-py/src/strands/tools/executors/sequential.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/executors/sequential.py)

### TypeScript

- [harness-sdk/strands-ts/src/tools/executors/concurrent.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/tools/executors/concurrent.ts)
- [harness-sdk/strands-ts/src/tools/executors/sequential.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/tools/executors/sequential.ts)
