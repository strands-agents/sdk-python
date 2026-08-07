Strands Agents SDK provides real-time streaming capabilities that allow you to monitor and process events as they occur during agent execution. This enables responsive user interfaces, real-time monitoring, and custom output formatting.

Strands has multiple approaches for handling streaming events:

-   **[Async Iterators](/docs/user-guide/concepts/streaming/async-iterators/index.md)**: Ideal for asynchronous server frameworks
-   **[Callback Handlers (Python only)](/docs/user-guide/concepts/streaming/callback-handlers/index.md)**: Perfect for synchronous applications and custom event processing

Both methods receive the same event types but differ in their execution model and use cases.

## Event Types

All streaming methods yield the same set of events:

### Lifecycle Events

(( tab "Python" ))
-   **`init_event_loop`**: True at the start of agent invocation initializing
-   **`start_event_loop`**: True when the event loop is starting
-   **`event_loop_throttled_delay`**: Delay, in seconds, that the agent loop waited before retrying a model call after a retryable failure (by default only throttling, see [Retry Strategies](/docs/user-guide/concepts/agents/retry-strategies/index.md))
-   **`message`**: Present when a new message is created
-   **`event`**: Raw event from the model stream
-   **`force_stop`**: True if the event loop was forced to stop
    -   **`force_stop_reason`**: Reason for forced stop
-   **`result`**: The final [`AgentResult`](/docs/api/python/strands.agent.agent_result#AgentResult)
(( /tab "Python" ))

(( tab "TypeScript" ))
Each event emitted from the TypeScript agent is a class with a `type` attribute that has a unique value. When determining an event, you can use `instanceof` on the class, or an equality check on the `event.type` value. All events extend `HookableEvent`, making them both streamable and subscribable via hook callbacks.

-   **`BeforeInvocationEvent`**: Start of agent loop (before any iterations)
    -   **`cancel`**: Set by hook callbacks to cancel the invocation (`boolean | string`)
-   **`AfterInvocationEvent`**: End of agent loop (after all iterations complete)
    -   **`error?`**: Optional error if loop terminated due to exception
-   **`BeforeModelCallEvent`**: Before model invocation
    -   **`messages`**: Array of messages being sent to model
    -   **`cancel`**: Set by hook callbacks to cancel the model call (`boolean | string`)
-   **`AfterModelCallEvent`**: After model invocation
    -   **`message`**: Assistant message returned by model
    -   **`stopReason`**: Why generation stopped
-   **`BeforeToolsEvent`**: Before tools execution
    -   **`message`**: Assistant message containing tool use blocks
-   **`AfterToolsEvent`**: After tools execution
    -   **`message`**: User message containing tool results
-   **`AgentResultEvent`**: Final agent result
    -   **`result`**: The `AgentResult` with `stopReason`, `lastMessage`, and optional `structuredOutput`
(( /tab "TypeScript" ))

### Model Stream Events

(( tab "Python" ))
-   **`data`**: Text chunk from the model’s output
-   **`delta`**: Raw delta content from the model
-   **`reasoning`**: True for reasoning events
    -   **`reasoningText`**: Text from reasoning process
    -   **`reasoning_signature`**: Signature from reasoning process
    -   **`redactedContent`**: Reasoning content redacted by the model
(( /tab "Python" ))

(( tab "TypeScript" ))
-   **`ModelStreamUpdateEvent`**: Wraps transient model streaming deltas. Access the inner event via `.event`:
    -   **`ModelMessageStartEvent`**: Start of a message from the model
    -   **`ModelContentBlockStartEvent`**: Start of a content block (text, toolUse, reasoning, etc.)
    -   **`ModelContentBlockDeltaEvent`**: Content deltas for text, tool input, or reasoning
    -   **`ModelContentBlockStopEvent`**: End of a content block
    -   **`ModelMessageStopEvent`**: End of a message
    -   **`ModelMetadataEvent`**: Usage and metrics metadata
-   **`ContentBlockEvent`**: Wraps a fully assembled content block (TextBlock, ToolUseBlock, ReasoningBlock). Access via `.contentBlock`
-   **`ModelMessageEvent`**: Wraps the complete model message after all blocks are assembled. Access via `.message`
(( /tab "TypeScript" ))

### Tool Events

(( tab "Python" ))
-   **`current_tool_use`**: Information about the current tool being used, including:
    -   **`toolUseId`**: Unique ID for this tool use
    -   **`name`**: Name of the tool
    -   **`input`**: Tool input parameters (accumulated as streaming occurs)
-   **`tool_stream_event`**: Information about [an event streamed from a tool](/docs/user-guide/concepts/tools/custom-tools/index.md#tool-streaming), including:
    -   **`tool_use`**: The [`ToolUse`](/docs/api/python/strands.types.tools#ToolUse) for the tool that streamed the event
    -   **`data`**: The data streamed from the tool
(( /tab "Python" ))

(( tab "TypeScript" ))
-   **`BeforeToolCallEvent`**: Before a tool is executed
    -   **`toolUse`**: The tool use block with `name` and `input`
-   **`AfterToolCallEvent`**: After a tool finishes execution
    -   **`toolUse`**: The tool use block
    -   **`result`**: The tool result block
-   **`ToolStreamUpdateEvent`**: Wraps streaming progress events from a tool. Access via `.event`:
    -   **`data`**: The data streamed from the tool
-   **`ToolResultEvent`**: Wraps a completed tool result. Access via `.result`
(( /tab "TypeScript" ))

### Multi-Agent Events

(( tab "Python" ))
Multi-agent systems ([Graph](/docs/user-guide/concepts/multi-agent/graph/index.md) and [Swarm](/docs/user-guide/concepts/multi-agent/swarm/index.md)) emit additional coordination events:

-   **`multiagent_node_start`**: When a node begins execution
    -   **`type`**: `"multiagent_node_start"`
    -   **`node_id`**: Unique identifier for the node
    -   **`node_type`**: Type of node (`"agent"`, `"swarm"`, `"graph"`)
-   **`multiagent_node_stream`**: Forwarded events from agents/multi-agents with node context
    -   **`type`**: `"multiagent_node_stream"`
    -   **`node_id`**: Identifier of the node generating the event
    -   **`event`**: The original agent event (nested)
-   **`multiagent_node_stop`**: When a node completes execution
    -   **`type`**: `"multiagent_node_stop"`
    -   **`node_id`**: Unique identifier for the node
    -   **`node_result`**: Complete NodeResult with execution details, metrics, and status
-   **`multiagent_handoff`**: When control is handed off between agents (Swarm) or batch transitions (Graph)
    -   **`type`**: `"multiagent_handoff"`
    -   **`from_node_ids`**: List of node IDs completing execution
    -   **`to_node_ids`**: List of node IDs beginning execution
    -   **`message`**: Optional handoff message (typically used in Swarm)
-   **`multiagent_result`**: Final multi-agent result
    -   **`type`**: `"multiagent_result"`
    -   **`result`**: The final GraphResult or SwarmResult

See [Graph streaming](/docs/user-guide/concepts/multi-agent/graph/index.md#streaming-events) and [Swarm streaming](/docs/user-guide/concepts/multi-agent/swarm/index.md#streaming-events) for usage examples.
(( /tab "Python" ))

(( tab "TypeScript" ))
Multi-agent systems ([Graph](/docs/user-guide/concepts/multi-agent/graph/index.md) and [Swarm](/docs/user-guide/concepts/multi-agent/swarm/index.md)) emit additional coordination events. Each event is a class with a `type` attribute, extending `HookableEvent` for both streaming and hook subscription.

-   **`MultiAgentInitializedEvent`**: When a multi-agent orchestrator has finished initialization
    -   **`orchestrator`**: The `MultiAgentBase` instance
-   **`BeforeMultiAgentInvocationEvent`**: Before orchestrator execution starts
    -   **`orchestrator`**: The `MultiAgentBase` instance
    -   **`state`**: The current `MultiAgentState`
-   **`BeforeNodeCallEvent`**: Before a node begins execution
    -   **`nodeId`**: Unique identifier for the node
    -   **`orchestrator`**: The `MultiAgentBase` instance
    -   **`state`**: The current `MultiAgentState`
    -   **`cancel`**: Set by hook callbacks to cancel node execution (`boolean | string`)
-   **`NodeStreamUpdateEvent`**: Forwarded events from agents or nested orchestrators with node context
    -   **`nodeId`**: Identifier of the node generating the event
    -   **`nodeType`**: Type of node (`"agentNode"`, `"multiAgentNode"`)
    -   **`state`**: The current `MultiAgentState`
    -   **`event`**: The inner `AgentStreamEvent` or `MultiAgentStreamEvent`
-   **`NodeCancelEvent`**: When a node is cancelled via `BeforeNodeCallEvent.cancel`
    -   **`nodeId`**: Unique identifier for the node
    -   **`state`**: The current `MultiAgentState`
    -   **`message`**: Cancel reason
-   **`AfterNodeCallEvent`**: After a node completes execution
    -   **`nodeId`**: Unique identifier for the node
    -   **`orchestrator`**: The `MultiAgentBase` instance
    -   **`state`**: The current `MultiAgentState`
    -   **`error?`**: Optional error if the node failed
-   **`NodeResultEvent`**: When a node finishes execution
    -   **`nodeId`**: Unique identifier for the node
    -   **`nodeType`**: Type of node (`"agentNode"`, `"multiAgentNode"`)
    -   **`state`**: The current `MultiAgentState`
    -   **`result`**: The `NodeResult` with `status`, `duration`, `content`, and optional `error`
-   **`MultiAgentHandoffEvent`**: When execution transitions between nodes
    -   **`source`**: Node ID completing execution
    -   **`targets`**: Array of node IDs beginning execution
    -   **`state`**: The current `MultiAgentState`
-   **`AfterMultiAgentInvocationEvent`**: After orchestrator execution completes
    -   **`orchestrator`**: The `MultiAgentBase` instance
    -   **`state`**: The current `MultiAgentState`
-   **`MultiAgentResultEvent`**: Final event in the multi-agent stream
    -   **`result`**: The `MultiAgentResult` with `status`, `results`, `content`, and `duration`

See [Graph streaming](/docs/user-guide/concepts/multi-agent/graph/index.md#streaming-events) and [Swarm streaming](/docs/user-guide/concepts/multi-agent/swarm/index.md#streaming-events) for usage examples.
(( /tab "TypeScript" ))

### Event Serialization

(( tab "Python" ))
Python streaming events are plain dictionaries. The SDK does not include a built-in serialization filter — you have full control over which events and fields to forward from your processes and servers.

When serving streamed responses (for example, over SSE or WebSockets), you can filter the yielded events to keep payloads compact:

```python
import json

def filter_event(event: dict) -> dict | None:
    """Filter streaming events to only forward relevant data over the wire."""
    # Forward text deltas for real-time display
    if "data" in event:
        return {"type": "text", "data": event["data"]}

    # Forward tool usage for progress indicators
    if "current_tool_use" in event and event["current_tool_use"].get("name"):
        return {"type": "tool", "name": event["current_tool_use"]["name"]}

    # Forward the final result
    if "result" in event:
        return {"type": "result", "stop_reason": str(event["result"].stop_reason)}

    # Skip everything else (lifecycle signals, raw deltas, reasoning, etc.)
    return None


async for event in agent.stream_async("Hello"):
    filtered = filter_event(event)
    if filtered:
        await response.write(f"data: {json.dumps(filtered)}\n\n")
```

This approach lets you tailor the streamed output to your use case — for example, forwarding only text deltas for a chat UI or including tool events for a progress dashboard.
(( /tab "Python" ))

(( tab "TypeScript" ))
Every event class implements a `toJSON()` method that `JSON.stringify()` calls automatically. Each serialized event retains its `type` discriminator and the relevant data fields — matching the general shape of the class — while excluding in-memory runtime references (`agent`, `orchestrator`, `state`, `tool`) and mutable hook properties (`cancel`, `retry`). `Error` objects are converted to `{ message: string }`. This applies to single-agent, multi-agent, and A2A events alike.

You can filter which events to forward to the client:

```typescript
for await (const event of agent.stream('Hello')) {
  switch (event.type) {
    // Forward text deltas for real-time display
    case 'modelStreamUpdateEvent':
      if (
        event.event.type === 'modelContentBlockDeltaEvent' &&
        event.event.delta.type === 'textDelta'
      ) {
        console.log(
          `data: ${JSON.stringify({ type: 'text', text: event.event.delta.text })}`
        )
      }
      break

    // Forward tool names for progress indicators
    case 'beforeToolCallEvent':
      console.log(`data: ${JSON.stringify({ type: 'tool', name: event.toolUse.name })}`)
      break

    // Forward the final result
    case 'agentResultEvent':
      console.log(`data: ${JSON.stringify(event)}`)
      break
  }
}
```
(( /tab "TypeScript" ))

## Quick Examples

(( tab "Python" ))
**Async Iterator Pattern**

```python
async for event in agent.stream_async("Calculate 2+2"):
    if "data" in event:
        print(event["data"], end="")
```

**Callback Handler Pattern**

```python
def handle_events(**kwargs):
    if "data" in kwargs:
        print(kwargs["data"], end="")

agent = Agent(callback_handler=handle_events)
agent("Calculate 2+2")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
**Async Iterator Pattern**

```typescript
const agent = new Agent({ tools: [notebook] })

for await (const event of agent.stream('Calculate 2+2')) {
  if (
    event.type === 'modelStreamUpdateEvent' &&
    event.event.type === 'modelContentBlockDeltaEvent' &&
    event.event.delta.type === 'textDelta'
  ) {
    // Print out the model text delta event data
    process.stdout.write(event.event.delta.text)
  }
}
console.log('\nDone!')
```
(( /tab "TypeScript" ))

## Identifying Events Emitted from Agent

This example demonstrates how to identify event emitted from an agent:

(( tab "Python" ))
```python
from strands import Agent
from strands_tools import calculator

def process_event(event):
    """Shared event processor for both async iterators and callback handlers"""
    # Track event loop lifecycle
    if event.get("init_event_loop", False):
        print("🔄 Event loop initialized")
    elif event.get("start_event_loop", False):
        print("▶️ Event loop cycle starting")
    elif "message" in event:
        print(f"📬 New message created: {event['message']['role']}")
    elif "result" in event:
        print("✅ Agent completed with result")
    elif event.get("force_stop", False):
        print(f"🛑 Event loop force-stopped: {event.get('force_stop_reason', 'unknown reason')}")

    # Track tool usage
    if "current_tool_use" in event and event["current_tool_use"].get("name"):
        tool_name = event["current_tool_use"]["name"]
        print(f"🔧 Using tool: {tool_name}")

    # Show text snippets
    if "data" in event:
        data_snippet = event["data"][:20] + ("..." if len(event["data"]) > 20 else "")
        print(f"📟 Text: {data_snippet}")

agent = Agent(tools=[calculator], callback_handler=None)
async for event in agent.stream_async("What is the capital of France and what is 42+7?"):
    process_event(event)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
function processEvent(event: AgentStreamEvent): void {
  // Track agent loop lifecycle
  switch (event.type) {
    case 'beforeInvocationEvent':
      console.log('🔄 Agent loop initialized')
      break
    case 'beforeModelCallEvent':
      console.log('▶️ Agent loop cycle starting')
      break
    case 'afterModelCallEvent':
      console.log(`📬 New message created: ${event.stopData?.message.role}`)
      break
    case 'beforeToolsEvent':
      console.log('About to execute tool!')
      break
    case 'afterToolsEvent':
      console.log('Finished execute tool!')
      break
    case 'afterInvocationEvent':
      console.log('✅ Agent loop completed')
      break
  }

  // Track tool usage
  if (
    event.type === 'modelStreamUpdateEvent' &&
    event.event.type === 'modelContentBlockStartEvent' &&
    event.event.start?.type === 'toolUseStart'
  ) {
    console.log(`\n🔧 Using tool: ${event.event.start.name}`)
  }

  // Show text snippets
  if (
    event.type === 'modelStreamUpdateEvent' &&
    event.event.type === 'modelContentBlockDeltaEvent' &&
    event.event.delta.type === 'textDelta'
  ) {
    process.stdout.write(event.event.delta.text)
  }
}
const responseGenerator = agent.stream(
  'What is the capital of France and what is 42+7? Record in the notebook.'
)
for await (const event of responseGenerator) {
  processEvent(event)
}
```
(( /tab "TypeScript" ))

## Sub-Agent Streaming Example

Utilizing both [agents as a tool](/docs/user-guide/concepts/multi-agent/agents-as-tools/index.md) and [tool streaming](/docs/user-guide/concepts/tools/custom-tools/index.md#tool-streaming), this example shows how to stream events from sub-agents:

(( tab "Python" ))
```python
from typing import AsyncIterator
from dataclasses import dataclass
from strands import Agent, tool
from strands_tools import calculator

@dataclass
class SubAgentResult:
    agent: Agent
    event: dict

@tool
async def math_agent(query: str) -> AsyncIterator:
    """Solve math problems using the calculator tool."""
    agent = Agent(
        name="Math Expert",
        system_prompt="You are a math expert. Use the calculator tool for calculations.",
        callback_handler=None,
        tools=[calculator]
    )

    result = None
    async for event in agent.stream_async(query):
        yield SubAgentResult(agent=agent, event=event)
        if "result" in event:
            result = event["result"]

    yield str(result)

def process_sub_agent_events(event):
    """Shared processor for sub-agent streaming events"""
    tool_stream = event.get("tool_stream_event", {}).get("data")

    if isinstance(tool_stream, SubAgentResult):
        current_tool = tool_stream.event.get("current_tool_use", {})
        tool_name = current_tool.get("name")

        if tool_name:
            print(f"Agent '{tool_stream.agent.name}' using tool '{tool_name}'")

    # Also show regular text output
    if "data" in event:
        print(event["data"], end="")

# Using with async iterators
orchestrator_async_iterator = Agent(
    system_prompt="Route math questions to the math_agent tool.",
    callback_handler=None,
    tools=[math_agent]
)


# With async-iterator
async for event in orchestrator_async_iterator.stream_async("What is 3+3?"):
    process_sub_agent_events(event)


# With callback handler
def handle_events(**kwargs):
    process_sub_agent_events(kwargs)

orchestrator_callback = Agent(
    system_prompt="Route math questions to the math_agent tool.",
    callback_handler=handle_events,
    tools=[math_agent]
)

orchestrator_callback("What is 3+3?")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
// Create the math agent
const mathAgent = new Agent({
  systemPrompt: 'You are a math expert. Answer a math problem in one sentence',
  printer: false,
})

const calculator = tool({
  name: 'mathAgent',
  description: 'Agent that calculates the answer to a math problem input.',
  inputSchema: z.object({ input: z.string() }),
  callback: async function* (input): AsyncGenerator<string, string, unknown> {
    // Stream from the sub-agent
    const generator = mathAgent.stream(input.input)
    let result = await generator.next()
    while (!result.done) {
      // Process events from the sub-agent
      if (
        result.value.type === 'modelStreamUpdateEvent' &&
        result.value.event.type === 'modelContentBlockDeltaEvent' &&
        result.value.event.delta.type === 'textDelta'
      ) {
        yield result.value.event.delta.text
      }
      result = await generator.next()
    }
    return result.value.lastMessage.content[0]!.type === 'textBlock'
      ? result.value.lastMessage.content[0]!.text
      : result.value.lastMessage.content[0]!.toString()
  },
})

const agent = new Agent({ tools: [calculator] })
for await (const event of agent.stream('What is 2 * 3? Use your tool.')) {
  if (event.type === 'toolStreamUpdateEvent') {
    console.log(`Tool Event: ${JSON.stringify(event.event.data)}`)
  }
}
console.log('\nDone!')
```
(( /tab "TypeScript" ))

## Next Steps

-   Learn about [Async Iterators](/docs/user-guide/concepts/streaming/async-iterators/index.md) for asynchronous streaming
-   Explore [Callback Handlers](/docs/user-guide/concepts/streaming/callback-handlers/index.md) for synchronous event processing
-   See the Agent API Reference for complete method documentation: [Python](/docs/api/python/strands.agent.agent) | [TypeScript](/docs/api/typescript/Agent/index.md)

## Related pages

- [Async Iterators for Streaming](/docs/user-guide/concepts/streaming/async-iterators/index.md) (1 shared tag)
- [Callback Handlers](/docs/user-guide/concepts/streaming/callback-handlers/index.md) (1 shared tag)
