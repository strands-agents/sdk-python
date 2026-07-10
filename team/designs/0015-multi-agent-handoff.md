# Multi-Agent Handoff

**Status**: Proposed

**Date**: 2026-07-09

**Issue**: https://github.com/strands-agents/harness-sdk/issues/911

---

[Problem](#problem) · [Proposal](#proposal) · [Alternatives Considered](#alternatives-considered) · [Developer Experience](#developer-experience) · [Consequences](#consequences)

---

## Problem

The SDK does not have a first-class handoff mechanism for fully delegating execution from an orchestrator agent to a sub-agent. Builders want to create agents that follow this pattern, but the current primitives either make unnecessary model calls or lose context.

### Current State

There are currently three workarounds for delegating agent execution to sub-agents, but each have notable flaws.

#### Agents-as-tools

Users can define a parent agent that delegates to sub-agents by calling them as tools:

```ts
const researchAgent = new Agent({...})
const productAgent = new Agent({...})
const travelAgent = new Agent({...})

const orchestrator = new Agent({
  tools: [researchAgent, productAgent, travelAgent],
})
```

This pattern has two problems:

1. **Redundant model calls.** The parent agent must post-process the sub-agent's response before returning it to the user, even when the sub-agent's output is ready to serve as-is. This doubles latency and token usage for scenarios where no additional reasoning is required.
2. **Response corruption.** If the sub-agent produces structured output (JSON, code, formatted data), the parent agent may paraphrase or reformat it during post-processing, breaking the structure.

#### Graphs

Users configure a graph of agent nodes with pre-defined edges:

```ts
const researcher = new Agent({...})
const analyst = new Agent({...})
const factChecker = new Agent({...})

const graph = new Graph({
  nodes: [researcher, analyst, factChecker],
  edges: [
    ['research', 'analysis'],
    ['research', 'fact_check'],
  ],
})
```

However, each downstream node receives only resultant content block (from upstream nodes) and the original user input. The rest of the message history is not transferred between nodes. Furthermore, the graph edges must be defined upfront. This is a poor fit for non-deterministic workflows where the source agent is responsible for deciding the sub-agent to invoke.

#### Swarms

Users configure a swarm of agents that can hand off to each other:

```ts
const researcher = new Agent({...})
const architect = new Agent({...})
const coder = new Agent({...})

const swarm = new Swarm({
  nodes: [researcher, architect, coder],
})
```

As with graphs, swarm agents lose context during handoffs. Only the handoff message and optional context are transferred between agents. The original agent's full conversation history is lost. Additionally, sub-agents can hand off back to the original agent, which breaks the delegation pattern. When the goal is route to a specialist, bidirectional flow adds complexity without value.

### What's Missing

All three patterns share a fundamental gap: there is no way for an orchestrator to transfer full context to a sub-agent while passing the final response directly to the user.

## Goals and Non-Goals

**Goals:**

- Intuitive API surface for configuring sub-agents
- Mechanism for sharing message history between two agents
- Full execution handoff, where the sub-agent's streams to the user without orchestrator post-processing
- Supports composable delegation (ex. A delegates to B, which delegates to C)
- Integrates with graphs and swarms
- Compatibility with both TypeScript and Python SDKs

**Non-goals:**

- Extending Graph/Swarm primitives. They serve a different use case where agents are peers with limited context sharing
- Replacing agents-as-tools. It already implements return semantics, whereas this proposal focuses on delegation semantics
- Multi-invocation delegation. The orchestrator should not automatically reuse the current sub-agent in the next invocation because the new user prompt may favor a different sub-agent
- Circular delegation detection. Execution may be handed to a previous agent in a cycle but with new information, which is a valid agent flow

## Proposal

`AgentConfig` gains an optional `subAgents` field accepting an array of `InvokableAgent`s. The orchestrator injects sub-agent names and descriptions into its system prompt so the model can make routing decisions.

```ts
const customerService = new Agent({ name: 'CustomerService', description: '...' })
const technicalSupport = new Agent({ name: 'TechnicalSupport', description: '...' })

const orchestrator = new Agent({
  subAgents: [customerService, technicalSupport],
})
```

```python
orchestrator = Agent(
    sub_agents=[customer_service, technical_support],
)
```

The orchestrator agent adds a `HandoffTool` instance to its tool registry if `subAgents` is non-empty. `HandoffTool` is a `Tool` subclass whose `inputSchema` exposes an `agent_name` string field and an optional `message` string field for routing context. It validates the agent's choice and echoes it back in a `ToolResultBlock`.

```ts
export const HANDOFF_TOOL_NAME = 'strands_handoff'

export class HandoffTool extends Tool {
  readonly name = HANDOFF_TOOL_NAME
  readonly toolSpec: ToolSpec

  constructor(subAgents: InvokableAgent[]) {
    super()
    const agentNames = subAgents.map(a => a.name ?? a.id)
    this.toolSpec = {
      name: HANDOFF_TOOL_NAME,
      description: 'Delegate the conversation to a specialist agent.',
      inputSchema: {
        type: 'object',
        properties: {
          agent_name: { type: 'string', enum: agentNames },
          message: { type: 'string', description: 'Optional routing context for the target agent.' },
        },
        required: ['agent_name'],
      },
    }
  }

  async *stream(toolContext: ToolContext): ToolStreamGenerator {
    // The selected sub-agent is passed from the model into the tool input in the agent loop
    const { agent_name, message } = toolContext.toolUse.input as { agent_name: string; message?: string }

    return new ToolResultBlock({
      toolUseId: toolContext.toolUse.toolUseId,
      status: 'success',
      content: [new JsonBlock({ json: { agentName: agent_name, message: message ?? null } })],
    })
  }
}
```

After `executeTools()` completes, the agent loop checks whether any `ToolUseBlock` in the assistant message has `name === HANDOFF_TOOL_NAME`.
- If it does not find one, the orchestrator handles the request normally, with no delegation overhead.
- If it does find one, the `agentName` is extracted from the JSON content and the sub-agent is resolved by name.
  - If the name is invalid, an error tool result is appended and the loop continues so the model can self-correct.

The following proposals diverge from this point onward.

### Recommended: Sub-invocation, Clone Orchestrator History

The orchestrator's full `messages` array is deep cloned as the input to the sub-agent's `stream()`. The sub-agent applies its own system prompt, tools, and conversation manager. From the sub-agent's perspective, this is a normal invocation. Its response streams directly to the caller without re-entering the orchestrator's loop.

Once the sub-agent completes, the orchestrator appends the sub-agent's final assistant message to its own `messages` array. This gives the orchestrator continuity on subsequent invocations: the user can follow up and the orchestrator sees the prior response in its history. Only the final assistant message is appended, not the sub-agent's internal tool calls or reasoning, which would reference tools the orchestrator does not have.

```ts
const handoff = this._extractHandoff(assistantMessage, toolResultMessage)
if (handoff) {
  const target = this._subAgents.find(a => (a.name ?? a.id) === handoff.agentName)

  const subGen = target.stream(this.messages.map(m => m.clone()))
  let next = await subGen.next()
  while (!next.done) {
    yield next.value
    next = await subGen.next()
  }

  const subResult = next.value
  this.messages.push(subResult.lastMessage)

  return new AgentResult({
    stopReason: 'endTurn',
    lastMessage: subResult.lastMessage,
    invocationState,
    traces: this._tracer.localTraces,
    metrics: this._meter.metrics,
  })
}
```

#### Interrupts

If the sub-agent raises an interrupt during execution (such as due to tool use), the interrupt propagates up through the orchestrator's stream generator and surfaces on the orchestrator's `AgentResult` with `stopReason: 'interrupt'`. The orchestrator records the interrupted sub-agent's name in `appState` under a reserved key (`handoff_target`). This follows the same pattern used by other SDK internals that need state to survive across invocations (Cedar stores rate-limit counters in `appState`, HITL stores trusted tool lists). Since `appState` is included in snapshots and session persistence, the handoff target remains durable in stateless deployments where the orchestrator instance may not survive between calls.

When the caller resumes by invoking the orchestrator with `InterruptResponseContent` blocks, the orchestrator checks `appState` for an active handoff target. If one is found, the orchestrator forwards the responses directly to that sub-agent without making another model call, re-entering the handoff path (resolve target, stream, return result). Once the sub-agent completes successfully, the handoff target is cleared from `appState`.

```ts
// Orchestrator records handoff target in appState on interrupt:
this.appState.set('handoff_target', handoff.agentName)

// Caller resumes after interrupt:
const result = await orchestrator.invoke([
  new InterruptResponseContent({ interruptId: 'confirm_action', response: 'yes' })
])
// Orchestrator reads 'handoff_target' from appState, forwards response, streams sub-agent to completion
// On completion:
this.appState.delete('handoff_target')
```

Similarly, sub-agent events are passed directly to the caller. While the sub-agent's hooks and middleware process the sub-agent events, the orchestrator does not touch them. This maintains the delegation pattern because the sub-agent retains full control over execution after the handoff. In the future, new events like `BeforeHandoffEvent` and `AfterHandoffEvent` may be introduced to the orchestrator for visibility into delegation boundaries.

**Pros:**
- Prevents the sub-agent from mutating the orchestrator's message history, maintaining the orchestrator's ownership of its messages
- Minimal context pollution, since the orchestrator's history stays clean of sub-agent tools
- Consistent with existing SDK patterns. Uses the same `InvokeArgs` for input and the same `invocationState` forwarding as graphs, swarms, and agent-as-tool.
- Composable without special handling. A sub-agent with its own `subAgents` chains naturally (A to B to C) because each handoff is a standard `stream()` call.

**Cons:**
- Builds the full message history in sub-agent context on each handoff. This increases the sub-agent's input token cost, especially for long conversations
- The sub-agent's intermediate work is lost on subsequent invocations, so the orchestrator cannot answer questions like "What tools did you use?"


### Alternative: Sub-invocation, Return Full Sub-Agent History

This is mostly the same as the recommended approach. The only difference is that instead of appending the sub-agent's final assistant message, the orchestrator appends the sub-agent's entire message history. This includes all tool calls, tool results, reasoning, and intermediate assistant messages generated during the sub-agent's execution.

```ts
const subResult = next.value
// Append everything the sub-agent produced
for (const msg of subAgent.messages.slice(originalMessageCount)) {
  this.messages.push(msg)
}
```

**Pros:**
- The orchestrator model can reason about the sub-agent's tool calls and other intermediate results, improving responses to subsequent orchestrator invocations

**Cons:**
- The orchestrator sees `toolUse` blocks for tools it does not have registered, so it may attempt to call those tools on subsequent invocations and fail
- Consumes significantly more orchestrator context than providing it with only the final sub-agent result

### Alternative: Sub-invocation, Summarize Orchestrator History

This option uses the same mechanism as the recommended approach. The only difference is that instead of cloning the orchestrator's full message history, the orchestrator generates a summary of the conversation and passes it to the sub-agent as a single user message.

The summary is produced by a model call before the handoff executes. The orchestrator sends its messages to the model with a system prompt instructing it to produce a concise summary of the conversation relevant to the handoff target.

```ts
const handoff = this._extractHandoff(assistantMessage, toolResultMessage)
if (handoff) {
  const target = this._subAgents.find(a => (a.name ?? a.id) === handoff.agentName)

  // Summarize the orchestrator's conversation for the sub-agent
  const summary = await this._summarizeForHandoff(this.messages, handoff.message)
  const subGen = target.stream(summary)
  // ... stream, append final message, return result
}
```

The sub-agent starts with a compact representation of the conversation rather than the full token-heavy history.

**Pros:**

- Lower input token cost on the sub-agent's model call, especially for long conversations
- The sub-agent's context window is not consumed by irrelevant orchestrator history (tool calls to other sub-agents, prior routing decisions)

**Cons:**

- Details from the conversation may be lost, degrading sub-agent performance on tasks that require precise context (no longer a lossless handoff)
- Adds an extra model call for summarization before every handoff, increasing latency and cost on the orchestrator side
- The quality of the handoff depends on the summarization model's ability to identify what's relevant for the target sub-agent, which is hard to get right generically
- Can be added later if sub-agent input token usage becomes a problem

### Alternative: Shared Message History

In Google's ADK, the orchestrator and sub-agents operate within the same `InvocationContext`, sharing a single session with a unified message history and state. All agents read from and write to the same event stream. When the orchestrator delegates, the sub-agent simply continues appending to the shared history as if it were the same agent.

In Strands, this would mean passing the orchestrator's `messages` array by reference to the sub-agent, so both agents mutate the same array during execution.

```ts
// Sub-agent writes directly to orchestrator's messages
subAgent.messages = this.messages
const subGen = subAgent.stream(userMessage)
```

**Pros:**

- Full-history continuity, since a single message history is maintained
- Full bidirectional visibility between orchestrator and sub-agent

**Cons:**

- Requires significant redesigns to state management, from a agent-owned mutable history to a shared state object
- Models can view messages generated under instructions and tools they cannot access, creating the possibility of the orchestrator calling tools it does not have access to
- Snapshots and context reduction from the sub-agent may silently mutate the orchestrator's state.

### Alternative: Agent Swap

In the OpenAI Agents SDK, the runner loop maintains a `currentAgent` variable. When a handoff occurs, the runner swaps `currentAgent` to the target agent and continues the same loop with the new agent's system prompt, tools, and handoff definitions, but the same conversation history. There is no sub-invocation; the loop just changes which "personality" drives the next model call.

In Strands, this would mean the orchestrator's agent loop detects the handoff tool, then swaps its own system prompt, tool registry, and model to those of the target sub-agent, and continues looping:

```ts
this.systemPrompt = target.systemPrompt
this._toolRegistry = target._toolRegistry
this.model = target.model
// Continue the while(true) loop with the new config
continue
```

**Pros:**

- Full-history continuity, since a single message history is maintained
- No sub-invocation overhead. The same loop and messages run on the next cycle, albeit different config.

**Cons:**

- Strands has no runner/agent separation. Mutating the agent's identity mid-loop breaks hooks, plugins, middleware, telemetry, and the printer, which all hold a reference to `this` and expect it to be stable.
- Identity swapping is inconsistent with the rest of the Strands SDK, since existing primitives are separate objects with separate lifecycles
- If the sub-agent finishes and the orchestrator should resume (ex. on the next orchestrator invocation), the swap must be reversed, requiring bookkeeping that the sub-invocation approach avoids entirely.

## Developer Experience

Developers delegate execution to sub-agents by passing them to the `subAgents` field on the orchestrator's constructor (`sub_agents` in Python). The orchestrator decides at runtime which sub-agent to invoke based on the user's request.

TypeScript
```ts
import { Agent } from '@strands-agents/sdk'

const customerService = new Agent({
    name: "CustomerService",
    description: "Handles customer service inquiries, billing questions, and account issues",
    systemPrompt: "You are a customer service specialist...",
})

const technicalSupport = new Agent({
    name: "TechnicalSupport",
    description: "Provides technical support for product issues and troubleshooting",
    systemPrompt: "You are a technical support specialist...",
})

const orchestrator = new Agent({
    name: "HelpDeskOrchestrator",
    description: "Routes customer requests to appropriate specialists",
    systemPrompt: `
    You are a help desk coordinator. Route customer requests to the appropriate specialist:
    - Use CustomerService for billing, account, or general service questions
    - Use TechnicalSupport for product issues, bugs, or technical problems
    - Handle simple greetings and general questions yourself`,
    subAgents: [customerService, technicalSupport],
})

orchestrator.invoke("My wifi doesn't work")
```

Python
```python
from strands import Agent

customer_service = Agent(
    name="CustomerService",
    description="Handles customer service inquiries, billing questions, and account issues",
    system_prompt="You are a customer service specialist...",
)

technical_support = Agent(
    name="TechnicalSupport",
    description="Provides technical support for product issues and troubleshooting",
    system_prompt="You are a technical support specialist...",
)

orchestrator = Agent(
    name="HelpDeskOrchestrator",
    description="Routes customer requests to appropriate specialists",
    system_prompt="""
    You are a help desk coordinator. Route customer requests to the appropriate specialist:
    - Use CustomerService for billing, account, or general service questions
    - Use TechnicalSupport for product issues, bugs, or technical problems
    - Handle simple greetings and general questions yourself""",
    sub_agents=[customer_service, technical_support],
)

orchestrator("My wifi doesn't work")
```

### Composable Delegation

Sub-agents can themselves declare `subAgents`, forming delegation chains.

TypeScript
```ts
import { Agent } from '@strands-agents/sdk'

const billingAgent = new Agent({
    name: "Billing",
    description: "Processes refunds, invoices, and payment disputes",
    systemPrompt: "You are a billing specialist...",
})

const customerService = new Agent({
    name: "CustomerService",
    description: "Handles customer service inquiries. Delegates billing and shipping questions to specialists.",
    systemPrompt: `You are a customer service coordinator.
    - Use Billing for refunds, invoices, or payment issues
    - Handle general account questions yourself`,
    subAgents: [billingAgent],
})

const technicalSupport = new Agent({
    name: "TechnicalSupport",
    description: "Provides technical support for product issues and troubleshooting",
    systemPrompt: "You are a technical support specialist...",
})

const orchestrator = new Agent({
    name: "HelpDesk",
    description: "Routes customer requests to appropriate departments",
    systemPrompt: `Route requests to the appropriate department:
    - Use CustomerService for billing, shipping, or account questions
    - Use TechnicalSupport for product issues or bugs`,
    subAgents: [customerService, technicalSupport],
})

orchestrator.invoke("Where's my refund for order #12345?")
```

## Consequences

- Builders get a simple path to multi-agent routing that avoids the latency/token cost of agents-as-tools and the context loss of swarms.
- Adding `subAgents` as a first-class concept introduces a new relationship type between agents. This needs clear documentation to help builders choose between `tools: [agent]`, `subAgents: [agent]`, and multi-agent primitives.

## Willingness to Implement

Yes
