# Agent-as-Tool Handoff

**Status**: Proposed

**Date**: 2026-07-13

**Issue**: https://github.com/strands-agents/harness-sdk/issues/911

## Problem

Builders want a way to delegate tasks from orchestrator agents to sub-agents without the orchestrator post-processing sub-agent responses. This minimizes extraneous model calls, latency, and token usage when the sub-agent response is ready to be served to the user as-is. However, there is currently no way to both enable delegation and disable orchestrator post-processing in the SDK.

### Desired Use Cases

The following summarizes desired use cases discussed in the linked issue thread.

**Domain routing without post-processing.** A help-desk orchestrator analyzes incoming requests and delegates to a specialist: billing questions to a CustomerService agent, technical issues to a TechnicalSupport agent, etc. The specialist's response is the final response. The orchestrator does not re-process it to avoid adding cost without value. This last point is reinforced in hierarchical routing scenarios, a natural extension of domain routing, where a manager agent delegates to senior agents which delegate to junior agents.

**Structured output preservation.** A supervisor agent delegates to a sub-agent that generates a JSON schema (or other structured data). The JSON must reach the user verbatim and cannot be paraphrased by the orchestrator agent.

### Current State

Agent-as-tools almost accomplishes the problem statement above. It implements delegation by letting the orchestrator model call sub-agents via the tools interface:

```ts
const researchAgent = new Agent({...})
const productAgent = new Agent({...})
const travelAgent = new Agent({...})

const orchestrator = new Agent({
  tools: [researchAgent, productAgent, travelAgent],
})
```

However, there is currently no way to stop the orchestrator model from processing sub-agent responses before passing it to the user. This adds unnecessary latency and token usage for simple routing scenarios. Furthermore, the orchestrator may paraphrase or reformat structured output (JSON, code, formatted data) from the sub-agent, breaking otherwise valid responses.

## Goals and Non-Goals

**Goals:**

- Intuitive API surface for telling agents to directly return an agent-as-tool response
- Mechanism for returning sub-agent results directly to the user without additional orchestrator reasoning
- Handle edge case where multiple agent-as-tools are handed off to in the same turn
- Can be extended in the future to directly return non-agent tool results
- Compatible with both TypeScript and Python SDKs

**Non-goals:**

- Multi-turn conversation handoff where the user is transferred to a specialized agent for extended interaction. This can be explored in a separate proposal.
- Extending Graph/Swarm primitives. They serve a different use case where agents are peers with limited context sharing.

## Proposal

### API Surface

The following API surfaces configure agent-as-tools to stream directly to the user without an additional orchestrator model call.

#### Recommended: `handoff` flag on `asTool()`

`Agent.asTool()` gains an optional `handoff` boolean. When `true`, the orchestrator loop treats the agent tool result as the final response and exits (not to be confused with multi-turn handoffs). It is orthogonal to existing options like `preserveContext`.

```typescript
const customerService = new Agent({...})

const technicalSupport = new Agent({...})

const orchestrator = new Agent({
  tools: [
    customerService.asTool({ handoff: true }),
    technicalSupport.asTool({ handoff: true }),
  ],
})
```

**Pros:**
- Minimal API surface change, since this only adds a single boolean to an existing method.
- Explicit per-tool opt-in gives fine-grained control where some sub-agents can delegate and others can return results for orchestrator reasoning

**Cons:**
- Forces builders to call `.asTool()`, which is verbose especially when creating many handoff agent tools

#### Alternative: `handoffTarget` flag on sub-agents

Sub-agents declare themselves as handoff targets via a constructor option. The orchestrator passes raw `Agent` instances (not wrapped tools) to its `tools` array. The SDK auto-wraps them as tools and passes the `handoffTarget` into the tool constructor.

```typescript
const customerService = new Agent({
  handoffTarget: true,
  ...
})

const technicalSupport = new Agent({
  handoffTarget: true,
  ...
})

const orchestrator = new Agent({
  tools: [customerService, technicalSupport],
})
```

**Pros:**
- Builders can declare dedicated handoff sub-agents which any orchestrator can handoff to.
- Clean pattern for the common case where all sub-agents in a routing scenario are handoff targets.

**Cons:**
- Cannot express cases where an agent is both a handoff target in one orchestrator and a regular tool in another.
- Adds a new config field to `Agent` that only has meaning when the agent is used as a tool, which leaks concerns.

#### Alternative: Dedicated `handoffAgentTools` field on orchestrator

`AgentConfig` gains a separate `handoffAgentTools` array alongside `tools`. Agents placed in `handoffAgentTools` are treated with handoff logic. Regular `tools` continue to work as before.

```typescript
const customerService = new Agent({...})

const technicalSupport = new Agent({...})

const orchestrator = new Agent({
  handoffAgentTools: [customerService, technicalSupport],
})
```

**Pros:**
- Clear separation of intent at the orchestrator level.
- No flag pollution on sub-agents or on `.asTool()` options.
- Easy to mix delegation agents with regular tools on the same orchestrator.

**Cons:**
- Two arrays that feed into the same tool registry may confuse builders.
- Introduces a new top-level config field, expanding the `AgentConfig` surface area.
- If non-agent tool handoffs are supported in the future, `handoffAgentTools` would need to be renamed and accept arbitrary `Tool` instances. This creates a deprecation path: either broaden the field's type (breaking the original contract) or introduce a replacement field and deprecate `handoffAgentTools`.

### Handoff Mechanism

Regardless of which API surface is chosen, `Agent.asTool({ handoff: true })` sets `directReturn: true` on the resulting tool instance. `directReturn` is a property on the base `Tool` class, so it can be extended in the future to support non-agent tools that want to return results directly to the user without additional orchestrator reasoning.

A description suffix is applied to the agent-as-tool during construction time if `handoff: true`. The suffix resembles the following:

> "Calling this tool will return its response directly to the user as the final answer. It should be the only tool called in the turn."

This nudges the model to treat the tool call as a terminal action, avoiding wasted tokens on preamble text, parallel tool calls, or post-processing plans. The builder-provided description (or the agent's description) forms the first part, and the handoff instruction is appended as a trailing sentence.

The potential handoff implementations are described in the proposals below. In both cases, the final AgentResult is passed a new `stopReason` of `handoff` for observability.

### Recommended: Direct Return Plugin

This implements the handoff mechanism using a vended plugin that combines hooks (for control flow) with middleware (for result transformation). The SDK vends a `DirectReturnPlugin` that is auto-registered when any tool in the agent's tool list has `directReturn: true`.

The plugin subscribes to `BeforeToolsEvent` (TS) or the per-tool `BeforeToolCallEvent` (Python). When the model emits a handoff tool call alongside other tool calls in the same assistant message, the hook cancels all tool execution:

```typescript
agent.addHook(BeforeToolsEvent, (event) => {
  const toolUseBlocks = event.message.content.filter(
    (b): b is ToolUseBlock => b.type === 'toolUseBlock'
  )
  const hasHandoff = toolUseBlocks.some(b => isHandoffTool(agent, b.name))

  if (hasHandoff && toolUseBlocks.length > 1) {
    event.cancel =
      'This tool call was not executed. A handoff tool must be the only ' +
      'tool called in a turn. Retry with a single handoff tool call or ' +
      'use only non-handoff tools.'
  }
})
```

`BeforeToolsEvent.cancel` (TS) produces error results for all tool-use blocks in the batch. In Python, per-tool `BeforeToolCallEvent.cancel` cancels each tool individually, achieving the same outcome. The model sees the error results and retries on the next turn. The retry counts against `limits.turns` as normal.

After the single handoff tool executes successfully, the plugin ends the turn using the existing `endTurn: string` field. It extracts the text representation of the handoff result and stops the loop.

```typescript
agent.addHook(AfterToolsEvent, (event) => {
  const handoffResult = findSuccessfulHandoff(agent, event.message)
  if (handoffResult) {
    this._handoffTriggered = true
    // endTurn accepts a string — stops the loop with a text representation
    event.endTurn = extractText(handoffResult)
  }
})
```

The plugin registers middleware on `AgentStreamStage` that wraps the entire agent stream. When a handoff was triggered, the middleware walks `agent.messages` to find the tool-result message containing the handoff `ToolResultBlock`, extracts its rich content, and replaces `AgentResult.lastMessage`:

```typescript
agent.addMiddleware(AgentStreamStage, async function* (context, next) {
  const streamResult = yield* next(context)

  if (!this._handoffTriggered) return streamResult
  this._handoffTriggered = false

  // The tool-result message is in history — find the handoff tool's result block
  const toolResultMsg = context.agent.messages.findLast(
    msg => msg.role === 'user' && msg.content.some(b => b.type === 'toolResultBlock')
  )
  const handoffBlock = toolResultMsg?.content.find(
    (b): b is ToolResultBlock =>
      b.type === 'toolResultBlock' && isHandoffToolId(context.agent, b.toolUseId)
  )

  if (handoffBlock?.status === 'success') {
    return {
      result: new AgentResult({
        ...streamResult.result,
        stopReason: 'handoff',
        lastMessage: new Message({ role: 'assistant', content: handoffBlock.content }),
      }),
    }
  }
  return streamResult
})
```

The middleware's sole responsibility is output transformation. It sets the AgentResult message to the rich content from the `ToolResultBlock` in the conversation history. The hook and middleware communicate only through a boolean flag (`_handoffTriggered`) on the plugin instance.

**Pros:**
- No handoff-specific branches in the agent loop. The loop stays general-purpose.
- Reuses existing hook primitives (`BeforeToolsEvent.cancel`, `AfterToolsEvent.endTurn`) and middleware (`AgentStreamStage`) — validates that the existing extension points are expressive enough for complex control flow.

**Cons:**
- History divergence. For structured output (JSON, multi-block content), `AgentResult.lastMessage` and `agent.messages[-1]` point to different messages. The caller gets the rich content; history retains a text approximation. This breaks the invariant that `lastMessage` is the same object as the last history entry.

### Alternative: Agent Loop Changes

This approach modifies the agent loop directly.

After `executeTools()` completes and the assistant message + tool-result message are appended to history, the loop checks for a successful handoff result:
- If the handoff tool's result has `status: 'error'`, skip it and continue the normal loop so the model can recover or try a different tool.
- If the `ToolResultBlock` has `status: 'success'`, return it as `AgentResult.lastMessage` with `stopReason: 'handoff'`. The tool-result message (role: `assistant`) is the `lastMessage`.

Despite the description suffix nudging the model to treat a handoff tool call as a terminal action, some models may still emit a handoff tool call alongside other tool calls in the same assistant message. When this happens, the agent loop rejects the turn **before executing any tools** and forces the model to retry with a single handoff call.

**Detection.** After the model returns a `toolUse` stop reason, the loop inspects the assistant message's `ToolUseBlock`s. If the message contains a handoff tool call *and* one or more other tool calls (handoff or non-handoff), the turn is invalid.

**Recovery.** The loop appends the assistant message to history along with a synthetic tool-result message containing error results for every tool use in the turn. Each error result carries a message explaining the constraint:

> "This tool call was not executed. A handoff tool must be the only tool called in a turn. Retry with a single handoff tool call or use only non-handoff tools."

The loop then continues to the next model call. The model sees the error results in context and can correct its behavior, either calling the handoff tool alone or falling back to non-handoff tools.

**Pros:**
- Full control over `AgentResult` construction. The loop can set any `stopReason`, attach arbitrary metadata, and shape the `lastMessage` directly without being constrained by the hook API surface.

**Cons:**
- Adds handoff-specific branches to the agent loop, increasing loop complexity and coupling it to a feature that not all agents use.
- Sets a precedent for future features to add their own loop branches rather than composing via hooks.

## Developer Experience

The following DevX snippets assume the first proposal is used.

### Basic Usage

An orchestrator delegates to specialist agents by wrapping them with `handoff: true`. The sub-agent's response is returned directly without an additional orchestrator model call.

TypeScript:
```typescript
import { Agent } from '@strands-agents/sdk'

const customerService = new Agent({
  name: 'CustomerService',
  description: 'Handles billing questions and account issues',
  systemPrompt: 'You are a customer service specialist. Help the user with their billing or account question.',
})

const technicalSupport = new Agent({
  name: 'TechnicalSupport',
  description: 'Provides technical support and troubleshooting',
  systemPrompt: 'You are a technical support specialist. Help the user troubleshoot their issue.',
})

const orchestrator = new Agent({
  name: 'HelpDesk',
  systemPrompt: 'You are a help desk coordinator. Route requests to the appropriate specialist.',
  tools: [
    customerService.asTool({ handoff: true }),
    technicalSupport.asTool({ handoff: true }),
  ],
})

const result = await orchestrator.invoke('My wifi does not work')
// result.lastMessage contains the TechnicalSupport agent's response directly
```

Python:
```python
from strands import Agent

customer_service = Agent(
    name="CustomerService",
    description="Handles billing questions and account issues",
    system_prompt="You are a customer service specialist. Help the user with their billing or account question.",
)

technical_support = Agent(
    name="TechnicalSupport",
    description="Provides technical support and troubleshooting",
    system_prompt="You are a technical support specialist. Help the user troubleshoot their issue.",
)

orchestrator = Agent(
    name="HelpDesk",
    system_prompt="You are a help desk coordinator. Route requests to the appropriate specialist.",
    tools=[
        customer_service.as_tool(handoff=True),
        technical_support.as_tool(handoff=True),
    ],
)

result = orchestrator("My wifi does not work")
# result contains the TechnicalSupport agent's response directly
```

### Mixing Handoff and Regular Tools

Handoff tools can coexist with regular tools on the same orchestrator. The orchestrator processes regular tool results normally but short-circuits when a handoff tool succeeds.

```typescript
import { Agent } from '@strands-agents/sdk'
import { calculatorTool } from './tools'

const researcher = new Agent({
  name: 'Researcher',
  description: 'Performs deep research on a topic and returns a comprehensive report',
})

const orchestrator = new Agent({
  tools: [
    calculatorTool,                          // regular tool — results feed back to orchestrator
    researcher.asTool({ handoff: true }),     // handoff — response goes directly to user
  ],
})
```

### Hierarchical Delegation

Sub-agents can themselves use handoff tools, forming delegation chains. Each level routes deeper without adding model calls on the way back up.

```typescript
const billingAgent = new Agent({
  name: 'Billing',
  description: 'Processes refunds, invoices, and payment disputes',
})

const customerService = new Agent({
  name: 'CustomerService',
  description: 'Handles customer service inquiries. Delegates billing to a specialist.',
  tools: [
    billingAgent.asTool({ handoff: true }),
  ],
})

const orchestrator = new Agent({
  name: 'HelpDesk',
  tools: [
    customerService.asTool({ handoff: true }),
  ],
})

// orchestrator → CustomerService → Billing (two handoffs, zero post-processing calls)
await orchestrator.invoke("Where's my refund for order #12345?")
```

### Structured Output Preservation

A sub-agent produces JSON that must reach the caller verbatim. The handoff flag ensures the orchestrator never paraphrases or reformats the response.

```typescript
const schemaGenerator = new Agent({
  name: 'SchemaGenerator',
  description: 'Generates a JSON Schema from a natural-language description of a data model',
  systemPrompt: `You are a JSON Schema expert. Return ONLY valid JSON Schema (no markdown fences, no commentary).`,
})

const orchestrator = new Agent({
  name: 'APIDesigner',
  systemPrompt: 'You help users design APIs. When the user describes a data model, delegate to the schema generator.',
  tools: [
    schemaGenerator.asTool({ handoff: true }),
  ],
})

const result = await orchestrator.invoke('I need a schema for a User with name, email, and an array of roles')
// result.lastMessage contains raw JSON Schema — no orchestrator paraphrasing
```

```python
from strands import Agent
import json

schema_generator = Agent(
    name="SchemaGenerator",
    description="Generates a JSON Schema from a natural-language description of a data model",
    system_prompt="You are a JSON Schema expert. Return ONLY valid JSON Schema (no markdown fences, no commentary).",
)

orchestrator = Agent(
    name="APIDesigner",
    system_prompt="You help users design APIs. When the user describes a data model, delegate to the schema generator.",
    tools=[
        schema_generator.as_tool(handoff=True),
    ],
)

result = orchestrator("I need a schema for a User with name, email, and an array of roles")
# result contains raw JSON Schema — no orchestrator paraphrasing
```

## Consequences

**What becomes easier:**

- Building multi-agent routing systems (help desks, domain specialists) without paying for an extra model call on every delegation.
- Preserving structured output from sub-agents. JSON, code, and formatted data reach the user verbatim.

**What becomes more difficult:**

- Understanding the difference between handoff agent tools and regular tools. When used incorrectly, users may encounter unexpected handoffs. The builder must ensure their system prompt and tool descriptions guide routing correctly.

## Willingness to Implement

Yes
