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
- Mechanism for streaming sub-agent results directly to the user without additional orchestrator reasoning
- Handle edge case where multiple agent-as-tools are handed off to in the same turn
- Can be extended in the future to directly return non-agent tool results
- Compatible with both TypeScript and Python SDKs

**Non-goals:**

- Multi-turn conversation handoff where the user is transferred to a specialized agent for extended interaction. This can be explored in a separate proposal.
- Extending Graph/Swarm primitives. They serve a different use case where agents are peers with limited context sharing.

## Proposal

The following API surfaces configure agent-as-tools to stream directly to the user without an additional orchestrator model call.

### Recommended: `handoff` flag on `asTool()`

`Agent.asTool()` gains an optional `handoff` boolean. When `true`, the orchestrator loop treats the agent tool result as the final response and exits.

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

### Alternative: `handoffTarget` flag on sub-agents

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

### Alternative: Dedicated `handoffAgentTools` field on orchestrator

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

### Agent Loop Changes

Regardless of which API surface is chosen, the handoff behavior works inside the agent loop. When the agent loop builds the `toolSpecs` array for a model call, it appends a suffix to the description of any tool marked as a handoff target. The suffix resembles the following:

> "Calling this tool will return its response directly to the user as the final answer. It should be the only tool called in the turn."

This nudges the model to treat the tool call as a terminal action, avoiding wasted tokens on preamble text, parallel tool calls, or post-processing plans. By appending the suffix at read time, the tool's description remains clean (reflecting only what the builder wrote or what was derived from the agent's description) and the handoff instruction remains at the model-calling boundary. This approach also extends naturally to non-agent tool handoffs in the future.

After `executeTools()` completes and the assistant message + tool-result message are appended to history, the loop checks for a successful handoff result:
- If the handoff tool's result has `status: 'error'`, skip it and continue the normal loop so the model can recover or try a different tool.
- If the `ToolResultBlock` has `status: 'success'`, return it as `AgentResult.lastMessage` with `stopReason: 'endTurn'`. The tool-result message (role: `user`) is the `lastMessage` — no synthetic assistant message is created.

#### Multiple Tool Calls in a Single Turn

When the model calls a handoff tool and other tools in the same turn despite the handoff tool description suffix, all tools execute to completion. Afterwards, all tool use and tool result blocks are appended to history.
- If there are non-handoff tools, they are not passed back to thea orchestrator model
- If there are multiple handoff tools, the first successful result in tool use order is returned. This can be achieved by iterating on the `ToolUseBlock`s in the assistant message and checking the corresponding `ToolResultBlock`. 

Other potential ways to handle multiple tool calls include:
- Continuing the agent loop if multiple handoff tool calls are detected. This forces the model to choose a single handoff target, but adds a model call and risks an infinite loop if the model uses the same set of tools on subsequent turns.
- Concatenating the handoff tool results together. Langchain uses a similar approach where they append each ToolMessage response to a list and return the entire list. However, `AgentResult.lastMessage` is currently a `Message`, not a `Message` array. If sub-agent responses are concatenated into the same message object, any structured data they produce breaks.

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

### Edge Cases

**Handoff tool returns an error.** The loop continues normally. The model receives the error in the tool result and can retry or choose a different tool.

**Model calls handoff tool alongside other tools.** All tools execute. The handoff result is returned; non-handoff results remain in history but are not fed back to the model.

**Same agent used as handoff in one orchestrator and regular tool in another.** Each `.asTool()` call creates an independent instance:

```typescript
const researcher = new Agent({ name: 'Researcher', description: '...' })

// Orchestrator A: researcher is a handoff target
const orchestratorA = new Agent({ tools: [researcher.asTool({ handoff: true })] })

// Orchestrator B: researcher is a regular tool (results feed back for synthesis)
const orchestratorB = new Agent({ tools: [researcher.asTool()] })
```

## Consequences

**What becomes easier:**

- Building multi-agent routing systems (help desks, domain specialists) without paying for an extra model call on every delegation.
- Preserving structured output from sub-agents. JSON, code, and formatted data reach the user verbatim.

**What becomes more difficult:**

- Understanding parallel tool-call behavior. If the model calls a handoff agent tool alongside other tools, the non-handoff results are "lost" from the model's perspective (they exist in history but are never reasoned over). Builders need to understand that the description suffix discourages this, but doesn't prevent it.
- Understanding the difference between handoff agent tools and regular tools. When used incorrectly, users may encounter unexpected handoffs. The builder must ensure their system prompt and tool descriptions guide routing correctly.

## Willingness to Implement

Yes
