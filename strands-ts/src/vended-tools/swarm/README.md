# Swarm Tool

Lets an agent spin up a small handoff-based sub-agent team at runtime. Thin shim over the SDK's Swarm class.

The parent supplies a list of sub-agent specs (name, system prompt, tool allowlist) and an initial input. Sub-agents inherit the parent's model and may only use tools whose names appear in the parent's tool registry. The tool participates in the shared multi-agent recursion counter (invocationState.multiagentDepth, default cap three) so nested delegation cannot reset it. Default caps on agent count (five), total wall-clock (three hundred seconds), and per-node wall-clock (one hundred twenty seconds) are set at the tool boundary and not model-configurable; use makeSwarm to construct a tool with a tighter envelope. See [_multiagent-conventions.md](../_multiagent-conventions.md) for the full shared dialect.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { swarm } from '@strands-agents/sdk/vended-tools/swarm'

const agent = new Agent({ tools: [swarm] })
await agent.invoke('Use a swarm to research and summarize recent octopus news.')
```

For a custom name or tighter caps, use the factory:

```typescript
import { makeSwarm } from '@strands-agents/sdk/vended-tools/swarm'

const tightSwarm = makeSwarm({ name: 'mini-swarm', maxAgents: 3, executionTimeoutMs: 60_000 })
```

The agent decides at runtime to call `swarm` with something like:

```json
{
  "agents": [
    {
      "name": "researcher",
      "systemPrompt": "You gather facts from public sources.",
      "tools": ["http_request"]
    },
    {
      "name": "writer",
      "systemPrompt": "You turn facts into a concise summary.",
      "tools": []
    }
  ],
  "initialInput": "Find and summarize recent octopus news.",
  "entryAgent": "researcher"
}
```

## Result shape

```typescript
{
  status: 'success' | 'error' | 'cancelled',
  output: string,             // final text from the terminal agent
  nodeHistory: string[],      // agent ids in completion order
  executionCount: number,
  executionTimeMs: number,
  usage: { inputTokens: number, outputTokens: number, totalTokens: number },
}
```

`status` is translated from the SDK's execution vocabulary into the shared multi-agent result dialect so downstream models see a consistent contract across every multi-agent tool.

## When to reach for something else

If the sub-agents do not need to hand off and you know the sequence up front, a `Graph` is cheaper and more predictable. If you need custom models per sub-agent, or per-sub-agent hooks and plugins, build the `Swarm` yourself — this tool is intentionally opinionated.
