# Graph Tool

Deterministic directed acyclic graph orchestration for multi-agent pipelines. The caller agent describes a graph of sub-agents at tool-call time; the tool constructs a Graph, runs it, and returns per-node text output.

Shim over [Graph](../../multiagent/graph.ts) — all execution semantics live there.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { BedrockModel } from '@strands-agents/sdk/models/bedrock'
import { graph } from '@strands-agents/sdk/vended-tools/graph'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'

const agent = new Agent({
  model: new BedrockModel({ region: 'us-east-1' }),
  // Any tool listed here is eligible to appear in a node's `tools` allow-list.
  tools: [graph, httpRequest],
})

await agent.invoke(
  'Research the top three JS frameworks and summarise them. ' +
    'Use the graph tool with a researcher and a summariser node.'
)
```

The model produces a call like:

```json
{
  "nodes": [
    { "id": "research", "systemPrompt": "Find recent facts.", "tools": ["http_request"] },
    { "id": "summarise", "systemPrompt": "Turn findings into 3 bullets." }
  ],
  "edges": [{ "fromId": "research", "toId": "summarise" }],
  "initialInput": "top three JS frameworks"
}
```

The tool builds two sub-agents, hooks research to summarise, runs the graph, and returns:

```json
{
  "status": "completed",
  "output": "...",
  "executionOrder": ["research", "summarise"],
  "results": {
    "research": { "status": "completed", "output": "...", "executionTimeMs": 812 },
    "summarise": { "status": "completed", "output": "...", "executionTimeMs": 604 }
  },
  "executionTimeMs": 1430
}
```

The top-level `output` is the concatenation of every terminal (leaf) node's text, joined by a blank line, so a graph without a single named sink still returns every final answer. See [the shared multi-agent conventions](../_multiagent-conventions.md) for the wider dialect.

## Input

```typescript
interface GraphInput {
  nodes: {
    id: string // [A-Za-z0-9_-], up to 64 chars, unique
    systemPrompt?: string // <= 8000 chars
    tools?: string[] // names from the parent agent's registry; no wildcards
  }[]
  edges: { fromId: string; toId: string }[]
  initialInput: string // <= 32000 chars
}
```

Caps:

| Field                     | Cap         |
| ------------------------- | ----------- |
| Nodes                     | 20          |
| Edges                     | 40          |
| Tools per node allow-list | 64          |
| Node id length            | 64          |
| System prompt length      | 8000        |
| Initial input length      | 32000       |
| Max node executions       | 40          |
| Total wall clock          | 300 seconds |
| Per-node wall clock       | 120 seconds |

The 300-second total is enforced as a hard ceiling: the invocation is wrapped in a timeout so a chain of long-running nodes cannot silently exceed it, even though the underlying Graph's own execution timeout is checked at node boundaries. If the ceiling fires, the tool returns a cancelled result with an empty output.

## Result

```typescript
interface GraphOutput {
  status: string // "completed" | "failed" | "cancelled" | "interrupted"
  output: string // concatenated terminal-node text (blank-line separated)
  executionOrder: string[] // node ids in completion order
  results: Record<
    string,
    {
      status: string
      output: string // text-only; non-text content blocks are dropped
      executionTimeMs: number
    }
  >
  executionTimeMs: number
}
```

Status strings are single-word lower-case values that match the Python side byte-for-byte, per the shared multi-agent dialect.

## Recursion depth

The tool participates in the shared `multiagentDepth` counter threaded through `invocationState`. Chains that cross tool boundaries (`use_agent -> graph -> use_agent`) are all capped at the same limit (default three). See [the shared multi-agent conventions](../_multiagent-conventions.md).

## Customising the tool

Use `makeGraph` when you need a different tool name, description, or depth cap:

```typescript
import { makeGraph } from '@strands-agents/sdk/vended-tools/graph'

const shallowGraph = makeGraph({ maxDepth: 2 })
```

The knobs are factory-only — nothing surfaces to the model.

## Non-goals

Conditional edges are not supported: a shim should not evaluate model-supplied predicates at every edge. Per-node model overrides are not supported: sub-agents inherit the parent's model instance so credentials are never accepted from the caller. Node outputs are flattened to text, so nodes returning images or tool-use blocks will not propagate those to the graph result. If you need any of these, author a Graph directly.
