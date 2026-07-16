# Multi-agent conventions, vended tools

This document defines the shared dialect that every multi-agent vended tool
implements. The scope covers tools that let an agent spin up child agents at
runtime: use_agent (single child), swarm (handoff team), graph (structured
DAG), and a2a_client (remote agent card).

## Agent spec

Every child-agent-defining tool accepts agent specs with the same shape:

```
{
    name: str,          # required, [a-zA-Z_][a-zA-Z0-9_]{0,63}
    system_prompt: str, # required, <= 8 KiB
    tools: [str, ...],  # allow-list of parent-registered tool names
                        # required (may be empty for a no-tool child)
                        # no wildcards, no `"*"`, no glob patterns
                        # unknown names are rejected at the boundary
}
```

No model_provider or model_settings field. Children inherit the parent's
model instance. Rationale: model-selection at spec time is a
credential-injection surface. Provider constructors accept api_key,
client_args, endpoint_url, base_url, which a compromised model can supply
to route to attacker-controlled endpoints or bill an attacker's account.
The convenience of per-child model choice does not offset that surface. If
a developer wants child-model variance they can register N pre-configured
factories rather than exposing model config to the model.

## Recursion depth

Every multi-agent tool participates in a single shared depth counter tracked
on the parent's invocation_state:

- Python: `invocation_state["multiagent_depth"]`
- TypeScript: `invocationState.multiagentDepth`

Default cap: three. Configurable at tool construction only, never by the
model.

At the tool's entry point:

1. Read the current depth from invocation_state (default 0 if absent).
2. If depth is at or above the cap, reject with a specific
   MultiagentDepthExceeded error.
3. Merge the parent's invocation_state into the child's invocation_state,
   then override multiagent_depth with the incremented value, and pass that
   through to the child's invoke call.

If a tool skips step 3, an adversarial model can walk use_agent then swarm
then use_agent and reset the counter at each hop. All participating tools
must merge-and-increment rather than replace.

## Tool allow-list

Every agent spec's `tools` field is an explicit name allow-list from the
parent's registry:

- Names are matched exactly, case-sensitive.
- Unknown names are rejected at the boundary before construction.
- No wildcards, no glob patterns, no "all my tools" shortcut.
- The multi-agent tools themselves (use_agent, swarm, graph, a2a_client)
  may not be listed in a child's tools (defense-in-depth against
  tool-registered-then-recursed patterns that bypass the depth cap).
  Reject them explicitly.

If a caller wants the child to have the multi-agent tools, they can register
variants themselves at construction time; the model cannot grant that
authority.

## Result shape

All participating tools return a result dict with this minimum shape:

```
{
    status: "completed" | "failed" | "cancelled" | "interrupted",
    output: str,               # the primary textual answer
    execution_time_ms: int,    # total wall-clock
}
```

Status strings are the lower-cased values of the SDK's `Status` enum, single
word, byte-identical across Python and TypeScript. The TypeScript field name
is `executionTimeMs` per the SDK's camelCase convention; the string values of
`status` itself are byte-identical across the two SDKs.

Extended fields (each tool may add its own, additive, not renaming):

- swarm: node_history (list of str), execution_count (int), usage
  ({input_tokens, output_tokens, total_tokens})
- graph: execution_order (list of str), results (per-node map of
  {status, output})
- use_agent: no additional fields required
- a2a_client: remote_card (subset of the resolved agent card)

Tools may return richer shapes but must keep status, output, and
execution_time_ms at the top level so downstream models get a consistent
contract regardless of which pattern they invoke.

## Cancellation

Parent cancellation must propagate to the child or children within 100 ms:

- Python: poll the parent's cancel signal at the tool's supervision level
  and call cancel on the child when set. For multi-child tools (swarm,
  graph), the SDK's BeforeNodeCallEvent hook is the correct extension
  point: set event.cancel_node when the parent is cancelled.
- TypeScript: forward the parent's cancelSignal into the child agent or
  swarm's invoke options and into any transport-level AbortSignal the
  child holds. AbortSignal.any is the standard composition.

Python return contract: on cancellation the tool returns
`{status: "cancelled", output: partial or empty, execution_time_ms}`. It
does not raise past the tool boundary.

TypeScript raise contract: on cancellation the tool re-raises the parent's
AbortError so callers can distinguish cancellation from other failures via
`error.name === 'AbortError'`, matching the sibling http-request tool. This
is the TypeScript SDK's cancellation idiom; the shape divergence from
Python is intentional and matches how transport-level cancellation flows
through the TypeScript SDK.

## Size caps

At the tool boundary, before constructing anything:

- system_prompt: 8 KiB per spec
- task or initial_input: 32 KiB
- name: 64 characters (regex above)
- tools list: 64 entries max
- Number of agents (swarm, graph nodes): 20 max
- Number of edges (graph): 40 max
- Total execution time: 300 seconds wall-clock

These are defaults, factory-configurable at tool construction but not
model-controllable.
