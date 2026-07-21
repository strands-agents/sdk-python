# Multi-agent conventions — vended tools

Reviewer consensus: four PRs (#65 graph, #66 swarm, #69 a2a_client, #73 use_agent) each shipped their own version of `_multiagent_conventions.md` and no two agree. This document is the resolution.

Scope: `use_agent`, `swarm`, `graph`, `a2a_client`. All four must implement this dialect. If a tool can't, its PR body must explain the deviation.

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

**No `model_provider` / `model_settings` field.** Children inherit the parent's model instance. Rationale: model-selection at spec time is a credential-injection surface — provider constructors accept `api_key`, `client_args`, `endpoint_url`, `base_url`, which a compromised model can supply to route to attacker-controlled endpoints or bill an attacker's account. The convenience of per-child model choice does not offset that surface. If a developer wants child-model variance they can register N pre-configured factories rather than exposing model config to the model.

This is a change from the sub-issues and from three of the four PRs. `use_agent` (#73) keeps a `model_settings` field today — that field must be removed as part of this convergence.

## Recursion depth

Every multi-agent tool participates in a single shared depth counter tracked on the parent's `invocation_state`:

- Python: `invocation_state["multiagent_depth"]`
- TypeScript: `invocationState.multiagentDepth`

Default cap: **3.** Configurable at tool construction only, never by the model.

At the tool's entry point:

1. Read the current depth from `invocation_state` (default 0 if absent).
2. If `depth >= cap`, reject with a specific `MultiagentDepthExceeded` error.
3. Construct the child's `invocation_state` with `multiagent_depth = depth + 1` and pass it into `Agent.invoke_async` (py) / `agent.invoke({ invocationState })` (ts).

If a tool skips step 3, an adversarial model can walk `use_agent → swarm → use_agent` and reset the counter at each hop. All four must participate.

## Tool allow-list

Every agent spec's `tools` field is an explicit name allow-list from the parent's registry:

- Names are matched exactly, case-sensitive.
- Unknown names are rejected at the boundary before construction.
- No wildcards, no glob patterns, no "all my tools" shortcut.
- Multi-agent tools (`use_agent`, `swarm`, `graph`, `a2a_client`) may not be listed in a child's tools (defense-in-depth against tool-registered-then-recursed patterns that bypass the depth cap). Reject them explicitly.

If a caller wants the child to have the multi-agent tools, they can register variants themselves at construction time; the model can't grant that authority.

## Result shape

All four tools return a result dict with this minimum shape:

```
{
    status: "completed" | "failed" | "cancelled" | "interrupted",
    output: str,               # the primary textual answer
    execution_time_ms: int,    # total wall-clock
}
```

Status strings are the lower-cased values of the SDK's `Status` enum, single word, byte-identical across Python and TypeScript. The TypeScript field name is `executionTimeMs` per the SDK's camelCase convention; the string values of `status` itself are byte-identical across the two SDKs.

Extended fields (each tool may add its own — should be additive, not renaming):

- `swarm`: `node_history: list[str]`, `execution_count: int`, `usage: {input_tokens, output_tokens}`
- `graph`: `execution_order: list[str]`, `results: {node_id: {status, output}}` (per-node; the top-level `output` is the terminal node's output or a concatenation)
- `use_agent`: no additional fields required (single child)
- `a2a_client`: `remote_card: {name, description, ...}` (subset of the resolved agent card)

The tools may return richer shapes but must keep `status`, `output`, `execution_time_ms` at the top level so downstream models get a consistent contract regardless of which pattern they invoke.

## Cancellation

Parent cancellation must propagate to the child(ren) within 100 ms:

- Python: poll `parent_agent._cancel_signal.is_set()` at the tool's supervision level (a 50 ms loop is fine) and call `child.cancel()` when set. For multi-child tools (swarm/graph), the SDK's `BeforeNodeCallEvent` hook is the correct extension point — set `event.cancel_node` when the parent is cancelled.
- TypeScript: forward `agent.cancelSignal` into `Agent.invoke(..., { cancelSignal })` and into any transport-level `AbortSignal` the child holds. `AbortSignal.any([parent, timeout])` is the standard composition.

On cancellation, the tool returns `{status: "cancelled", output: "<partial or empty>", execution_time_ms}` — it does not raise past the tool boundary. A cancelled tool result reaching the loop is a signal to the parent to stop, not an exception to propagate.

## Size caps

At the tool boundary, before constructing anything:

- `system_prompt`: 8 KiB per spec
- `task` / `initial_input`: 32 KiB
- `name`: 64 chars (regex above)
- `tools` list: 64 entries max
- Number of agents (swarm, graph nodes): 20 max
- Number of edges (graph): 40 max
- Total execution time: 300 s wall-clock

These are hard caps, factory-configurable but not model-controllable.

## What each PR needs to change to conform

- **#65 graph:** add `multiagent_depth` participation (BLOCK). Replace `finally`-block cancellation with a `BeforeNodeCallEvent` hook (BLOCK — current code is a no-op in py). Rename `execution_order` → `execution_order` (keep) but add top-level `output` and `execution_time_ms` for shape consistency. Ship a copy of this file at `strands-py/src/strands/vended_tools/_multiagent_conventions.md` / `strands-ts/src/vended-tools/_multiagent-conventions.md`.
- **#66 swarm:** add `multiagent_depth` participation. Add `.strict()` to the top-level Zod schema (contradicts PR body's own claim). Cap `initial_input` and `system_prompt`. Ship the shared conventions doc.
- **#69 a2a_client:** the tool doesn't spawn child agents so most conventions don't apply, but its addendum currently claims to increment `multiagent_depth` when it doesn't (BLOCK). Either implement or delete the claim. Add BLOCK-level fixes from SSRF spec.
- **#73 use_agent:** **remove `model_settings` field entirely** (credential injection MAJOR). Align provider list across py/ts, or drop the provider-name field along with `model_settings` and just inherit the parent's model instance (recommended — matches this spec).

The `_multiagent_conventions.md` files across all four PRs will conflict on merge. That's expected. Human resolves to the version in this spec.
