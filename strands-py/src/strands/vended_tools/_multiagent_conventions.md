# Multi-agent conventions for vended tools

This document defines the dialect shared by the multi-agent vended tools: `use_agent`, `swarm`, `graph`, and `a2a_client`. Every tool in scope must implement it. If a tool cannot, its addendum records the deviation and the rationale.

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

Tools that historically exposed model configuration on their agent-spec input must remove that surface before conforming.

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
    status: "success" | "error" | "cancelled",
    output: str,               # the primary textual answer
    execution_time_ms: int,    # total wall-clock
}
```

Extended fields (each tool may add its own — should be additive, not renaming):
- `swarm`: `node_history: list[str]`, `execution_count: int`, `usage: {input_tokens, output_tokens}`
- `graph`: `execution_order: list[str]`, `results: {node_id: {status, output}}` (per-node; the top-level `output` is the terminal node's output or a concatenation)
- `use_agent`: no additional fields required (single child)
- `a2a_client`: `remote_card: {name, description, ...}` (subset of the resolved agent card)

The tools may return richer shapes but must keep `status`, `output`, `execution_time_ms` at the top level so downstream models get a consistent contract regardless of which pattern they invoke.

Field names are written in Python `snake_case` in this document. Each SDK renders them in its own convention per the root `AGENTS.md` cross-SDK parity rule: Python keeps `snake_case` (`execution_time_ms`, `remote_card`) and TypeScript renders `camelCase` (`executionTimeMs`, `remoteCard`). The `status` and `output` string literals stay byte-identical across SDKs.

## Cancellation

Parent cancellation must propagate to the child(ren) within 100 ms:

- Python: poll `parent_agent._cancel_signal.is_set()` at the tool's supervision level (a 50 ms loop is fine) and call `child.cancel()` when set. For multi-child tools (swarm/graph), the SDK's `BeforeNodeCallEvent` hook is the correct extension point — set `event.cancel_node` when the parent is cancelled.
- TypeScript: forward `agent.cancelSignal` into `Agent.invoke(..., { cancelSignal })` and into any transport-level `AbortSignal` the child holds. `AbortSignal.any([parent, timeout])` is the standard composition.

On cancellation, the tool returns `{status: "cancelled", output: "<partial or empty>", execution_time_ms}` — it does not raise past the tool boundary. A cancelled tool result reaching the loop is a signal to the parent to stop, not an exception to propagate. (The `a2a_client` addendum below documents an explicit deviation from this rule; the other three tools follow it as written.)

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

## Conformance notes per tool

- `graph` and `swarm` participate in the shared `multiagent_depth` counter and enforce the size caps at their tool boundary. Cancellation flows through the SDK's before-node hook so a parent-cancelled run stops promptly at the next node boundary rather than raising.
- `use_agent` invokes a single child; the child's `invocation_state` is constructed with `multiagent_depth = depth + 1` so nested calls respect the cap. No model-controllable model-selection surface is exposed.
- `a2a_client` reads and enforces the depth cap at the tool boundary. It does not spawn a local child agent, so there is nothing to increment into: the counter guard is one-way for this tool, and the addendum below documents the wire-boundary limitation.

---

## `a2a_client` addendum

The `a2a_client` tool invokes a **remote** A2A agent instead of constructing a child locally. Most fields in the agent spec are inapplicable (the remote picks its own model, prompt, and tools at deploy time). The tool receives a `url` + `message` from the model rather than an agent spec.

- **Model dialect** (`model_provider`, `model_settings`, `system_prompt`): not applicable — the remote agent chose these at deploy time. The model has no say.
- **Task input**: passed as `message` (not `task`) to match A2A protocol terminology. Capped at 64 KiB.
- **Tool allow-list**: not applicable — the remote agent brings its own tools; the parent's registry is invisible to it.
- **URL allow-list**: `allowed_url_prefixes` / `allowedUrlPrefixes` set at construction time (developer, never the model). If unset, only SSRF checks apply. The allowlist is re-applied to the card-advertised URL, so a remote that resolves to a public host outside the allowlist is rejected before the message is sent. On TypeScript the allowlist is also re-applied to every hop the guarded fetch walks, so a 3xx from an allowlisted origin cannot steer the client onto an off-list public host.
- **Redirect handling**:
  - TypeScript: the tool wraps the underlying A2A SDK's fetch (`makeGuardedFetch`) with `redirect: 'manual'`, re-runs the URL guard and allowlist on every hop, caps at five hops, and strips `Authorization` / `Cookie` / `Proxy-Authorization` on any cross-origin change (compared on full origin, not host).
  - Python: the underlying A2A SDK uses httpx, which defaults `follow_redirects=False`. Redirects are therefore not walked by default; a 3xx surfaces to the caller. Developers who supply their own `httpx_client` in `client_config` own the redirect discipline. There is no per-hop guard equivalent to the TypeScript wrapper on Python today; that parity is a follow-up if a use case emerges.
- **Recursion depth**:
  - The tool participates in the shared counter at the **local boundary**: a parent that calls `a2a_client` counts as depth+1, so a chain of `use_agent → a2a_client → ...` still respects the cap.
  - The counter is **not propagated across the wire**. The remote agent is opaque — the tool has no way to write into its `invocation_state`, and the remote may not even be a Strands agent. Depth resets from the remote's perspective; that's an accepted limitation of the network boundary.
- **Cancellation** (deviation from the shared spec): parent cancellation and total-timeout both **raise past the tool boundary** — `asyncio.CancelledError` / `TimeoutError` on Python, `DOMException` with `name === 'AbortError'` on TypeScript — rather than returning `status: "cancelled"`. Rationale: `a2a_client` is a network tool, not a child-agent supervisor. Its sibling network tools (`http_request`, `web_fetch`) raise cancellation the same way, and callers already special-case `AbortError` to distinguish it from other failures. Result-shape cancellation would need a new `status: 'cancelled'` variant that no existing caller reads. This is an accepted local divergence; the other three multi-agent tools continue to return the shaped result.
- **Result shape**: `{"status": "success", "output": str, "execution_time_ms": int, "remote_card": {"name", "description", "url"}}`. The top-level triple matches the shared multi-agent contract; `remote_card` mirrors a subset of the resolved agent card so the parent can distinguish remote endpoints. `status` is `Literal["success"]` on the success path; cancellation and timeout raise rather than populate a `cancelled` variant (see above).
- **Size caps as memory guard, not DoS defense**: `max_card_bytes` and `max_response_bytes` are enforced *after* the underlying A2A SDK has fetched and materialized the body. They bound what the model sees, not what the process reads into memory. A malicious server can still stream a large body before the cap fires; fronting the tool with an egress-side response-size limit is the mitigation.
