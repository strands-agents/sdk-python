# Divergences

Two lists.

- **Proposed TS SDK changes** — places where the TS surface should move toward
  what the WIT contract already says. Open proposals, not landed changes.
- **`strands-py-wasm` vs upstream `sdk-python`** — deliberate differences in our
  Python package, given the agent loop runs in WASM here.

---

## Proposed TS SDK changes

- `StopData` → `StopEvent`. Every other terminal arm in the stream is `*Event` (`MetadataEvent`, `LifecycleEvent`). WIT is already `stop-event`.
- Merge `StreamEvent` + `AgentStreamEvent`. The abstract `StreamEvent` base class has no runtime behavior worth keeping; make `StreamEvent` the union directly.
- `InterruptResponseContent` → `InterruptResponseBlock`. Every other content block class ends in `Block` (`TextBlock`, `ImageBlock`, `ReasoningBlock`). WIT is `interrupt-response-block`.
- Split multi-agent `Status` into `OrchestrationStatus` (in-flight arms) and `TerminalStatus` (final arms). WIT has the split; TS conflated them.
- Consider dropping `*Event` suffix on hook event classes where it reads as "event-event" (`ModelStreamUpdateEvent`, `ContentBlockEvent`). Low priority.
- `SaveLatestStrategy` should become a variant, not a string union. Currently `'message' | 'invocation' | 'trigger'` with the trigger callback registered separately; WIT carries the handler id inline on `trigger(string)`. TS should follow: `{ tag: 'trigger'; handlerId: string } | 'message' | 'invocation'`.
- `Usage.totalTokens` is redundant with `inputTokens + outputTokens`. Either drop, or doc it as the canonical provider-reported value that may diverge. Same issue exists in WIT; decide once and apply both places.
- `Message.metadata.custom` is `Record<string, JSONValue>` in TS, opaque JSON string in WIT. Type WIT (tracks with the typed-snapshot work) or drop the TS structure. Don't leave them mismatched.
- Group `name`, `id`, `description` into an `agentIdentity` sub-record on `AgentConfig`. WIT already nests them; TS has three top-level fields.
- `ToolResultBlock.status` should be an enum, not `'success' | 'error'` string union. Matches WIT `tool-result-status`.
- `ToolResultBlock.content` should be a typed discriminated union (matching WIT `tool-result-content`), not a raw union reduced at runtime.
- `CachePointBlock.type: 'default'` should be a `CacheKind` enum. Matches WIT `cache-kind`.
- `ConversationManagerConfig` should be a typed variant (`none` | `slidingWindow` | `summarizing`), not a flat record with a string `strategy` discriminator. The wasmtime-py limitation that motivated the flat shape has been lifted.
- `TraceContext` should be a typed record (`traceparent`, `tracestate?`), not JSON W3C headers.
- `CustomModelConfig.stateful` should be a static field on the config, not a per-invocation `Model.stateful` getter. Stateful providers are identified once at registration.
- MCP transport names: if TS still uses a generic `http` transport, rename to `streamableHttp` per the current MCP spec. WIT is `streamable-http`.
- Consider replacing millisecond `number` fields (retries, graph/swarm timeouts, MCP task polling) with a `Duration` type. WIT uses `wasi:clocks/monotonic-clock.duration`.
- Add a `StrandsError extends Error` base class and reparent every SDK-thrown error to it. Today `ModelError` extends `Error`; `JsonValidationError`, `ToolValidationError`, `StructuredOutputError`, `ConcurrentInvocationError`, `SessionError` extend `Error` directly with no shared root. Users can't `catch (e instanceof StrandsError)`.
- Export `SessionError` and `ProviderTokenCountError` from `src/index.ts`. Both exist in `src/errors.ts` but neither is in the package index or any subpath.
- Surface hook event errors as typed objects rather than strings. `AfterModelCallEvent.error` should carry a `ModelError` (or an `unknown` widened to the `ModelError` union); same for `AfterToolCallEvent.error` → `ToolError`. WIT already carries typed errors into the hook payloads.
- `Plugin` should carry a doc banner that it's TS-only. Host-side SDKs can't implement it without a `custom-plugin` WIT interface that doesn't exist yet; users reading the type today have no way to know.
- Adopt the jco-generated types verbatim for `Usage`, `Metrics`, and `StopReason`. Today the WASM bridge maps camelCase TS types to kebab-case WIT variants on every event; having the SDK's runtime types *be* the generated types deletes those translators.
- `Agent.stream()` should yield the WIT `stream-event` variant directly, not the current class hierarchy of `ModelStreamUpdateEvent` / `ContentBlockEvent` / `ToolResultEvent` / etc. The hook-registration API (`agent.addHook(BeforeModelCallEvent, cb)`) stays; only the stream payload changes shape. Collapses ~300 lines of guest-to-host translation.
- SDK constructors for `Agent`, `SessionManager`, `ConversationManager`, tool registry, and built-in model providers (Bedrock/Anthropic/OpenAI/Google) should accept the WIT-shaped config records (`agent-config`, `session-config`, `conversation-manager-config`, `tool-spec`, `model-config` arms) directly. Eliminates the `buildSystemPrompt` / `createSessionManager` / `createConversationManager` / `createToolChoiceProxy` / `createTools` / `createModel` translators in the WASM bridge.
- `Agent` should implement the WIT `api` interface directly (or componentize-js should generate the resource shims). Removes `AgentImpl` / `ResponseStreamImpl` from the bridge. Enables the final test: running componentize-js on the SDK with no `entry.ts` produces an equivalent `.wasm`.

---

## `strands-py-wasm` vs upstream `sdk-python`

- `types/content.py` + `types/tools.py` → all generated types live in `_generated.py` (one file, auto-written from the WIT bindings).
- `agent/conversation_manager/*.py` → `SlidingWindowConversationManager` / `SummarizingConversationManager` are config-only dataclasses; execution is in the WASM guest.
- `session/file_session_manager.py` + `s3_session_manager.py` → `FileStorage` / `S3Storage` are config-only passthroughs to the WIT `storage_config`.
- `telemetry/`, `plugins/`, `hooks/`, `handlers/`, `event_loop/` — agent loop runs in the guest; these modules are either removed or collapsed into the thin SDK surface.
- Users never see base64 — binary content arrives as `bytes`.
- Users never format or parse ISO-8601 — snapshot timestamps come through as `wasi:clocks/wall-clock.datetime` records.
- Two error surfaces: the WIT tagged-variant records (`StorageError`, `ModelError`, `ToolError`, etc.) for pattern matching, plus `Exception` classes (`StrandsError`, `ContextWindowOverflowError`, `ToolValidationError`, …) for raise/catch. No stringly-typed error payloads.
- No custom `FileSessionManager` / `S3SessionManager` classes. Users pass `FileStorage(base_dir=...)` or `S3Storage(bucket=...)`; the guest instantiates the backend.
- Custom storage: set `StorageConfig_Custom(backend_id=...)` and implement the `snapshot-storage` host interface. No extra config record needed.
- `save_latest_policy.trigger` holds the handler id inline. Upstream's optional trigger-callback-on-config field is gone.
- `Graph`, `Swarm`, and `McpClient` are config-builder subclasses of the generated WIT records, not host-side orchestration runtimes. Orchestration and MCP transport management run in the guest.
- Interrupts are stream events, not exceptions. Upstream raises `InterruptException` from hooks and aggregates them in the registry; strands-py-wasm emits `StreamEventInterrupt(value=Interrupt)` on the event stream and resumes via `agent.respond(interrupt_id, payload)`. The `HookRegistry` does not interpret or aggregate interrupts.
- `HookRegistry` has no `order=` / `HookOrder` knobs; LIFO dispatch for `After*` arms is inferred from the class name. Upstream has a `should_reverse_callbacks` property on each event; our inference replaces the hand-set property.
- No type-hint inference on `add_callback`. Users pass `event_type` explicitly. Upstream's `add_callback(None, fn)` auto-inference added a `_type_inference.py` module we consider more trouble than it's worth.
- No `BaseHookEvent.__setattr__` immutability gate. Our hook events come from the WIT generator as `@dataclass` records; if immutability becomes required we'll add `frozen=True` at the generator level for both wire and hook consumers.
- New Python-layer types that aren't in upstream: `PydanticTool` (analog to TS's `ZodTool`), `McpClient` + `StdioMcpTransport` / `StreamableHttpMcpTransport` / `SseMcpTransport` (subclass config builders over the generated WIT records), `AgentResult` (matches the TS SDK class for the terminal invocation value).
