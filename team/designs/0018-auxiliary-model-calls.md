# Instrumenting Auxiliary Model Calls

**Status**: Proposed

**Date**: 2026-08-27

**Issue**: [#3863](https://github.com/strands-agents/harness-sdk/issues/3863)

**Scope**: Python SDK (the same shape ports to the TypeScript SDK; TS already has the `model=` knob on its conversation manager).

## Problem

The SDK makes model calls on the user's behalf that are not part of the main event loop: it summarizes history to stay under the context window, classifies tool calls for human-in-the-loop review, classifies request complexity to route between models, and extracts memories from a conversation. These *auxiliary* calls cost real tokens, but they are invisible to the machinery that tracks the main loop — and each one is instrumented differently, because there is no shared path for "the SDK is calling a model that isn't the agent's turn."

Token usage is only systematically tracked for main-loop calls: `EventLoopMetrics.update_usage` is invoked solely from `event_loop.py`, which feeds `accumulated_usage`, per-invocation usage, and the OTel token histograms. Every auxiliary call sidesteps this in its own way. The result is that `result.metrics.accumulated_usage` does not mean "what this agent invocation spent" — it silently omits summarization and HITL classification today, and would omit routing classification once [#3846](https://github.com/strands-agents/harness-sdk/pull/3846) lands. Any cost feature built on that number ([#1216](https://github.com/strands-agents/harness-sdk/issues/1216), [#1428](https://github.com/strands-agents/harness-sdk/issues/1428), [PR #3212](https://github.com/strands-agents/harness-sdk/pull/3212)) inherits the blind spots. Anyone doing cost accounting or capacity planning sees their observed token counts diverge from their provider bill with no way to explain the gap short of provider-side logging.

### Current State — the gap table

Each of the four auxiliary call sites decided its own telemetry story, and they have already diverged four ways:

| Auxiliary call | Call site | `EventLoopMetrics` | OTel span | Hooks fire | Retry |
|---|---|---|---|---|---|
| **Summarization** (default) | `compression/context_compression.py:156` `generate_summary` → raw `model.stream()` | ✗ usage from the stop event is discarded (`_, result_message, _, _`) | ✗ none | ✗ none | ✗ |
| **Summarization** (user `summarization_agent`) | user-supplied `Agent` | ✗ lands on *that* agent's metrics, never merged into the parent | that agent's own span | that agent's own hooks | that agent's |
| **HITL LLM classifier** | `vended_interventions/hitl/classifier.py:117` — throwaway inner `Agent` per tool call | ✗ inner agent's metrics discarded → parent undercounts | ✓ reaches global histograms, but unattributed | inner agent's hooks | inner agent's |
| **Routing** `InputComplexityStrategy` ([#3846](https://github.com/strands-agents/harness-sdk/pull/3846)) | `_classify` → raw `structured_output()`; lazily creates a Haiku 4.5 model | ✗ none | ✗ none | ✗ none | ✗ |
| **Memory** `ModelExtractor` | `memory/extraction/model_extractor.py:87` — the closest to correct | ✗ none | ✓ span with usage attributes | ✗ none | ✗ |

Two things break as a result. First, cost/usage is wrong or orphaned: the default summarizer's tokens vanish entirely, the routing classifier's recurring Haiku spend is completely invisible, and the HITL inner agent's usage is stranded on a discarded metrics object. Second, customer hooks silently miss these calls. A cost-tracking or logging hook subscribed to `BeforeModelCallEvent` fires for the agent's turns but never for summarization — which is exactly the paper cut the issue reports.

## Goals

- One shared, instrumented entry point for SDK-internal model calls, so a new auxiliary feature does not re-decide its telemetry story.
- Auxiliary spend is attributable per source (`summarization`, `hitl_classifier`, `routing_classifier`, `memory_extraction`) in both `result.metrics` and OTel.
- Existing model-call hooks fire for auxiliary calls, so a hook that means "the agent is calling the model" stops missing a whole class of calls.
- Do this without minting a parallel telemetry stack: reuse the span types, histograms, hooks, and retry that already exist.
- Let each auxiliary feature choose its own (cheaper) model, and let the whole-agent features (HITL, goal judge) inherit the parent's interventions/limits/trace instead of spawning a detached `Agent`.

## Proposal

The proposal is **one internal *runner* for SDK-internal model calls, plus two follow-up knobs, shipped in phases.** The runner is ~90% of the value and lands first; the knobs are additive and can follow without further design review. There is no new mechanism here — the runner just gives the four call sites the same instrumentation the main event loop already has.

**The runner — the core fix, merged first.** One internal helper — call it `run_aux_model_call(...)` — that every site which today calls `model.stream()` raw (summarizer, compactor, routing classifier, memory extractor) calls instead. It does what the event loop already does around a model call: opens the model-invoke span, fires the model-call hooks, adds usage to the owning agent's metrics, and applies retry. It is the `stream_messages` + `start_model_invoke_span`/`end_model_invoke_span` pattern that `ModelExtractor` already hand-rolls, centralized so all four sites share it. The Hooks, Metrics, and OTel subsections below spell out what the runner does at each layer.

**Knob 1 (follow-up) — `model=` per auxiliary feature.** Each feature takes an optional `model=` (a cheap model for summaries; inherits the agent's model when omitted). The TS conversation manager already has this; Python does not.

**Knob 2 (follow-up) — worker inheritance.** The features that spawn a whole agent (HITL classifier, goal judge) get a fresh worker per call that inherits the parent's interventions, limits, and trace context, instead of today's detached `Agent(...)`.

The phasing is deliberate: the runner can merge and deliver the entire cost-visibility fix on its own, and the two knobs are independent enough to land later as separate PRs without revisiting this design.

### Hooks — reuse the existing events, add a `source` field

At the model-call level, **reuse `BeforeModelCallEvent` / `AfterModelCallEvent`; do not mint `BeforeAuxModelCallEvent`.** These events already mean "the agent is calling the model." An auxiliary call *not* firing them is a gap, not a new concept — a customer's cost-tracking or logging hook *wants* to fire for summarization. Minting a parallel event instead rebuilds today's gap as API: every existing hook keeps missing auxiliary calls until its author subscribes a second time. (This is ADK's exact disease — `before_model_callback` exists, and the summarizer bypasses it.)

The behavior change is explicit and worth calling out on review:

> **`Before/AfterModelCallEvent` now fire for summarization, routing classification, and memory extraction as well as the agent's own turns.** Existing subscribers will see new traffic.

To let loop-only subscribers opt out of the new traffic, the events gain a `source` field:

```python
@dataclass
class BeforeModelCallEvent(HookEvent):
    invocation_state: dict[str, Any] = field(default_factory=dict)
    projected_input_tokens: int | None = None
    cancel: bool | str = False
    source: ModelCallSource = "main"   # "main" | "summarization" | "routing_classifier" | "hitl_classifier" | "memory_extraction"
```

A subscriber that only wants main-loop calls filters with `if event.source != "main": return`. In-tree there is exactly one such subscriber — `ModelRouter` (`models/routing/router.py:210`, subscribed to `AfterModelCallEvent`) — and it is fixed in the same PR. As a bonus, `ModelRetryStrategy` already subscribes to `AfterModelCallEvent`, so **auxiliary calls get retry for free** the moment the events fire through the runner.

At the *operation* level the answer is the opposite — but it is a later follow-up. No existing event means "about to compact your history," so `Before/AfterSummarizationEvent` (like Claude Code's `PreCompact`/`PostCompact`) is genuinely new and correct there. That is not needed for the core fix.

### Metrics — roll auxiliary usage into the owning agent, tagged by source

Auxiliary usage rolls up into the owning agent's `EventLoopMetrics.accumulated_usage`, tagged by `source` so it stays attributable per feature. This makes `result.metrics.accumulated_usage` finally mean "what this invocation spent."

This does change observed numbers for existing users — a summarizing agent's `accumulated_usage` goes up because it now includes the summarization tokens it was always spending. That is the point (the number was wrong before), but it is the one behavior change a reviewer should sign off on deliberately. The per-`source` breakdown means anyone who wants the old "main-loop only" number can still recover it.

### OTel / Tracing — zero new telemetry machinery

The tracer API already exists and is already used exactly this way (hand-copied today in `memory/extraction/model_extractor.py:87-111`); the runner just centralizes it:

```python
tracer = get_tracer()
span = tracer.start_model_invoke_span(messages=..., model_id=...)   # existing method
with trace_api.use_span(span, end_on_exit=False):                   # OTel context = auto-parenting
    ... stream the model ...
tracer.end_model_invoke_span(span, message, usage, metrics, stop_reason)  # existing method
```

- **Parenting is automatic.** OTel parents the span under whatever span is active in context. Summarization runs inside an agent invocation (`reduce_context` is called from `agent.py`), so the model span lands under the live `invoke_agent` span in the same trace — no span handles to thread around.
- **Same span type as main-loop calls.** Dashboards that sum `gen_ai.usage.*` over model-invoke spans become correct with zero changes. The only new thing is one attribute, `strands.source="summarization"`.
- **Optional wrapper span.** A `summarize_conversation` span (with before/after size attributes) can wrap the model span later; precedent is already in-tree with `start_memory_extract_span` (`memory/extraction/coordinator.py:214`).
- TS mirrors this via `Tracer.startModelInvokeSpan` (`telemetry/tracer.ts:358`).

**Total new observability surface:** zero new span types, one new span attribute, one new field on two existing hook events — plus, optionally and later, one wrapper-span method and two operation-level events.

## Developer Experience

**Cost tracking now sees every call, attributed by source.** A hook subscribed to `AfterModelCallEvent` fires for summarization and routing, and can bucket spend:

```python
class CostTracker(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterModelCallEvent, self.on_model_call)

    def on_model_call(self, event: AfterModelCallEvent) -> None:
        usage = event.stop_response.usage
        self.by_source[event.source] += usage["totalTokens"]
        # {"main": 41_320, "summarization": 2_880, "routing_classifier": 190}
```

**A loop-only hook opts out with one line:**

```python
    def on_model_call(self, event: BeforeModelCallEvent) -> None:
        if event.source != "main":
            return   # only care about the agent's own turns
        ...
```

**`result.metrics` finally reconciles with the provider bill:**

```python
result = agent("...")
result.metrics.accumulated_usage            # includes summarization + routing spend
result.metrics.accumulated_usage_by_source  # {"main": ..., "summarization": ...}
```

**Knob 1 — a cheap model for summaries (follow-up):**

```python
agent = Agent(
    model=sonnet,
    conversation_manager=SummarizingConversationManager(model=haiku),  # inherits agent.model when omitted
)
```

## Consequences

- Auxiliary spend becomes attributable everywhere — `result.metrics`, OTel spans, and OTel histograms — with a per-source breakdown.
- New auxiliary features route through one helper and inherit correct telemetry, hooks, and retry by default, instead of re-deriving it.
- **Behavior changes** to flag on review: (1) existing `Before/AfterModelCallEvent` subscribers see new traffic (mitigated by `source`); (2) `accumulated_usage` rises for agents that summarize/route because it now counts spend it always incurred.

## Willingness to Implement

Yes.

---

<details>
<summary><strong>Appendix A — What other frameworks do</strong></summary>

- **Google ADK** exposes `before_model_callback` / `after_model_callback`, but its own summarizer bypasses them — so a user callback registered for model calls silently misses summarization. This is the failure mode a `BeforeAuxModelCallEvent` would recreate for us, and the reason we reuse the existing events with a `source` field instead.
- **Claude Code** ships distinct `PreCompact` / `PostCompact` hooks for the *operation* of compacting history, separate from model-call hooks. This is the precedent for the (later, follow-up) `Before/AfterSummarizationEvent` — an operation with no existing event is where a new event type is genuinely warranted.
- **LiteLLM / gateways** track spend at the proxy, below the SDK. That catches auxiliary tokens on the bill but cannot attribute them to a source or surface them in `result.metrics`, which is what SDK-side accounting needs.

</details>

<details>
<summary><strong>Appendix B — Alternatives considered</strong></summary>

- **Do nothing / document that auxiliary calls are out-of-band spend.** Cheap, but makes accurate SDK-side cost reporting permanently impossible and leaves the four call sites diverging further.
- **Per-site OTel spans only — don't touch metrics.** Extend the `ModelExtractor` span pattern to the other three sites but leave `EventLoopMetrics` alone. Non-breaking and attributable in tracing backends, but invisible to anyone who reads `result.metrics` — the most common cost surface.
- **The shared runner without the two knobs.** This is the core proposal minus the follow-ups; it fully fixes cost visibility on its own. The knobs (per-feature `model=`, worker inheritance) are additive quality-of-life improvements, which is exactly why they are sequenced as later work rather than dropped.
- **New `BeforeAuxModelCallEvent` hook events.** Keeps existing subscribers' traffic unchanged, but requires every cost/logging hook to subscribe twice and rebuilds today's gap as a permanent API seam. Rejected in favor of reuse + `source`.
- **Record auxiliary usage on a separate surface (not `accumulated_usage`).** Preserves today's numbers exactly, but needs a new public surface and still leaves the headline number misleading. The per-`source` breakdown gives the same recoverability without a second surface.

</details>

<details>
<summary><strong>Appendix C — Related work</strong></summary>

- Cost on `AgentResult` (main-loop only): [#1216](https://github.com/strands-agents/harness-sdk/issues/1216), [#1428](https://github.com/strands-agents/harness-sdk/issues/1428)
- LiteLLM cost tracking (open): [PR #3212](https://github.com/strands-agents/harness-sdk/pull/3212)
- Cost-aware routing strategy metadata: [#3860](https://github.com/strands-agents/harness-sdk/issues/3860)
- Per-tool token counts: [#1503](https://github.com/strands-agents/harness-sdk/issues/1503)
- Where `gen_ai.usage.*` attaches on spans: [#3602](https://github.com/strands-agents/harness-sdk/issues/3602)
- Model routing (in flight, source of call site #3): [PR #3846](https://github.com/strands-agents/harness-sdk/pull/3846)

</details>
