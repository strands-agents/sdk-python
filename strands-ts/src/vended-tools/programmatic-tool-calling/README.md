# programmatic-tool-calling: recommendation to not vend

This directory intentionally contains no implementation. The recommendation
that programmatic-tool-calling should not be vended in its currently spec'd
form is recorded in
[`team/DECISIONS.md`](../../../../team/DECISIONS.md) under
"Do Not Vend programmatic_tool_calling Yet", which is the single source of
truth for the rationale and the preconditions for revisiting the decision.

Sub-issue: https://github.com/strands-agents/harness-sdk/issues/3251

Developers who want this pattern today can register their own tool that
closes over a specific `Agent` instance and calls its tools directly through
the tool-caller accessor exposed on `agent.tool` (see
`strands-ts/src/agent/tool-caller.ts` for the accessor and
`strands-ts/src/agent/agent.ts` for its wiring on `Agent`). That is a
per-application escape hatch and does not require the SDK to make
sandbox-level security guarantees it is not yet ready to make.

When the code-execution dependency lands and a design one-pager has been
reviewed, this directory should be replaced with the actual implementation.
Reopen [#3251](https://github.com/strands-agents/harness-sdk/issues/3251)
at that point.
