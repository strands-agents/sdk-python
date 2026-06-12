---
sdk: harness
language: python
version: "1.43.0"
tag: python/v1.43.0
date: 2026-06-12
releaseUrl: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.43.0
packageUrl: https://pypi.org/project/strands-agents/1.43.0/
entries:
  - { type: feat, breaking: false, scope: strands-py, areas: [hooks], title: "add optional hook order", pr: 2559, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2559", commit: "6dd249d", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/6dd249d", author: lizradway }
  - { type: feat, breaking: false, scope: checkpoint, areas: [persistence], title: "wire checkpointing into agent event loop", pr: 2190, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2190", commit: "cbd4f03", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/cbd4f03", author: JackYPCOnline }
  - { type: chore, breaking: false, scope: null, areas: [], title: "update repository references to harness-sdk", pr: 2618, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2618", commit: "ff37eb0", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/ff37eb0", author: zastrowm }
  - { type: fix, breaking: false, scope: null, areas: [], title: "include json blocks in counting tokens", pr: 2639, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2639", commit: "8f4a8eb", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/8f4a8eb", author: opieter-aws }
  - { type: feat, breaking: false, scope: strands-py, areas: [], title: "add Sandbox core abstraction (TS→Python port, core only 1/N)", pr: 2665, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2665", commit: "fb4a2e4", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/fb4a2e4", author: agent-of-mkmeral }
  - { type: feat, breaking: false, scope: context, areas: [context], title: "add context_manager=\"auto\" facade on Agent", pr: 2643, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2643", commit: "4ba1f19", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/4ba1f19", author: lizradway }
  - { type: feat, breaking: false, scope: null, areas: [context], title: "add claude-opus-4-8 to context window limits", pr: 2676, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2676", commit: "dc10b50", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/dc10b50", author: lizradway }
  - { type: fix, breaking: false, scope: a2a, areas: [a2a], title: "introduce agent factory for isolating agent context from different callers", pr: 2628, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2628", commit: "398343f", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/398343f", author: JackYPCOnline }
  - { type: feat, breaking: false, scope: null, areas: [], title: "add model_state as a snapshot field", pr: 2680, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2680", commit: "9a6be27", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/9a6be27", author: zastrowm }
  - { type: refactor, breaking: false, scope: a2a, areas: [a2a], title: "use native model_state snapshot field", pr: 2694, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2694", commit: "f133bbf", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/f133bbf", author: JackYPCOnline }
  - { type: feat, breaking: false, scope: context, areas: [context], title: "add message pinning to conversation managers", pr: 2644, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2644", commit: "43cc86e", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/43cc86e", author: lizradway }
  - { type: feat, breaking: false, scope: python, areas: [server], title: "add Docker/SSH Sandbox implementations", pr: 2691, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2691", commit: "5f4257f", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/5f4257f", author: gautamsirdeshmukh }
  - { type: fix, breaking: false, scope: a2a, areas: [a2a, sessions], title: "isolate conversation state per context in TypeScript", pr: 2696, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2696", commit: "403d878", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/403d878", author: JackYPCOnline }
  - { type: fix, breaking: false, scope: strands-py, areas: [community], title: "default to us-east-1 for integration tests", pr: 2720, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2720", commit: "61c1695", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/61c1695", author: zastrowm }
  - { type: feat, breaking: false, scope: null, areas: [interventions], title: "implement intervention primitive in python with cancellation support", pr: 2693, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2693", commit: "7e4f5cb", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/7e4f5cb", author: mehtarac }
  - { type: fix, breaking: false, scope: strands-py, areas: [multiagent, community], title: "remove flaky test_graph_parallel_execution test", pr: 2743, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2743", commit: "d0c4503", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/d0c4503", author: lizradway }
newContributors:
  - { login: senthilkumarmohan, pr: 2623 }
  - { login: ianholtz, pr: 2651 }
---
