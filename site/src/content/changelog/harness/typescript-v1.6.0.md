---
sdk: harness
language: typescript
version: "1.6.0"
tag: typescript/v1.6.0
date: 2026-06-16
releaseUrl: https://github.com/strands-agents/harness-sdk/releases/tag/typescript/v1.6.0
packageUrl: https://www.npmjs.com/package/@strands-agents/sdk/v/1.6.0
entries:
  - { type: feat, breaking: true, scope: null, areas: [hooks, agent], title: "copy middleware context inputs to prevent accidental mutation", pr: 2742, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2742", commit: "ee4e69e", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/ee4e69e", author: zastrowm }
  - { type: feat, breaking: false, scope: null, areas: [context, agent], title: "memory injection", pr: 2631, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2631", commit: "dacb784", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/dacb784", author: opieter-aws }
  - { type: feat, breaking: false, scope: memory, areas: [community], title: "add memory injection", pr: 2797, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2797", commit: "a111c5d", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/a111c5d", author: opieter-aws }
  - { type: feat, breaking: false, scope: null, areas: [context, agent], title: "add agentic context management with model-driven compression tools", pr: 2754, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2754", commit: "7ccf733", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/7ccf733", author: notowen333 }
  - { type: feat, breaking: false, scope: null, areas: [context], title: "port agentic context management", pr: 2808, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2808", commit: "fbfd898", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/fbfd898", author: notowen333 }
  - { type: feat, breaking: false, scope: interventions, areas: [interventions], title: "add Cedar vended intervention handler", pr: 2365, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2365", commit: "7564ccd", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/7564ccd", author: lizradway }
  - { type: feat, breaking: false, scope: null, areas: [hooks, model], title: "add internal middleware system for `InvokeModelStage`", pr: 2760, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2760", commit: "2eafc76", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/2eafc76", author: zastrowm }
  - { type: feat, breaking: false, scope: context-offloader, areas: [context], title: "add turn-based eviction to `InMemoryStorage`", pr: 2648, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2648", commit: "fd2daad", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/fd2daad", author: lizradway }
  - { type: refactor, breaking: false, scope: sandbox, areas: [server], title: "inline tool prefixing into `getTools` implementations", pr: 2806, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2806", commit: "49d797a", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/49d797a", author: gautamsirdeshmukh }
  - { type: fix, breaking: false, scope: cedar, areas: [community], title: "correct fixture path in integration test", pr: 2810, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2810", commit: "1719c63", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/1719c63", author: lizradway }
  - { type: fix, breaking: false, scope: null, areas: [context, devx], title: "remove pin messaging barrel export", pr: 2767, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2767", commit: "ef58825", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/ef58825", author: lizradway }
  - { type: test, breaking: false, scope: null, areas: [context], title: "add integration test for memory manager", pr: 2764, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2764", commit: "96fec40", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/96fec40", author: opieter-aws }
  - { type: test, breaking: false, scope: memory, areas: [tool], title: "isolate KB search/add tool tests from auto-injection", pr: 2825, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2825", commit: "21ed754", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/21ed754", author: pgrayy }
  - { type: test, breaking: false, scope: memory, areas: [persistence], title: "assert S3 sidecar metadata and scope round-trip", pr: 2840, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2840", commit: "7170a70", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/7170a70", author: pgrayy }
---
