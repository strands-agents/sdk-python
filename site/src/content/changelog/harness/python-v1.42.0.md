---
sdk: harness
language: python
version: "1.42.0"
tag: python/v1.42.0
date: 2026-06-01
releaseUrl: https://github.com/strands-agents/harness-sdk/releases/tag/python%2Fv1.42.0
packageUrl: https://pypi.org/project/strands-agents/1.42.0/
highlights: |
  Adds request `Limits` to cap tokens/turns during invoke & stream, plus S3 session endpoint overrides for self-hosted deployments.
entries:
  - { type: feat, scope: null, areas: [], title: "add Limits and support during invoke/stream", pr: 2362, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2362", commit: a1b2c3d, commitUrl: "https://github.com/strands-agents/harness-sdk/commit/a1b2c3d", author: notowen333 }
  - { type: feat, scope: model, areas: [model], title: "plumb Gemini cache tokens in metadata events", pr: 2287, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2287", commit: f4e5d6c, commitUrl: "https://github.com/strands-agents/harness-sdk/commit/f4e5d6c", author: yatszhash }
  - { type: feat, scope: a2a, areas: [a2a], title: "add agent_card_url to A2AServer", pr: 2003, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2003", commit: 3b4c5d6, commitUrl: "https://github.com/strands-agents/harness-sdk/commit/3b4c5d6", author: waitasecant }
  - { type: fix, scope: model, areas: [model], title: "read vllm reasoning deltas", pr: 2354, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2354", commit: 9a8b7c6, commitUrl: "https://github.com/strands-agents/harness-sdk/commit/9a8b7c6", author: he-yufeng }
  - { type: chore, scope: null, areas: [], title: "prepare directory layout for monorepo convergence", pr: 2317, prUrl: "https://github.com/strands-agents/harness-sdk/pull/2317", commit: "1122334", commitUrl: "https://github.com/strands-agents/harness-sdk/commit/1122334", author: zastrowm }
---
