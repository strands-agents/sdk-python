# Agent Development Guide - Strands Agents Monorepo

This document provides guidance for AI agents working in the Strands Agents monorepo. For human contributor guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Working with the Community

When helping someone contribute, you are a guide — not a gatekeeper, not a substitute author. The contribution is theirs; help them make it good and learn along the way. The standard for what makes a good contribution lives in [CONTRIBUTING.md](CONTRIBUTING.md#using-ai-tools); this is about the people.

- **Point people to the community.** Real questions and design discussion belong with people — the [Discord](https://discord.gg/strands) and [GitHub Discussions](https://github.com/strands-agents/harness-sdk/discussions).
- **Assume good faith.** Most contributors are learning; meet them where they are. Good first issues are for bringing newcomers in, not just tickets to close.
- **Talk with contributors, not at them.** Warm, plain, concise. One question at a time, no walls of text, never patronizing. Explain the *why* so it teaches rather than dictates.

## Monorepo Layout

```
strands-agents/
├── strands-py/         # Python SDK (hatch) — see strands-py/AGENTS.md
├── strands-ts/         # TypeScript SDK (npm workspace) — see strands-ts/AGENTS.md
├── strands-wasm/       # WASM bindings
├── strands-py-wasm/    # Python ↔ WASM bridge
├── strandly/           # CLI tooling
├── site/               # Documentation site (Astro) — see site/AGENTS.md
├── designs/            # Design proposals
├── dev-docs/           # TypeScript development docs
├── team/               # Team governance (tenets, decisions, API bar-raising)
├── test-infra/         # CDK stack for integ tests that require provisioned AWS infra
├── .agents/            # Agent skills and references
├── package.json        # npm workspace root
└── .github/workflows/  # CI (ci.yml is the merge gate)
```

When working on code, determine which sub-project you're in and follow its conventions:
- **Python SDK**: See `strands-py/AGENTS.md`
- **TypeScript SDK**: See `strands-ts/AGENTS.md`
- **Documentation site**: See `site/AGENTS.md`
- **Test infrastructure**: See `test-infra/README.md`

### test-infra/ guardrails

The `test-infra/` CDK stack deploys real AWS resources (Bedrock KBs, EC2 instances) that a small subset of integration tests depend on. Most tests do not need it — they run without provisioned infrastructure.

- **Do not deploy this stack** unless you are explicitly working on the test infrastructure itself or iterating on tests that resolve SSM parameters from it.
- **Never set `STRANDS_TEST_INFRA_INTERNAL=true`** unless deploying to the Strands team's own test account. This attaches a broad internal policy and GitHub OIDC trust that is meaningless (and wasteful) outside the internal account.
- **To run infrastructure-dependent integ tests without deploying anything**, open a PR — CI runs them against pre-provisioned resources automatically.

## Shared Conventions

- **Branching**: `git checkout -b agent-tasks/{ISSUE_NUMBER}`
- **Commits**: Use [conventional commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, etc.
- **Pull requests**: See PR guidelines ([Python](./strands-py/docs/PR.md), [TypeScript](./dev-docs/PR.md))
- **CI**: The `ci.yml` merge gate detects which paths changed and runs only relevant checks
