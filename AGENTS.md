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
- **Pull requests**: See PR guidelines ([Python](./strands-py/docs/PR.md), [TypeScript](./dev-docs/PR.md)). Use the `pr-create` and `pr-writer` skills under `.agents/skills/` to draft and open PRs.
- **CI**: The `ci.yml` merge gate detects which paths changed and runs only relevant checks

## Quality Bar for PRs

If you are opening a PR on behalf of a contributor, the human is the author and is accountable for everything you submit. A small, focused change that its author fully understands is the single biggest predictor of a fast review and an accepted PR. (See [CONTRIBUTING.md](./CONTRIBUTING.md#using-ai-tools) for the human-facing version.)

- **Understand before you submit.** The contributor must be able to explain why every line works and defend the design. If you produced code you cannot explain plainly, simplify or explain it before opening the PR.
- **Keep it small and focused.** One logical change per PR. A branch that touches several sub-projects (`strands-py/`, `strands-ts/`, `site/`) is almost always several PRs. Smaller PRs are easier to understand, guide, and merge.
- **Open an issue first for anything significant**, so maintainers can align on the approach before time is invested.
- **Don't pad the change.** No drive-by reformatting, unrelated refactors, or speculative abstractions — they make the diff hard to review and the change hard to trust.
- **Verify before opening.** Run the relevant sub-project's checks (see [Development Environment](./CONTRIBUTING.md#development-environment), or the sub-project's own `AGENTS.md`) and make sure the change passes the `ci.yml` merge gate locally. Don't open a PR with known lint, type, or test failures.
- **Actually exercise the change, don't just rely on the gate.** Automated checks confirm the code is *valid*, not that the feature *works*. Run the behavior end to end — a manual script, a REPL snippet, the CLI, or an example — and confirm it does what the PR claims, including edge cases. If you can't exercise it (e.g. requires provisioned infra), say so explicitly in the PR rather than implying it was tested. Where it helps a reviewer, include the script or commands you ran.
- **Self-review the diff** end to end as if you were the reviewer, and confirm you can truthfully check every box in the PR template — including the item attesting that you have reviewed and understand every line of code in the PR, including any generated by AI tools. Then use the `pr-writer` skill so the description explains the **why**.
