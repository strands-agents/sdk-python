# Agent Development Guide - Strands Agents Monorepo

This document provides guidance for AI agents working in the Strands Agents monorepo. For human contributor guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

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
├── .agents/            # Agent skills and references
├── package.json        # npm workspace root
└── .github/workflows/  # CI (ci.yml is the merge gate)
```

When working on code, determine which sub-project you're in and follow its conventions:
- **Python SDK**: See `strands-py/AGENTS.md`
- **TypeScript SDK**: See `strands-ts/AGENTS.md`
- **Documentation site**: See `site/AGENTS.md`

## Shared Conventions

- **Branching**: `git checkout -b agent-tasks/{ISSUE_NUMBER}`
- **Commits**: Use [conventional commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, etc.
- **Pull requests**: See PR guidelines ([Python](./strands-py/docs/PR.md), [TypeScript](./dev-docs/PR.md))
- **CI**: The `ci.yml` merge gate detects which paths changed and runs only relevant checks
