---
name: pr-create
description: Creates a GitHub pull request using the gh CLI. Use when the user asks to create, open, or submit a PR on GitHub.
---

# GitHub PR Create Skill

## Process

### 1. Get the PR Description

- Check for `.local/pr-body.md`. If it exists, re-read it fresh (the user may have edited it) and use its contents as the title and body.
- Else if the user already generated a PR description earlier in the conversation, use that.
- Otherwise, fall back to the `pr-writer` skill to generate the title and body.

### 2. Pre-flight Checks

Check for `CONTRIBUTING.md` (at the repo root or in `docs/`) and `.github/PULL_REQUEST_TEMPLATE.md`. Scan them for any recommended steps before opening a PR — common examples:

- Running a linter or formatter (e.g. `npm run lint`, `cargo fmt`)
- Running tests (e.g. `npm test`, `pytest`)
- Building the project
- Updating documentation or changelogs

If you find any such steps, **list them and ask the user which ones they'd like you to run**. Run whichever the user approves. If any step fails, show the output and ask how to proceed before continuing.

If neither file exists or they contain no actionable pre-PR steps, skip this and move on.

### 3. Push the Branch

```bash
git push -u origin HEAD
```

If the push fails, show the error and stop.

### 4. Create the PR

Write the PR body to `.local/pr-body.md` (create `.local/` if needed) and create the PR:

```bash
gh pr create \
  --title "<title>" \
  --body-file .local/pr-body.md \
  --web
```

- Do NOT pass `--repo` — let `gh` infer the upstream from the git remotes so the PR targets the correct upstream repository.
- Always pass `--web` to open the PR in the browser for final review before submission.

## Rules

- Never create the PR without `--web` — the user must confirm in the browser.
- If `gh` is not authenticated or the command fails, show the error and stop.
