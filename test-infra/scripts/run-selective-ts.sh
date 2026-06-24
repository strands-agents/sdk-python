#!/usr/bin/env bash
set -euo pipefail

# Selective TypeScript integration testing.
# Runs only the integ specs whose module graph depends on changed files.
# Falls back to the full suite on structural changes; skips when no TS source changed.
# Shared by local dev (npm run test:integ:selective) and CI.

# --- Resolve repo root so the script is callable from anywhere ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# --- Determine the base ref to diff against ---
# CI passes SELECTIVE_BASE_REF (the PR base SHA). Locally, discover the
# closest of {origin/main, main, master, */main} — mirrors get-diff.sh.
BASE="${SELECTIVE_BASE_REF:-}"
if [[ -z "$BASE" ]]; then
  candidates=()
  for ref in main master; do
    git rev-parse --verify "$ref" &>/dev/null && candidates+=("$ref")
    for remote_ref in $(git for-each-ref --format='%(refname:short)' "refs/remotes/*/$ref" 2>/dev/null); do
      candidates+=("$remote_ref")
    done
  done
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "WARNING: no base branch found; running full integration suite." >&2
    npm run test:integ:all
    exit $?
  fi
  BASE="${candidates[0]}"
  best=$(git rev-list --count "$BASE"..HEAD 2>/dev/null || echo 999999)
  for ref in "${candidates[@]:1}"; do
    d=$(git rev-list --count "$ref"..HEAD 2>/dev/null || echo 999999)
    if [[ "$d" -lt "$best" ]]; then BASE="$ref"; best="$d"; fi
  done
fi

# --- Compute changed files ---
# merge-base diff INCLUDING uncommitted working-tree changes, so the local
# inner loop tests what you just edited. (Two-arg form: base..working-tree.)
MERGE_BASE="$(git merge-base "$BASE" HEAD 2>/dev/null || echo "$BASE")"
CHANGED="$(git diff --name-only "$MERGE_BASE" 2>/dev/null)" || {
  echo "WARNING: cannot diff against $MERGE_BASE; running full integration suite." >&2
  npm run test:integ:all
  exit $?
}

if [[ -z "$CHANGED" ]]; then
  echo "No changes detected vs $BASE — skipping integration tests."
  exit 0
fi

# DRY_RUN prints the chosen branch and exits before invoking any test command.
# Used by verification scenarios so they never trigger live AWS integ runs.
DRY_RUN="${SELECTIVE_DRY_RUN:-}"

# --- Branch 1: structural fallback ---
STRUCTURAL='^package\.json$|^package-lock\.json$|^strands-ts/package\.json$|^strands-ts/tsconfig.*\.json$|^strands-ts/vitest\.config\.ts$|^strands-ts/test/integ/__fixtures__/|^\.github/workflows/typescript-'
if echo "$CHANGED" | grep -qE "$STRUCTURAL"; then
  echo "Structural change detected — running full integration suite."
  [[ -n "$DRY_RUN" ]] && exit 0
  npm run test:integ:all
  exit $?
fi

# --- Branch 2: no TS source changed ---
TS_SOURCE="$(echo "$CHANGED" | grep -E '^(strands-ts|strands-wasm|wit)/' || true)"
if [[ -z "$TS_SOURCE" ]]; then
  echo "No strands-ts source changes — skipping integration tests."
  exit 0
fi

# --- Branch 3: selective ---
# Pass changed source files to Vitest's module-graph tracer, scoped to both
# integ projects. vitest related exits 0 with "No test files found" when none
# depend on the changes — a valid skip.
echo "Selective run for changed files:"
echo "$TS_SOURCE" | sed 's/^/  /'
[[ -n "$DRY_RUN" ]] && exit 0
# shellcheck disable=SC2046  # intentional word-splitting of the file list
( cd strands-ts && npx vitest related $(echo "$TS_SOURCE" | sed -E 's#^(strands-ts|strands-wasm|wit)/##') \
    --project integ-node --project integ-browser --run )
