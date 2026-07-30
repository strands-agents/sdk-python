#!/usr/bin/env python3
"""Group release-note commits by conventional-commit type.

Reads `<short-hash><TAB><subject>` lines on stdin (one commit each, as
produced by `git log --pretty=format:'%h%x09%s'`) and writes Markdown
release notes to stdout, bucketed by conventional-commit type (`feat`,
`fix`, `docs`, ...) instead of by author.

The header is drawn from the `NEW_TAG` / `PREV_TAG` environment variables
so the output matches the previous shortlog-based notes.
"""

import os
import re
import sys

# Conventional-commit types in the order they should appear, mapped to
# their section headings. Types not listed here fall through to "Other".
SECTIONS = [
    ("feat", "🚀 Features"),
    ("fix", "🐛 Fixes"),
    ("perf", "⚡ Performance"),
    ("refactor", "♻️ Refactoring"),
    ("docs", "📚 Documentation"),
    ("test", "✅ Tests"),
    ("build", "📦 Build"),
    ("ci", "👷 CI"),
    ("chore", "🔧 Chores"),
    ("revert", "⏪ Reverts"),
]
SECTION_TITLES = dict(SECTIONS)
OTHER_KEY = "other"

# `type(optional scope)!: subject` — captures the type and the remaining
# subject. The optional `!` and scope are conventional-commit syntax.
COMMIT_RE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?!?:\s*(?P<subject>.*)$")


def classify(subject):
    """Return the `section_key` for a commit subject line.

    Only the type is parsed off, for bucketing; the subject itself is kept
    verbatim (prefix included) by the caller so the rendered notes still
    show the `feat:` / `fix:` convention.
    """
    match = COMMIT_RE.match(subject)
    if match and match.group("type") in SECTION_TITLES:
        return match.group("type")
    return OTHER_KEY


def main():
    buckets = {}
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        short_hash, _, subject = line.partition("\t")
        key = classify(subject)
        buckets.setdefault(key, []).append((short_hash, subject.strip()))

    new_tag = os.environ.get("NEW_TAG", "")
    prev_tag = os.environ.get("PREV_TAG", "")

    # An empty prev_tag means the initial release: there is no baseline to
    # diff against, so describe the range as "all commits up to" the new tag
    # instead of rendering a dangling `..{new_tag}` range.
    range_desc = f"`{prev_tag}..{new_tag}`" if prev_tag else f"all commits up to `{new_tag}`"

    out = []
    out.append(f"## {new_tag}")
    out.append("")
    out.append(
        f"_Auto-drafted from commits in {range_desc}, grouped by "
        "conventional-commit type. Edit on the release page after publish if "
        "you want a polished writeup; the canonical release notes live on the "
        "website._"
    )
    out.append("")

    # Known types first, in declared order; then the "Other" catch-all.
    for key, title in SECTIONS + [(OTHER_KEY, "🔖 Other")]:
        commits = buckets.get(key)
        if not commits:
            continue
        out.append(f"### {title}")
        out.append("")
        for short_hash, subject in commits:
            out.append(f"- {subject} ({short_hash})")
        out.append("")

    if not any(buckets.values()):
        out.append("_No commits in range._")
        out.append("")

    sys.stdout.write("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main()
