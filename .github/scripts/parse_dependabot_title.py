"""Parse a dependabot PR title into structured metadata.

Runs in the dependabot-auto-merge workflow BEFORE the agent sees anything,
so the agent receives only validated structured fields, never the raw title.

Usage: python parse_dependabot_title.py "<title>"
Outputs JSON to stdout.
"""

import json
import re
import sys

# Maps dependabot commit prefixes to ecosystems.
PREFIX_TO_ECOSYSTEM = {
    "ci(python)": "python",
    "ci(typescript)": "typescript",
    "ci(docs)": "docs",
}

# Matches "bump <pkg> from <old> to <new>" (single-package updates).
SINGLE_RE = re.compile(
    r"bump\s+(?P<package>[\w@/\-\.]+)\s+from\s+(?P<old>[\w\.\-]+)\s+to\s+(?P<new>[\w\.\-]+)",
    re.IGNORECASE,
)

# Matches grouped updates: "bump the <group> group ...".
GROUPED_RE = re.compile(r"bump\s+the\s+.+\s+group", re.IGNORECASE)


def parse(title: str) -> dict:
    result = {
        "ecosystem": "unknown",
        "package": "",
        "old_version": "",
        "new_version": "",
        "grouped": False,
    }

    # Determine ecosystem from prefix.
    prefix_match = re.match(r"^(ci\([\w]+\)|ci)(?=:)", title)
    if prefix_match:
        prefix = prefix_match.group(1)
        if prefix == "ci":
            result["ecosystem"] = "actions"
        else:
            result["ecosystem"] = PREFIX_TO_ECOSYSTEM.get(prefix, "unknown")

    if GROUPED_RE.search(title):
        result["grouped"] = True
        return result

    m = SINGLE_RE.search(title)
    if m:
        result["package"] = m.group("package")
        result["old_version"] = m.group("old")
        result["new_version"] = m.group("new")

    return result


if __name__ == "__main__":
    # Prefer stdin (avoids any shell interpretation of attacker-influenceable
    # title text); fall back to argv for convenience/testing.
    if len(sys.argv) > 1:
        title = sys.argv[1]
    else:
        title = sys.stdin.read()
    print(json.dumps(parse(title.strip())))
