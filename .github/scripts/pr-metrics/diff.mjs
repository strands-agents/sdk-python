// Unified-diff parsing: which lines did this PR actually change?
//
// The complexity labeler scores only the functions a diff touches, so a file
// that already contains a very complex function does not tag every PR that
// edits an unrelated part of it.

/**
 * Parse a unified diff into added/changed line numbers per file, using
 * post-image (new file) numbering.
 *
 * Deleted lines are deliberately ignored: they no longer exist in the head
 * revision, so no function range can contain them.
 *
 * @param {string} diff - output of `git diff -U0`
 * @returns {Map<string, Set<number>>} new path -> changed line numbers
 */
export function parseChangedLines(diff) {
  const changed = new Map()
  let path = null
  let newLine = 0

  for (const line of diff.split('\n')) {
    // `diff --git a/x b/x` resets state; rename/mode-only entries never get a hunk.
    if (line.startsWith('diff --git ')) {
      path = null
      continue
    }
    if (line.startsWith('+++ ')) {
      const target = line.slice(4).trim()
      path = target === '/dev/null' ? null : stripPrefix(target)
      continue
    }
    if (line.startsWith('@@')) {
      const m = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))?/.exec(line)
      if (!m) continue
      newLine = Number(m[1])
      // A hunk with a zero-length post-image is a pure deletion.
      if (m[2] === '0') newLine = -1
      continue
    }
    if (path === null || newLine < 0) continue

    if (line.startsWith('+')) {
      if (!changed.has(path)) changed.set(path, new Set())
      changed.get(path).add(newLine)
      newLine += 1
    } else if (line.startsWith(' ')) {
      newLine += 1
    }
  }
  return changed
}

// git prefixes paths with a/ and b/ unless --no-prefix is used.
function stripPrefix(p) {
  return p.startsWith('b/') || p.startsWith('a/') ? p.slice(2) : p
}

/**
 * Does a function's line range overlap any changed line?
 * Both bounds are inclusive and 1-indexed.
 */
export function rangeTouched(changedLines, startLine, endLine) {
  if (!changedLines) return false
  for (let n = startLine; n <= endLine; n += 1) {
    if (changedLines.has(n)) return true
  }
  return false
}
