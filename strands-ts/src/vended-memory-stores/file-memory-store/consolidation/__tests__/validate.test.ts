import { describe, it, expect } from 'vitest'
import { validatePlan, writeTargetOf } from '../validate.js'
import { CONSOLIDATION_CHANGELOG } from '../execute.js'
import type { ConsolidationAction, ConsolidationPlan } from '../plan.js'
import type { ConsolidateOperation } from '../../types.js'

/** A well-formed file body: frontmatter with a quoted description, then a non-empty body. */
function fileBody(description: string, body: string): string {
  return `---\ndescription: "${description}"\n---\n\n${body}\n`
}

/** Wrap actions in a plan, defaulting the required summary. */
function planOf(actions: ConsolidationAction[]): ConsolidationPlan {
  return { actions, summary: 'test plan' }
}

/** Run validation with sensible defaults, overridable per test. */
function validate(
  plan: ConsolidationPlan,
  files: Map<string, string>,
  operations: ConsolidateOperation[] = [
    'deduplicate',
    'resolveContradictions',
    'deriveInsights',
    'prune',
    'reorganize',
  ],
  maxDirectories = 8
): string[] {
  return validatePlan(plan, files, operations, maxDirectories)
}

describe('validatePlan', () => {
  describe('operation allow-list', () => {
    it('accepts an action permitted by the requested operations', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
      const plan = planOf([{ action: 'delete', path: 'facts/a.md', reason: 'stale' }])

      expect(validate(plan, files, ['prune'])).toEqual([])
    })

    it('rejects an action not permitted by the requested operations', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
      const plan = planOf([{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' }])

      expect(validate(plan, files, ['deduplicate'])).toContain(
        "Action 'move' is not allowed for operations: deduplicate"
      )
    })

    it('lets deriveInsights emit an update as well as a merge', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
      const plan = planOf([{ action: 'update', path: 'facts/a.md', content: fileBody('A', 'Refined A'), reason: 'x' }])

      expect(validate(plan, files, ['deriveInsights'])).toEqual([])
    })
  })

  describe('merge source requirements', () => {
    it('accepts a merge with two distinct existing sources', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'Content A')],
        ['facts/b.md', fileBody('B', 'Content B')],
      ])
      const plan = planOf([
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/b.md'],
          target: 'facts/merged.md',
          content: fileBody('Merged', 'A and B'),
          reason: 'dedup',
        },
      ])

      expect(validate(plan, files, ['deduplicate'])).toEqual([])
    })

    it('rejects a merge naming fewer than two distinct sources', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
      const plan = planOf([
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/a.md'],
          target: 'facts/a.md',
          content: fileBody('A', 'Rewritten'),
          reason: 'dedup',
        },
      ])

      expect(validate(plan, files, ['deduplicate'])).toContain('Merge action requires at least 2 distinct source paths')
    })

    it('rejects a merge whose source does not exist', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
      const plan = planOf([
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/ghost.md'],
          target: 'facts/merged.md',
          content: fileBody('Merged', 'body'),
          reason: 'dedup',
        },
      ])

      expect(validate(plan, files, ['deduplicate'])).toContain("Merge source 'facts/ghost.md' does not exist")
    })
  })

  describe('read-target existence', () => {
    it('rejects an update whose target does not exist', () => {
      const plan = planOf([{ action: 'update', path: 'facts/ghost.md', content: fileBody('G', 'b'), reason: 'x' }])

      expect(validate(plan, new Map(), ['resolveContradictions'])).toContain(
        "Update target 'facts/ghost.md' does not exist"
      )
    })

    it('rejects a delete whose target does not exist', () => {
      const plan = planOf([{ action: 'delete', path: 'facts/ghost.md', reason: 'x' }])

      expect(validate(plan, new Map(), ['prune'])).toContain("Delete target 'facts/ghost.md' does not exist")
    })

    it('rejects a move whose source does not exist', () => {
      const plan = planOf([{ action: 'move', from: 'facts/ghost.md', to: 'ops/ghost.md', reason: 'x' }])

      expect(validate(plan, new Map(), ['reorganize'])).toContain("Move source 'facts/ghost.md' does not exist")
    })
  })

  describe('write-target path rules', () => {
    const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
    const moveTo = (to: string): ConsolidationPlan => planOf([{ action: 'move', from: 'facts/a.md', to, reason: 'x' }])

    it('rejects a path containing a backslash', () => {
      expect(validate(moveTo('..\\..\\escaped.md'), files, ['reorganize'])).toContain(
        'Path must not contain backslashes: ..\\..\\escaped.md'
      )
    })

    it.each([
      ['..', '../escape.md'],
      ['.', './facts/a.md'],
    ])('rejects a path containing a %s segment', (_label, path) => {
      expect(validate(moveTo(path), files, ['reorganize'])).toContain(
        `Path must not contain dot segments ('.' or '..'): ${path}`
      )
    })

    it('rejects the reserved changelog path', () => {
      expect(validate(moveTo(CONSOLIDATION_CHANGELOG), files, ['reorganize'])).toContain(
        `Path must not be the reserved '${CONSOLIDATION_CHANGELOG}' file: ${CONSOLIDATION_CHANGELOG}`
      )
    })

    it('rejects a path not ending in .md', () => {
      expect(validate(moveTo('facts/a.txt'), files, ['reorganize'])).toContain('Path must end with .md: facts/a.txt')
    })

    it('rejects a bare .md filename with an empty stem', () => {
      expect(validate(moveTo('facts/.md'), files, ['reorganize'])).toContain(
        'Filename must have a non-empty stem before .md: facts/.md'
      )
    })

    it('rejects an over-long filename stem', () => {
      const path = `facts/${'a'.repeat(81)}.md`
      expect(validate(moveTo(path), files, ['reorganize'])).toContain(`Filename stem exceeds 80 characters: ${path}`)
    })

    it('rejects a stem with path-hostile characters', () => {
      const path = 'facts/a:b.md'
      expect(validate(moveTo(path), files, ['reorganize'])).toContain(
        `Filename stem contains path-hostile characters: ${path}`
      )
    })

    it('rejects a stem with leading or trailing whitespace', () => {
      const path = 'facts/ a .md'
      expect(validate(moveTo(path), files, ['reorganize'])).toContain(
        `Filename stem must not have leading or trailing whitespace: ${path}`
      )
    })

    it('rejects more than one level of nesting', () => {
      const path = 'l1/l2/deep.md'
      expect(validate(moveTo(path), files, ['reorganize'])).toContain(`Only one level of nesting allowed: ${path}`)
    })

    it('rejects an invalid directory name', () => {
      expect(validate(moveTo('BAD_DIR/a.md'), files, ['reorganize'])).toContain(
        "Directory name must be lowercase alphanumeric + hyphens, ≤30 chars: 'BAD_DIR'"
      )
    })

    it.each([
      ['ASCII with hyphens and digits', 'my-note-2024'],
      ['Japanese', '日本語'],
    ])('accepts a legitimate stem: %s', (_label, stem) => {
      expect(validate(moveTo(`facts/${stem}.md`), files, ['reorganize'])).toEqual([])
    })
  })

  describe('directory budget', () => {
    it('rejects creating a directory beyond maxDirectories', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'A')],
        ['ops/b.md', fileBody('B', 'B')],
      ])
      const plan = planOf([{ action: 'move', from: 'facts/a.md', to: 'newdir/a.md', reason: 'x' }])

      expect(validate(plan, files, ['reorganize'], 2)).toContain(
        "Cannot create directory 'newdir': maximum of 2 directories reached"
      )
    })

    it('allows a move into a directory that already exists', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'A')],
        ['ops/b.md', fileBody('B', 'B')],
      ])
      const plan = planOf([{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'x' }])

      expect(validate(plan, files, ['reorganize'], 2)).toEqual([])
    })

    it('counts the cumulative effect of a plan, not each action in isolation', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'A')],
        ['facts/b.md', fileBody('B', 'B')],
      ])
      // maxDirectories=2, existing dir: facts. Two moves into new dirs would total 3.
      const plan = planOf([
        { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'x' },
        { action: 'move', from: 'facts/b.md', to: 'team/b.md', reason: 'x' },
      ])

      expect(validate(plan, files, ['reorganize'], 2)).toContain(
        "Cannot create directory 'team': maximum of 2 directories reached"
      )
    })
  })

  describe('write-content shape', () => {
    const files = new Map([['facts/a.md', fileBody('A', 'Content A')]])
    const update = (content: string): ConsolidationPlan =>
      planOf([{ action: 'update', path: 'facts/a.md', content, reason: 'x' }])

    it('rejects empty content', () => {
      expect(validate(update('   '), files, ['resolveContradictions'])).toContain(
        "Update target 'facts/a.md' has empty content — a write must not blank out a file"
      )
    })

    it('rejects content that does not start with frontmatter', () => {
      expect(validate(update('no frontmatter here'), files, ['resolveContradictions'])).toContain(
        "Update target 'facts/a.md' must start with YAML frontmatter ('---' on the first line)"
      )
    })

    it('rejects content whose frontmatter is never closed', () => {
      expect(validate(update('---\ndescription: "x"\nunclosed'), files, ['resolveContradictions'])).toContain(
        "Update target 'facts/a.md' is missing the closing frontmatter delimiter ('---' on its own line)"
      )
    })

    it('rejects frontmatter without a quoted description', () => {
      expect(validate(update('---\ntitle: x\n---\n\nbody\n'), files, ['resolveContradictions'])).toContain(
        'Update target \'facts/a.md\' frontmatter needs a quoted description field (description: "a short summary")'
      )
    })

    it('rejects frontmatter-only content with no body', () => {
      expect(validate(update('---\ndescription: "x"\n---\n'), files, ['resolveContradictions'])).toContain(
        "Update target 'facts/a.md' has no body after its frontmatter"
      )
    })

    it('accepts well-formed content', () => {
      expect(validate(update(fileBody('A', 'Refined content')), files, ['resolveContradictions'])).toEqual([])
    })
  })

  describe('target collisions', () => {
    it('rejects two actions writing the same path', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'A')],
        ['facts/b.md', fileBody('B', 'B')],
        ['facts/c.md', fileBody('C', 'C')],
      ])
      const plan = planOf([
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/b.md'],
          target: 'facts/combined.md',
          content: fileBody('M', 'AB'),
          reason: 'x',
        },
        {
          action: 'merge',
          sources: ['facts/b.md', 'facts/c.md'],
          target: 'facts/combined.md',
          content: fileBody('M', 'BC'),
          reason: 'x',
        },
      ])

      expect(validate(plan, files, ['deduplicate'])).toContain(
        "Multiple actions write to the same path 'facts/combined.md'"
      )
    })

    it('rejects a path both written and vacated by the same plan', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'A')]])
      const plan = planOf([
        { action: 'update', path: 'facts/a.md', content: fileBody('A', 'new'), reason: 'x' },
        { action: 'delete', path: 'facts/a.md', reason: 'x' },
      ])

      expect(validate(plan, files, ['resolveContradictions', 'prune'])).toContain(
        "Path 'facts/a.md' is both written to and removed by the same plan"
      )
    })

    it('rejects a write onto an existing file no action vacates', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'A')],
        ['ops/a.md', fileBody('OpsA', 'existing ops')],
      ])
      const plan = planOf([{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'x' }])

      expect(validate(plan, files, ['reorganize'])).toContain(
        "Target path 'ops/a.md' already exists and is not vacated by another action in the plan"
      )
    })

    it('allows an update to overwrite its own file in place', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'A')]])
      const plan = planOf([{ action: 'update', path: 'facts/a.md', content: fileBody('A', 'new'), reason: 'x' }])

      expect(validate(plan, files, ['resolveContradictions'])).toEqual([])
    })

    it('allows a merge to overwrite one of its own sources', () => {
      const files = new Map([
        ['facts/a.md', fileBody('A', 'A')],
        ['facts/b.md', fileBody('B', 'B')],
      ])
      const plan = planOf([
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/b.md'],
          target: 'facts/a.md',
          content: fileBody('M', 'AB'),
          reason: 'x',
        },
      ])

      expect(validate(plan, files, ['deduplicate'])).toEqual([])
    })
  })

  describe('violation accumulation', () => {
    it('reports every violation at once rather than stopping at the first', () => {
      const files = new Map([['facts/a.md', fileBody('A', 'A')]])
      const plan = planOf([
        { action: 'delete', path: 'facts/ghost.md', reason: 'x' },
        { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'x' },
      ])

      const violations = validate(plan, files, ['prune'])

      expect(violations).toEqual(
        expect.arrayContaining([
          "Delete target 'facts/ghost.md' does not exist",
          "Action 'move' is not allowed for operations: prune",
        ])
      )
    })
  })

  it('never throws on a plan naming a huge sources array (appends, never spreads)', () => {
    const files = new Map([['facts/a.md', fileBody('A', 'A')]])
    const sources = Array.from({ length: 200_000 }, (_, index) => `facts/ghost-${index}.md`)
    const plan = planOf([
      { action: 'merge', sources, target: 'facts/merged.md', content: fileBody('M', 'body'), reason: 'x' },
    ])

    expect(() => validate(plan, files, ['deduplicate'])).not.toThrow()
  })
})

describe('writeTargetOf', () => {
  it('returns the target path an action writes to, or undefined for a delete', () => {
    const merge: ConsolidationAction = {
      action: 'merge',
      sources: ['facts/a.md', 'facts/b.md'],
      target: 'facts/m.md',
      content: 'c',
      reason: 'r',
    }
    const update: ConsolidationAction = { action: 'update', path: 'facts/a.md', content: 'c', reason: 'r' }
    const move: ConsolidationAction = { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'r' }
    const del: ConsolidationAction = { action: 'delete', path: 'facts/a.md', reason: 'r' }

    expect(writeTargetOf(merge)).toBe('facts/m.md')
    expect(writeTargetOf(update)).toBe('facts/a.md')
    expect(writeTargetOf(move)).toBe('ops/a.md')
    expect(writeTargetOf(del)).toBeUndefined()
  })
})
