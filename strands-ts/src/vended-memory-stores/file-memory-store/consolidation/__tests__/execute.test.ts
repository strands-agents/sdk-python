import { describe, it, expect, vi } from 'vitest'
import { CONSOLIDATION_CHANGELOG, executePlan, readAllFiles, recordChangelog } from '../execute.js'
import { InMemoryStorage } from '../../../../storage/in-memory-storage.js'
import type { ConsolidationPlan } from '../plan.js'
import type { Storage } from '../../../../storage/storage.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

function fileBody(description: string, body: string): string {
  return `---\ndescription: "${description}"\n---\n\n${body}\n`
}

async function seed(storage: Storage, path: string, body: string): Promise<void> {
  await storage.write(path, encoder.encode(body))
}

async function readText(storage: Storage, path: string): Promise<string | null> {
  const bytes = await storage.read(path)
  return bytes ? decoder.decode(bytes) : null
}

describe('readAllFiles', () => {
  it('reads every knowledge file into a path→content map', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A body')
    await seed(storage, 'ops/b.md', 'B body')

    const files = await readAllFiles(storage, 100)

    expect(files).toEqual(
      new Map([
        ['facts/a.md', 'A body'],
        ['ops/b.md', 'B body'],
      ])
    )
  })

  it('excludes the reserved changelog from the working set', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A body')
    await seed(storage, CONSOLIDATION_CHANGELOG, '# Changelog\n')

    const files = await readAllFiles(storage, 100)

    expect([...files.keys()]).toEqual(['facts/a.md'])
  })

  it('returns an empty map for an empty store', async () => {
    expect(await readAllFiles(new InMemoryStorage(), 100)).toEqual(new Map())
  })

  it('propagates a backend read error', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/good.md', 'good')
    await seed(storage, 'facts/bad.md', 'bad')
    vi.spyOn(storage, 'read').mockImplementation(async (key) => {
      if (key === 'facts/bad.md') throw new Error('backend failure')
      return encoder.encode('good')
    })

    await expect(readAllFiles(storage, 100)).rejects.toThrow('backend failure')
  })

  it('rejects an oversized store on the key count, before reading any content', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A body')
    await seed(storage, 'facts/b.md', 'B body')
    await seed(storage, 'facts/c.md', 'C body')
    const readSpy = vi.spyOn(storage, 'read')

    await expect(readAllFiles(storage, 2)).rejects.toThrow(
      'Knowledge store exceeds consolidation file limit: 3 files (maxFiles: 2)'
    )
    // The guard must fire before the content fan-out — reading the corpus just to reject it
    // defeats the purpose of the limit.
    expect(readSpy).not.toHaveBeenCalled()
  })

  it('excludes the changelog from the maxFiles count', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A body')
    await seed(storage, 'facts/b.md', 'B body')
    await seed(storage, CONSOLIDATION_CHANGELOG, '# Changelog\n')

    // Two knowledge files + the changelog: at maxFiles=2 the changelog must not tip it over.
    const files = await readAllFiles(storage, 2)

    expect([...files.keys()].sort()).toEqual(['facts/a.md', 'facts/b.md'])
  })
})

describe('executePlan', () => {
  it('writes a merge target before deleting its sources', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A')
    await seed(storage, 'facts/b.md', 'B')
    const files = new Map([
      ['facts/a.md', 'A'],
      ['facts/b.md', 'B'],
    ])
    const plan: ConsolidationPlan = {
      actions: [
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/b.md'],
          target: 'facts/combined.md',
          content: fileBody('M', 'AB'),
          reason: 'x',
        },
      ],
      summary: 's',
    }

    const failures = await executePlan(storage, plan, files)

    expect(failures).toEqual([])
    expect(await readText(storage, 'facts/combined.md')).toContain('AB')
    expect(await readText(storage, 'facts/a.md')).toBeNull()
    expect(await readText(storage, 'facts/b.md')).toBeNull()
  })

  it('applies an update in place without deleting the file', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'old')
    const files = new Map([['facts/a.md', 'old']])
    const plan: ConsolidationPlan = {
      actions: [{ action: 'update', path: 'facts/a.md', content: fileBody('A', 'new'), reason: 'x' }],
      summary: 's',
    }

    await executePlan(storage, plan, files)

    expect(await readText(storage, 'facts/a.md')).toContain('new')
  })

  it('moves a file using content from the snapshot, not from storage', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'live content')
    // The snapshot deliberately differs from what is on disk, proving the move reads the snapshot.
    const files = new Map([['facts/a.md', 'snapshot content']])
    const plan: ConsolidationPlan = {
      actions: [{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'x' }],
      summary: 's',
    }

    await executePlan(storage, plan, files)

    expect(await readText(storage, 'ops/a.md')).toBe('snapshot content')
    expect(await readText(storage, 'facts/a.md')).toBeNull()
  })

  it('throws when a move source is missing from the working set (unvalidated plan)', async () => {
    const storage = new InMemoryStorage()
    const plan: ConsolidationPlan = {
      actions: [{ action: 'move', from: 'facts/ghost.md', to: 'ops/ghost.md', reason: 'x' }],
      summary: 's',
    }

    await expect(executePlan(storage, plan, new Map())).rejects.toThrow(
      "Invariant violated: move source 'facts/ghost.md' missing from working set — plan not validated"
    )
  })

  it('does not delete a merge source that is also the target', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A')
    await seed(storage, 'facts/b.md', 'B')
    const files = new Map([
      ['facts/a.md', 'A'],
      ['facts/b.md', 'B'],
    ])
    const plan: ConsolidationPlan = {
      actions: [
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/b.md'],
          target: 'facts/a.md',
          content: fileBody('M', 'AB'),
          reason: 'x',
        },
      ],
      summary: 's',
    }

    await executePlan(storage, plan, files)

    expect(await readText(storage, 'facts/a.md')).toContain('AB')
    expect(await readText(storage, 'facts/b.md')).toBeNull()
  })

  it('attempts every delete and reports each failure instead of throwing', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, 'facts/a.md', 'A')
    await seed(storage, 'facts/b.md', 'B')
    await seed(storage, 'facts/c.md', 'C')
    const files = new Map([
      ['facts/a.md', 'A'],
      ['facts/b.md', 'B'],
      ['facts/c.md', 'C'],
    ])
    const attempted: string[] = []
    vi.spyOn(storage, 'delete').mockImplementation(async (key) => {
      attempted.push(key)
      if (key === 'facts/a.md' || key === 'facts/c.md') throw new Error(`denied: ${key}`)
    })
    const plan: ConsolidationPlan = {
      actions: [
        { action: 'delete', path: 'facts/a.md', reason: 'x' },
        { action: 'delete', path: 'facts/b.md', reason: 'x' },
        { action: 'delete', path: 'facts/c.md', reason: 'x' },
      ],
      summary: 's',
    }

    const failures = await executePlan(storage, plan, files)

    expect(attempted).toEqual(['facts/a.md', 'facts/b.md', 'facts/c.md'])
    expect(failures.map((failure) => failure.path)).toEqual(['facts/a.md', 'facts/c.md'])
    expect(failures[0]!.error).toBeInstanceOf(Error)
  })
})

describe('recordChangelog', () => {
  const plan: ConsolidationPlan = {
    actions: [
      {
        action: 'merge',
        sources: ['facts/a.md', 'facts/b.md'],
        target: 'facts/m.md',
        content: fileBody('M', 'AB'),
        reason: 'duplicate facts',
      },
      { action: 'delete', path: 'facts/c.md', reason: 'stale' },
    ],
    summary: 'tidied the store',
  }

  it('creates the changelog with a header, operations, actions, and summary', async () => {
    const storage = new InMemoryStorage()

    await recordChangelog(storage, ['deduplicate', 'prune'], plan, [])

    const text = (await readText(storage, CONSOLIDATION_CHANGELOG))!
    expect(text).toContain('# Consolidation Changelog')
    expect(text).toContain('Operations: deduplicate, prune')
    expect(text).toContain('Actions (2):')
    expect(text).toContain('merge: facts/a.md + facts/b.md → facts/m.md (duplicate facts)')
    expect(text).toContain('delete: facts/c.md (stale)')
    expect(text).toContain('Summary: tidied the store')
  })

  it('appends to an existing changelog rather than overwriting it', async () => {
    const storage = new InMemoryStorage()
    await seed(storage, CONSOLIDATION_CHANGELOG, '# Consolidation Changelog\n\n## Prior run\n')

    await recordChangelog(storage, ['prune'], plan, [])

    const text = (await readText(storage, CONSOLIDATION_CHANGELOG))!
    expect(text).toContain('## Prior run')
    expect(text).toContain('Operations: prune')
  })

  it('records failed deletes so a partial run is not reported as clean', async () => {
    const storage = new InMemoryStorage()

    await recordChangelog(storage, ['prune'], plan, [{ path: 'facts/c.md', error: new Error('denied') }])

    const text = (await readText(storage, CONSOLIDATION_CHANGELOG))!
    expect(text).toContain('Failed deletes (1)')
    expect(text).toContain('facts/c.md: Error: denied')
  })

  it('swallows a changelog write error so it never masks the run outcome', async () => {
    const storage = new InMemoryStorage()
    vi.spyOn(storage, 'write').mockRejectedValue(new Error('disk full'))

    await expect(recordChangelog(storage, ['prune'], plan, [])).resolves.toBeUndefined()
  })
})
