import { describe, it, expect, beforeEach, vi } from 'vitest'
import { FileMemoryStore } from '../file-memory-store.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { logger } from '../../../logging/logger.js'
import { NAMESPACED } from '../../../storage/storage.js'
import type { JSONValue } from '../../../types/json.js'
import type { Storage } from '../../../storage/storage.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

function writeFile(storage: Storage, path: string, description: string, body: string): Promise<void> {
  const content = `---\ndescription: "${description}"\n---\n\n${body}\n`
  return storage.write(path, encoder.encode(content))
}

function buildPlanTurn(plan: { actions: JSONValue[]; summary: string }): {
  type: 'toolUseBlock'
  name: string
  toolUseId: string
  input: JSONValue
} {
  return {
    type: 'toolUseBlock',
    name: 'strands_structured_output',
    toolUseId: 'plan-fixture-id',
    input: plan as JSONValue,
  }
}

describe('FileMemoryStore.consolidate', () => {
  // A view scoped to the same `memory/<name>/` namespace the store applies internally, so tests
  // address entries by their namespace-relative keys (e.g. `facts/...`) and directly assert on the
  // keys the store reads and writes.
  let storage: Storage
  let store: FileMemoryStore

  beforeEach(() => {
    storage = new InMemoryStorage().namespace('memory/test-store')
    store = new FileMemoryStore({ name: 'test-store', storage })
  })

  describe('basic behavior', () => {
    it('returns immediately when the store is empty', async () => {
      const model = new MockMessageModel()
      await store.consolidate({ model })
    })

    // Guards the MemoryStore contract: writable:false means searchable only, never written to.
    it('rejects consolidate() on a read-only store without reading files or invoking the model', async () => {
      const readOnlyStorage = new InMemoryStorage().namespace('memory/readonly-store')
      await writeFile(readOnlyStorage, 'facts/a.md', 'Fact A', 'Content A')

      const readOnlyStore = new FileMemoryStore({ name: 'readonly-store', storage: readOnlyStorage, writable: false })
      const model = new MockMessageModel()
      const listSpy = vi.spyOn(readOnlyStorage, 'list')
      const streamSpy = vi.spyOn(model, 'stream')

      await expect(readOnlyStore.consolidate({ model })).rejects.toThrow('consolidate requires a writable store')

      // The guard fires before any storage read or model invocation
      expect(listSpy).not.toHaveBeenCalled()
      expect(streamSpy).not.toHaveBeenCalled()
    })

    it('rejects a second consolidate() that overlaps a run in flight', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'OK.' }))

      // The first call runs synchronously up to its first await (setting the guard); the second,
      // started before the first resolves, observes the guard and rejects.
      const first = store.consolidate({ model, operations: ['deduplicate'] })
      const second = store.consolidate({ model, operations: ['deduplicate'] })

      await expect(second).rejects.toThrow('A consolidation is already running on this store instance')
      await expect(first).resolves.toBeUndefined()

      // The guard clears after the first run, so a later call succeeds
      await expect(store.consolidate({ model, operations: ['deduplicate'] })).resolves.toBeUndefined()
    })

    it('executes a plan with no actions', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({ actions: [], summary: 'All files are well-organized.' })
      )

      await store.consolidate({ model, operations: ['deduplicate'] })

      const file = await storage.read('facts/a.md')
      expect(file).not.toBeNull()
      const changelog = await storage.read('consolidation/changelog.md')
      expect(changelog).not.toBeNull()
      expect(decoder.decode(changelog!)).toContain('Actions (0)')
    })

    it('records operations and summary in changelog', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'Nothing to do.' }))

      await store.consolidate({ model, operations: ['deduplicate', 'prune'] })

      const changelog = decoder.decode((await storage.read('consolidation/changelog.md'))!)
      expect(changelog).toContain('deduplicate, prune')
      expect(changelog).toContain('Nothing to do.')
    })

    it('appends to existing changelog', async () => {
      await storage.write('consolidation/changelog.md', encoder.encode('# Consolidation Changelog\n\n## Prior\n'))
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'OK.' }))

      await store.consolidate({ model, operations: ['deduplicate'] })

      const changelog = decoder.decode((await storage.read('consolidation/changelog.md'))!)
      expect(changelog).toContain('## Prior')
      expect(changelog).toContain('deduplicate')
    })

    it('makes a single model call over all files (holistic plan)', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'ops/b.md', 'Op B', 'Content B')

      // A single plan turn handles files from multiple directories — no per-directory calls
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'delete', path: 'facts/a.md', reason: 'Cross-directory dedup with ops/b' }],
          summary: 'Single call saw both directories.',
        })
      )

      await store.consolidate({ model, operations: ['prune'] })

      // The plan executed successfully across directories in one call
      expect(await storage.read('facts/a.md')).toBeNull()
      expect(await storage.read('ops/b.md')).not.toBeNull()
    })

    it('warns but does not throw when the changelog write fails on an otherwise successful run', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'delete', path: 'facts/b.md', reason: 'prune' }],
          summary: 'test',
        })
      )

      const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      // Fail only the changelog write; the plan's own mutations still land
      const originalWrite = storage.write.bind(storage)
      vi.spyOn(storage, 'write').mockImplementation(async (key, data) => {
        if (key === 'consolidation/changelog.md') throw new Error('disk full')
        return originalWrite(key, data)
      })

      // The run succeeds — a lost audit log does not fail consolidation
      await expect(store.consolidate({ model, operations: ['prune'] })).resolves.toBeUndefined()

      // The delete still executed, and the failure was logged
      expect(await storage.read('facts/b.md')).toBeNull()
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('failed to record consolidation changelog'))

      warnSpy.mockRestore()
    })

    it('records the changelog and reports the failure when a delete fails', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'delete', path: 'facts/a.md', reason: 'prune' }],
          summary: 'test',
        })
      )

      // Make storage.delete throw to simulate a backend delete failure
      vi.spyOn(storage, 'delete').mockRejectedValueOnce(new Error('Storage write failed'))

      await expect(store.consolidate({ model, operations: ['prune'] })).rejects.toThrow(
        /1 delete\(s\) failed.*facts\/a\.md/
      )

      // The changelog records the partial run, including the failed delete
      const changelog = await storage.read('consolidation/changelog.md')
      expect(changelog).not.toBeNull()
      const text = decoder.decode(changelog!)
      expect(text).toContain('Failed deletes (1)')
      expect(text).toContain('facts/a.md')
    })
  })

  describe('plan execution', () => {
    it('executes merge actions (write target, delete sources)', async () => {
      await writeFile(storage, 'facts/a.md', 'Dark mode', 'User prefers dark mode')
      await writeFile(storage, 'facts/b.md', 'Theme dark', 'Theme preference: dark')

      const mergedContent = '---\ndescription: "Theme preference"\n---\n\nUser prefers dark mode in all editors\n'
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'merge',
              sources: ['facts/a.md', 'facts/b.md'],
              target: 'facts/a.md',
              content: mergedContent,
              reason: 'Both express the same dark mode preference',
            },
          ],
          summary: 'Merged duplicate dark mode files.',
        })
      )

      await store.consolidate({ model, operations: ['deduplicate'] })

      const merged = await storage.read('facts/a.md')
      expect(merged).not.toBeNull()
      expect(decoder.decode(merged!)).toContain('dark mode in all editors')

      const deleted = await storage.read('facts/b.md')
      expect(deleted).toBeNull()
    })

    it('executes delete actions', async () => {
      await writeFile(storage, 'facts/old.md', 'Old deploy', 'Use Jenkins')
      await writeFile(storage, 'facts/new.md', 'New deploy', 'Use GitHub Actions')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'delete',
              path: 'facts/old.md',
              reason: 'Superseded by new deploy process',
            },
          ],
          summary: 'Pruned old deploy process.',
        })
      )

      await store.consolidate({ model, operations: ['prune'] })

      expect(await storage.read('facts/old.md')).toBeNull()
      expect(await storage.read('facts/new.md')).not.toBeNull()
    })

    it('executes update actions', async () => {
      await writeFile(storage, 'facts/indent.md', 'Indent style', 'Uses 4 spaces (April)')

      const newContent = '---\ndescription: "Indent style"\n---\n\nUses 2 spaces (June)\n'
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'update',
              path: 'facts/indent.md',
              content: newContent,
              reason: 'Updated to reflect current 2-space preference',
            },
          ],
          summary: 'Resolved contradiction.',
        })
      )

      await store.consolidate({ model, operations: ['resolveContradictions'] })

      const updated = decoder.decode((await storage.read('facts/indent.md'))!)
      expect(updated).toContain('2 spaces')
      expect(updated).not.toContain('4 spaces')
    })

    it('executes move actions', async () => {
      await writeFile(storage, 'facts/runbook.md', 'Oncall runbook', 'Check dashboards first')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'move',
              from: 'facts/runbook.md',
              to: 'ops/runbook.md',
              reason: 'Runbook belongs in ops/',
            },
          ],
          summary: 'Reorganized misplaced file.',
        })
      )

      await store.consolidate({ model, operations: ['reorganize'] })

      expect(await storage.read('facts/runbook.md')).toBeNull()
      const moved = await storage.read('ops/runbook.md')
      expect(moved).not.toBeNull()
      expect(decoder.decode(moved!)).toContain('Check dashboards first')
    })

    it('executes writes before deletes (merge target lands before sources removed)', async () => {
      await writeFile(storage, 'facts/a.md', 'A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'B', 'Content B')

      const mergedContent = '---\ndescription: "Merged"\n---\n\nCombined A and B\n'
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'merge',
              sources: ['facts/a.md', 'facts/b.md'],
              target: 'facts/combined.md',
              content: mergedContent,
              reason: 'Merging related facts',
            },
          ],
          summary: 'Combined.',
        })
      )

      await store.consolidate({ model, operations: ['deduplicate'] })

      expect(await storage.read('facts/combined.md')).not.toBeNull()
      expect(await storage.read('facts/a.md')).toBeNull()
      expect(await storage.read('facts/b.md')).toBeNull()
    })

    it('move reads from snapshot, not from storage', async () => {
      const originalContent = '---\ndescription: "original"\n---\n\noriginal body'
      await storage.write('facts/source.md', encoder.encode(originalContent))

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'move',
              from: 'facts/source.md',
              to: 'ops/source.md',
              reason: 'reorganize',
            },
          ],
          summary: 'test',
        })
      )

      await store.consolidate({ model, operations: ['reorganize'] })

      const movedBytes = await storage.read('ops/source.md')
      expect(movedBytes).not.toBeNull()
      const movedContent = decoder.decode(movedBytes!)
      expect(movedContent).toBe(originalContent)
    })

    it('partial delete failure reports all failures', async () => {
      await writeFile(storage, 'facts/a.md', 'A', 'content a')
      await writeFile(storage, 'facts/b.md', 'B', 'content b')
      await writeFile(storage, 'facts/c.md', 'C', 'content c')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            { action: 'delete', path: 'facts/a.md', reason: 'prune' },
            { action: 'delete', path: 'facts/b.md', reason: 'prune' },
            { action: 'delete', path: 'facts/c.md', reason: 'prune' },
          ],
          summary: 'test',
        })
      )

      const deletedPaths: string[] = []
      vi.spyOn(storage, 'delete').mockImplementation(async (key) => {
        deletedPaths.push(key)
        if (key === 'facts/a.md' || key === 'facts/c.md') {
          throw new Error(`permission denied: ${key}`)
        }
      })

      await expect(store.consolidate({ model, operations: ['prune'] })).rejects.toThrow(
        /2 delete\(s\) failed.*facts\/a\.md.*facts\/c\.md/
      )

      // All three deletes were attempted
      expect(deletedPaths).toContain('facts/a.md')
      expect(deletedPaths).toContain('facts/b.md')
      expect(deletedPaths).toContain('facts/c.md')
    })
  })

  describe('plan validation', () => {
    it('throws after failed retry when plan is invalid both times', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      // Both attempts reference a non-existent file
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'delete', path: 'facts/nonexistent.md', reason: 'test' }],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'delete', path: 'facts/still-nonexistent.md', reason: 'test' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['prune'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      // File untouched — nothing executed
      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('move with undefined source throws', async () => {
      await writeFile(storage, 'facts/exists.md', 'Exists', 'exists')

      // Marked NAMESPACED so the store uses this view as-is (no extra memory/<name>/ wrapping),
      // keeping the phantom key namespace-relative like every other path in these tests.
      const phantomStorage: Storage & { [NAMESPACED]: true } = {
        async write(key: string, data: Uint8Array): Promise<void> {
          return storage.write(key, data)
        },
        async read(key: string): Promise<Uint8Array | null> {
          if (key === 'facts/phantom.md') return null
          return storage.read(key)
        },
        async delete(key: string): Promise<void> {
          return storage.delete(key)
        },
        async list(prefix: string): Promise<string[]> {
          const keys = await storage.list(prefix)
          return [...keys, 'facts/phantom.md']
        },
        [NAMESPACED]: true,
      }

      const phantomStore = new FileMemoryStore({ name: 'phantom', storage: phantomStorage })

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'move',
                from: 'facts/phantom.md',
                to: 'ops/phantom.md',
                reason: 'reorganize',
              },
            ],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'move',
                from: 'facts/phantom.md',
                to: 'ops/phantom.md',
                reason: 'reorganize',
              },
            ],
            summary: 'retry',
          })
        )

      await expect(phantomStore.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /validation failed after retry/
      )
    })

    it('rejects plan writing into the reserved consolidation/ directory', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'consolidation/place.md', reason: 'hack' }],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'consolidation/place.md', reason: 'hack' }],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('rejects plan with too-deep nesting', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'level1/level2/deep.md', reason: 'test' }],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'level1/level2/deep.md', reason: 'test' }],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('rejects plan with invalid directory name', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'BAD_DIR/a.md', reason: 'test' }],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'BAD_DIR/a.md', reason: 'test' }],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('rejects plan exceeding maxDirectories', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'A')
      await writeFile(storage, 'ops/b.md', 'Op B', 'B')
      await writeFile(storage, 'team/c.md', 'Team C', 'C')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'new-dir/a.md', reason: 'test' }],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'new-dir/a.md', reason: 'test' }],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'], maxDirectories: 3 })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('allows move to existing directory within limit', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'A')
      await writeFile(storage, 'ops/b.md', 'Op B', 'B')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'Better fit' }],
          summary: 'Moved to existing directory.',
        })
      )

      await store.consolidate({ model, operations: ['reorganize'], maxDirectories: 2 })

      expect(await storage.read('facts/a.md')).toBeNull()
      expect(await storage.read('ops/a.md')).not.toBeNull()
    })

    it('rejects plan with non-existent merge source', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/nonexistent.md'],
                target: 'facts/merged.md',
                content: 'merged',
                reason: 'test',
              },
            ],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/nonexistent.md'],
                target: 'facts/merged.md',
                content: 'merged',
                reason: 'test',
              },
            ],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )
    })

    it('collects multiple distinct violations into a single error', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      // Plan with two distinct violations: nonexistent delete target AND disallowed move action
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'delete', path: 'facts/nonexistent.md', reason: 'test' },
              { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'test' },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'delete', path: 'facts/nonexistent.md', reason: 'test' },
              { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'test' },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['prune'] })).rejects.toSatisfy((error: Error) => {
        const message = error.message
        // Both violations must appear in the error message
        return (
          message.includes("Delete target 'facts/nonexistent.md' does not exist") &&
          message.includes("Action 'move' is not allowed for operations: prune")
        )
      })

      // File untouched — nothing executed
      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('succeeds on retry when the revised plan is valid', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      // First plan is invalid; revision is valid
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'delete', path: 'facts/nonexistent.md', reason: 'test' }],
            summary: 'bad',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'delete', path: 'facts/b.md', reason: 'Pruning duplicate' }],
            summary: 'Fixed plan.',
          })
        )

      await store.consolidate({ model, operations: ['prune'] })

      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).toBeNull()
    })

    it('rejects plan where move source is also an update target', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'update', path: 'facts/a.md', content: 'Updated A', reason: 'fix' },
              { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'update', path: 'facts/a.md', content: 'Updated A', reason: 'fix' },
              { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['resolveContradictions', 'reorganize'] })).rejects.toThrow(
        /both written to and removed by the same plan/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('rejects plan where multiple actions together exceed maxDirectories', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'B')

      // maxDirectories=2, existing dirs: [facts]. Two moves to new dirs would create 3 total.
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' },
              { action: 'move', from: 'facts/b.md', to: 'team/b.md', reason: 'reorg' },
            ],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' },
              { action: 'move', from: 'facts/b.md', to: 'team/b.md', reason: 'reorg' },
            ],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'], maxDirectories: 2 })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
    })
  })

  describe('operation scoping', () => {
    it('rejects update action when only deduplicate is allowed', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'update', path: 'facts/a.md', content: 'new', reason: 'test' }],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'update', path: 'facts/a.md', content: 'new', reason: 'test' }],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      // Update was rejected — file unchanged
      const content = decoder.decode((await storage.read('facts/a.md'))!)
      expect(content).toContain('Content A')
    })

    it('rejects move action when only deduplicate is allowed', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'test' }],
            summary: 'test',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'test' }],
            summary: 'test',
          })
        )

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('allows update when resolveContradictions is active', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const newContent = '---\ndescription: "Updated"\n---\n\nNew content\n'
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'update', path: 'facts/a.md', content: newContent, reason: 'Resolved' }],
          summary: 'Fixed.',
        })
      )

      await store.consolidate({ model, operations: ['resolveContradictions'] })

      const content = decoder.decode((await storage.read('facts/a.md'))!)
      expect(content).toContain('New content')
    })

    it('allows update when deriveInsights is active', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const newContent = '---\ndescription: "Insight"\n---\n\nDerived insight\n'
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'update', path: 'facts/a.md', content: newContent, reason: 'Insight' }],
          summary: 'Derived.',
        })
      )

      await store.consolidate({ model, operations: ['deriveInsights'] })

      const content = decoder.decode((await storage.read('facts/a.md'))!)
      expect(content).toContain('Derived insight')
    })

    it('defaults to all operations when none specified', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'All good.' }))

      await store.consolidate({ model })

      const changelog = decoder.decode((await storage.read('consolidation/changelog.md'))!)
      expect(changelog).toContain('deduplicate')
      expect(changelog).toContain('resolveContradictions')
      expect(changelog).toContain('deriveInsights')
      expect(changelog).toContain('prune')
      expect(changelog).toContain('reorganize')
    })
  })

  describe('structured output guard', () => {
    it('throws when model returns no structured output', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      // Plain text response (no structured output)
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'I cannot produce a plan.' })

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        'The model failed to invoke the structured output tool'
      )

      // File untouched
      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('rejects merge with fewer than 2 sources via guardrail validation', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      // Both attempts produce a merge with only 1 source — guardrail rejects both
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md'],
                target: 'facts/a.md',
                content: 'rewritten',
                reason: 'test',
              },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md'],
                target: 'facts/a.md',
                content: 'rewritten again',
                reason: 'test',
              },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        'Consolidation plan validation failed after retry'
      )
    })
  })

  describe('target-collision guard', () => {
    it('rejects plan with two actions writing the same target path', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')
      await writeFile(storage, 'facts/c.md', 'Fact C', 'Content C')

      // Two merges both target the same path
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/b.md'],
                target: 'facts/combined.md',
                content: 'merged AB',
                reason: 'dedup',
              },
              {
                action: 'merge',
                sources: ['facts/c.md'],
                target: 'facts/combined.md',
                content: 'merged C',
                reason: 'dedup',
              },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/b.md'],
                target: 'facts/combined.md',
                content: 'merged AB',
                reason: 'dedup',
              },
              {
                action: 'merge',
                sources: ['facts/c.md'],
                target: 'facts/combined.md',
                content: 'merged C',
                reason: 'dedup',
              },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        /Multiple actions write to the same path 'facts\/combined\.md'/
      )

      // Files untouched
      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
    })

    it('rejects move onto an existing file not vacated by the plan', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'ops/a.md', 'Op A', 'Ops content')

      // Move onto ops/a.md which already exists and is not deleted/moved away
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' }],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'ops/a.md', reason: 'reorg' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /Target path 'ops\/a\.md' already exists and is not vacated/
      )

      // Both files untouched
      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('ops/a.md')).not.toBeNull()
    })

    it('rejects merge onto an existing file that is not one of its sources', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')
      await writeFile(storage, 'facts/c.md', 'Fact C', 'Important content C')

      // Merge [a, b] → c.md, where c.md exists but was not a source. The model never saw c.md's
      // content, so overwriting it would silently destroy it.
      const mergedContent = '---\ndescription: "Merged"\n---\n\nCombined A and B\n'
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/b.md'],
                target: 'facts/c.md',
                content: mergedContent,
                reason: 'dedup',
              },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/b.md'],
                target: 'facts/c.md',
                content: mergedContent,
                reason: 'dedup',
              },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(
        /Target path 'facts\/c\.md' already exists and is not vacated/
      )

      // c.md's content survived — nothing executed
      expect(decoder.decode((await storage.read('facts/c.md'))!)).toContain('Important content C')
    })

    it('rejects a move onto a path another action deletes as a merge source', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')
      await writeFile(storage, 'ops/target.md', 'Op target', 'Ops content')

      // The merge consumes ops/target.md as a source (deleting it), while the move writes ops/target.md.
      // Because deletes run after writes, the merge's source cleanup would clobber the moved content.
      const mergedContent = '---\ndescription: "Merged"\n---\n\nCombined\n'
      const badPlan = buildPlanTurn({
        actions: [
          {
            action: 'merge',
            sources: ['ops/target.md', 'facts/b.md'],
            target: 'facts/b.md',
            content: mergedContent,
            reason: 'dedup',
          },
          { action: 'move', from: 'facts/a.md', to: 'ops/target.md', reason: 'reorg' },
        ],
        summary: 'contradictory plan',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['deduplicate', 'reorganize'] })).rejects.toThrow(
        /both written to and removed by the same plan/
      )
    })

    it('rejects chained moves where a write target is another move source', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      // move A→B (writes to B), move B→C (reads from B) — chained move conflict
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'move', from: 'facts/a.md', to: 'facts/b.md', reason: 'reorg' },
              { action: 'move', from: 'facts/b.md', to: 'ops/b.md', reason: 'reorg' },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'move', from: 'facts/a.md', to: 'facts/b.md', reason: 'reorg' },
              { action: 'move', from: 'facts/b.md', to: 'ops/b.md', reason: 'reorg' },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /both written to and removed by the same plan/
      )

      // Both files untouched
      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
    })

    it('rejects a plan that both writes to and deletes the same path', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      // update writes a.md in the write pass; delete removes it in the delete pass — content lost
      const badPlan = buildPlanTurn({
        actions: [
          { action: 'update', path: 'facts/a.md', content: 'Updated A', reason: 'fix' },
          { action: 'delete', path: 'facts/a.md', reason: 'prune' },
        ],
        summary: 'contradictory',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['resolveContradictions', 'prune'] })).rejects.toThrow(
        /both written to and removed by the same plan/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('rejects an identity move where source and target are the same path', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      // move writes a.md (.to) then deletes a.md (.from) — the file is destroyed
      const badPlan = buildPlanTurn({
        actions: [{ action: 'move', from: 'facts/a.md', to: 'facts/a.md', reason: 'reorg' }],
        summary: 'no-op move',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /both written to and removed by the same plan/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    it('handles delete of a path that is also a merge source (redundant delete)', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const mergedContent = '---\ndescription: "Merged"\n---\n\nCombined\n'
      // merge [A, B] → new combined.md, then explicit delete A (redundant — merge already vacates A)
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'merge',
              sources: ['facts/a.md', 'facts/b.md'],
              target: 'facts/combined.md',
              content: mergedContent,
              reason: 'dedup',
            },
            { action: 'delete', path: 'facts/a.md', reason: 'redundant cleanup' },
          ],
          summary: 'test',
        })
      )

      // This should succeed — the delete is redundant but not harmful
      await store.consolidate({ model, operations: ['deduplicate', 'prune'] })

      expect(await storage.read('facts/a.md')).toBeNull()
      expect(await storage.read('facts/b.md')).toBeNull()
      expect(decoder.decode((await storage.read('facts/combined.md'))!)).toContain('Combined')
    })
  })

  describe('revision prompt context', () => {
    it('includes the original plan and targeted-repair instruction in the revision message', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const invalidPlan = {
        actions: [{ action: 'delete', path: 'facts/nonexistent.md', reason: 'test' }],
        summary: 'bad plan',
      }
      const validPlan = {
        actions: [{ action: 'delete', path: 'facts/b.md', reason: 'Pruned' }],
        summary: 'fixed',
      }

      const model = new MockMessageModel().addTurn(buildPlanTurn(invalidPlan)).addTurn(buildPlanTurn(validPlan))

      const streamSpy = vi.spyOn(model, 'stream')

      await store.consolidate({ model, operations: ['prune'] })

      // The second call (revision) should contain the original plan JSON
      expect(streamSpy).toHaveBeenCalledTimes(2)
      const revisionMessages = streamSpy.mock.calls[1]![0]
      const userMessages = revisionMessages.filter((message) => message.role === 'user')
      const lastUserMessage = userMessages[userMessages.length - 1]
      expect(lastUserMessage).toBeDefined()

      // Extract the text from the user message content (may be string or content blocks)
      const content = lastUserMessage!.content
      const messageText =
        typeof content === 'string'
          ? content
          : (content as Array<{ type: string; text?: string }>)
              .filter((block) => block.type === 'textBlock')
              .map((block) => block.text)
              .join('')

      expect(messageText).toContain(JSON.stringify(invalidPlan))
      expect(messageText).toContain('Your plan was rejected')
      expect(messageText).toContain('Modify ONLY the offending actions to fix the violations above')
      expect(messageText).toContain('Keep all other actions unchanged')
    })

    it('logs rejected plans with structured format', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})

      // Both attempts produce invalid plans
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'delete', path: 'facts/nonexistent.md', reason: 'test' }],
            summary: 'bad',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'delete', path: 'facts/still-nonexistent.md', reason: 'test' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['prune'] })).rejects.toThrow()

      // Both the initial rejection and the post-retry rejection are logged
      expect(warnSpy).toHaveBeenCalledTimes(2)
      expect(warnSpy.mock.calls[0]![0]).toContain('consolidation plan rejected on initial attempt')
      expect(warnSpy.mock.calls[0]![0]).toContain('validation_errors=<')
      expect(warnSpy.mock.calls[1]![0]).toContain('consolidation plan rejected after retry')
      expect(warnSpy.mock.calls[1]![0]).toContain('plan=<')

      warnSpy.mockRestore()
    })
  })

  describe('input-size guardrails', () => {
    it('throws when file count exceeds maxFiles', async () => {
      for (let index = 0; index < 4; index++) {
        await writeFile(storage, `facts/file-${index}.md`, `File ${index}`, `Content ${index}`)
      }

      const model = new MockMessageModel()

      await expect(store.consolidate({ model, maxFiles: 3 })).rejects.toThrow(
        'Knowledge store exceeds consolidation file limit: 4 files (maxFiles: 3)'
      )

      // No model calls made — guardrail fires before planner
      expect(model.callCount).toBe(0)
    })

    it('throws when total content bytes exceed maxInputBytes', async () => {
      // Each file has frontmatter + body; compute a limit smaller than the total
      await writeFile(storage, 'facts/a.md', 'Fact A', 'A'.repeat(100))
      await writeFile(storage, 'facts/b.md', 'Fact B', 'B'.repeat(100))

      const model = new MockMessageModel()

      // The total byte size of the two files (including frontmatter) exceeds 50 bytes
      await expect(store.consolidate({ model, maxInputBytes: 50 })).rejects.toThrow(
        /Knowledge store exceeds consolidation input size limit: \d+ bytes \(maxInputBytes: 50\)/
      )

      // No model calls made — guardrail fires before planner
      expect(model.callCount).toBe(0)
    })

    it('succeeds at exactly the file limit', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')
      await writeFile(storage, 'facts/c.md', 'Fact C', 'Content C')

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'All good.' }))

      // maxFiles === files.size → should NOT throw (> is the check, not >=)
      await expect(store.consolidate({ model, maxFiles: 3, operations: ['deduplicate'] })).resolves.not.toThrow()
    })

    it('succeeds at exactly the byte limit', async () => {
      const body = 'X'
      await writeFile(storage, 'facts/a.md', 'Fact A', body)

      // Compute exact byte size of the stored file to set maxInputBytes precisely
      const stored = await storage.read('facts/a.md')
      const exactBytes = encoder.encode(new TextDecoder().decode(stored!)).byteLength

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'All good.' }))

      await expect(
        store.consolidate({ model, maxInputBytes: exactBytes, operations: ['deduplicate'] })
      ).resolves.not.toThrow()
    })

    it('defaults allow a normal small store', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const model = new MockMessageModel().addTurn(buildPlanTurn({ actions: [], summary: 'All good.' }))

      // No maxFiles/maxInputBytes specified — defaults (100 files, 128 KiB) should pass
      await expect(store.consolidate({ model, operations: ['deduplicate'] })).resolves.not.toThrow()
    })
  })

  describe('output-size guardrail', () => {
    it('throws when the plan action count exceeds maxActionsPerPlan', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')
      await writeFile(storage, 'facts/c.md', 'Fact C', 'Content C')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            { action: 'delete', path: 'facts/a.md', reason: 'prune' },
            { action: 'delete', path: 'facts/b.md', reason: 'prune' },
            { action: 'delete', path: 'facts/c.md', reason: 'prune' },
          ],
          summary: 'oversized plan',
        })
      )

      await expect(store.consolidate({ model, operations: ['prune'], maxActionsPerPlan: 2 })).rejects.toThrow(
        'Consolidation plan exceeds action limit: 3 actions (maxActionsPerPlan: 2)'
      )

      // Files untouched — the guard fires before any execution
      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
      expect(await storage.read('facts/c.md')).not.toBeNull()
    })

    it('does not attempt a revision when the plan exceeds the action limit', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            { action: 'delete', path: 'facts/a.md', reason: 'prune' },
            { action: 'delete', path: 'facts/b.md', reason: 'prune' },
          ],
          summary: 'oversized plan',
        })
      )
      const streamSpy = vi.spyOn(model, 'stream')

      await expect(store.consolidate({ model, operations: ['prune'], maxActionsPerPlan: 1 })).rejects.toThrow(
        /exceeds action limit/
      )

      // Only the initial plan call was made — an oversized plan is rejected outright, never re-sent
      expect(streamSpy).toHaveBeenCalledTimes(1)
    })

    it('succeeds at exactly the action limit', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            { action: 'delete', path: 'facts/a.md', reason: 'prune' },
            { action: 'delete', path: 'facts/b.md', reason: 'prune' },
          ],
          summary: 'at limit',
        })
      )

      // maxActionsPerPlan === actions.length → should NOT throw (> is the check, not >=)
      await expect(store.consolidate({ model, operations: ['prune'], maxActionsPerPlan: 2 })).resolves.not.toThrow()

      expect(await storage.read('facts/a.md')).toBeNull()
      expect(await storage.read('facts/b.md')).toBeNull()
    })
  })

  describe('path-identity validation', () => {
    // Guards against directory traversal: backslashes bypass POSIX-only segment splitting,
    // allowing paths like '..\\..\\escaped.md' to resolve outside the store boundary.
    it('rejects a plan action whose path contains a backslash', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: '..\\..\\escaped.md', reason: 'hack' }],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: '..\\..\\escaped.md', reason: 'hack' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /must not contain backslashes/
      )

      // No mutation occurred
      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    // Guards against path traversal via dot segments that escape the store namespace.
    it('rejects a path with a ".." segment', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: '../escape.md', reason: 'hack' }],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: '../escape.md', reason: 'hack' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /must not contain dot segments/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    // Guards against current-directory dot segments which are ambiguous and non-canonical.
    it('rejects a path with a "." segment', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: './facts/a.md', reason: 'no-op' }],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: './facts/a.md', reason: 'no-op' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /must not contain dot segments/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
    })

    // Guards against data loss from case-only moves: on a case-insensitive filesystem,
    // 'topic.md' and 'Topic.md' resolve to the same file. Deleting the source after writing
    // the target would destroy the file's content.
    it('case-only move does not delete the source data', async () => {
      await writeFile(storage, 'facts/topic.md', 'Topic', 'Important content about topic')

      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [{ action: 'move', from: 'facts/topic.md', to: 'facts/Topic.md', reason: 'capitalize' }],
          summary: 'Rename to capitalize.',
        })
      )

      await store.consolidate({ model, operations: ['reorganize'] })

      // The content is still accessible — the source was not deleted because the paths
      // resolve to the same identity on a case-insensitive backend
      const content = await storage.read('facts/Topic.md')
      expect(content).not.toBeNull()
      expect(decoder.decode(content!)).toContain('Important content about topic')
    })

    // Guards against silent clobber: two actions writing case-variant paths would resolve to
    // the same file on case-insensitive backends, with the second silently overwriting the first.
    it('rejects two actions writing case-variant targets', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')
      await writeFile(storage, 'facts/c.md', 'Fact C', 'Content C')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'move', from: 'facts/a.md', to: 'facts/merged.md', reason: 'reorg' },
              { action: 'move', from: 'facts/b.md', to: 'facts/Merged.md', reason: 'reorg' },
            ],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              { action: 'move', from: 'facts/a.md', to: 'facts/merged.md', reason: 'reorg' },
              { action: 'move', from: 'facts/b.md', to: 'facts/Merged.md', reason: 'reorg' },
            ],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /Multiple actions write to the same path/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
    })

    // Guards against overwriting an existing file when paths differ only by case — on a
    // case-insensitive backend, writing to a case-variant of an existing file would clobber it.
    it('rejects a write colliding case-insensitively with an existing file', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/existing.md', 'Existing', 'Important existing content')

      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'facts/Existing.md', reason: 'reorg' }],
            summary: 'bad plan',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [{ action: 'move', from: 'facts/a.md', to: 'facts/Existing.md', reason: 'reorg' }],
            summary: 'still bad',
          })
        )

      await expect(store.consolidate({ model, operations: ['reorganize'] })).rejects.toThrow(
        /already exists and is not vacated/
      )

      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/existing.md')).not.toBeNull()
    })
  })

  describe('write-content validation', () => {
    // Guards against knowledge erasure: a schema-valid plan whose merge content is empty would
    // write a zero-byte target and then delete its populated sources.
    it('rejects merge with empty content, leaving sources intact', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const badPlan = buildPlanTurn({
        actions: [
          {
            action: 'merge',
            sources: ['facts/a.md', 'facts/b.md'],
            target: 'facts/combined.md',
            content: '',
            reason: 'dedup',
          },
        ],
        summary: 'empty merge',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['deduplicate'] })).rejects.toThrow(/has empty content/)

      // Sources untouched and no zero-byte target written — nothing mutated
      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
      expect(await storage.read('facts/combined.md')).toBeNull()
    })

    // Content without frontmatter is unparseable by parseFrontmatter, so the file's description
    // would be lost and its body would no longer be indexed the way search expects.
    it('rejects content missing frontmatter', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const badPlan = buildPlanTurn({
        actions: [
          { action: 'update', path: 'facts/a.md', content: 'Just plain text without frontmatter', reason: 'update' },
        ],
        summary: 'no frontmatter',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['resolveContradictions'] })).rejects.toThrow(
        /must start with YAML frontmatter/
      )

      expect(decoder.decode((await storage.read('facts/a.md'))!)).toContain('Content A')
    })

    it('rejects content whose frontmatter is never closed', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const badPlan = buildPlanTurn({
        actions: [
          {
            action: 'update',
            path: 'facts/a.md',
            content: '---\ndescription: "test"\nno closing delimiter here',
            reason: 'update',
          },
        ],
        summary: 'unclosed frontmatter',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['resolveContradictions'] })).rejects.toThrow(
        /missing the closing frontmatter delimiter/
      )

      expect(decoder.decode((await storage.read('facts/a.md'))!)).toContain('Content A')
    })

    // Frontmatter-only content is the subtler erasure case: structurally valid, but every fact
    // the file held is gone.
    it('rejects content with no body after its frontmatter', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')

      const badPlan = buildPlanTurn({
        actions: [
          { action: 'update', path: 'facts/a.md', content: '---\ndescription: "test"\n---\n', reason: 'update' },
        ],
        summary: 'no body',
      })
      const model = new MockMessageModel().addTurn(badPlan).addTurn(badPlan)

      await expect(store.consolidate({ model, operations: ['resolveContradictions'] })).rejects.toThrow(
        /has no body after its frontmatter/
      )

      expect(decoder.decode((await storage.read('facts/a.md'))!)).toContain('Content A')
    })

    // Malformed content is a formatting slip, so it routes through the same revise-retry as every
    // other guardrail rather than aborting the run.
    it('recovers when the model fixes malformed content on revision', async () => {
      await writeFile(storage, 'facts/a.md', 'Dark mode', 'User prefers dark mode')
      await writeFile(storage, 'facts/b.md', 'Theme dark', 'Theme preference: dark')

      const fixedContent = '---\ndescription: "Theme preference"\n---\n\nUser prefers dark mode everywhere\n'
      const model = new MockMessageModel()
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/b.md'],
                target: 'facts/combined.md',
                content: '',
                reason: 'dedup',
              },
            ],
            summary: 'empty merge',
          })
        )
        .addTurn(
          buildPlanTurn({
            actions: [
              {
                action: 'merge',
                sources: ['facts/a.md', 'facts/b.md'],
                target: 'facts/combined.md',
                content: fixedContent,
                reason: 'dedup',
              },
            ],
            summary: 'repaired merge',
          })
        )

      await store.consolidate({ model, operations: ['deduplicate'] })

      expect(decoder.decode((await storage.read('facts/combined.md'))!)).toContain('dark mode everywhere')
      expect(await storage.read('facts/a.md')).toBeNull()
      expect(await storage.read('facts/b.md')).toBeNull()
    })

    // Bounds planner output volume: a plan within the action limit can still generate unbounded
    // content. Like the action-count guard, this throws without a retry — an oversized plan is a
    // runaway signal, and re-prompting would re-incur the same cost.
    it('rejects a plan exceeding maxGeneratedBytes without attempting a revision', async () => {
      await writeFile(storage, 'facts/a.md', 'Fact A', 'Content A')
      await writeFile(storage, 'facts/b.md', 'Fact B', 'Content B')

      const largeContent = `---\ndescription: "Large"\n---\n\n${'X'.repeat(200)}\n`
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'merge',
              sources: ['facts/a.md', 'facts/b.md'],
              target: 'facts/combined.md',
              content: largeContent,
              reason: 'dedup',
            },
          ],
          summary: 'large merge',
        })
      )
      const streamSpy = vi.spyOn(model, 'stream')

      await expect(store.consolidate({ model, operations: ['deduplicate'], maxGeneratedBytes: 50 })).rejects.toThrow(
        /exceeds generated content limit/
      )

      // Only the initial plan call was made — the guard does not route into the revise-retry
      expect(streamSpy).toHaveBeenCalledTimes(1)
      expect(await storage.read('facts/a.md')).not.toBeNull()
      expect(await storage.read('facts/b.md')).not.toBeNull()
      expect(await storage.read('facts/combined.md')).toBeNull()
    })

    // Content validation must not block legitimate dedup: a well-formed merge still writes its
    // target and deletes the redundant sources.
    it('allows a valid merge that deduplicates sources into a well-formed target', async () => {
      await writeFile(storage, 'facts/a.md', 'Dark mode preference', 'User prefers dark mode in editors')
      await writeFile(storage, 'facts/b.md', 'Theme is dark', 'User uses dark mode theme')

      const mergedContent =
        '---\ndescription: "Theme preference"\n---\n\nUser prefers dark mode in all editors and applications\n'
      const model = new MockMessageModel().addTurn(
        buildPlanTurn({
          actions: [
            {
              action: 'merge',
              sources: ['facts/a.md', 'facts/b.md'],
              target: 'facts/a.md',
              content: mergedContent,
              reason: 'Both express the same dark mode preference',
            },
          ],
          summary: 'Merged duplicate dark mode files.',
        })
      )

      await store.consolidate({ model, operations: ['deduplicate'] })

      const target = await storage.read('facts/a.md')
      expect(target).not.toBeNull()
      expect(decoder.decode(target!)).toContain('dark mode in all editors and applications')

      // Redundant source was deleted
      expect(await storage.read('facts/b.md')).toBeNull()
    })
  })
})
