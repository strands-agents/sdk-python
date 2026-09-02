import { describe, it, expect } from 'vitest'
import { notebook } from '../notebook.js'
import type { NotebookInput, NotebookState } from '../types.js'
import { MAX_NOTEBOOKS, MAX_NOTEBOOK_NAME_LENGTH, MAX_NOTEBOOK_SIZE_BYTES, MAX_TOTAL_SIZE_BYTES } from '../types.js'
import type { ToolContext } from '../../../index.js'
import { StateStore } from '../../../state-store.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'

describe('notebook tool', () => {
  // Helper to create fresh state and context for each test
  const createFreshContext = (): { state: StateStore; context: ToolContext } => {
    const agent = createMockAgent({ appState: { notebooks: {} } })
    const context: ToolContext = {
      toolUse: {
        name: 'notebook',
        toolUseId: 'test-id',
        input: {},
      },
      agent,
      invocationState: {},
      cancelSignal: agent.cancelSignal,
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
    return { state: agent.appState, context }
  }

  describe('create oper ation', () => {
    it('creates an empty notebook with default name', async () => {
      const { state, context } = createFreshContext()
      const result = await notebook.invoke({ mode: 'create' }, context)
      expect(result).toBe("Created notebook 'default' (empty)")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('')
    })

    it('creates an empty notebook with custom name', async () => {
      const { state, context } = createFreshContext()
      const result = await notebook.invoke({ mode: 'create', name: 'notes' }, context)
      expect(result).toBe("Created notebook 'notes' (empty)")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('')
    })

    it('creates a notebook with initial content', async () => {
      const { state, context } = createFreshContext()
      const content = '# My Notes\n\nFirst entry'
      const result = await notebook.invoke({ mode: 'create', name: 'notes', newStr: content }, context)
      expect(result).toBe("Created notebook 'notes' with specified content")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe(content)
    })

    it('overwrites existing notebook on create', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'Old content' })
      const result = await notebook.invoke({ mode: 'create', name: 'notes', newStr: 'New content' }, context)
      expect(result).toBe("Created notebook 'notes' with specified content")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('New content')
    })
  })

  describe('list operation', () => {
    it('lists default notebook when initialized', async () => {
      const { state, context } = createFreshContext()
      // Initialize notebooks with default
      state.set('notebooks', { default: '' })
      const result = await notebook.invoke({ mode: 'list' }, context)
      expect(result).toContain('default: Empty')
    })

    it('lists multiple notebooks with line counts', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', {
        default: '',
        notes: 'Line 1\nLine 2\nLine 3',
        todo: 'Single line',
      })

      const result = await notebook.invoke({ mode: 'list' }, context)
      expect(result).toContain('default: Empty')
      expect(result).toContain('notes: 3 lines')
      expect(result).toContain('todo: 1 lines')
    })
  })

  describe('read operation', () => {
    it('reads entire notebook with default name', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5' })
      const result = await notebook.invoke({ mode: 'read' }, context)
      expect(result).toBe('Line 1\nLine 2\nLine 3\nLine 4\nLine 5')
    })

    it('reads entire notebook with custom name', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'Content here' })
      const result = await notebook.invoke({ mode: 'read', name: 'notes' }, context)
      expect(result).toBe('Content here')
    })

    it('reads empty notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { empty: '' })
      const result = await notebook.invoke({ mode: 'read', name: 'empty' }, context)
      expect(result).toBe("Notebook 'empty' is empty")
    })

    it('throws error for non-existent notebook', async () => {
      const { context } = createFreshContext()
      await expect(notebook.invoke({ mode: 'read', name: 'missing' }, context)).rejects.toThrow(
        "Notebook 'missing' not found"
      )
    })

    it('reads specific line range', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5' })
      const result = await notebook.invoke({ mode: 'read', readRange: [2, 4] }, context)
      expect(result).toBe('2: Line 2\n3: Line 3\n4: Line 4')
    })

    it('reads line range with negative start index', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5' })
      const result = await notebook.invoke({ mode: 'read', readRange: [-3, 5] }, context)
      expect(result).toBe('3: Line 3\n4: Line 4\n5: Line 5')
    })

    it('reads line range with negative end index', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5' })
      const result = await notebook.invoke({ mode: 'read', readRange: [1, -2] }, context)
      expect(result).toBe('1: Line 1\n2: Line 2\n3: Line 3\n4: Line 4')
    })

    it('reads line range with both negative indices', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5' })
      const result = await notebook.invoke({ mode: 'read', readRange: [-2, -1] }, context)
      expect(result).toBe('4: Line 4\n5: Line 5')
    })

    it('returns no valid lines for out of range', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5' })
      const result = await notebook.invoke({ mode: 'read', readRange: [10, 20] }, context)
      expect(result).toBe('No valid lines found in range')
    })
  })

  describe('write operation - string replacement', () => {
    it('replaces text in default notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: '# Todo List\n\n[ ] Task 1\n[ ] Task 2\n[x] Task 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          oldStr: '[ ] Task 1',
          newStr: '[x] Task 1',
        },
        context
      )
      expect(result).toBe("Replaced text in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('# Todo List\n\n[x] Task 1\n[ ] Task 2\n[x] Task 3')
    })

    it('replaces text in custom notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'Original text' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          name: 'notes',
          oldStr: 'Original',
          newStr: 'Updated',
        },
        context
      )
      expect(result).toBe("Replaced text in notebook 'notes'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('Updated text')
    })

    it('replaces multiline text', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: '# Todo List\n\n[ ] Task 1\n[ ] Task 2\n[x] Task 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          oldStr: '[ ] Task 1\n[ ] Task 2',
          newStr: '[x] Task 1\n[x] Task 2',
        },
        context
      )
      expect(result).toBe("Replaced text in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('# Todo List\n\n[x] Task 1\n[x] Task 2\n[x] Task 3')
    })

    it('preserves dollar sign patterns in newStr literally', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'const value = getPrice()' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          oldStr: 'getPrice()',
          newStr: '$& is not $1 or $$',
        },
        context
      )
      expect(result).toBe("Replaced text in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('const value = $& is not $1 or $$')
    })

    it('throws error if old string not found', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: '# Todo List\n\n[ ] Task 1\n[ ] Task 2\n[x] Task 3' })
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            oldStr: 'Nonexistent',
            newStr: 'New',
          },
          context
        )
      ).rejects.toThrow("String 'Nonexistent' not found in notebook 'default'")
    })

    it('throws error for non-existent notebook', async () => {
      const { context } = createFreshContext()
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            name: 'missing',
            oldStr: 'Old',
            newStr: 'New',
          },
          context
        )
      ).rejects.toThrow("Notebook 'missing' not found")
    })
  })

  describe('write operation - append', () => {
    it('appends newStr to a populated notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'First entry' })
      const input: NotebookInput = { mode: 'write', name: 'notes', newStr: 'Second entry' }

      const result = await notebook.invoke(input, context)

      expect(result).toBe("Appended text to notebook 'notes'")
      expect(state.get<NotebookState>('notebooks')!.notes).toBe('First entry\nSecond entry')
    })

    it('writes newStr directly to an empty notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: '' })

      const result = await notebook.invoke({ mode: 'write', name: 'notes', newStr: 'First entry' }, context)

      expect(result).toBe("Appended text to notebook 'notes'")
      expect(state.get<NotebookState>('notebooks')!.notes).toBe('First entry')
    })

    it('does not add an extra newline when the notebook already ends with one', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'First entry\n' })

      const result = await notebook.invoke(
        { mode: 'write', name: 'notes', newStr: 'Second entry\nThird entry' },
        context
      )

      expect(result).toBe("Appended text to notebook 'notes'")
      expect(state.get<NotebookState>('notebooks')!.notes).toBe('First entry\nSecond entry\nThird entry')
    })

    it('leaves the notebook unchanged when newStr is empty', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'First entry' })

      const result = await notebook.invoke({ mode: 'write', name: 'notes', newStr: '' }, context)

      expect(result).toBe("No changes made to notebook 'notes'")
      expect(state.get<NotebookState>('notebooks')!.notes).toBe('First entry')
    })

    it('throws for a notebook that does not exist', async () => {
      const { context } = createFreshContext()

      await expect(notebook.invoke({ mode: 'write', name: 'missing', newStr: 'Entry' }, context)).rejects.toThrow(
        "Notebook 'missing' not found"
      )
    })
  })

  describe('write operation - line insertion', () => {
    it('inserts after line number', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          insertLine: 2,
          newStr: 'Inserted line',
        },
        context
      )
      expect(result).toBe("Inserted text at line 3 in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('Line 1\nLine 2\nInserted line\nLine 3')
    })

    it('inserts at beginning (after line 0)', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          insertLine: 0,
          newStr: 'First line',
        },
        context
      )
      expect(result).toBe("Inserted text at line 1 in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('First line\nLine 1\nLine 2\nLine 3')
    })

    it('appends to end with negative index', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          insertLine: -1,
          newStr: 'Last line',
        },
        context
      )
      expect(result).toBe("Inserted text at line 4 in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('Line 1\nLine 2\nLine 3\nLast line')
    })

    it('inserts after negative line index', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          insertLine: -2,
          newStr: 'Before last',
        },
        context
      )
      expect(result).toBe("Inserted text at line 3 in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('Line 1\nLine 2\nBefore last\nLine 3')
    })

    it('inserts after text search', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          insertLine: 'Line 1',
          newStr: 'After Line 1',
        },
        context
      )
      expect(result).toBe("Inserted text at line 2 in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('Line 1\nAfter Line 1\nLine 2\nLine 3')
    })

    it('inserts after partial text match', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          insertLine: '2',
          newStr: 'After match',
        },
        context
      )
      expect(result).toBe("Inserted text at line 3 in notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('Line 1\nLine 2\nAfter match\nLine 3')
    })

    it('throws error if search text not found', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            insertLine: 'Nonexistent',
            newStr: 'New line',
          },
          context
        )
      ).rejects.toThrow("Text 'Nonexistent' not found in notebook 'default'")
    })

    it('throws error for line number out of range', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Line 1\nLine 2\nLine 3' })
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            insertLine: 100,
            newStr: 'New line',
          },
          context
        )
      ).rejects.toThrow('Line number out of range')
    })

    it('inserts into custom notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'First\nSecond' })
      const result = await notebook.invoke(
        {
          mode: 'write',
          name: 'notes',
          insertLine: 1,
          newStr: 'Middle',
        },
        context
      )
      expect(result).toBe("Inserted text at line 2 in notebook 'notes'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('First\nMiddle\nSecond')
    })
  })

  describe('clear operation', () => {
    it('clears default notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Some content' })
      const result = await notebook.invoke({ mode: 'clear' }, context)
      expect(result).toBe("Cleared notebook 'default'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('')
    })

    it('clears custom notebook', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { notes: 'More content' })
      const result = await notebook.invoke({ mode: 'clear', name: 'notes' }, context)
      expect(result).toBe("Cleared notebook 'notes'")
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('')
    })

    it('throws error for non-existent notebook', async () => {
      const { context } = createFreshContext()
      await expect(notebook.invoke({ mode: 'clear', name: 'missing' }, context)).rejects.toThrow(
        "Notebook 'missing' not found"
      )
    })

    it('clearing does not affect other notebooks', async () => {
      const { state, context } = createFreshContext()
      state.set('notebooks', { default: 'Some content', notes: 'More content' })
      await notebook.invoke({ mode: 'clear', name: 'notes' }, context)
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.default).toBe('Some content')
    })
  })

  describe('state persistence', () => {
    it('persists notebooks across operations', async () => {
      const { state, context } = createFreshContext()
      // Create notebook
      await notebook.invoke({ mode: 'create', name: 'notes', newStr: 'Initial' }, context)
      let notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('Initial')

      // Append to notebook
      await notebook.invoke({ mode: 'write', name: 'notes', newStr: 'Added' }, context)
      notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('Initial\nAdded')

      // Read notebook
      const content = await notebook.invoke({ mode: 'read', name: 'notes' }, context)
      expect(content).toBe('Initial\nAdded')

      // Verify state is still intact
      notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks!.notes).toBe('Initial\nAdded')
    })

    it('does not persist state when listing on empty state', async () => {
      const { state, context } = createFreshContext()
      const result = await notebook.invoke({ mode: 'list' }, context)
      expect(result).toContain('default: Empty')
      const notebooks = state.get<NotebookState>('notebooks')
      expect(notebooks).toEqual({})
    })
  })

  describe('name validation', () => {
    it.each([
      ['../etc/passwd', 'path separators'],
      ['..\\evil', 'path separators'],
      ['/absolute/path', 'path separators'],
      ['nested/name', 'path separators'],
      ['back\\slash', 'path separators'],
      ['..', 'not allowed'],
      ['.', 'not allowed'],
      ['notes\x00nul', 'NUL bytes'],
      ['   ', 'whitespace'],
      [' leading', 'whitespace'],
      ['trailing ', 'whitespace'],
      ['a'.repeat(MAX_NOTEBOOK_NAME_LENGTH + 1), 'maximum length'],
    ])('rejects %s (%s)', async (badName, _reason) => {
      const { context } = createFreshContext()
      await expect(notebook.invoke({ mode: 'create', name: badName }, context)).rejects.toThrow()
    })

    it('rejects empty name', async () => {
      const { context } = createFreshContext()
      await expect(notebook.invoke({ mode: 'create', name: '' }, context)).rejects.toThrow(
        'Notebook name must be a non-empty string'
      )
    })
  })

  describe('session caps', () => {
    it('rejects creating too many notebooks', async () => {
      const { state, context } = createFreshContext()
      const initial: Record<string, string> = {}
      for (let i = 0; i < MAX_NOTEBOOKS; i++) {
        initial[`nb-${i}`] = ''
      }
      state.set('notebooks', initial)
      await expect(notebook.invoke({ mode: 'create', name: 'over-the-line' }, context)).rejects.toThrow(
        'notebook count'
      )
    })

    it('rejects notebook content over size limit', async () => {
      const { context } = createFreshContext()
      const oversized = 'a'.repeat(MAX_NOTEBOOK_SIZE_BYTES + 1)
      await expect(notebook.invoke({ mode: 'create', name: 'big', newStr: oversized }, context)).rejects.toThrow(
        'maximum of'
      )
    })

    it('rejects total session size over limit', async () => {
      const { state, context } = createFreshContext()
      const perNotebook = MAX_NOTEBOOK_SIZE_BYTES
      const count = Math.floor(MAX_TOTAL_SIZE_BYTES / perNotebook) + 1
      const initial: Record<string, string> = {}
      for (let i = 0; i < count - 1; i++) {
        initial[`nb-${i}`] = 'a'.repeat(perNotebook)
      }
      state.set('notebooks', initial)
      await expect(
        notebook.invoke({ mode: 'create', name: 'last', newStr: 'a'.repeat(perNotebook) }, context)
      ).rejects.toThrow('session maximum')
    })
  })

  describe('validation errors', () => {
    it('requires context', async () => {
      await expect(notebook.invoke({ mode: 'list' })).rejects.toThrow('Tool context is required')
    })

    it('rejects write without newStr for replacement', async () => {
      const { context } = createFreshContext()
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            oldStr: 'Old',
            // Missing newStr
          } as any,
          context
        )
      ).rejects.toThrow()
    })

    it('rejects write without newStr for insertion', async () => {
      const { context } = createFreshContext()
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            insertLine: 1,
            // Missing newStr
          } as any,
          context
        )
      ).rejects.toThrow()
    })

    it('rejects write without valid operation parameters', async () => {
      const { context } = createFreshContext()
      await expect(
        notebook.invoke(
          {
            mode: 'write',
            // Missing both replacement and insertion params
          } as any,
          context
        )
      ).rejects.toThrow()
    })
  })
})
