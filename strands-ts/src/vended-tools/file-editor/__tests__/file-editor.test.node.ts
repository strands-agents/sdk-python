import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { fileEditor } from '../file-editor.js'
import type { ToolContext } from '../../../index.js'
import { StateStore } from '../../../state-store.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { promises as fs } from 'fs'
import * as path from 'path'
import { tmpdir } from 'os'

describe('fileEditor tool', () => {
  let testDir: string
  let context: ToolContext

  // Helper to create fresh state and context for each test
  const createFreshContext = (): { state: StateStore; context: ToolContext } => {
    const agent = createMockAgent()
    const toolContext: ToolContext = {
      toolUse: {
        name: 'fileEditor',
        toolUseId: 'test-id',
        input: {},
      },
      agent,
      invocationState: {},
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
    return { state: agent.appState, context: toolContext }
  }

  // Helper to create a test file
  const createTestFile = async (filename: string, content: string): Promise<string> => {
    const filePath = path.join(testDir, filename)
    const dir = path.dirname(filePath)
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(filePath, content, 'utf-8')
    return filePath
  }

  // Helper to create a test directory with files
  const createTestDirectory = async (dirName: string, files: Record<string, string>): Promise<string> => {
    const dirPath = path.join(testDir, dirName)
    await fs.mkdir(dirPath, { recursive: true })
    for (const [filename, content] of Object.entries(files)) {
      const filePath = path.join(dirPath, filename)
      const fileDir = path.dirname(filePath)
      await fs.mkdir(fileDir, { recursive: true })
      await fs.writeFile(filePath, content, 'utf-8')
    }
    return dirPath
  }

  beforeEach(async () => {
    // Create a temporary test directory
    testDir = path.join(tmpdir(), `file-editor-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    await fs.mkdir(testDir, { recursive: true })

    // Create fresh state and context
    const fresh = createFreshContext()
    context = fresh.context
  })

  afterEach(async () => {
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true })
    } catch {
      // Ignore cleanup errors
    }
  })

  describe('view command', () => {
    describe('when viewing entire file', () => {
      it('returns file content with line numbers', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
        const result = await fileEditor.invoke({ command: 'view', path: filePath }, context)
        expect(result).toContain("Here's the result of running `cat -n`")
        expect(result).toContain('     1  Line 1')
        expect(result).toContain('     2  Line 2')
        expect(result).toContain('     3  Line 3')
      })

      it('handles empty file', async () => {
        const filePath = await createTestFile('empty.txt', '')
        const result = await fileEditor.invoke({ command: 'view', path: filePath }, context)
        expect(result).toContain("Here's the result of running `cat -n`")
      })

      it('handles single line file', async () => {
        const filePath = await createTestFile('single.txt', 'Only one line')
        const result = await fileEditor.invoke({ command: 'view', path: filePath }, context)
        expect(result).toContain('     1  Only one line')
      })
    })

    describe('when viewing with line range', () => {
      it('returns specified lines with line numbers', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5')
        const result = await fileEditor.invoke({ command: 'view', path: filePath, view_range: [2, 4] }, context)
        expect(result).toContain('     2  Line 2')
        expect(result).toContain('     3  Line 3')
        expect(result).toContain('     4  Line 4')
        expect(result).not.toContain('     1  ')
        expect(result).not.toContain('     5  ')
      })

      it('handles negative end index (-1 means to end)', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5')
        const result = await fileEditor.invoke({ command: 'view', path: filePath, view_range: [3, -1] }, context)
        expect(result).toContain('     3  Line 3')
        expect(result).toContain('     4  Line 4')
        expect(result).toContain('     5  Line 5')
        expect(result).not.toContain('     1  ')
        expect(result).not.toContain('     2  ')
      })

      it('handles single line range', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
        const result = await fileEditor.invoke({ command: 'view', path: filePath, view_range: [2, 2] }, context)
        expect(result).toContain('     2  Line 2')
        expect(result).not.toContain('     1  ')
        expect(result).not.toContain('     3  ')
      })
    })

    describe('when viewing directory', () => {
      it('lists files up to 2 levels deep', async () => {
        const dirPath = await createTestDirectory('testdir', {
          'file1.txt': 'content',
          'file2.txt': 'content',
          'subdir/file3.txt': 'content',
          'subdir/nested/file4.txt': 'content',
        })
        const result = await fileEditor.invoke({ command: 'view', path: dirPath }, context)
        expect(result).toContain('file1.txt')
        expect(result).toContain('file2.txt')
        expect(result).toContain('subdir')
        expect(result).toContain('file3.txt')
        expect(result).toContain('file4.txt')
      })

      it('excludes hidden files', async () => {
        const dirPath = await createTestDirectory('testdir', {
          'visible.txt': 'content',
          '.hidden.txt': 'content',
          'subdir/.hidden-dir/file.txt': 'content',
        })
        const result = await fileEditor.invoke({ command: 'view', path: dirPath }, context)
        expect(result).toContain('visible.txt')
        expect(result).not.toContain('.hidden')
      })
    })

    describe('error cases', () => {
      it('throws when file not found', async () => {
        const nonExistentPath = path.join(testDir, 'nonexistent.txt')
        await expect(fileEditor.invoke({ command: 'view', path: nonExistentPath }, context)).rejects.toThrow(
          'does not exist'
        )
      })

      it('throws when path is not absolute', async () => {
        await expect(fileEditor.invoke({ command: 'view', path: 'relative/path.txt' }, context)).rejects.toThrow(
          'not an absolute path'
        )
      })

      it('throws when view_range has invalid start line', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
        await expect(
          fileEditor.invoke({ command: 'view', path: filePath, view_range: [0, 2] }, context)
        ).rejects.toThrow('view_range')
      })

      it('throws when view_range end is beyond file length', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
        await expect(
          fileEditor.invoke({ command: 'view', path: filePath, view_range: [1, 10] }, context)
        ).rejects.toThrow('view_range')
      })

      it('throws when view_range end is before start', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
        await expect(
          fileEditor.invoke({ command: 'view', path: filePath, view_range: [3, 1] }, context)
        ).rejects.toThrow('view_range')
      })

      it('throws when view_range is provided for directory', async () => {
        const dirPath = await createTestDirectory('testdir', { 'file.txt': 'content' })
        await expect(
          fileEditor.invoke({ command: 'view', path: dirPath, view_range: [1, 2] }, context)
        ).rejects.toThrow('not allowed when')
      })
    })
  })

  describe('create command', () => {
    it('creates new file with content', async () => {
      const filePath = path.join(testDir, 'new-file.txt')
      const content = 'Hello World\nLine 2'
      const result = await fileEditor.invoke({ command: 'create', path: filePath, file_text: content }, context)
      expect(result).toContain('File created successfully')
      expect(result).toContain(filePath)

      // Verify file was created
      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe(content)
    })

    it('creates file in non-existent directory', async () => {
      const filePath = path.join(testDir, 'newdir', 'subdir', 'new-file.txt')
      const content = 'Content'
      const result = await fileEditor.invoke({ command: 'create', path: filePath, file_text: content }, context)
      expect(result).toContain('File created successfully')

      // Verify file and directories were created
      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe(content)
    })

    it('creates empty file', async () => {
      const filePath = path.join(testDir, 'empty.txt')
      const result = await fileEditor.invoke({ command: 'create', path: filePath, file_text: '' }, context)
      expect(result).toContain('File created successfully')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('')
    })

    describe('error cases', () => {
      it('throws when file already exists', async () => {
        const filePath = await createTestFile('existing.txt', 'content')
        await expect(
          fileEditor.invoke({ command: 'create', path: filePath, file_text: 'new content' }, context)
        ).rejects.toThrow('already exists')
      })

      it('throws when path is not absolute', async () => {
        await expect(
          fileEditor.invoke({ command: 'create', path: 'relative/path.txt', file_text: 'content' }, context)
        ).rejects.toThrow('not an absolute path')
      })

      it('throws when path contains traversal', async () => {
        const filePath = '..outside.txt'
        await expect(
          fileEditor.invoke({ command: 'create', path: filePath, file_text: 'content' }, context)
        ).rejects.toThrow()
      })

      it('throws when trying to create in directory as path', async () => {
        const dirPath = await createTestDirectory('testdir', {})
        await expect(
          fileEditor.invoke({ command: 'create', path: dirPath, file_text: 'content' }, context)
        ).rejects.toThrow('already exists')
      })
    })
  })

  describe('str_replace command', () => {
    it('replaces unique string occurrence', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2 OLD\nLine 3\nLine 4')
      const result = await fileEditor.invoke(
        { command: 'str_replace', path: filePath, old_str: 'OLD', new_str: 'NEW' },
        context
      )
      expect(result).toContain('The file')
      expect(result).toContain('has been edited')
      expect(result).toContain('NEW')

      // Verify file was updated
      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('Line 1\nLine 2 NEW\nLine 3\nLine 4')
    })

    it('shows snippet with 4 lines before and after change', async () => {
      const content = 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5 OLD\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10'
      const filePath = await createTestFile('test.txt', content)
      const result = await fileEditor.invoke(
        { command: 'str_replace', path: filePath, old_str: 'OLD', new_str: 'NEW' },
        context
      )
      // Should show lines 1-9 (4 before + line 5 + 4 after)
      expect(result).toContain('Line 1')
      expect(result).toContain('Line 9')
      expect(result).not.toContain('Line 10')
    })

    it('handles empty new_str (deletion)', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2 DELETE_ME\nLine 3')
      const result = await fileEditor.invoke(
        { command: 'str_replace', path: filePath, old_str: ' DELETE_ME', new_str: '' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('Line 1\nLine 2\nLine 3')
    })

    it('handles multi-line old_str', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nOLD LINE 1\nOLD LINE 2\nLine 4')
      const result = await fileEditor.invoke(
        { command: 'str_replace', path: filePath, old_str: 'OLD LINE 1\nOLD LINE 2', new_str: 'NEW LINE' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('Line 1\nNEW LINE\nLine 4')
    })

    it('preserves dollar sign patterns in new_str literally', async () => {
      const filePath = await createTestFile('test.txt', 'const value = getPrice()')
      await fileEditor.invoke(
        { command: 'str_replace', path: filePath, old_str: 'getPrice()', new_str: '$& is not $1 or $$' },
        context
      )

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('const value = $& is not $1 or $$')
    })

    describe('error cases', () => {
      it('throws when old_str not found', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
        await expect(
          fileEditor.invoke({ command: 'str_replace', path: filePath, old_str: 'NOTFOUND', new_str: 'NEW' }, context)
        ).rejects.toThrow('did not appear')
      })

      it('throws when multiple occurrences of old_str', async () => {
        const filePath = await createTestFile('test.txt', 'DUP Line 1\nLine 2\nDUP Line 3')
        await expect(
          fileEditor.invoke({ command: 'str_replace', path: filePath, old_str: 'DUP', new_str: 'NEW' }, context)
        ).rejects.toThrow('Multiple occurrences')
      })

      it('throws when file not found', async () => {
        const nonExistentPath = path.join(testDir, 'nonexistent.txt')
        await expect(
          fileEditor.invoke({ command: 'str_replace', path: nonExistentPath, old_str: 'OLD', new_str: 'NEW' }, context)
        ).rejects.toThrow('does not exist')
      })

      it('throws when path is directory', async () => {
        const dirPath = await createTestDirectory('testdir', {})
        await expect(
          fileEditor.invoke({ command: 'str_replace', path: dirPath, old_str: 'OLD', new_str: 'NEW' }, context)
        ).rejects.toThrow('directory')
      })
    })
  })

  describe('insert command', () => {
    it('inserts at beginning (line 0)', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
      const result = await fileEditor.invoke(
        { command: 'insert', path: filePath, insert_line: 0, new_str: 'NEW LINE' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('NEW LINE\nLine 1\nLine 2\nLine 3')
    })

    it('inserts in middle', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
      const result = await fileEditor.invoke(
        { command: 'insert', path: filePath, insert_line: 2, new_str: 'NEW LINE' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('Line 1\nLine 2\nNEW LINE\nLine 3')
    })

    it('inserts at end', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
      const result = await fileEditor.invoke(
        { command: 'insert', path: filePath, insert_line: 3, new_str: 'NEW LINE' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('Line 1\nLine 2\nLine 3\nNEW LINE')
    })

    it('shows snippet with 4 lines before and after insertion', async () => {
      const content = 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9'
      const filePath = await createTestFile('test.txt', content)
      const result = await fileEditor.invoke(
        { command: 'insert', path: filePath, insert_line: 5, new_str: 'INSERTED' },
        context
      )
      // Inserting at line 5 (0-indexed) means after Line 5
      // Snippet shows 4 lines before (lines 2-5) + inserted + 4 lines after (lines 6-9)
      expect(result).toContain('Line 2')
      expect(result).toContain('Line 9')
      expect(result).toContain('INSERTED')
    })

    it('handles multi-line insertion', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2')
      const result = await fileEditor.invoke(
        { command: 'insert', path: filePath, insert_line: 1, new_str: 'NEW 1\nNEW 2\nNEW 3' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('Line 1\nNEW 1\nNEW 2\nNEW 3\nLine 2')
    })

    it('handles insertion in empty file', async () => {
      const filePath = await createTestFile('empty.txt', '')
      const result = await fileEditor.invoke(
        { command: 'insert', path: filePath, insert_line: 0, new_str: 'First line' },
        context
      )
      expect(result).toContain('has been edited')

      const fileContent = await fs.readFile(filePath, 'utf-8')
      expect(fileContent).toBe('First line')
    })

    describe('error cases', () => {
      it('throws when insert_line is negative', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2')
        await expect(
          fileEditor.invoke({ command: 'insert', path: filePath, insert_line: -1, new_str: 'NEW' }, context)
        ).rejects.toThrow('insert_line')
      })

      it('throws when insert_line is beyond file length', async () => {
        const filePath = await createTestFile('test.txt', 'Line 1\nLine 2')
        await expect(
          fileEditor.invoke({ command: 'insert', path: filePath, insert_line: 10, new_str: 'NEW' }, context)
        ).rejects.toThrow('insert_line')
      })

      it('throws when file not found', async () => {
        const nonExistentPath = path.join(testDir, 'nonexistent.txt')
        await expect(
          fileEditor.invoke({ command: 'insert', path: nonExistentPath, insert_line: 0, new_str: 'NEW' }, context)
        ).rejects.toThrow('does not exist')
      })

      it('throws when path is directory', async () => {
        const dirPath = await createTestDirectory('testdir', {})
        await expect(
          fileEditor.invoke({ command: 'insert', path: dirPath, insert_line: 0, new_str: 'NEW' }, context)
        ).rejects.toThrow('directory')
      })
    })
  })

  describe('path validation and security', () => {
    it('rejects relative paths', async () => {
      await expect(fileEditor.invoke({ command: 'view', path: 'relative/path.txt' }, context)).rejects.toThrow(
        'not an absolute path'
      )
    })
  })

  describe('file size limits', () => {
    it('throws when file exceeds default size limit', async () => {
      // Create a file larger than 1MB
      const largeContent = 'x'.repeat(1048577) // 1MB + 1 byte
      const filePath = await createTestFile('large.txt', largeContent)

      await expect(fileEditor.invoke({ command: 'view', path: filePath }, context)).rejects.toThrow('exceeds')
    })
  })

  describe('edge cases', () => {
    it('handles files with special characters in content', async () => {
      const content = 'Special chars: @#$%^&*()_+-={}[]|:;"<>,.?/~`'
      const filePath = await createTestFile('special.txt', content)
      const result = await fileEditor.invoke({ command: 'view', path: filePath }, context)
      expect(result).toContain('Special chars:')
    })

    it('handles files with unicode characters', async () => {
      const content = '你好世界\n🚀 Emoji test\nΣ Greek letters'
      const filePath = await createTestFile('unicode.txt', content)
      const result = await fileEditor.invoke({ command: 'view', path: filePath }, context)
      expect(result).toContain('你好世界')
      expect(result).toContain('🚀')
    })

    it('handles files with tabs (expands tabs)', async () => {
      const content = 'Line 1\tTab\tSeparated'
      const filePath = await createTestFile('tabs.txt', content)
      const result = await fileEditor.invoke({ command: 'view', path: filePath }, context)
      // Tabs should be expanded to spaces
      expect(result).not.toContain('\t')
    })
  })
})
