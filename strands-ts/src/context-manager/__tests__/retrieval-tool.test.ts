import { describe, it, expect } from 'vitest'
import { createRetrievalTool, RETRIEVAL_TOOL_NAME } from '../retrieval-tool.js'
import { Stash } from '../stash.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'

function invoke(tool: ReturnType<typeof createRetrievalTool>, input: unknown): Promise<unknown> {
  return (tool as unknown as { invoke(input: unknown): Promise<unknown> }).invoke(input)
}

describe('retrieval tool', () => {
  function makeStashWithContent(text: string): { stash: Stash; refPromise: Promise<string> } {
    const stash = new Stash(new InMemoryStorage())
    const refPromise = stash.store('tool-1', 0, new TextEncoder().encode(text), 'text/plain')
    return { stash, refPromise }
  }

  it('has the correct tool name', () => {
    const stash = new Stash(new InMemoryStorage())
    const tool = createRetrievalTool(stash)
    expect(tool.name).toBe(RETRIEVAL_TOOL_NAME)
  })

  it('retrieves full text content', async () => {
    const { stash, refPromise } = makeStashWithContent('hello world\nline two')
    const ref = await refPromise
    const tool = createRetrievalTool(stash)

    const result = await invoke(tool, { reference: ref })
    expect(result).toBe('hello world\nline two')
  })

  it('searches with pattern', async () => {
    const text = Array.from({ length: 20 }, (_, index) => `line ${index + 1}: ${index % 3 === 0 ? 'ERROR' : 'ok'}`).join(
      '\n'
    )
    const { stash, refPromise } = makeStashWithContent(text)
    const ref = await refPromise
    const tool = createRetrievalTool(stash)

    const result = (await invoke(tool, { reference: ref, pattern: 'ERROR' })) as string
    expect(result).toContain('ERROR')
    expect(result).toContain('match')
  })

  it('returns line range', async () => {
    const text = Array.from({ length: 50 }, (_, index) => `line ${index + 1}`).join('\n')
    const { stash, refPromise } = makeStashWithContent(text)
    const ref = await refPromise
    const tool = createRetrievalTool(stash)

    const result = (await invoke(tool, { reference: ref, line_range: { start: 5, end: 10 } })) as string
    expect(result).toContain('line 5')
    expect(result).toContain('line 10')
    expect(result).not.toContain('line 11')
  })

  it('returns error for unknown reference', async () => {
    const stash = new Stash(new InMemoryStorage())
    const tool = createRetrievalTool(stash)

    const result = (await invoke(tool, { reference: 'nonexistent' })) as string
    expect(result).toContain('Error: reference not found')
  })
})
