import { describe, it, expect } from 'vitest'
import { createRetrievalTool, RETRIEVAL_TOOL_NAME } from '../retrieval-tool.js'
import { Stash } from '../stash.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'
import { ImageBlock } from '../../types/media.js'

function invoke(retrievalTool: ReturnType<typeof createRetrievalTool>, input: unknown): Promise<unknown> {
  return (retrievalTool as unknown as { invoke(input: unknown): Promise<unknown> }).invoke(input)
}

describe('retrieval tool', () => {
  function makeStashWithContent(text: string): { stash: Stash; refPromise: Promise<string> } {
    const stash = new Stash(new InMemoryStorage())
    const refPromise = stash.store('tool-1', 0, new TextEncoder().encode(text), 'text/plain')
    return { stash, refPromise }
  }

  it('has the correct tool name', () => {
    const stash = new Stash(new InMemoryStorage())
    const retrievalTool = createRetrievalTool(stash)
    expect(retrievalTool.name).toBe(RETRIEVAL_TOOL_NAME)
  })

  it('retrieves full text content', async () => {
    const { stash, refPromise } = makeStashWithContent('hello world\nline two')
    const ref = await refPromise
    const retrievalTool = createRetrievalTool(stash)

    const result = await invoke(retrievalTool, { reference: ref })
    expect(result).toBe('hello world\nline two')
  })

  it('searches with pattern', async () => {
    const text = Array.from(
      { length: 20 },
      (_, index) => `line ${index + 1}: ${index % 3 === 0 ? 'ERROR' : 'ok'}`
    ).join('\n')
    const { stash, refPromise } = makeStashWithContent(text)
    const ref = await refPromise
    const retrievalTool = createRetrievalTool(stash)

    const result = (await invoke(retrievalTool, { reference: ref, pattern: 'ERROR' })) as string
    expect(result).toContain('ERROR')
    expect(result).toContain('match')
  })

  it('returns line range', async () => {
    const text = Array.from({ length: 50 }, (_, index) => `line ${index + 1}`).join('\n')
    const { stash, refPromise } = makeStashWithContent(text)
    const ref = await refPromise
    const retrievalTool = createRetrievalTool(stash)

    const result = (await invoke(retrievalTool, { reference: ref, line_range: { start: 5, end: 10 } })) as string
    expect(result).toContain('line 5')
    expect(result).toContain('line 10')
    expect(result).not.toContain('line 11')
  })

  it('returns error for unknown reference', async () => {
    const stash = new Stash(new InMemoryStorage())
    const retrievalTool = createRetrievalTool(stash)

    const result = (await invoke(retrievalTool, { reference: 'nonexistent' })) as string
    expect(result).toContain('Error: reference not found')
  })

  it('retrieves binary content as native ImageBlock', async () => {
    const stash = new Stash(new InMemoryStorage())
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47])
    const ref = await stash.store('tool-img', 0, imageBytes, 'image/png')
    const retrievalTool = createRetrievalTool(stash)

    const result = await invoke(retrievalTool, { reference: ref })
    expect(result).toBeInstanceOf(ImageBlock)
    expect((result as ImageBlock).format).toBe('png')
    expect((result as ImageBlock).source.type).toBe('imageSourceBytes')
  })

  it('retrieves JSON content as parsed object', async () => {
    const stash = new Stash(new InMemoryStorage())
    const json = { key: 'value', nested: [1, 2, 3] }
    const ref = await stash.store('tool-json', 0, new TextEncoder().encode(JSON.stringify(json)), 'application/json')
    const retrievalTool = createRetrievalTool(stash)

    const result = await invoke(retrievalTool, { reference: ref })
    expect(result).toEqual(json)
  })

  it('returns error when searching binary content', async () => {
    const stash = new Stash(new InMemoryStorage())
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47])
    const ref = await stash.store('tool-img', 0, imageBytes, 'image/png')
    const retrievalTool = createRetrievalTool(stash)

    const result = (await invoke(retrievalTool, { reference: ref, pattern: 'test' })) as string
    expect(result).toContain('cannot search binary content')
  })
})
