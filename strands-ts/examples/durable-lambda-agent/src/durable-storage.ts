import type { DurableContext } from '@aws/durable-execution-sdk-js'
import type { Storage } from '@strands-agents/sdk/storage'

/** Journals storage operations so Lambda replay never repeats S3 side effects. */
export class DurableStorage implements Storage {
  private operationIndex = 0

  constructor(
    private readonly context: DurableContext,
    private readonly delegate: Storage
  ) {}

  async write(key: string, data: Uint8Array): Promise<void> {
    await this.context.step<boolean>(this.nextStepName('write'), async () => {
      await this.delegate.write(key, data)
      return true
    })
  }

  async read(key: string): Promise<Uint8Array | null> {
    const encoded = await this.context.step<string | null>(this.nextStepName('read'), async () => {
      const data = await this.delegate.read(key)
      return data === null ? null : Buffer.from(data).toString('base64')
    })
    return encoded === null ? null : new Uint8Array(Buffer.from(encoded, 'base64'))
  }

  async delete(key: string): Promise<void> {
    await this.context.step<boolean>(this.nextStepName('delete'), async () => {
      await this.delegate.delete(key)
      return true
    })
  }

  async list(prefix: string): Promise<string[]> {
    return this.context.step<string[]>(this.nextStepName('list'), async () => this.delegate.list(prefix))
  }

  private nextStepName(operation: string): string {
    const index = this.operationIndex
    this.operationIndex += 1
    return `session-storage:${index}:${operation}`
  }
}
