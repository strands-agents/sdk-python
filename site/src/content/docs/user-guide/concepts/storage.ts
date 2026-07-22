import { Agent } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'
import { InMemoryStorage, LocalFileStorage, S3Storage } from '@strands-agents/sdk/storage'
import type { Storage } from '@strands-agents/sdk/storage'

async function basicUsage() {
  // --8<-- [start:basic_usage]
  const storage = new LocalFileStorage()

  const agent = new Agent({
    plugins: [new ContextOffloader({ storage })],
  })
  // --8<-- [end:basic_usage]
}

async function inMemory() {
  // --8<-- [start:in_memory]
  const storage = new InMemoryStorage()
  // --8<-- [end:in_memory]
}

async function localFile() {
  // --8<-- [start:local_file]
  const storage = new LocalFileStorage('./my-data/')
  // --8<-- [end:local_file]
}

async function s3() {
  // --8<-- [start:s3]
  const storage = new S3Storage('my-bucket', {
    prefix: 'agents/prod/',
  })
  // --8<-- [end:s3]
}

async function namespaceExample() {
  // --8<-- [start:namespace]
  const storage = new LocalFileStorage()

  const scoped = storage.namespace('project-alpha')
  // Writes to "project-alpha/config.json" in the underlying store
  // --8<-- [end:namespace]
}

// --8<-- [start:custom_backend]
interface RedisClient {
  set(key: string, value: Buffer): Promise<unknown>
  getBuffer(key: string): Promise<Buffer | null>
  del(key: string): Promise<number>
  keys(pattern: string): Promise<string[]>
}

class RedisStorage implements Storage {
  private client: RedisClient

  constructor(client: RedisClient) {
    this.client = client
  }

  async write(key: string, data: Uint8Array): Promise<void> {
    await this.client.set(key, Buffer.from(data))
  }

  async read(key: string): Promise<Uint8Array | null> {
    const value = await this.client.getBuffer(key)
    return value ? new Uint8Array(value) : null
  }

  async delete(key: string): Promise<void> {
    await this.client.del(key)
  }

  async list(query: string): Promise<string[]> {
    const keys = await this.client.keys(`${query}*`)
    return keys.sort()
  }
}
// --8<-- [end:custom_backend]
