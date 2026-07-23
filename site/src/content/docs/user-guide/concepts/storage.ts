import { Agent } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'
import { InMemoryStorage, LocalFileStorage, S3Storage } from '@strands-agents/sdk/storage'

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
