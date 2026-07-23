# File Read Tool

A read-only view of the filesystem: the agent can view a file (with an optional line range) or list a directory. The tool is a thin shim over the file editor's view command with a two-parameter surface (path and view_range), so a read-only agent's tool schema does not include the write-side parameters (file_text, old_str, new_str, insert_line) at all. All validation is delegated to the file editor; updates to its security surface propagate here automatically.

## Usage

```typescript
import { fileRead } from '@strands-agents/sdk/vended-tools/file-read'
import { Agent, BedrockModel } from '@strands-agents/sdk'

const agent = new Agent({
  model: new BedrockModel({ region: 'us-east-1' }),
  tools: [fileRead],
})

await agent.invoke('Read the first twenty lines of /tmp/config.json')
```

## Parameters

- `path` (string, required): Absolute path to a file or directory.
- `view_range` (`[number, number]`, optional): Line range to view. 1-indexed; `end` may be `-1` for end-of-file. Not allowed when `path` points to a directory.

For any write operation, use [`fileEditor`](../file-editor/README.md).
