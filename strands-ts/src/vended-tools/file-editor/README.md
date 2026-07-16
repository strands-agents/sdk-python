# File Editor Tool

A filesystem editor for viewing, creating, and editing files. All I/O routes through the agent's configured `Sandbox`, so isolation is a property of the sandbox rather than the tool.

The tool exposes seven commands: `view` (file contents with line numbers, or a two-level directory listing), `create` (refuses to overwrite), `str_replace` (exact-match; ambiguous matches require `replace_all: true`), `insert` (0-indexed line insertion), `pattern_replace` (regex; `new_str` honours `$&` / `$1` backreferences), `find_line` (first-occurrence line lookup with optional whitespace-tolerant `fuzzy` matching), and `undo_edit` (single-step, in-memory revert of the most recent edit for a path). Paths must be absolute and free of `..` segments. Binary files and files above the size cap are rejected with a clean error. `pattern_replace` bounds the pattern length, the match count, and rejects the classic `(...+)+` catastrophic-backtracking shape before compilation.

## Usage

```typescript
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { Agent, BedrockModel } from '@strands-agents/sdk'

const agent = new Agent({
  model: new BedrockModel({ region: 'us-east-1' }),
  tools: [fileEditor],
})

await agent.invoke('Create a file /tmp/notes.txt with "# My Notes"')
```

## Customization

`makeFileEditor` accepts:

- `root` (string, optional): confines every operation to an absolute directory. Applies both a string-level check and, when the local filesystem sees the target, a `realpath`-based symlink check.
- `maxFileSize` (number, optional): byte cap on files that can be read or written. Defaults to 1 MB.
- `maxUndoEntries` (number, optional): maximum distinct paths retained in the in-memory undo history. Defaults to 32.
- `maxUndoBytes` (number, optional): approximate byte cap on the undo history. Defaults to 32 MB.
- `name`, `description`: override the tool's registered name and description.

```typescript
import { makeFileEditor } from '@strands-agents/sdk/vended-tools/file-editor'

const editor = makeFileEditor({ root: '/workspace', maxFileSize: 5 * 1024 * 1024 })
```
