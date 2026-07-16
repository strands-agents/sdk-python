# file-write: recommendation to not vend

This directory contains no code. It records the decision not to ship a separate file-write vended tool and points at the correct migration target for the two capabilities the sub-issue asks for.

Sub-issue: https://github.com/strands-agents/harness-sdk/issues/3237

## What was proposed

A file-write shim exposing `path` and `content`, internally delegating to file-editor's `create` command and adding overwrite semantics on top so a single call could either create a new file or replace an existing one.

## Why we recommend against vending it

The create surface is already vended. The file-editor tool at `@strands-agents/sdk/vended-tools/file-editor` exposes `command: 'create'` with `path` and `file_text`, the same input surface a file-write shim would carry. It is sandbox-routed and rejects relative paths and parent-directory traversal. A second entrypoint into the same write path adds an API surface without adding a capability. Size enforcement on the create path is currently a gap in file-editor's create handler on both language sides, and the fix belongs there, not in a new tool.

The create command deliberately refuses to overwrite, and that refusal is a safety property rather than a limitation. Overwrite in place erases whatever the agent or the human had there before, with no undo. The file-editor tool forces the model to see the existing content via `view` before mutating it with `str_replace` or `insert`, which is the choice most agent-editor tools make. A separate file-write that silently overwrote would undo that choice from the outside.

If we want an overwrite command in the vended tool set, it should live inside file-editor with the same validation and audit surface as `create`, `str_replace`, and `insert`. Two tools sharing the write path means two boundaries to keep tight forever; one tool means one. No owner is chartered for that expansion today; the related reconciliation work at https://github.com/strands-agents/harness-sdk/issues/3235 covers view, create, str_replace, insert, undo, find, and pattern, but not overwrite.

This decision is not symmetric with the parallel file-read one. The sibling file-read shim (sub-issue https://github.com/strands-agents/harness-sdk/issues/3236) is being proposed for vending, with the same "narrower schema is better model UX" motivation. The difference is shape, not preference. file-read is a strict subset of file-editor's view command, so an agent that only needs to read can be given fileRead *instead of* fileEditor: the strict subset lets us remove capability from the model's tool set, which is a real security win. file-write cannot replace file-editor: an agent that needs to write also needs view, str_replace, and insert to change anything that already exists, so file-write would always ship alongside file-editor rather than in place of it. The tool count goes up, the capability set does not, and the model gets two names for the same create path. Concretely, `new Agent({ tools: [fileRead] })` is a coherent, less-privileged configuration; `new Agent({ tools: [fileWrite] })` without fileEditor is not, because the agent could only create files and never modify them.

## Migration target

Create a new file with fileEditor and `command: 'create'`:

```typescript
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { Agent, BedrockModel } from '@strands-agents/sdk'

const agent = new Agent({
  model: new BedrockModel({ region: 'us-east-1' }),
  tools: [fileEditor],
})
// The model calls fileEditor with command='create', path=..., file_text=...
```

Overwrite an existing file with `command: 'view'` followed by `command: 'str_replace'`, the same pattern used for any other targeted edit. There is no explicit overwrite command today.

For programmatic, non-agent writes from TypeScript code, call the sandbox directly: `agent.sandbox.writeText(path, content)`. Vended tools are for the model; TypeScript code has the SDK primitive. Direct sandbox writes bypass the path and content validation that fileEditor layers on top, which is by design, since the caller is trusted TypeScript code, not a model.

## If you disagree

Reopen https://github.com/strands-agents/harness-sdk/issues/3237 with a concrete case where fileEditor create is not enough. The bar is a capability that cannot be expressed through the existing tool without confusing the model, not an aesthetic preference for a shorter tool name.
