# file_write: recommendation to not vend

This directory contains no code. It records the decision not to ship a separate file_write vended tool and points at the correct migration target for the two capabilities the sub-issue asks for.

Sub-issue: https://github.com/strands-agents/harness-sdk/issues/3237

## What was proposed

A file_write shim exposing `path` and `content`, internally delegating to file_editor's `create` command and adding overwrite semantics on top so a single call could either create a new file or replace an existing one.

## Why we recommend against vending it

The create surface is already vended. The file_editor tool exposes `command="create"` with `path` and `file_text`, the same input surface a file_write shim would carry. It is sandbox-routed and rejects relative paths and parent-directory traversal. A second entrypoint into the same write path adds an API surface without adding a capability. Size enforcement on the create path is currently a gap in file_editor's create handler and the fix belongs there, not in a new tool.

The create command deliberately refuses to overwrite, and that refusal is a safety property rather than a limitation. Overwrite in place erases whatever the agent or the human had there before, with no undo. The file_editor tool forces the model to see the existing content via `view` before mutating it with `str_replace` or `insert`, which is the choice most agent-editor tools make. A separate file_write that silently overwrote would undo that choice from the outside.

If we want an overwrite command in the vended tool set, it should live inside file_editor with the same validation and audit surface as `create`, `str_replace`, and `insert`. Two tools sharing the write path means two boundaries to keep tight forever; one tool means one. No owner is chartered for that expansion today; the related reconciliation work at https://github.com/strands-agents/harness-sdk/issues/3235 covers view, create, str_replace, insert, undo, find, and pattern, but not overwrite.

This decision is not symmetric with the parallel file_read one. The sibling file_read shim (sub-issue https://github.com/strands-agents/harness-sdk/issues/3236) is being proposed for vending, with the same "narrower schema is better model UX" motivation. The difference is shape, not preference. file_read is a strict subset of file_editor's view command, so an agent that only needs to read can be given file_read *instead of* file_editor: the strict subset lets us remove capability from the model's tool set, which is a real security win. file_write cannot replace file_editor: an agent that needs to write also needs view, str_replace, and insert to change anything that already exists, so file_write would always ship alongside file_editor rather than in place of it. The tool count goes up, the capability set does not, and the model gets two names for the same create path. Concretely, `Agent(tools=[file_read])` is a coherent, less-privileged configuration; `Agent(tools=[file_write])` without file_editor is not, because the agent could only create files and never modify them.

## Migration target

Create a new file with file_editor and `command="create"`:

```python
from strands import Agent
from strands.vended_tools import file_editor

agent = Agent(tools=[file_editor])
# The model calls file_editor with command="create", path=..., file_text=...
```

Overwrite an existing file with `command="view"` followed by `command="str_replace"`, the same pattern used for any other targeted edit. There is no explicit overwrite command today.

For programmatic, non-agent writes from Python code, call the sandbox directly: `agent.sandbox.write_text(path, content)`. Vended tools are for the model; Python code has the SDK primitive. Direct sandbox writes bypass the path and content validation that file_editor layers on top, which is by design, since the caller is trusted Python code, not a model.

## If you disagree

Reopen https://github.com/strands-agents/harness-sdk/issues/3237 with a concrete case where file_editor create is not enough. The bar is a capability that cannot be expressed through the existing tool without confusing the model, not an aesthetic preference for a shorter tool name.
