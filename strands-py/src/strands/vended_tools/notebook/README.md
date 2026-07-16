# Notebook Tool

A session-scoped scratchpad an agent can read, write, and clear. Notebook state lives on `agent.state` under the `notebooks` key, so it persists across invocations within a session and across sessions when the caller wires up a durable state store.

## Usage

```python
from strands import Agent
from strands.vended_tools import notebook

agent = Agent(tools=[notebook])
agent('Create a notebook called "ideas" with "# Project Ideas"')
agent('Add "- Build a web scraper" to the ideas notebook')
agent('Read the ideas notebook')
```

State is accessible directly:

```python
print(agent.state.get("notebooks"))
# {'default': '', 'ideas': '# Project Ideas\n- Build a web scraper'}
```

A `default` notebook is materialized the first time the tool sees empty state so read and list have something to point at. Any `create` that names another notebook keeps the default alongside it.

## Operations

- `create` — create or overwrite a notebook, optionally with initial content.
- `list` — list all notebooks with line counts.
- `read` — read a whole notebook or a line range. Ranges are one-indexed and support negative indices.
- `write` — replace a substring (`old_str` plus `new_str`) or insert a line (`insert_line` plus `new_str`). `insert_line` accepts a line number or a search string.
- `clear` — empty a notebook without deleting it.

## Session limits

The Python port enforces per-session caps so a prompt injection cannot grow state without bound:

- `MAX_NOTEBOOKS` — sixty-four notebooks per session.
- `MAX_NOTEBOOK_SIZE_BYTES` — one mebibyte per notebook, measured after UTF-8 encoding.
- `MAX_TOTAL_SIZE_BYTES` — eight mebibytes total.

Notebook names are validated: path separators, single- and double-dot names, NUL bytes, Windows-reserved device names, and Unicode invisible-format characters are all rejected. Names are NFKC-normalized before checks so fullwidth solidus and similar look-alikes cannot bypass validation.
