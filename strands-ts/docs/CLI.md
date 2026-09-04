# CLI and Terminal Output

The SDK ships two layers of terminal output:

1. **`CliPrinter`** (`@strands-agents/sdk/cli`) — a drop-in replacement for the default `AgentPrinter` that adds color theming (Chalk), Markdown rendering, and spinners. Attach it by setting `_printer` on an agent created with `printer: false`:

   ```typescript
   import { Agent } from '@strands-agents/sdk'
   import { CliPrinter } from '@strands-agents/sdk/cli'

   const agent = new Agent({ printer: false })
   ;(agent as unknown as { _printer: CliPrinter })._printer = new CliPrinter()
   ```

2. **`strands`** — a command-line entry point:

   ```
   strands                     # interactive REPL
   strands run "prompt"        # one-shot invocation
   strands --help
   ```

## Plain-text and machine-readable output

All styling degrades automatically so output stays usable when piped or logged:

- ANSI codes are disabled when stdout is not a TTY (Chalk auto-detection) or when `NO_COLOR` is set.
- `--plain` (or `STRANDS_PLAIN=1`) forces raw text; Markdown is emitted in a readable unstyled form.
- `--json` emits stream events as newline-delimited JSON with no ANSI codes, for scripting.
- Errors go to stderr; content goes to stdout.
