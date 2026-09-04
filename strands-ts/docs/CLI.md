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

## Model providers via environment

The `strands` CLI selects its model from the environment. Copy `strands-ts/.env.example`
to `.env` in the directory you run from and fill in the values you need.

Selection order is: `--provider` flag → `STRANDS_PROVIDER` → auto-detect
from `GROQ_API_KEY` then `OPENAI_API_KEY` → the SDK's default model (Bedrock).
The generic `PROVIDER` variable is deliberately ignored — it belongs to other tooling.

| Provider  | API key          | Model (default)                          | Base URL                                                     |
| --------- | ---------------- | ---------------------------------------- | ------------------------------------------------------------ |
| `groq`    | `GROQ_API_KEY`   | `GROQ_MODEL` (`llama-3.3-70b-versatile`) | `GROQ_API_BASE_URL` (`https://api.groq.com/openai/v1`)       |
| `openai`  | `OPENAI_API_KEY` | `OPENAI_MODEL`                           | `OPENAI_BASE_URL` (optional — any OpenAI-compatible gateway) |
| `bedrock` | — (SDK default)  | —                                        | —                                                            |

Examples:

```bash
# Groq
GROQ_API_KEY=gsk_... strands run "Explain vector databases"

# OpenAI
PROVIDER=openai OPENAI_API_KEY=sk-... strands

# A generic OpenAI-compatible endpoint
PROVIDER=openai OPENAI_API_KEY=... OPENAI_BASE_URL=https://gateway.example.com/v1 strands
```

Groq and any `OPENAI_BASE_URL` gateway require `api: 'chat'` (OpenAI's Responses API is
not implemented by these gateways); the CLI configures this automatically. A selected
provider with a missing key fails fast and names the exact env var to set. API keys are
never logged — the startup line shows only provider, model id, and base URL.
