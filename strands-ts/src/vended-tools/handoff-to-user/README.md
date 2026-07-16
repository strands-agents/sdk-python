# handoff_to_user

Hands off to the user when the agent cannot proceed without human input. A thin
shim over the SDK's interrupt primitive:

1. The model calls `handoff_to_user` with a question and optional options.
2. The tool validates the input and raises an `InterruptError`, halting the
   agent with `stopReason: 'interrupt'`.
3. The caller inspects `result.interrupts`, renders the question in whatever
   UI it owns, collects the user's answer, and resumes the agent with an
   `InterruptResponseContent`.
4. On resume, the tool normalizes the response into a `HandoffAnswer` and
   returns it to the model.

The tool does not read from stdin, does not render anything, and does not talk
to a session store. Presentation is the consumer's job.

## Usage

```typescript
import { Agent, InterruptResponseContent } from '@strands-agents/sdk'
import { handoffToUser } from '@strands-agents/sdk/vended-tools/handoff-to-user'

const agent = new Agent({ tools: [handoffToUser] })

const result = await agent.invoke('Ask me which environment to deploy to.')

if (result.stopReason === 'interrupt') {
  const interrupt = result.interrupts[0]
  // interrupt.reason is the HandoffQuestion payload:
  //   { question, options, allow_free_text }
  const userAnswer = await promptUserSomehow(interrupt.reason)

  await agent.invoke([
    new InterruptResponseContent({
      interruptId: interrupt.id,
      response: { answer: userAnswer, chose: userAnswer },
    }),
  ])
}
```

## Event schema

The interrupt's `name` is always `strands:handoff-to-user`. Its `reason` field
carries the question payload:

```typescript
interface HandoffQuestion {
  question: string
  options: string[] | null
  allow_free_text: boolean
}
```

The caller resumes with either a bare string (interpreted as `answer`) or an
object matching:

```typescript
interface HandoffAnswer {
  answer: string
  chose?: string
}
```

Bare-string resumes are wrapped into `{ answer: <string> }` before the model
sees them. The same shape applies to the Python tool
(`strands.vended_tools.handoff_to_user`).

## Input schema

```typescript
interface HandoffToUserInput {
  question: string
  options?: string[]
  allow_free_text?: boolean // defaults to true
}
```

Validated at the tool boundary before the interrupt is raised:

- `question`: non-empty, at most 4096 characters.
- `options`: at most 20 entries, each non-empty (after trimming), at most 256
  characters, no duplicates (compared on trimmed value).
- At least one answer channel must be enabled: either `options` is provided,
  or `allow_free_text` is not explicitly `false`.

## When to reach for something else

- To gate a specific tool call on approval, use the `HumanInTheLoop`
  intervention (`@strands-agents/sdk/vended-interventions/hitl`). That is a
  before-tool policy layer, not an in-turn question.
- For interactive stdin/stdout, wire a consumer around this tool's interrupt.
  Do not fork the tool to add a readline reader.
