# Strands Agents — Cache-Aligned Idle Summarization Example

Proactively summarize a conversation while the agent sits idle — using the
`cacheAligned` option of `SummarizingConversationManager` so the summarization
request reads the history from the provider prompt cache instead of paying
full input price for it.

## The idea

Provider prompt caches are short-lived: Anthropic's prompt cache TTL is about
5 minutes, and the same applies to Anthropic models on Amazon Bedrock, where
cached content expires after 5 minutes of inactivity.

`cacheAligned: true` makes proactive summarization reuse the agent's live
system prompt, tool specs, and message history so the request prefix is
byte-identical to the live conversation's request — the provider serves the
whole history read from its prompt cache. But that only works while the cache
entry written by the last live turn is still warm.

That makes idle time the perfect moment to summarize:

- **Summarize at minute 3 of idleness** → the cache is still warm, the history
  is read at the cached-token price (roughly a 10x discount on those tokens).
- **Wait until the user comes back an hour later** → the cache has expired, and
  summarization (or the next live turn against the full history) pays full
  input cost for every token.

This example wires a 3-minute idle timer around a small interactive chat
agent. After each completed turn, the timer is armed; if the user replies
first, it is cancelled. When it fires, it runs the conversation manager's
proactive reduction directly — compressing the oldest messages into a summary
while the read is still cheap, so the *next* turn starts from a smaller
history.

The trigger is careful about two things:

- **It never fires mid-turn.** The timer is only armed while the agent is
  waiting for human input, and is cancelled as soon as the user submits the
  next message. If the user replies while a summarization is already in
  flight, the new turn waits for it to finish before running.
- **It skips an unresolved tool-use tail.** If the last message is an
  assistant tool use still awaiting its result, the summarization instruction
  can't be delivered without breaking the tool-use/result pair — the manager's
  documented fallback to slice-based (cache-missing) summarization — so the
  idle pass skips instead of paying for an unaligned summary.

## Prerequisites

- Node.js 20+
- AWS credentials configured (for the default Bedrock model provider), or an
  Anthropic API key (see below)

## Quick Start

```bash
npm install
npm start
```

Chat with the agent for a few turns, then leave it idle. By default the timer
is 3 minutes; to see the pattern without waiting, shorten it:

```bash
IDLE_SUMMARIZE_MS=15000 npm start
```

Watch the `[usage]` line after each turn — `cacheWrite` shows the provider
storing the prefix, `cacheRead` shows it being reused — and the `[idle]` lines
when the timer fires.

## Model providers

- **Amazon Bedrock (default)**: uses `cacheConfig: { strategy: 'auto' }`, so
  the SDK places prompt-cache points automatically and every live turn leaves
  a warm cache entry.
- **Anthropic**: run with `MODEL_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
  set. The direct Anthropic provider has no automatic cache strategy — you
  place explicit `CachePointBlock` markers yourself. The idle-summarize wiring
  is identical; without cache points the summary still works but reads at the
  uncached price.

## Configuration shown

```typescript
const conversationManager = new SummarizingConversationManager({
  cacheAligned: true,
  proactiveCompression: { compressionThreshold: 0.7 },
  summaryRatio: 0.5,
  preserveRecentMessages: 4,
})
```

Two things to keep in mind when enabling `cacheAligned`:

- Don't pass a `model` override — a different model cannot reuse the live
  model's prompt cache, so the manager skips the aligned path entirely (and
  logs a warning) when the two are combined.
- `cacheAligned` only takes effect on proactive compression. Reactive overflow
  recovery always uses slice-based summarization, because the cache-aligned
  request is a superset of the live request — on overflow it would overflow
  again.

See the [Conversation Management documentation](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/conversation-management/)
for the full feature reference.
