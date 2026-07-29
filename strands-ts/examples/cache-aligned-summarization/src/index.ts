/**
 * Cache-aligned summarization with an idle-summarize trigger.
 *
 * This example demonstrates the `cacheAligned` option of
 * `SummarizingConversationManager`, plus a pattern it enables: proactively
 * summarizing the conversation while the agent is idle and the provider
 * prompt cache is still warm.
 *
 * Why idle time matters: provider prompt caches are short-lived — Anthropic's
 * cache TTL is ~5 minutes (the same applies to Anthropic models on Amazon
 * Bedrock, where cached content expires after 5 minutes of inactivity). A
 * cache-aligned summarization request reuses the agent's live system prompt,
 * tool specs, and message history byte-for-byte, so it reads the whole
 * conversation at the cached-token price — but only while the cache entry from
 * the last live turn is still alive. Summarizing at minute 3 of idleness pays
 * cached price for the history read; waiting until the user comes back an hour
 * later pays full input cost for every token.
 *
 * The idle trigger below:
 * - arms a timer when a turn completes and the agent is waiting for input
 * - cancels the timer as soon as the user starts a new turn
 * - never fires mid-turn (it is only armed between turns)
 * - skips summarization when the message tail is an assistant tool use still
 *   awaiting its result — the feature's documented fallback case, where the
 *   summarization instruction cannot be delivered without breaking the
 *   tool-use/result pair, so the manager falls back to a slice-based
 *   (cache-missing) summary anyway
 */
import * as readline from 'node:readline/promises'
import { env, stdin, stdout } from 'node:process'
import { Agent, BedrockModel, SummarizingConversationManager, tool, type Model } from '@strands-agents/sdk'
import { AnthropicModel } from '@strands-agents/sdk/models/anthropic'
import { z } from 'zod'

/**
 * How long the agent must sit idle before we summarize. Three minutes leaves a
 * comfortable margin inside the ~5 minute prompt-cache TTL: late enough that a
 * quick follow-up question usually cancels it, early enough that the cache
 * entry written by the last live turn is still warm when the summary request
 * reads the history. Override with IDLE_SUMMARIZE_MS (e.g. 15000) to try the
 * pattern without waiting three minutes.
 */
const IDLE_SUMMARIZE_MS = Number(env.IDLE_SUMMARIZE_MS ?? 3 * 60 * 1000)

/**
 * Pick a model provider. Both work with cache alignment:
 *
 * - Bedrock (default): `cacheConfig: { strategy: 'auto' }` makes the SDK place
 *   cache points automatically on the tool specs and the latest user message,
 *   so each live turn leaves a warm cache entry covering the whole prefix.
 * - Anthropic (MODEL_PROVIDER=anthropic, needs ANTHROPIC_API_KEY): the direct
 *   Anthropic provider has no automatic cache strategy — you place explicit
 *   `CachePointBlock` markers yourself. The idle-summarize wiring is identical;
 *   without cache points the summary still works but reads at uncached price.
 */
function createModel(): Model {
  if (env.MODEL_PROVIDER === 'anthropic') {
    return new AnthropicModel()
  }
  return new BedrockModel({
    cacheConfig: { strategy: 'auto' },
  })
}

const weatherTool = tool({
  name: 'get_weather',
  description: 'Get the current weather for a specific location.',
  inputSchema: z.object({
    location: z.string().describe('The city and state, e.g., San Francisco, CA'),
  }),
  callback: (input) => {
    return `The weather in ${input.location} is 72°F and sunny.`
  },
})

// Keep a reference to the conversation manager: the idle trigger calls its
// public reduce() directly, outside the agent loop.
//
// Do NOT pass a `model` override here — a different model cannot reuse the
// live model's prompt cache, so the manager skips the aligned path entirely
// and logs a warning if you combine the two.
const conversationManager = new SummarizingConversationManager({
  cacheAligned: true,
  // cacheAligned only takes effect on proactive compression. Reactive overflow
  // recovery always uses slice-based summarization: the cache-aligned request
  // is a superset of the live request, so on overflow it would overflow again.
  proactiveCompression: { compressionThreshold: 0.7 },
  // Small values so the demo summarizes after a handful of turns. Production
  // agents usually keep the defaults (summaryRatio 0.3, preserveRecentMessages 10).
  summaryRatio: 0.5,
  preserveRecentMessages: 4,
})

const agent = new Agent({
  model: createModel(),
  systemPrompt: 'You are a helpful, concise assistant. Use the get_weather tool when the user asks about the weather.',
  tools: [weatherTool],
  conversationManager,
})

/**
 * Returns true when the last message is an assistant tool use still awaiting
 * its tool result. The summarization instruction cannot be delivered at such
 * a tail, so the manager would fall back to slice-based summarization — no
 * cache reuse — and the idle pass skips instead, waiting for the next
 * completed turn.
 *
 * (Between completed turns this is normally false; it protects against idle
 * timers armed around interrupted or resumed sessions.)
 */
function tailAwaitsToolResult(): boolean {
  const last = agent.messages[agent.messages.length - 1]
  if (!last) {
    return false
  }
  const hasToolUse = last.content.some((block) => block.type === 'toolUseBlock')
  const hasToolResult = last.content.some((block) => block.type === 'toolResultBlock')
  return hasToolUse && !hasToolResult
}

/** Aborting this controller cancels the pending idle timer. */
let idleController: AbortController | undefined
/** Set while an idle summarization is in flight, so a new turn can await it. */
let idleTask: Promise<void> | undefined

/**
 * Arm the idle timer. Called only when a turn has completed and the agent is
 * waiting for human input — so the timer can never fire mid-turn.
 */
function armIdleSummarize(): void {
  idleController?.abort()
  const controller = new AbortController()
  idleController = controller

  const timer = setTimeout(() => {
    idleTask = summarizeWhileCacheIsWarm().finally(() => {
      idleTask = undefined
    })
  }, IDLE_SUMMARIZE_MS)

  controller.signal.addEventListener('abort', () => clearTimeout(timer), { once: true })
}

/** Cancel a pending (not yet fired) idle summarization. */
function cancelIdleSummarize(): void {
  idleController?.abort()
  idleController = undefined
}

/**
 * The idle pass: run the manager's proactive reduction directly while the
 * prompt cache written by the last live turn is still warm. Because `error`
 * is not set, this takes the proactive path — which is where cacheAligned
 * applies — and it is best-effort: reduce() returns false when there is not
 * enough history to summarize yet.
 */
async function summarizeWhileCacheIsWarm(): Promise<void> {
  if (tailAwaitsToolResult()) {
    console.log('\n[idle] tail is an unresolved tool use — skipping (cache alignment would fall back to slice-based)')
    return
  }

  const before = agent.messages.length
  console.log(`\n[idle] ${IDLE_SUMMARIZE_MS / 1000}s idle — summarizing while the prompt cache is warm...`)
  try {
    const reduced = await conversationManager.reduce({ agent, model: agent.model })
    if (reduced) {
      console.log(`[idle] summarized: ${before} -> ${agent.messages.length} messages`)
    } else {
      console.log('[idle] nothing to summarize yet')
    }
  } catch (error) {
    console.log(`[idle] summarization failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}

async function main(): Promise<void> {
  console.log('Cache-aligned idle summarization demo')
  console.log(`Idle timer: ${IDLE_SUMMARIZE_MS / 1000}s (override with IDLE_SUMMARIZE_MS)`)
  console.log('Chat with the agent; leave it idle to see the summarization fire. Type "exit" to quit.\n')

  const rl = readline.createInterface({ input: stdin, output: stdout })

  for (;;) {
    const line = (await rl.question('you> ')).trim()

    // The human is back: cancel the pending idle summarization...
    cancelIdleSummarize()
    // ...and if one already fired and is mid-flight, let it finish before the
    // new turn mutates the message history.
    if (idleTask) {
      await idleTask
    }

    if (line === 'exit' || line === 'quit') {
      break
    }
    if (line.length === 0) {
      continue
    }

    const result = await agent.invoke(line)

    // cacheWrite on a turn means the provider stored the prefix; cacheRead on
    // the next request (including an idle summarization) means it was reused.
    const usage = result.metrics?.accumulatedUsage
    if (usage) {
      console.log(
        `\n[usage] input=${usage.inputTokens} output=${usage.outputTokens} ` +
          `cacheRead=${usage.cacheReadInputTokens ?? 0} cacheWrite=${usage.cacheWriteInputTokens ?? 0} ` +
          `| ${agent.messages.length} messages in history\n`
      )
    }

    // Turn complete, waiting for the human again — start the idle clock.
    armIdleSummarize()
  }

  cancelIdleSummarize()
  rl.close()
}

await main().catch(console.error)
