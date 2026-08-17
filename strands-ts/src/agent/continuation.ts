import { AfterInvocationEvent, BeforeModelCallEvent, MessageAddedEvent } from '../hooks/events.js'
import { logger } from '../logging/logger.js'
import { Message } from '../types/messages.js'

import type { AgentResult, InvokeArgs, InvocationState, LocalAgent } from '../types/agent.js'
import type { StopReason } from '../types/messages.js'

/**
 * One internal input contribution to an agent or model invocation.
 *
 * When both settlement callbacks are supplied, exactly one runs once. Callback
 * failures are logged and do not change the agent result.
 *
 * @internal
 */
export interface ContinuationInput {
  /** Input that normalizes to one or more complete messages. */
  readonly args: InvokeArgs
  /** Runs after the input is committed. */
  readonly onConsumed?: () => void | Promise<void>
  /** Runs when the input cannot be included in a successful model-stage result. */
  readonly onRejected?: (reason: unknown) => void | Promise<void>
}

/**
 * Prepared internal input for one agent or model invocation.
 *
 * @internal
 */
export interface InvocationContinuation {
  args: InvokeArgs
  internal: readonly {
    readonly input: ContinuationInput
    readonly messages: readonly Message[]
  }[]
  readonly publicResume: boolean
  publicMessages: readonly Message[]
  requestGroups: readonly (readonly Message[])[]
  settled: boolean
}

const pendingInputs = new WeakMap<AfterInvocationEvent | BeforeModelCallEvent, ContinuationInput[]>()
const REJECTION_STOP_REASONS: ReadonlySet<StopReason> = new Set([
  'cancelled',
  'checkpoint',
  'interrupt',
  'limitTurns',
  'limitTotalTokens',
  'limitOutputTokens',
])

/**
 * Internal continuation operations used by the agent loop.
 *
 * @internal
 */
export const continuations = {
  add,
  buildRequest,
  cleanup,
  isRejectionStopReason,
  prepareInvocation,
  publish,
  rejectInputs,
  rejectInvocation,
  requestSourceIds,
  settleResult,
  take,
  updateArgs,
}

function add(event: AfterInvocationEvent | BeforeModelCallEvent, input: ContinuationInput): void {
  const inputs = pendingInputs.get(event) ?? []
  inputs.push(input)
  pendingInputs.set(event, inputs)
}

function take(event: AfterInvocationEvent | BeforeModelCallEvent): readonly ContinuationInput[] {
  const inputs = pendingInputs.get(event) ?? []
  pendingInputs.delete(event)
  return inputs
}

function isRejectionStopReason(stopReason: StopReason): boolean {
  return REJECTION_STOP_REASONS.has(stopReason)
}

async function prepareInvocation(
  inputs: readonly ContinuationInput[],
  resume: InvokeArgs | undefined,
  normalizeInput: (args: InvokeArgs) => Message[]
): Promise<InvocationContinuation | undefined> {
  const internal = await prepareInputs(inputs, normalizeInput)
  if (internal.length === 0 && resume === undefined) return undefined

  const invocation: InvocationContinuation = {
    args: resume ?? [],
    internal,
    publicResume: resume !== undefined,
    publicMessages: [],
    requestGroups: [],
    settled: false,
  }
  if (resume !== undefined) {
    await updateArgs(invocation, resume, normalizeInput)
  }
  return invocation
}

async function updateArgs(
  invocation: InvocationContinuation,
  args: InvokeArgs,
  normalizeInput: (args: InvokeArgs) => Message[]
): Promise<InvokeArgs> {
  try {
    invocation.publicMessages = normalizeInput(args)
  } catch (error) {
    await rejectInvocation(invocation, error)
    throw error
  }

  const emptyInput = Array.isArray(args) && args.length === 0
  if (invocation.internal.length > 0 && !emptyInput && !isComposableUserInput(invocation.publicMessages)) {
    await rejectPrepared(
      invocation.internal,
      new TypeError('Continuation input cannot be combined with this public resume')
    )
    invocation.internal = []
  }

  invocation.args = invocation.publicMessages.length > 0 ? [...invocation.publicMessages] : args
  return invocation.args
}

function buildRequest(
  history: readonly Message[],
  ...invocations: readonly (InvocationContinuation | undefined)[]
): Message[] {
  const activeInvocations = invocations.filter(
    (invocation): invocation is InvocationContinuation =>
      invocation !== undefined && invocation.internal.length > 0 && !invocation.settled
  )

  let sourceMessages = [...history]
  for (const invocation of activeInvocations) {
    const publicIds = new Set(invocation.publicMessages.map((message) => message.trackingId))
    sourceMessages = [
      ...sourceMessages.filter((message) => !publicIds.has(message.trackingId)),
      ...invocation.internal.flatMap((input) => input.messages),
      ...invocation.publicMessages,
    ]
  }

  const requestGroups = groupMessages(sourceMessages)
  const messages = requestGroups.map((group) => {
    const last = group.at(-1)!
    return group.length === 1
      ? last
      : new Message({
          role: last.role,
          content: group.flatMap((message) => message.content),
          trackingId: last.trackingId,
        })
  })
  for (const invocation of activeInvocations) {
    invocation.requestGroups = requestGroups.map((group) => group.map((message) => message.clone()))
  }
  return messages
}

function requestSourceIds(
  requestMessage: Message,
  request: readonly Message[],
  ...invocations: readonly (InvocationContinuation | undefined)[]
): ReadonlySet<string> | undefined {
  const sourceIds = new Set<string>()
  for (const invocation of invocations) {
    if (!invocation) continue
    const group = matchRequestGroups(invocation, request).get(requestMessage.trackingId)
    if (!group) continue
    for (const sourceId of matchedSourceIds(group, serializedContent(requestMessage), true)) sourceIds.add(sourceId)
  }
  return sourceIds.size > 0 ? sourceIds : undefined
}

async function publish(
  agent: LocalAgent,
  invocation: InvocationContinuation | undefined,
  modelRequests: readonly (readonly Message[])[] | undefined,
  invocationState: InvocationState,
  insertBeforeTrackingId?: string,
  includePublic = true
): Promise<readonly MessageAddedEvent[]> {
  if (!invocation || invocation.settled) return []

  const prepared = invocation.internal
  const acceptedInputs = findAcceptedInputs(invocation, modelRequests ?? [])
  const accepted = prepared.filter((entry) => acceptedInputs.has(entry))
  invocation.settled = true
  const inserted = integrateMessages(agent, invocation, accepted, insertBeforeTrackingId, includePublic)

  await rejectPrepared(
    prepared.filter((entry) => !acceptedInputs.has(entry)),
    new Error('Continuation was not retained in the successful model request')
  )
  await notifyConsumed(accepted)

  return inserted.map((message) => new MessageAddedEvent({ agent, message, invocationState }))
}

async function settleResult(
  agent: LocalAgent,
  invocation: InvocationContinuation | undefined,
  result: AgentResult,
  invocationState: InvocationState,
  passRanAgentLoop: boolean
): Promise<readonly MessageAddedEvent[]> {
  if (!invocation || invocation.settled) return []
  if (isRejectionStopReason(result.stopReason)) {
    await rejectInvocation(invocation, new Error(`Continuation stopped by ${result.stopReason}`))
    return []
  }
  if (!passRanAgentLoop && invocation.internal.length === 0) {
    invocation.settled = true
    return []
  }

  const events = [
    ...(await publish(
      agent,
      invocation,
      undefined,
      invocationState,
      passRanAgentLoop ? result.lastMessage.trackingId : undefined,
      passRanAgentLoop
    )),
  ]
  if (!agent.messages.some((message) => message.trackingId === result.lastMessage.trackingId)) {
    agent.messages.push(result.lastMessage)
    events.push(new MessageAddedEvent({ agent, message: result.lastMessage, invocationState }))
  }

  return events
}

async function rejectInvocation(invocation: InvocationContinuation | undefined, reason: unknown): Promise<void> {
  if (!invocation || invocation.settled) return
  invocation.settled = true
  await rejectPrepared(invocation.internal, reason)
}

async function rejectInputs(inputs: readonly ContinuationInput[], reason: unknown): Promise<void> {
  for (const input of inputs) {
    try {
      await input.onRejected?.(reason)
    } catch (error) {
      logger.warn(`error=<${error}> | continuation rejection callback failed`)
    }
  }
}

async function cleanup(
  invocation: InvocationContinuation | undefined,
  inputs: readonly ContinuationInput[] = []
): Promise<void> {
  const reason = new Error('Agent stream closed before continuation consumption')
  await rejectInvocation(invocation, reason)
  await rejectInputs(inputs, reason)
}

async function prepareInputs(
  inputs: readonly ContinuationInput[],
  normalizeInput: (args: InvokeArgs) => Message[]
): Promise<InvocationContinuation['internal']> {
  const prepared: Array<InvocationContinuation['internal'][number]> = []

  for (const input of inputs) {
    try {
      const messages = normalizeInput(input.args)
      if (!isCompleteMessageInput(messages)) {
        throw new TypeError('Continuation input must contain a complete message sequence')
      }
      prepared.push({ input, messages })
    } catch (error) {
      await rejectInputs([input], error)
    }
  }
  return prepared
}

function isComposableUserInput(messages: readonly Message[]): boolean {
  return (
    messages.length > 0 &&
    messages.every(
      (message) =>
        message.role === 'user' &&
        message.content.every((block) => block.type !== 'toolUseBlock' && block.type !== 'toolResultBlock')
    )
  )
}

function isCompleteMessageInput(messages: readonly Message[]): boolean {
  if (messages.length === 0) return false

  let pendingToolUseIds = new Set<string>()
  for (const message of messages) {
    if (pendingToolUseIds.size > 0 && message.role !== 'user') return false

    const nextToolUseIds = new Set<string>()
    for (const block of message.content) {
      if (block.type === 'toolUseBlock') {
        if (message.role !== 'assistant' || nextToolUseIds.has(block.toolUseId)) return false
        nextToolUseIds.add(block.toolUseId)
      } else if (block.type === 'toolResultBlock') {
        if (message.role !== 'user' || !pendingToolUseIds.delete(block.toolUseId)) return false
      }
    }

    if (message.role === 'user' && pendingToolUseIds.size > 0) return false
    pendingToolUseIds = nextToolUseIds
  }
  return pendingToolUseIds.size === 0
}

function findAcceptedInputs(
  invocation: InvocationContinuation,
  modelRequests: readonly (readonly Message[])[]
): ReadonlySet<InvocationContinuation['internal'][number]> {
  const accepted = new Set<InvocationContinuation['internal'][number]>()
  if (modelRequests.length === 0) {
    for (const entry of invocation.internal) accepted.add(entry)
    return accepted
  }
  for (const request of modelRequests) {
    const sourceIds = retainedSourceIds(invocation, request)
    for (const entry of invocation.internal) {
      if (!accepted.has(entry) && entry.messages.every((message) => sourceIds.has(message.trackingId))) {
        accepted.add(entry)
      }
    }
  }
  return accepted
}

function retainedSourceIds(invocation: InvocationContinuation, request: readonly Message[]): ReadonlySet<string> {
  const retained = new Set<string>()
  const groups = matchRequestGroups(invocation, request)
  for (const message of request) {
    const group = groups.get(message.trackingId)
    if (!group) continue
    for (const sourceId of matchedSourceIds(group, serializedContent(message), false)) retained.add(sourceId)
  }
  return retained
}

function matchRequestGroups(
  invocation: InvocationContinuation,
  request: readonly Message[]
): ReadonlyMap<string, readonly Message[]> {
  const matched = new Map<string, readonly Message[]>()
  const used = new Set<number>()
  let nextIndex = 0

  for (const message of request) {
    let index = invocation.requestGroups.findIndex(
      (group, groupIndex) => !used.has(groupIndex) && group.at(-1)?.trackingId === message.trackingId
    )
    if (index < 0) {
      const content = serializedContent(message)
      index = invocation.requestGroups.findIndex(
        (group, groupIndex) =>
          groupIndex >= nextIndex &&
          group[0]?.role === message.role &&
          group.some((source) => findSequence(content, serializedContent(source), 0) >= 0)
      )
    }
    if (index < 0) continue
    used.add(index)
    nextIndex = Math.max(nextIndex, index + 1)
    matched.set(message.trackingId, invocation.requestGroups[index]!)
  }
  return matched
}

function serializedContent(message: Message): string[] {
  return message.content.map((block) => JSON.stringify(block.toJSON()))
}

function matchedSourceIds(
  group: readonly Message[],
  content: readonly string[],
  includeAmbiguous: boolean
): ReadonlySet<string> {
  const sourceGroups = new Map<string, { content: readonly string[]; count: number }>()
  for (const source of group) {
    const sourceContent = serializedContent(source)
    const key = JSON.stringify(sourceContent)
    const existing = sourceGroups.get(key)
    sourceGroups.set(key, { content: sourceContent, count: (existing?.count ?? 0) + 1 })
  }
  const ambiguous = new Set(
    [...sourceGroups].flatMap(([key, sourceGroup]) => {
      const occurrences = countSequence(content, sourceGroup.content)
      return sourceGroup.count > 1 && occurrences > 0 && occurrences < sourceGroup.count ? [key] : []
    })
  )

  const matched = new Set<string>()
  let contentIndex = 0
  for (const source of group) {
    const sourceContent = serializedContent(source)
    const key = JSON.stringify(sourceContent)
    if (ambiguous.has(key)) {
      if (includeAmbiguous) matched.add(source.trackingId)
      continue
    }
    const sourceIndex = findSequence(content, sourceContent, contentIndex)
    if (sourceIndex < 0) continue
    matched.add(source.trackingId)
    contentIndex = sourceIndex + sourceContent.length
  }
  return matched
}

function countSequence(haystack: readonly string[], needle: readonly string[]): number {
  let count = 0
  let index = 0
  while ((index = findSequence(haystack, needle, index)) >= 0) {
    count += 1
    index += needle.length
  }
  return count
}

function findSequence(haystack: readonly string[], needle: readonly string[], start: number): number {
  if (needle.length === 0) return -1
  for (let index = start; index <= haystack.length - needle.length; index++) {
    if (needle.every((value, offset) => haystack[index + offset] === value)) return index
  }
  return -1
}

function integrateMessages(
  agent: LocalAgent,
  invocation: InvocationContinuation,
  accepted: InvocationContinuation['internal'],
  insertBeforeTrackingId?: string,
  includePublic = true
): readonly Message[] {
  const publicMessages = includePublic ? invocation.publicMessages : []
  const publicIds = new Set(publicMessages.map((message) => message.trackingId))
  const existingPublicIds = new Set(
    agent.messages.filter((message) => publicIds.has(message.trackingId)).map((message) => message.trackingId)
  )
  const publicIndex = agent.messages.findIndex((message) => publicIds.has(message.trackingId))
  if (accepted.length === 0 && (publicMessages.length === 0 || publicIndex >= 0)) return []

  const baseMessages = agent.messages.filter((message) => !publicIds.has(message.trackingId))
  let insertionIndex = baseMessages.length
  if (publicIndex >= 0) {
    insertionIndex = agent.messages.slice(0, publicIndex).filter((message) => !publicIds.has(message.trackingId)).length
  } else if (insertBeforeTrackingId !== undefined) {
    const resultIndex = baseMessages.findIndex((message) => message.trackingId === insertBeforeTrackingId)
    if (resultIndex >= 0) insertionIndex = resultIndex
  }

  const internalMessages = accepted.flatMap((entry) => entry.messages)
  const messages = [...internalMessages, ...publicMessages]
  baseMessages.splice(insertionIndex, 0, ...messages)
  agent.messages.splice(0, agent.messages.length, ...baseMessages)
  return [...internalMessages, ...publicMessages.filter((message) => !existingPublicIds.has(message.trackingId))]
}

function groupMessages(messages: readonly Message[]): Message[][] {
  const groups: Message[][] = []
  for (const message of messages) {
    const group = groups.at(-1)
    if (group?.[0]?.role === message.role) {
      group.push(message)
    } else {
      groups.push([message])
    }
  }
  return groups
}

async function rejectPrepared(inputs: InvocationContinuation['internal'], reason: unknown): Promise<void> {
  await rejectInputs(
    inputs.map((entry) => entry.input),
    reason
  )
}

async function notifyConsumed(entries: InvocationContinuation['internal']): Promise<void> {
  for (const entry of entries) {
    try {
      await entry.input.onConsumed?.()
    } catch (error) {
      logger.warn(`error=<${error}> | continuation consumption callback failed`)
    }
  }
}
