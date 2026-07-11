Defined in: [src/retry/model-retry-strategy.ts:33](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/model-retry-strategy.ts#L33)

Abstract base class for model-retry strategies.

A ModelRetryStrategy is a [Plugin](/docs/api/typescript/Plugin/index.md) that retries failed model calls. Subclasses implement [computeRetryDecision](#computeretrydecision) to answer *whether* to retry and *how long* to wait; the base class orchestrates the rest:

1.  Short-circuits if another hook already set `event.retry` (no stacked delay).
2.  Short-circuits on success events (`event.error === undefined`).
3.  Calls [onFirstModelAttempt](#onfirstmodelattempt) on turn boundaries (`event.attemptCount === 1`), letting stateful subclasses clear per-turn state.
4.  Invokes [computeRetryDecision](#computeretrydecision); on `retry: true`, sleeps for `waitMs` then sets `event.retry = true`.

Other retry kinds (e.g. tool retries) will land as *sibling* abstract classes, not as additional methods on this one — different retry kinds have different unit-of-work boundaries and don’t share a single state contract.

Single-agent attachment: instances typically carry per-turn state, so sharing one instance across agents would let their calls trample each other. The base class throws on attempts to attach to a different agent.

## Extended by

-   [`DefaultModelRetryStrategy`](/docs/api/typescript/DefaultModelRetryStrategy/index.md)

## Implements

-   [`Plugin`](/docs/api/typescript/Plugin/index.md)

## Constructors

### Constructor

```ts
new ModelRetryStrategy(): ModelRetryStrategy;
```

#### Returns

`ModelRetryStrategy`

## Properties

### name

```ts
abstract readonly name: string;
```

Defined in: [src/retry/model-retry-strategy.ts:37](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/model-retry-strategy.ts#L37)

A stable string identifier for this retry strategy.

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`name`](/docs/api/typescript/Plugin/index.md#name)

## Methods

### computeRetryDecision()

```ts
abstract protected computeRetryDecision(event):
  | RetryDecision
| Promise<RetryDecision>;
```

Defined in: [src/retry/model-retry-strategy.ts:53](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/model-retry-strategy.ts#L53)

Decide whether to retry the failed model call, and how long to wait first.

Called only for error events that have not already been marked for retry by another hook. The base class has already filtered out successes and short-circuited events where `event.retry` is true, so implementations only need to reason about `event.error`.

Return `{ retry: false }` to let the error propagate. Return `{ retry: true, waitMs }` to retry after sleeping for `waitMs` milliseconds.

#### Parameters

| Parameter | Type |
| --- | --- |
| `event` | [`AfterModelCallEvent`](/docs/api/typescript/AfterModelCallEvent/index.md) |

#### Returns

| [`RetryDecision`](/docs/api/typescript/RetryDecision/index.md) | `Promise`<[`RetryDecision`](/docs/api/typescript/RetryDecision/index.md)\>

---

### onFirstModelAttempt()

```ts
protected onFirstModelAttempt(): void;
```

Defined in: [src/retry/model-retry-strategy.ts:63](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/model-retry-strategy.ts#L63)

Called when `event.attemptCount === 1`, i.e. at the start of a fresh turn. Subclasses with per-turn state override this to clear it; the default is a no-op.

The agent loop guarantees `attemptCount === 1` on every new turn, so this is a reliable turn-boundary signal.

#### Returns

`void`

---

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/retry/model-retry-strategy.ts:99](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/model-retry-strategy.ts#L99)

Initialize the retry strategy with the agent instance.

Enforces the single-agent attachment guard and registers the [AfterModelCallEvent](/docs/api/typescript/AfterModelCallEvent/index.md) hook that drives retry orchestration.

Subclasses that override this method MUST call `super.initAgent(agent)` to preserve the attachment guard and hook registration. Additional hooks may be registered after the `super` call.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent to register hooks with |

#### Returns

`void`

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`initAgent`](/docs/api/typescript/Plugin/index.md#initagent)