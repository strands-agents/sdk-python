Defined in: [src/retry/default-model-retry-strategy.ts:70](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/default-model-retry-strategy.ts#L70)

Retries failed model calls classified by the SDK as retryable.

Today, only [ModelThrottledError](/docs/api/typescript/ModelThrottledError/index.md) is treated as retryable — subclass and override [isRetryable](#isretryable) to expand or narrow that set without reimplementing the rest of the retry policy.

State is per-turn: backoff timing state resets in [onFirstModelAttempt](#onfirstmodelattempt), which the base class calls when `event.attemptCount === 1`. The attempt counter itself is owned by the agent loop and read off [AfterModelCallEvent.attemptCount](/docs/api/typescript/AfterModelCallEvent/index.md#attemptcount).

Hook precedence: [AfterModelCallEvent](/docs/api/typescript/AfterModelCallEvent/index.md) fires hooks in reverse registration order, so user-registered hooks run before this strategy. If a user hook sets `event.retry = true` first, the base class returns early and does not stack additional backoff on top.

Sharing: a given instance tracks its own backoff state and must not be shared across multiple agents. Create a separate instance per agent.

## Example

```ts
const agent = new Agent({
  model,
  retryStrategy: new DefaultModelRetryStrategy({ maxAttempts: 4 }),
})
```

## Extends

-   [`ModelRetryStrategy`](/docs/api/typescript/ModelRetryStrategy/index.md)

## Constructors

### Constructor

```ts
new DefaultModelRetryStrategy(opts?): DefaultModelRetryStrategy;
```

Defined in: [src/retry/default-model-retry-strategy.ts:79](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/default-model-retry-strategy.ts#L79)

#### Parameters

| Parameter | Type |
| --- | --- |
| `opts` | [`DefaultModelRetryStrategyOptions`](/docs/api/typescript/DefaultModelRetryStrategyOptions/index.md) |

#### Returns

`DefaultModelRetryStrategy`

#### Overrides

[`ModelRetryStrategy`](/docs/api/typescript/ModelRetryStrategy/index.md).[`constructor`](/docs/api/typescript/ModelRetryStrategy/index.md#constructor)

## Properties

### name

```ts
readonly name: string = 'strands:default-model-retry-strategy';
```

Defined in: [src/retry/default-model-retry-strategy.ts:71](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/default-model-retry-strategy.ts#L71)

A stable string identifier for this retry strategy.

#### Overrides

[`ModelRetryStrategy`](/docs/api/typescript/ModelRetryStrategy/index.md).[`name`](/docs/api/typescript/ModelRetryStrategy/index.md#name)

## Methods

### isRetryable()

```ts
protected isRetryable(error): boolean;
```

Defined in: [src/retry/default-model-retry-strategy.ts:94](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/default-model-retry-strategy.ts#L94)

Whether `error` should be retried. Override to extend or narrow the retryable set (e.g. to also retry transient 5xx errors).

#### Parameters

| Parameter | Type |
| --- | --- |
| `error` | `Error` |

#### Returns

`boolean`

---

### computeRetryDecision()

```ts
protected computeRetryDecision(event): RetryDecision;
```

Defined in: [src/retry/default-model-retry-strategy.ts:98](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/default-model-retry-strategy.ts#L98)

Decide whether to retry the failed model call, and how long to wait first.

Called only for error events that have not already been marked for retry by another hook. The base class has already filtered out successes and short-circuited events where `event.retry` is true, so implementations only need to reason about `event.error`.

Return `{ retry: false }` to let the error propagate. Return `{ retry: true, waitMs }` to retry after sleeping for `waitMs` milliseconds.

#### Parameters

| Parameter | Type |
| --- | --- |
| `event` | [`AfterModelCallEvent`](/docs/api/typescript/AfterModelCallEvent/index.md) |

#### Returns

[`RetryDecision`](/docs/api/typescript/RetryDecision/index.md)

#### Overrides

[`ModelRetryStrategy`](/docs/api/typescript/ModelRetryStrategy/index.md).[`computeRetryDecision`](/docs/api/typescript/ModelRetryStrategy/index.md#computeretrydecision)

---

### onFirstModelAttempt()

```ts
protected onFirstModelAttempt(): void;
```

Defined in: [src/retry/default-model-retry-strategy.ts:126](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/default-model-retry-strategy.ts#L126)

Called when `event.attemptCount === 1`, i.e. at the start of a fresh turn. Subclasses with per-turn state override this to clear it; the default is a no-op.

The agent loop guarantees `attemptCount === 1` on every new turn, so this is a reliable turn-boundary signal.

#### Returns

`void`

#### Overrides

[`ModelRetryStrategy`](/docs/api/typescript/ModelRetryStrategy/index.md).[`onFirstModelAttempt`](/docs/api/typescript/ModelRetryStrategy/index.md#onfirstmodelattempt)

---

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/retry/model-retry-strategy.ts:99](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/retry/model-retry-strategy.ts#L99)

Initialize the retry strategy with the agent instance.

Enforces the single-agent attachment guard and registers the [AfterModelCallEvent](/docs/api/typescript/AfterModelCallEvent/index.md) hook that drives retry orchestration.

Subclasses that override this method MUST call `super.initAgent(agent)` to preserve the attachment guard and hook registration. Additional hooks may be registered after the `super` call.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent to register hooks with |

#### Returns

`void`

#### Inherited from

[`ModelRetryStrategy`](/docs/api/typescript/ModelRetryStrategy/index.md).[`initAgent`](/docs/api/typescript/ModelRetryStrategy/index.md#initagent)