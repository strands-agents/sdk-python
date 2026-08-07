Defined in: [src/errors.ts:103](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/errors.ts#L103)

Error thrown when attempting to invoke an agent that is already processing an invocation.

This error indicates that invoke() or stream() was called while the agent is already executing. Agents can only process one invocation at a time to prevent state corruption.

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new ConcurrentInvocationError(message): ConcurrentInvocationError;
```

Defined in: [src/errors.ts:109](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/errors.ts#L109)

Creates a new ConcurrentInvocationError.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `message` | `string` | Error message describing the concurrent invocation attempt |

#### Returns

`ConcurrentInvocationError`

#### Overrides

```ts
Error.constructor
```