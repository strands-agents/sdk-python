Defined in: [src/sandbox/errors.ts:11](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/sandbox/errors.ts#L11)

Thrown by sandbox execution when the configured `timeout` elapses.

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new SandboxTimeoutError(seconds): SandboxTimeoutError;
```

Defined in: [src/sandbox/errors.ts:12](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/sandbox/errors.ts#L12)

#### Parameters

| Parameter | Type |
| --- | --- |
| `seconds` | `number` |

#### Returns

`SandboxTimeoutError`

#### Overrides

```ts
Error.constructor
```