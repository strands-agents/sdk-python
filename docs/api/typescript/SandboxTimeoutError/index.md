Defined in: [src/sandbox/errors.ts:11](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/sandbox/errors.ts#L11)

Thrown by sandbox execution when the configured `timeout` elapses.

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new SandboxTimeoutError(seconds): SandboxTimeoutError;
```

Defined in: [src/sandbox/errors.ts:12](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/sandbox/errors.ts#L12)

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