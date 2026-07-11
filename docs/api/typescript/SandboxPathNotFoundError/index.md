Defined in: [src/sandbox/errors.ts:32](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/sandbox/errors.ts#L32)

Thrown by [Sandbox.listFiles](/docs/api/typescript/Sandbox/index.md#listfiles) when the path does not exist, distinguishing genuine absence from permission or transport failures (which throw plain errors).

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new SandboxPathNotFoundError(path): SandboxPathNotFoundError;
```

Defined in: [src/sandbox/errors.ts:33](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/sandbox/errors.ts#L33)

#### Parameters

| Parameter | Type |
| --- | --- |
| `path` | `string` |

#### Returns

`SandboxPathNotFoundError`

#### Overrides

```ts
Error.constructor
```