```ts
type HookableEventConstructor<T> = (...args) => T;
```

Defined in: [src/hooks/types.ts:7](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/types.ts#L7)

Type for a constructor function that creates HookableEvent instances.

## Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `T` *extends* [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md) | [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md) |

## Parameters

| Parameter | Type |
| --- | --- |
| …`args` | `any`\[\] |

## Returns

`T`