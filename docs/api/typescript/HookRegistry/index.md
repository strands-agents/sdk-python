Defined in: [src/hooks/registry.ts:38](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/registry.ts#L38)

Implementation of the hook registry for managing hook callbacks. Maintains mappings between event types and callback functions.

## Implements

-   `HookRegistry`

## Constructors

### Constructor

```ts
new HookRegistry(): HookRegistryImplementation;
```

Defined in: [src/hooks/registry.ts:41](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/registry.ts#L41)

#### Returns

`HookRegistryImplementation`

## Methods

### addCallback()

```ts
addCallback<T>(
   eventType,
   callback,
   options?): HookCleanup;
```

Defined in: [src/hooks/registry.ts:46](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/registry.ts#L46)

HookRegistry.addCallback

#### Type Parameters

| Type Parameter |
| --- |
| `T` *extends* [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md) |

#### Parameters

| Parameter | Type |
| --- | --- |
| `eventType` | [`HookableEventConstructor`](/docs/api/typescript/HookableEventConstructor/index.md)<`T`\> |
| `callback` | [`HookCallback`](/docs/api/typescript/HookCallback/index.md)<`T`\> |
| `options?` | [`HookCallbackOptions`](/docs/api/typescript/HookCallbackOptions/index.md) |

#### Returns

`HookCleanup`

#### Implementation of

```ts
HookRegistry.addCallback
```

---

### invokeCallbacks()

```ts
invokeCallbacks<T>(event): Promise<T>;
```

Defined in: [src/hooks/registry.ts:86](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/registry.ts#L86)

Invoke all registered callbacks for the given event. Awaits each callback, supporting both sync and async.

InterruptErrors are collected across callbacks rather than immediately thrown, allowing all hooks to register their interrupts. Non-interrupt errors propagate immediately.

#### Type Parameters

| Type Parameter |
| --- |
| `T` *extends* [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md) |

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `event` | `T` | The event to invoke callbacks for |

#### Returns

`Promise`<`T`\>

The event after all callbacks have been invoked

#### Throws

InterruptError with all collected interrupts after all callbacks complete