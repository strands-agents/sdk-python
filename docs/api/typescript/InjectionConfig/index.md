Defined in: [src/injection/types.ts:35](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/injection/types.ts#L35)

Configuration common to every injection consumer: when to inject. What text to inject is a consumer concern, added by the interfaces that extend this one (e.g. [MemoryInjectionConfig](/docs/api/typescript/MemoryInjectionConfig/index.md)).

## Extended by

-   [`MemoryInjectionConfig`](/docs/api/typescript/MemoryInjectionConfig/index.md)

## Properties

### trigger?

```ts
optional trigger?:
  | InjectionTrigger
  | ((context) => boolean);
```

Defined in: [src/injection/types.ts:43](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/injection/types.ts#L43)

When injection runs. An [InjectionTrigger](/docs/api/typescript/InjectionTrigger/index.md) name selects a built-in policy; a predicate is the escape hatch — it receives the [InjectionContext](/docs/api/typescript/InjectionContext/index.md) and returns whether to inject this call. A predicate that throws fails open (injection is skipped, the model call proceeds).

#### Default Value

```ts
'userTurn'
```