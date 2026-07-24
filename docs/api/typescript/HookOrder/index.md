```ts
const HookOrder: {
  SDK_FIRST: -100;
  INTERVENTION_OUTPUT: -90;
  DEFAULT: 0;
  INTERVENTION_INPUT: 90;
  SDK_LAST: 100;
};
```

Defined in: [src/hooks/types.ts:48](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/types.ts#L48)

Presets for hook execution order. Lower values run first. Any number is a valid order — these presets are not bounds, just convenient reference points. SDK\_FIRST/SDK\_LAST mark where the SDK’s own hooks run, so you can position yours relative to them.

## Type Declaration

| Name | Type | Default value | Defined in |
| --- | --- | --- | --- |
| `SDK_FIRST` | `-100` | `-100` | [src/hooks/types.ts:49](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/types.ts#L49) |
| `INTERVENTION_OUTPUT` | `-90` | `-90` | [src/hooks/types.ts:50](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/types.ts#L50) |
| `DEFAULT` | `0` | `0` | [src/hooks/types.ts:51](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/types.ts#L51) |
| `INTERVENTION_INPUT` | `90` | `90` | [src/hooks/types.ts:52](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/types.ts#L52) |
| `SDK_LAST` | `100` | `100` | [src/hooks/types.ts:53](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/types.ts#L53) |

## Example

```typescript
agent.addHook(BeforeToolCallEvent, callback, { order: HookOrder.SDK_FIRST }) // run with the SDK's earliest hooks
agent.addHook(BeforeToolCallEvent, callback, { order: HookOrder.SDK_FIRST - 1 }) // run before the SDK's earliest hooks
```