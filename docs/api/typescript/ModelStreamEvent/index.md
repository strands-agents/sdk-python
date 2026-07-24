```ts
type ModelStreamEvent =
  | ModelMessageStartEventData
  | ModelContentBlockStartEventData
  | ModelContentBlockDeltaEventData
  | ModelContentBlockStopEventData
  | ModelMessageStopEventData
  | ModelMetadataEventData
  | ModelRedactionEventData;
```

Defined in: [src/models/streaming.ts:19](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L19)

Union type representing all possible streaming events from a model provider. This is a discriminated union where each event has a unique type field.

This allows for type-safe event handling using switch statements.