Defined in: [src/models/streaming.ts:231](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L231)

Data for a metadata event.

## Properties

### type

```ts
type: "modelMetadataEvent";
```

Defined in: [src/models/streaming.ts:235](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L235)

Discriminator for metadata events.

---

### usage?

```ts
optional usage?: Usage;
```

Defined in: [src/models/streaming.ts:240](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L240)

Token usage information.

---

### metrics?

```ts
optional metrics?: Metrics;
```

Defined in: [src/models/streaming.ts:245](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L245)

Performance metrics.

---

### trace?

```ts
optional trace?: unknown;
```

Defined in: [src/models/streaming.ts:250](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L250)

Trace information for observability.