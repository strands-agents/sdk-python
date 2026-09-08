Defined in: [src/models/streaming.ts:231](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/streaming.ts#L231)

Data for a metadata event.

## Properties

### type

```ts
type: "modelMetadataEvent";
```

Defined in: [src/models/streaming.ts:235](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/streaming.ts#L235)

Discriminator for metadata events.

---

### usage?

```ts
optional usage?: Usage;
```

Defined in: [src/models/streaming.ts:240](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/streaming.ts#L240)

Token usage information.

---

### metrics?

```ts
optional metrics?: Metrics;
```

Defined in: [src/models/streaming.ts:245](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/streaming.ts#L245)

Performance metrics.

---

### trace?

```ts
optional trace?: unknown;
```

Defined in: [src/models/streaming.ts:250](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/streaming.ts#L250)

Trace information for observability.