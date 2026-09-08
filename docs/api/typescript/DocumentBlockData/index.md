Defined in: [src/types/media.ts:497](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L497)

Data for a document block.

## Properties

### name

```ts
name: string;
```

Defined in: [src/types/media.ts:501](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L501)

Document name.

---

### format

```ts
format: DocumentFormat;
```

Defined in: [src/types/media.ts:506](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L506)

Document format.

---

### source

```ts
source: DocumentSourceData;
```

Defined in: [src/types/media.ts:511](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L511)

Document source.

---

### citations?

```ts
optional citations?: {
  enabled: boolean;
};
```

Defined in: [src/types/media.ts:516](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L516)

Citation configuration.

#### enabled

```ts
enabled: boolean;
```

---

### context?

```ts
optional context?: string;
```

Defined in: [src/types/media.ts:521](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L521)

Context information for the document.