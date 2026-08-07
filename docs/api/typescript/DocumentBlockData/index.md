Defined in: [src/types/media.ts:397](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L397)

Data for a document block.

## Properties

### name

```ts
name: string;
```

Defined in: [src/types/media.ts:401](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L401)

Document name.

---

### format

```ts
format: DocumentFormat;
```

Defined in: [src/types/media.ts:406](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L406)

Document format.

---

### source

```ts
source: DocumentSourceData;
```

Defined in: [src/types/media.ts:411](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L411)

Document source.

---

### citations?

```ts
optional citations?: {
  enabled: boolean;
};
```

Defined in: [src/types/media.ts:416](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L416)

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

Defined in: [src/types/media.ts:421](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L421)

Context information for the document.