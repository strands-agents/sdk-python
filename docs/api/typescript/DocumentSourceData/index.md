```ts
type DocumentSourceData =
  | {
  bytes: Uint8Array;
}
  | {
  text: string;
}
  | {
  content: DocumentContentBlockData[];
}
  | {
  location: S3LocationData;
};
```

Defined in: [src/types/media.ts:379](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/media.ts#L379)

Source for a document (Data version). Supports multiple formats including structured content.