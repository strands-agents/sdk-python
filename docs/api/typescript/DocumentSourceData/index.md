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

Defined in: [src/types/media.ts:479](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/types/media.ts#L479)

Source for a document (Data version). Supports multiple formats including structured content.