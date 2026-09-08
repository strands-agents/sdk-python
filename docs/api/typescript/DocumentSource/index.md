```ts
type DocumentSource =
  | {
  type: "documentSourceBytes";
  bytes: Uint8Array;
}
  | {
  type: "documentSourceText";
  text: string;
}
  | {
  type: "documentSourceContentBlock";
  content: DocumentContentBlock[];
}
  | {
  type: "documentSourceS3Location";
  location: S3Location;
};
```

Defined in: [src/types/media.ts:488](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L488)

Source for a document (Class version).