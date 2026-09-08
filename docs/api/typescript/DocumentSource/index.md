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

Defined in: [src/types/media.ts:488](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/types/media.ts#L488)

Source for a document (Class version).