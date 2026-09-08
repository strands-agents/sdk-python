```ts
type ImageSource =
  | {
  type: "imageSourceBytes";
  bytes: Uint8Array;
}
  | {
  type: "imageSourceS3Location";
  location: S3Location;
}
  | {
  type: "imageSourceUrl";
  url: string;
};
```

Defined in: [src/types/media.ts:249](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L249)

Source for an image (Class version).