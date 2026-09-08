```ts
type AudioSourceData =
  | {
  bytes: Uint8Array;
}
  | {
  location: S3LocationData;
};
```

Defined in: [src/types/media.ts:140](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L140)

Source for an audio block (Data version).