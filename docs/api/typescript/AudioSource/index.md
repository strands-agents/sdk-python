```ts
type AudioSource =
  | {
  type: "audioSourceBytes";
  bytes: Uint8Array;
}
  | {
  type: "audioSourceS3Location";
  location: S3Location;
};
```

Defined in: [src/types/media.ts:145](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/types/media.ts#L145)

Source for an audio block (Class version).