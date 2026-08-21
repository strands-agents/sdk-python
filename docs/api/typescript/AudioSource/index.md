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

Defined in: [src/types/media.ts:145](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L145)

Source for an audio block (Class version).