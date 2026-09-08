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

Defined in: [src/types/media.ts:145](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/types/media.ts#L145)

Source for an audio block (Class version).