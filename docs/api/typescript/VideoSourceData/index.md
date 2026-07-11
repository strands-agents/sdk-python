```ts
type VideoSourceData =
  | {
  bytes: Uint8Array;
}
  | {
  location: S3LocationData;
};
```

Defined in: [src/types/media.ts:265](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/media.ts#L265)

Source for a video (Data version).