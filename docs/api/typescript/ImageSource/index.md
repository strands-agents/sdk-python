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

Defined in: [src/types/media.ts:149](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/media.ts#L149)

Source for an image (Class version).