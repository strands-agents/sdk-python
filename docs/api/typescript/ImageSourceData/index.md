```ts
type ImageSourceData =
  | {
  bytes: Uint8Array;
}
  | {
  location: S3LocationData;
}
  | {
  url: string;
};
```

Defined in: [src/types/media.ts:241](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L241)

Source for an image (Data version). Supports multiple formats for different providers.