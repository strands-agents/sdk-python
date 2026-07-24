A **storage backend** persists raw bytes under `/`\-separated string keys. The SDK uses the `Storage` interface internally for session snapshots, context offloading, and any construct that needs durable key-value persistence. Any class that implements the four-method `Storage` protocol (`write`, `read`, `delete`, `list`) can plug in.

The SDK ships reference backends: [`InMemoryStorage`](/docs/user-guide/concepts/storage/index.md#inmemorystorage), [`LocalFileStorage`](/docs/user-guide/concepts/storage/index.md#localfilestorage), and [`S3Storage`](/docs/user-guide/concepts/storage/index.md#s3storage). The packages below are **community-built** storage backends you can install and use wherever a `Storage` instance is accepted.

Community maintained

These packages are maintained by their authors, not the Strands team. Review packages before using them in production. Quality and support may vary.

## Browse the catalog

See the [Storage section of the community catalog](/docs/community/community-packages/index.md#storage) for the current list, with language support and links to each package.

## Add your storage backend

Built a `Storage` implementation? See the [Get Featured guide](/docs/community/get-featured/index.md) to list it here, and the [Extensions guide](/docs/contribute/contributing/extensions/index.md) for how to build and publish a package.