Defined in: [src/sandbox/base.ts:42](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L42)

Abstract execution environment.

A Sandbox provides the runtime context where tools execute code, run commands, and interact with a filesystem. Multiple tools share the same Sandbox instance, giving them a common working directory and filesystem.

Streaming methods (`executeStreaming`, `executeCodeStreaming`) are the abstract primitives. Non-streaming convenience methods (`execute`, `executeCode`) consume the stream and return the final result.

## Extended by

-   [`PosixShellSandbox`](/docs/api/typescript/PosixShellSandbox/index.md)

## Constructors

### Constructor

```ts
new Sandbox(): Sandbox;
```

#### Returns

`Sandbox`

## Methods

### executeStreaming()

```ts
abstract executeStreaming(command, options?): AsyncIterable<
  | StreamChunk
| ExecutionResult>;
```

Defined in: [src/sandbox/base.ts:54](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L54)

Execute a shell command, streaming output.

Yields [StreamChunk](/docs/api/typescript/StreamChunk/index.md) objects for stdout and stderr as output arrives. The final yield is an [ExecutionResult](/docs/api/typescript/ExecutionResult/index.md) with the exit code and complete output.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `command` | `string` | The shell command to execute. |
| `options?` | [`ExecuteOptions`](/docs/api/typescript/ExecuteOptions/index.md) | Execution options. |

#### Returns

`AsyncIterable`< | [`StreamChunk`](/docs/api/typescript/StreamChunk/index.md) | [`ExecutionResult`](/docs/api/typescript/ExecutionResult/index.md)\>

Async iterable yielding StreamChunks followed by a final ExecutionResult.

---

### executeCodeStreaming()

```ts
abstract executeCodeStreaming(
   code,
   language,
   options?): AsyncIterable<
  | StreamChunk
| ExecutionResult>;
```

Defined in: [src/sandbox/base.ts:64](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L64)

Execute source code via a language interpreter, streaming output.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `code` | `string` | The source code to execute. |
| `language` | `string` | The interpreter to use (e.g., `"python3"`, `"node"`). |
| `options?` | [`ExecuteOptions`](/docs/api/typescript/ExecuteOptions/index.md) | Execution options. |

#### Returns

`AsyncIterable`< | [`StreamChunk`](/docs/api/typescript/StreamChunk/index.md) | [`ExecutionResult`](/docs/api/typescript/ExecutionResult/index.md)\>

Async iterable yielding StreamChunks followed by a final ExecutionResult.

---

### readFile()

```ts
abstract readFile(path): Promise<Uint8Array<ArrayBufferLike>>;
```

Defined in: [src/sandbox/base.ts:80](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L80)

Read a file from the sandbox filesystem as raw bytes.

Returns `Uint8Array` to support both text and binary files. Use [readText](#readtext) for a convenience wrapper that decodes to a string.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `string` | Path to the file to read. |

#### Returns

`Promise`<`Uint8Array`<`ArrayBufferLike`\>>

The file contents as raw bytes.

#### Throws

Error if the file does not exist.

---

### writeFile()

```ts
abstract writeFile(path, content): Promise<void>;
```

Defined in: [src/sandbox/base.ts:91](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L91)

Write raw bytes to a file in the sandbox filesystem.

Implementations should create parent directories if they do not exist. Use [writeText](#writetext) for a convenience wrapper that encodes a string.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `string` | Path to the file to write. |
| `content` | `Uint8Array` | The content to write. |

#### Returns

`Promise`<`void`\>

---

### removeFile()

```ts
abstract removeFile(path): Promise<void>;
```

Defined in: [src/sandbox/base.ts:99](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L99)

Remove a file from the sandbox filesystem.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `string` | Path to the file to remove. |

#### Returns

`Promise`<`void`\>

#### Throws

Error if the file does not exist.

---

### listFiles()

```ts
abstract listFiles(path): Promise<FileInfo[]>;
```

Defined in: [src/sandbox/base.ts:112](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L112)

List files in a sandbox directory.

Returns [FileInfo](/docs/api/typescript/FileInfo/index.md) entries with name, isDir, and size metadata. Fields `isDir` and `size` may be `undefined` if the backend cannot determine them.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `string` | Path to the directory to list. |

#### Returns

`Promise`<[`FileInfo`](/docs/api/typescript/FileInfo/index.md)\[\]>

Array of FileInfo entries for the directory contents.

#### Throws

Error if the directory does not exist.

---

### getTools()

```ts
getTools(): Tool[];
```

Defined in: [src/sandbox/base.ts:119](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L119)

Tools this sandbox vends to an agent, registered during `Agent.initialize()`. A tool is skipped if the user already registered one with the same name. Override to provide them.

#### Returns

[`Tool`](/docs/api/typescript/Tool/index.md)\[\]

---

### execute()

```ts
execute(command, options?): Promise<ExecutionResult>;
```

Defined in: [src/sandbox/base.ts:135](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L135)

Execute a shell command and return the result.

Consumes [executeStreaming](#executestreaming) and returns the final [ExecutionResult](/docs/api/typescript/ExecutionResult/index.md). Use `executeStreaming` when you need to process output as it arrives.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `command` | `string` | The shell command to execute. |
| `options?` | [`ExecuteOptions`](/docs/api/typescript/ExecuteOptions/index.md) | Execution options. |

#### Returns

`Promise`<[`ExecutionResult`](/docs/api/typescript/ExecutionResult/index.md)\>

The execution result with exit code and output.

---

### executeCode()

```ts
executeCode(
   code,
   language,
options?): Promise<ExecutionResult>;
```

Defined in: [src/sandbox/base.ts:155](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L155)

Execute source code and return the result.

Consumes [executeCodeStreaming](#executecodestreaming) and returns the final [ExecutionResult](/docs/api/typescript/ExecutionResult/index.md). Use `executeCodeStreaming` when you need to process output as it arrives.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `code` | `string` | The source code to execute. |
| `language` | `string` | The interpreter to use. |
| `options?` | [`ExecuteOptions`](/docs/api/typescript/ExecuteOptions/index.md) | Execution options. |

#### Returns

`Promise`<[`ExecutionResult`](/docs/api/typescript/ExecutionResult/index.md)\>

The execution result with exit code and output.

---

### readText()

```ts
readText(path): Promise<string>;
```

Defined in: [src/sandbox/base.ts:173](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L173)

Read a text file from the sandbox filesystem.

Convenience wrapper over [readFile](#readfile) that decodes bytes as UTF-8. For other encodings, call `readFile` and decode manually.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `string` | Path to the file to read. |

#### Returns

`Promise`<`string`\>

The file contents decoded as a UTF-8 string.

---

### writeText()

```ts
writeText(path, content): Promise<void>;
```

Defined in: [src/sandbox/base.ts:186](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/sandbox/base.ts#L186)

Write a text file to the sandbox filesystem.

Convenience wrapper over [writeFile](#writefile) that encodes a string as UTF-8. For other encodings, encode manually and call `writeFile`.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `string` | Path to the file to write. |
| `content` | `string` | The text content to write. |

#### Returns

`Promise`<`void`\>