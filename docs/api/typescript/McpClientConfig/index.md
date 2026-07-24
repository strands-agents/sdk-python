```ts
type McpClientConfig = McpClientOptions & {
  transport?: McpTransport;
  url?: string | URL;
  auth?: McpClientCredentials;
  authProvider?: OAuthClientProvider;
  headers?: Record<string, string>;
};
```

Defined in: [src/mcp/client.ts:105](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L105)

Arguments for configuring an MCP Client.

## Type Declaration

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `transport?` | [`McpTransport`](/docs/api/typescript/McpTransport/index.md) | Pre-constructed transport. Mutually exclusive with `url`. | [src/mcp/client.ts:107](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L107) |
| `url?` | `string` | `URL` | Server URL. When provided, a StreamableHTTP transport is constructed automatically. | [src/mcp/client.ts:110](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L110) |
| `auth?` | [`McpClientCredentials`](/docs/api/typescript/McpClientCredentials/index.md) | Client credentials for OAuth machine-to-machine auth. Requires `url`. | [src/mcp/client.ts:113](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L113) |
| `authProvider?` | `OAuthClientProvider` | Custom OAuth provider for advanced auth flows. Requires `url`. Mutually exclusive with `auth`. | [src/mcp/client.ts:116](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L116) |
| `headers?` | `Record`<`string`, `string`\> | Custom headers to include on every request to the server. Requires `url`. | [src/mcp/client.ts:119](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L119) |