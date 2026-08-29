```ts
type McpClientConfig = McpClientOptions & {
  transport?: McpTransport;
  url?: string | URL;
  auth?: McpClientCredentials;
  authProvider?: OAuthClientProvider;
  headers?: Record<string, string>;
};
```

Defined in: [src/mcp/client.ts:136](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/mcp/client.ts#L136)

Arguments for configuring an MCP Client.

## Type Declaration

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `transport?` | [`McpTransport`](/docs/api/typescript/McpTransport/index.md) | Pre-constructed transport. Mutually exclusive with `url`. | [src/mcp/client.ts:138](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/mcp/client.ts#L138) |
| `url?` | `string` | `URL` | Server URL. When provided, a StreamableHTTP transport is constructed automatically. | [src/mcp/client.ts:141](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/mcp/client.ts#L141) |
| `auth?` | [`McpClientCredentials`](/docs/api/typescript/McpClientCredentials/index.md) | Client credentials for OAuth machine-to-machine auth. Requires `url`. | [src/mcp/client.ts:144](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/mcp/client.ts#L144) |
| `authProvider?` | `OAuthClientProvider` | Custom OAuth provider for advanced auth flows. Requires `url`. Mutually exclusive with `auth`. | [src/mcp/client.ts:147](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/mcp/client.ts#L147) |
| `headers?` | `Record`<`string`, `string`\> | Custom headers to include on every request to the server. Requires `url`. | [src/mcp/client.ts:150](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/mcp/client.ts#L150) |