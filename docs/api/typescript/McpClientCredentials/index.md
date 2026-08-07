Defined in: [src/mcp/client.ts:71](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L71)

OAuth client credentials for machine-to-machine authentication.

## Properties

### clientId

```ts
clientId: string;
```

Defined in: [src/mcp/client.ts:72](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L72)

---

### clientSecret

```ts
clientSecret: string;
```

Defined in: [src/mcp/client.ts:73](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L73)

---

### scopes?

```ts
optional scopes?: string[];
```

Defined in: [src/mcp/client.ts:75](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L75)

OAuth scopes to request. Joined with spaces before sending to the token endpoint.