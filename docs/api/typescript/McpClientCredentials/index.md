Defined in: [src/mcp/client.ts:71](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/mcp/client.ts#L71)

OAuth client credentials for machine-to-machine authentication.

## Properties

### clientId

```ts
clientId: string;
```

Defined in: [src/mcp/client.ts:72](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/mcp/client.ts#L72)

---

### clientSecret

```ts
clientSecret: string;
```

Defined in: [src/mcp/client.ts:73](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/mcp/client.ts#L73)

---

### scopes?

```ts
optional scopes?: string[];
```

Defined in: [src/mcp/client.ts:75](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/mcp/client.ts#L75)

OAuth scopes to request. Joined with spaces before sending to the token endpoint.