Defined in: [src/sandbox/base.ts:15](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/sandbox/base.ts#L15)

Options for command and code execution.

## Properties

### timeout?

```ts
optional timeout?: number;
```

Defined in: [src/sandbox/base.ts:17](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/sandbox/base.ts#L17)

Maximum execution time in seconds. `undefined` means no timeout.

---

### cwd?

```ts
optional cwd?: string;
```

Defined in: [src/sandbox/base.ts:19](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/sandbox/base.ts#L19)

Working directory for execution. `undefined` means use the sandbox default.

---

### signal?

```ts
optional signal?: AbortSignal;
```

Defined in: [src/sandbox/base.ts:21](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/sandbox/base.ts#L21)

Abort signal to cancel execution. The process is killed when the signal fires.

---

### env?

```ts
optional env?: Record<string, string>;
```

Defined in: [src/sandbox/base.ts:27](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/sandbox/base.ts#L27)

Environment variables to set for this command. Built-in sandboxes always apply these, though the mechanism differs (Docker `-e` flags, SSH `env` prefix); custom `Sandbox` implementations must handle env explicitly or it has no effect.