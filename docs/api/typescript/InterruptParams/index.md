Defined in: [src/types/interrupt.ts:14](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/interrupt.ts#L14)

Parameters for raising an interrupt.

## Properties

### name

```ts
name: string;
```

Defined in: [src/types/interrupt.ts:19](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/interrupt.ts#L19)

User-defined name for the interrupt. Must be unique within a single hook callback or tool execution.

---

### reason?

```ts
optional reason?: JSONValue;
```

Defined in: [src/types/interrupt.ts:24](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/interrupt.ts#L24)

User-provided reason for the interrupt.

---

### response?

```ts
optional response?: JSONValue;
```

Defined in: [src/types/interrupt.ts:42](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/interrupt.ts#L42)

Preemptive response to use if available. When provided, the interrupt returns this value immediately without halting agent execution. Useful for session-managed trust responses where a previous user response can be reused.

#### Example

```typescript
// If user already approved in a previous session, skip the interrupt
const approval = context.interrupt({
  name: 'confirm_delete',
  reason: 'Confirm deletion?',
  response: agent.appState['savedApproval'],
})
```