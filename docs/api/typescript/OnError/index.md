```ts
type OnError = "throw" | "proceed" | "deny";
```

Defined in: [src/interventions/handler.ts:19](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/interventions/handler.ts#L19)

What to do when a handler throws during evaluation.

-   `'throw'` — rethrow the error (default, safest: a broken policy check blocks execution)
-   `'proceed'` — log the error and continue as if the handler returned Proceed
-   `'deny'` — log the error and treat it as a Deny (fail-closed)