```ts
type ElicitationContext = RequestHandlerExtra<ClientRequest, ClientNotification>;
```

Defined in: [src/types/elicitation.ts:12](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/elicitation.ts#L12)

Context provided to an elicitation callback, including the abort signal for the in-flight request.