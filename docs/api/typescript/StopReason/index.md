```ts
type StopReason =
  | "cancelled"
  | "checkpoint"
  | "contentFiltered"
  | "endTurn"
  | "guardrailIntervened"
  | "interrupt"
  | "maxTokens"
  | "limitOutputTokens"
  | "limitTotalTokens"
  | "limitTurns"
  | "pauseTurn"
  | "refusal"
  | "stopSequence"
  | "toolUse"
  | "modelContextWindowExceeded"
  | string & {
};
```

Defined in: [src/types/messages.ts:708](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/types/messages.ts#L708)

Reason why the model stopped generating content.

-   `cancelled` - Agent invocation was cancelled via `agent.cancel()`
-   `contentFiltered` - Content was filtered by safety mechanisms
-   `endTurn` - Natural end of the model’s turn
-   `guardrailIntervened` - A guardrail policy stopped generation
-   `checkpoint` - Agent paused at a cycle boundary for durable execution (experimental; see experimental checkpoint module)
-   `interrupt` - Agent execution was interrupted for human input
-   `maxTokens` - The model provider’s per-call token cap was reached
-   `limitOutputTokens` - Agent loop stopped because `InvokeOptions.limits.outputTokens` was reached
-   `limitTotalTokens` - Agent loop stopped because `InvokeOptions.limits.totalTokens` was reached
-   `limitTurns` - Agent loop stopped because `InvokeOptions.limits.turns` was reached
-   `pauseTurn` - Model paused a long-running turn; the response should be sent back to continue
-   `refusal` - A streaming classifier intervened to handle a potential policy violation
-   `stopSequence` - A stop sequence was encountered
-   `toolUse` - Model wants to use a tool
-   `modelContextWindowExceeded` - Input exceeded the model’s context window