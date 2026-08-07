```ts
type GuardQualifier = "grounding_source" | "query" | "guard_content";
```

Defined in: [src/types/messages.ts:821](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/messages.ts#L821)

Qualifier for guard content. Specifies how the content should be evaluated by guardrails.

-   `grounding_source` - Content to check for grounding/factuality
-   `query` - User query to evaluate
-   `guard_content` - General content for guardrail evaluation