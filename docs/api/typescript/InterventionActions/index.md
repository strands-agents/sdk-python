```ts
const InterventionActions: {
  proceed: (options?) => Proceed;
  deny: (reason) => Deny;
  guide: (feedback, options?) => Guide;
  confirm: (prompt, options?) => Confirm;
  transform: (apply, options?) => Transform;
};
```

Defined in: [src/interventions/index.ts:3](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/interventions/index.ts#L3)

## Type Declaration

| Name | Type | Defined in |
| --- | --- | --- |
| `proceed()` | (`options?`) => `Proceed` | [src/interventions/index.ts:3](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/interventions/index.ts#L3) |
| `deny()` | (`reason`) => `Deny` | [src/interventions/index.ts:3](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/interventions/index.ts#L3) |
| `guide()` | (`feedback`, `options?`) => `Guide` | [src/interventions/index.ts:3](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/interventions/index.ts#L3) |
| `confirm()` | (`prompt`, `options?`) => `Confirm` | [src/interventions/index.ts:3](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/interventions/index.ts#L3) |
| `transform()` | (`apply`, `options?`) => `Transform` | [src/interventions/index.ts:3](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/interventions/index.ts#L3) |