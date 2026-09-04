Defined in: [src/tools/tool.ts:44](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L44)

Data for a tool stream event.

## Properties

### type

```ts
type: "toolStreamEvent";
```

Defined in: [src/tools/tool.ts:48](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L48)

Discriminator for tool stream events.

---

### data?

```ts
optional data?: unknown;
```

Defined in: [src/tools/tool.ts:54](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L54)

Caller-provided data for the progress update. Can be any type of data the tool wants to report.