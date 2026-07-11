Defined in: [src/tools/types.ts:41](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/tools/types.ts#L41)

Represents a tool usage request from the model. The model generates this when it wants to use a tool.

## Properties

### name

```ts
name: string;
```

Defined in: [src/tools/types.ts:45](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/tools/types.ts#L45)

The name of the tool to execute.

---

### toolUseId

```ts
toolUseId: string;
```

Defined in: [src/tools/types.ts:51](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/tools/types.ts#L51)

Unique identifier for this tool use instance. Used to match tool results back to their requests.

---

### input

```ts
input: JSONValue;
```

Defined in: [src/tools/types.ts:57](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/tools/types.ts#L57)

The input parameters for the tool. Must be JSON-serializable.