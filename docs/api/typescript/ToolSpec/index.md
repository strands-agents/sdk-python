Defined in: [src/tools/types.ts:13](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/tools/types.ts#L13)

Specification for a tool that can be used by the model. Defines the tool’s name, description, and input schema.

## Properties

### name

```ts
name: string;
```

Defined in: [src/tools/types.ts:17](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/tools/types.ts#L17)

The unique name of the tool.

---

### description

```ts
description: string;
```

Defined in: [src/tools/types.ts:23](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/tools/types.ts#L23)

A description of what the tool does. This helps the model understand when to use the tool.

---

### inputSchema?

```ts
optional inputSchema?: JSONSchema7;
```

Defined in: [src/tools/types.ts:29](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/tools/types.ts#L29)

JSON Schema defining the expected input structure for the tool. If omitted, defaults to an empty object schema allowing no input parameters.

---

### outputSchema?

```ts
optional outputSchema?: JSONSchema7;
```

Defined in: [src/tools/types.ts:34](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/tools/types.ts#L34)

JSON Schema defining the expected output structure for the tool.