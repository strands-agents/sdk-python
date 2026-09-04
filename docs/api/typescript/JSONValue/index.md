```ts
type JSONValue =
  | string
  | number
  | boolean
  | null
  | {
[key: string]: JSONValue;
}
  | JSONValue[];
```

Defined in: [src/types/json.ts:26](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/types/json.ts#L26)

Represents any valid JSON value. This type ensures type safety for JSON-serializable data.

## Example

```typescript
const value: JSONValue = { key: 'value', nested: { arr: [1, 2, 3] } }
const text: JSONValue = 'hello'
const num: JSONValue = 42
const bool: JSONValue = true
const nothing: JSONValue = null
```