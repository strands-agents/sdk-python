# Compatibility Policy

This document outlines the Strands TypeScript SDK's policy on changes that are **not considered breaking changes** under semantic versioning. Understanding these policies helps you anticipate how the SDK may evolve without requiring major version bumps.

## Field to Getter/Setter Conversion

Converting a public mutable field to a property with getter/setter methods **is not considered a breaking change**, even when adding validation or side effects.

### Policy

The SDK may convert public mutable fields to getter/setter properties in minor or patch releases. This includes adding:
- Validation logic that throws errors for invalid values
- Side effects during assignment (logging, notifications, state updates)
- Computed or transformed values in getters

### Rationale

In TypeScript and JavaScript, getter/setter properties are syntactically and behaviorally identical to direct field access from the consumer's perspective:

```typescript
// Before: Direct field access
agent.model = newModel
const currentModel = agent.model

// After: Getter/setter (identical usage)
agent.model = newModel  // Calls setter
const currentModel = agent.model  // Calls getter
```

Consumers cannot distinguish between direct field access and property access at the call site. The implementation change is transparent to user code.

### Example

The `Agent.model` property is currently a public mutable field. In a future release, it may be converted to a getter/setter to add validation:

```typescript
// Current implementation (field)
public model: Model<BaseModelConfig>

// Possible future implementation (getter/setter with validation)
private _model: Model<BaseModelConfig>
public get model(): Model<BaseModelConfig> {
  return this._model
}
public set model(value: Model<BaseModelConfig>) {
  if (!value) {
    throw new Error('Model cannot be null or undefined')
  }
  this._model = value
}
```

User code remains unchanged and continues to work as before.

## Union Type Extensions

Adding new types or classes to union types **is not considered a breaking change**, unless the union explicitly declares that it will no longer change.

### Policy

The SDK may add new event types, result variants, or other union members in minor or patch releases. This includes:
- New event types in streaming results
- Additional error types in result unions
- New configuration options in config unions
- Extended enum-like union types

### Rationale

Union type extensions are additive changes that don't break existing code.

Consumers handle union types through type guards, switch statements, or pattern matching that focus on known variants.

New union members are simply ignored by existing logic.

### Example

The `AgentStreamEvent` type returned by `Agent.stream()` may receive new event types:

```typescript
// Current usage (continues to work)
for await (const event of agent.stream('Hello')) {
  if (event.type === 'textDelta') {
    console.log(event.text)
  }
  // New event types are ignored by existing code
}
```

New event types added to the union don't affect existing event handling logic.

## Feedback

If you have questions or concerns about this compatibility policy, please [open an issue](https://github.com/strands-agents/sdk-typescript/issues) on GitHub.
