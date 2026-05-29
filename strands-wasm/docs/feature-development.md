# WASM Feature Development Guide

Follow this guide when developing new features or modifying existing implementations across the WASM bridge. Changes that cross the WASM boundary touch multiple files across layers.

For general development standards (conventional commits, test coverage, formatting, linting, TSDoc), see [CONTRIBUTING.md](../../CONTRIBUTING.md). For the SDK's compatibility policy on non-breaking changes (union type extensions, getter/setter conversions), see [COMPATIBILITY.MD](../../COMPATIBILITY.MD).

## File ownership

Know which file owns which concern. Read the relevant files before modifying them.

| File | Owns | When to modify |
|---|---|---|
| `wit/agent.wit` | Boundary types and contract between guest and host | Adding new config fields, new WIT records, new resource methods, or new import/export interfaces |
| `strands-wasm/entry.ts` | Config deserialization, TS SDK instantiation, event mapping | Changing how config is read from WIT and passed to TS SDK constructors, adding new `createXxx()` functions, modifying stream event mapping |
| `strands-py-wasm/strands/_wasm_host.py` | Config serialization (Python → WIT records), WASM runtime management (`WasmAgent`), raw wasmtime `Variant` → Python `StreamEvent` dataclass conversion | Adding `_build_xxx()` serialization functions, modifying `WasmAgent` methods, changing how raw wasmtime variants are converted to `StreamEvent` dataclasses |
| `strands-py-wasm/strands/agent/__init__.py` | Python user-facing API, config extraction from Python objects to dicts | Adding/modifying constructor parameters, extracting config from Python class instances |
| `strands-py-wasm/strands/_conversions.py` | `StreamEvent` dataclass → Python SDK dict format, TS SDK message format → Python SDK message format | Modifying how `StreamEvent` dataclasses are converted to dicts (`event_to_dict`), how TS messages are converted to Python format (`convert_message`), or how lifecycle events are mapped to hook events (`lifecycle_event_from_wit`) |

### Files you must not edit manually

| File | Why |
|---|---|
| `strands-py-wasm/strands/_generated/types.py` | Auto-generated from `wit/agent.wit` by `strands-py-wasm/scripts/generate_types.py`. Regenerate with `npm run dev -- generate`. |
| `strands-wasm/generated/` | Auto-generated WIT type bindings. Regenerate with `npm run dev -- generate`. |
| `strands-wasm/build.js` | Build pipeline script. Rarely needs changes unless adding a new esbuild plugin or changing the componentize step. |
| `strands-wasm/patches/getChunkedStream.js` | WASI buffer reuse workaround. Only modify if fixing the specific componentize-js buffering bug it addresses. |

## Naming conventions across layers

Each layer uses a different case convention. Use the correct case for the layer you are writing in.

| Layer | Convention | Example |
|---|---|---|
| WIT (`wit/agent.wit`) | `kebab-case` | `window-size`, `should-truncate-results` |
| TS (`strands-wasm/entry.ts`) | `camelCase` | `windowSize`, `shouldTruncateResults` |
| Python (`strands-py-wasm/`) | `snake_case` | `window_size`, `should_truncate_results` |

componentize-js translates WIT `kebab-case` to JS `camelCase` automatically. When `entry.ts` reads `cmConfig.windowSize`, it is accessing the WIT field `window-size`. Do not convert manually in `entry.ts`.

wasmtime-py does **not** translate automatically. Use `kebab-case` keys directly when building or reading WIT records in `_wasm_host.py`:

```python
_rec(**{"window-size": 40, "should-truncate-results": True})
```

Reading a WIT record returned from the guest:

```python
getattr(rec, "window-size")
```

## Decision: where does the feature run?

Answer these questions before writing any code. Read the TS SDK implementation of the feature first.

**Does the feature need to execute Python user code at runtime?** (e.g., calling a Python function when the model requests a tool)
- Yes → Needs a WIT **import** interface. The guest calls back to the host. See `tool-provider` in `wit/agent.wit` for the pattern.
- No → Feature runs entirely in the WASM guest.

**Is the feature configured once at construction, or invoked at runtime?**
- Construction → **Config holder pattern.** Python class stores config, serialized through WIT, TS instantiates the real implementation. See conversation manager for the pattern.
- Runtime → Needs WIT **export** methods on the `agent` or `response-stream` resource. See `get-messages`, `set-messages` for the pattern.

**Is the feature a Plugin in the TS SDK?**
- Yes → Pass via the appropriate Agent constructor field (`conversationManager`, `plugins`, `sessionManager`). The TS `PluginRegistry` calls `initAgent()` automatically. Do **not** register the Python config holder as a hook provider.
- No → Wire directly in the `AgentImpl` constructor in `entry.ts`.

## Workflow: adding a new feature

Follow these steps in order. Each step includes a verification checkpoint.

### Step 1: Read the TS SDK implementation

Read the TS source files for the feature. Identify:
- What config does the constructor accept? (types, defaults, required vs optional)
- What runtime behavior does it have? (hooks registered, methods called, events emitted)
- Is it a Plugin? (extends `Plugin`, has `initAgent()`)
- What does the public API look like for TS users?

Do not proceed until you can answer all four questions from the code you read.

### Step 2: Update the WIT contract

Read `wit/agent.wit` in full before modifying it. Add the new record(s) and/or fields.

**Pattern: flat record with string discriminator.** When a feature has multiple strategies (like conversation manager), use a flat record with a `strategy: string` field rather than a WIT `variant`. This works around a wasmtime-py limitation where `option<variant>` types are not properly supported.

```wit
record my-feature-config {
    strategy: string,
    field-a: s32,
    field-b: option<string>,
}
```

**Pattern: adding a field to `agent-config`.** Add the new config as `option<my-feature-config>` to the `agent-config` record.

**Extending existing WIT variants.** Adding a new variant case to an existing WIT `variant` type (e.g., a new model provider to `model-config`, or a new tag to `stream-event`) is a non-breaking change per the project's [compatibility policy](../../COMPATIBILITY.MD). Existing host code that pattern-matches on known tags will ignore the new tag. Do not add backwards-compatibility shims for new variant cases.

**Regenerate types** after updating `wit/agent.wit`: run `npm run dev -- generate`. This updates `strands-wasm/generated/` and `strands-py-wasm/strands/_generated/types.py` to match the new contract.

**Verification:** Run `npm run dev -- validate wit`. Fix any compile errors in downstream layers before proceeding.

### Step 3: Update `strands-wasm/entry.ts`

Read `entry.ts` in full before modifying it. Add imports for the TS SDK classes you will instantiate to the top-level import block. All imports must be at the top of the file. Then add a `createXxx()` function that:
1. Reads the config from `(config as any).myField` (the `as any` cast is necessary because WIT-generated `AgentConfig` types may not include new fields until regenerated)
2. Returns `undefined` when no config is provided, letting the TS `Agent` constructor apply its own default
3. Instantiates the real TS SDK class with the config values
4. Returns the proper TS SDK type (not `any`)

```typescript
function createMyFeature(config: AgentConfig): MyFeatureClass | undefined {
  const cfg = (config as any).myField
  if (!cfg) {
    return undefined
  }
  return new MyFeatureClass({
    fieldA: cfg.fieldA,
    fieldB: cfg.fieldB ?? undefined,
  })
}
```

Use `?? undefined` for WIT `option<T>` fields. The componentize-js runtime passes `undefined` for absent options, but `null` can appear in some edge cases. The `??` operator normalizes both to `undefined`.

Pass the result to the `Agent` constructor in `AgentImpl`.

**Do not duplicate TS SDK defaults.** If the TS SDK constructor defaults `fieldA` to `40`, do not also hardcode `40` in `entry.ts`. Return `undefined` and let the TS SDK apply its own default.

**Verification:** Run `npm run dev -- validate wasm`. Ensure the WASM component builds.

### Step 4: Update the Python host

Read each file in full before modifying it.

**`strands-py-wasm/strands/_wasm_host.py`** — Add a `_build_xxx_variant()` function that serializes a Python config dict to a WIT record. Add the parameter to `_build_agent_config()` and `WasmAgent.__init__()`.

```python
def _build_my_feature_variant(config: dict[str, typing.Any] | None) -> Record | None:
    if config is None:
        return None
    return _rec(
        strategy=config["type"],
        **{
            "field-a": config.get("field_a"),
            "field-b": config.get("field_b"),
        },
    )
```

Pass through values the user provided. Do not insert defaults here — let the TS SDK apply its own defaults for absent fields.

**`strands-py-wasm/strands/agent/__init__.py`** — Add the parameter to `Agent.__init__()` with a proper type hint. Add config extraction logic that inspects the instance type and builds a config dict. Always include a `dict` passthrough and an `else` warning for unknown types.

```python
feat_config: dict[str, Any] | None = None
if my_feature is not None:
    from strands.agent.my_feature import MyFeatureA as _A, MyFeatureB as _B

    if isinstance(my_feature, _A):
        feat_config = {"type": "strategy-a", "field_a": my_feature.field_a}
    elif isinstance(my_feature, _B):
        feat_config = {"type": "strategy-b", "field_b": my_feature.field_b}
    elif isinstance(my_feature, dict):
        feat_config = my_feature
    else:
        log.warning("unknown my_feature type: %s, ignoring", type(my_feature).__name__)
```

**Feature module** (e.g., `strands-py-wasm/strands/agent/my_feature/`) — Create config holder classes that store user-provided config and nothing else. They extend `HookProvider` for type compatibility with the `Agent` constructor, but must **not** register any hooks. Hook registration happens in the TS SDK's `initAgent()` inside the WASM guest.

```python
class MyFeatureManager(HookProvider):
    def __init__(self, field_a: int = 40, field_b: str | None = None) -> None:
        self.field_a = field_a
        self.field_b = field_b
```

**Verification:** Run `python -m pytest strands-py-wasm/tests_unit/` to validate serialization.

### Step 5: Write tests

The project requires 80% test coverage (see [CONTRIBUTING.md](../../CONTRIBUTING.md)).

**Unit tests** (`strands-py-wasm/tests_unit/`): Test the serialization boundary. Verify that config holder classes store the right values, that `_build_xxx_variant()` produces correct WIT records, and that edge cases (missing fields, invalid values) are handled.

**Integration tests** (`strands-py-wasm/tests_integ/`): Test end-to-end behavior. Create an agent with the feature configured, invoke it, and verify observable behavior. Do **not** test by calling internal methods on config holder classes — the implementation runs in the TS guest, so test through the agent's public API.

### Step 6: Document the change

**`strands-wasm/docs/python-api-changes.md`** — For each Python API change, document:
1. The TS SDK design (with code)
2. The WASM bridge implementation
3. The Python API (before/after code snippets)
4. How the functionality is preserved if the API surface differs from the standalone Python SDK

**`AGENTS.md`** — If the change adds new directories, files, or significantly restructures existing modules, update the directory structure section in [AGENTS.md](../../AGENTS.md).

## Workflow: modifying an existing bridged feature

Modifications (adding a parameter, fixing a bug, changing a default) are more common than new features. Discover the full data flow before changing anything.

### Step 1: Trace the data flow

Grep for the feature across all layers to find every file involved:

```bash
grep -rn 'feature_name\|featureName\|feature-name' wit/ strands-wasm/entry.ts strands-py-wasm/strands/
```

Read every file that appears in the results. Trace the full path: Python construction → WIT serialization → TS instantiation. Identify every function, record, and field involved before making changes.

### Step 2: Identify the change scope

Determine which layers your change affects:

- **Adding a config parameter**: All layers change (WIT record, `entry.ts` reader, `_wasm_host.py` serializer, `agent/__init__.py` extractor, config holder class, tests).
- **Changing a default value**: Usually only the layer that owns the default. If the WASM bridge delegates to the TS SDK default (returns `undefined`), changes to the TS SDK default propagate automatically. If the bridge hardcodes a default, it must be updated.
- **Fixing a serialization bug**: Usually `_wasm_host.py` (Python → WIT) or `_conversions.py` (WIT → Python), plus tests.
- **Fixing a type mismatch**: May involve multiple layers. Trace the type from Python through WIT to TS to find where the mismatch originates.

### Step 3: Make changes in dependency order

Changes cascade through the pipeline. Make changes in this order so each layer compiles against the updated layer above it:

1. `wit/agent.wit` (if the contract changes)
2. Regenerate types: `npm run dev -- generate`
3. `strands-wasm/entry.ts`
4. `strands-py-wasm/strands/_wasm_host.py`
5. `strands-py-wasm/strands/agent/__init__.py` and feature modules
6. Tests

### Step 4: Verify at each layer

After modifying each layer, run the appropriate validation:

| Layer changed | Validation command |
|---|---|
| `wit/agent.wit` | `npm run dev -- validate wit` |
| `strands-wasm/entry.ts` | `npm run dev -- validate wasm` |
| `strands-py-wasm/` | `python -m pytest strands-py-wasm/tests_unit/` |
| All layers | `npm run dev -- ci` |

## Common pitfalls

**Read before you write.** Always read a file before modifying it. Do not assume what a function signature, WIT record, or config dict looks like. The codebase changes across PRs. Stale assumptions cause incorrect edits.

**Do not duplicate TS SDK defaults.** If the TS SDK defaults `windowSize` to `40`, do not hardcode `40` in `entry.ts` or `_wasm_host.py`. Return `undefined` and let the TS SDK own its defaults. Hardcoded values silently diverge when the TS SDK changes.

**Do not register hooks in Python config holders.** Config holder classes extend `HookProvider` for type compatibility only. All hook registration happens in the TS SDK's `initAgent()` inside the WASM guest. Registering hooks on the Python side creates duplicate behavior.

**Do not edit generated files.** `strands-py-wasm/strands/_generated/types.py` and `strands-wasm/generated/` are auto-generated. Edits are overwritten on the next `npm run dev -- generate`.

**Separate formatting from feature changes.** Keep formatting (Prettier, ruff) in separate commits or PRs. Mixed diffs obscure functional changes.

**Update `_conversions.py` for return-path changes.** Data returning from the WASM guest (messages, stream events) passes through `_conversions.py`, not `_wasm_host.py`. If the TS SDK changes message format or event types, update `_conversions.py`.

**Keep serialization types explicit.** If a Python constructor accepts `dict[str, Any]` but the serialized form is `str` (JSON), store the user-provided type on the class and serialize in a dedicated method at the bridge boundary. Do not silently convert types in the constructor.

**Set all WIT record fields.** When using the flat record pattern with a strategy discriminator, every field must be present in every record instance, even if unused. wasmtime-py requires all fields of a record to be set. Use zero values or `None` for unused fields.
