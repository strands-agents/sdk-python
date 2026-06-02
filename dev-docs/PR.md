# Pull Request Description Guidelines

Good PR descriptions help reviewers understand the context and impact of your changes. They enable faster reviews, better decision-making, and serve as valuable historical documentation.

When creating a PR, follow the [GitHub PR template](../.github/PULL_REQUEST_TEMPLATE.md) and use these guidelines to fill it out effectively.

## Who's Reading Your PR?

Write for senior engineers familiar with the SDK. Assume your reader:

- Understands the SDK's architecture and patterns
- Has context about the broader system
- Can read code diffs to understand implementation details
- Values concise, focused communication

## What to Include

Every PR description should have:

1. **Motivation** — Why is this change needed?
2. **Public API Changes** — What changes to the public API (with code snippets)?
3. **Use Cases** (optional) — When would developers use this feature? Only include for non-obvious functionality; skip for trivial changes or obvious fixes.
4. **Breaking Changes** (if applicable) — What breaks and how to migrate?

## Writing Principles

**Focus on WHY, not HOW:**

- ✅ "The OpenAI SDK supports dynamic API keys, but we don't expose this capability"
- ❌ "Added ApiKeySetter type import from openai/client"

**Document public API changes with example code snippets:**

- ✅ Show before/after code snippets for API changes
- ❌ List every file or line changed

**Be concise:**

- ✅ Use prose over bullet lists when possible
- ❌ Create exhaustive implementation checklists

**Emphasize user impact:**

- ✅ "Enables secret manager integration for credential rotation"
- ❌ "Updated error message to mention 'string or function'"

## What to Skip

Leave these out of your PR description:

- **Implementation details** — Code comments and commit messages cover this
- **Test coverage notes** — CI will catch issues; assume tests are comprehensive
- **Line-by-line change lists** — The diff provides this
- **Build/lint/coverage status** — CI handles verification
- **Commit hashes** — GitHub links commits automatically

## Anti-patterns

❌ **Over-detailed checklists:**

```markdown
### Type Definition Updates

- Added ApiKeySetter type import from 'openai/client'
- Updated OpenAIModelOptions interface apiKey type
```

❌ **Implementation notes reviewers don't need:**

```markdown
## Implementation Notes

- No breaking changes - all existing string-based usage continues to work
- OpenAI SDK handles validation of function return values
```

❌ **Test coverage bullets:**

```markdown
### Test Coverage

- Added test: accepts function-based API key
- Added test: accepts async function-based API key
```

## Good Examples

✅ **Motivation section:**

```markdown
## Motivation

The OpenAI SDK supports dynamic API key resolution through async functions,
enabling use cases like credential rotation and secret manager integration.
However, our SDK currently only accepts static strings for the apiKey parameter,
preventing users from leveraging these capabilities.
```

✅ **Public API Changes section:**

````markdown
## Public API Changes

The `OpenAIModelOptions.apiKey` parameter now accepts either a string or an
async function:

```typescript
// Before: only string supported
const model = new OpenAIModel({
  modelId: 'gpt-4o',
  apiKey: 'sk-...',
})

// After: function also supported
const model = new OpenAIModel({
  modelId: 'gpt-4o',
  apiKey: async () => await secretManager.getApiKey(),
})
```

The change is backward compatible—all existing string-based usage continues
to work without modification.

````

✅ **Use Cases section:**
```markdown
## Use Cases

- **API key rotation**: Rotate keys without application restart
- **Secret manager integration**: Fetch credentials from AWS Secrets Manager, Vault, etc.
- **Multi-tenant systems**: Dynamically select API keys based on context
````

## Template

````markdown
## Motivation

[Explain WHY this change is needed. What problem does it solve? What limitation
does it address? What user need does it fulfill?]

Resolves: #[issue-number]

## Public API Changes

[Document changes to public APIs with before/after code snippets. If no public
API changes, state "No public API changes."]

```typescript
// Before
[existing API usage]

// After
[new API usage]
```

[Explain behavior, parameters, return values, and backward compatibility.]

## Use Cases (optional)

[Only include for non-obvious functionality. Provide 1-3 concrete use cases
showing when developers would use this feature. Skip for trivial changes obvious fixes..]

## Breaking Changes (if applicable)

[If this is a breaking change, explain what breaks and provide migration guidance.]

### Migration

```typescript
// Before
[old code]

// After
[new code]
```

````

## Why These Guidelines?

**Focus on WHY over HOW** because code diffs show implementation details, commit messages document granular changes, and PR descriptions provide the broader context reviewers need.

**Skip test/lint/coverage details** because CI pipelines verify these automatically. Including them adds noise without value.

**Write for senior engineers** to enable concise, technical communication without redundant explanations.

## References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google's Code Review Guidelines](https://google.github.io/eng-practices/review/)

## Checklist Items

 - [ ] Does the PR description target a Senior Engineer familiar with the project?
 - [ ] Does the PR description give an overview of the feature being implemented, including any notes on key implemention decisions
 - [ ] Does the PR include a "Resolves #<ISSUE NUMBER>" in the body and is not bolded?
 - [ ] Does the PR contain the motivation or use-cases behind the change?
 - [ ] Does the PR omit irrelevent details not needed for historical reference?
