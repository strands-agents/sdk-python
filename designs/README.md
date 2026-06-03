# Design Documents

This folder contains design documents for significant features and changes to Strands Agents. These documents capture the context, decision, and consequences of architectural choices.

For lightweight architecture decision records, see [DECISIONS.md](../team/DECISIONS.md).

## What is a design document?

A design document proposes a significant change or new feature. It describes the problem, proposed solution, and tradeoffs. Once approved and merged, it becomes an accepted design and part of the project's decision history.

## When to write a design document

Write a design document for:

- New major features affecting multiple parts of the SDK
- Breaking changes to existing APIs
- Architectural changes requiring design discussion
- Large contributions (> 1 week of work)
- Features that introduce new concepts

Skip the design process for bug fixes, small improvements, documentation updates, and new extensions in your own repository.

## How to submit a feature proposal

1. **Check the [roadmap](https://github.com/orgs/strands-agents/projects/8/views/1)** — See if your idea aligns with our direction
2. **Create a branch in this repository**
3. **Create your design** — Add a new file: `designs/NNNN-feature-name.md` using the template below
4. **Submit a pull request** — We'll review and discuss
5. **Iterate based on feedback** — Address comments and questions
6. **Get approval** — Once approved and merged, implement the feature
7. **Reference the design** — Link to it in your implementation PR

## Design document template

```markdown
# [Feature Name]

**Status**: Proposed | Accepted | Deprecated | Superseded

**Date**: YYYY-MM-DD

**Issue**: Issue Link

## Context

What is the issue that we're seeing that is motivating this decision or change?

- What task are you trying to accomplish?
- What makes it difficult or impossible today?
- Who experiences this problem?

## Decision

What is the change that we're proposing and/or doing?

- Changes to the API (with code examples)
- How it integrates with existing features
- Expected behavior and usage patterns

## Developer Experience

Show what the developer experience looks like:

- Code examples showing typical usage
- Configuration or setup required
- Error messages and edge cases

## Alternatives Considered

What other approaches did you consider? Why did you choose this one?

## Consequences

What becomes easier or more difficult to do because of this change?

## Willingness to Implement

Are you willing to implement this if approved?

Yes / No / Maybe with guidance
```

## Accepted Designs

*No designs have been accepted yet.*
