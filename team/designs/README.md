# Design Documents

This folder contains design documents for significant features and changes to Strands Agents. These documents capture the problem, the proposal and the alternatives weighed against it, and the consequences of architectural choices.

For lightweight architecture decision records, see [DECISIONS.md](../DECISIONS.md).

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

## Problem

In a few sentences, what is the issue motivating this change?

- What task are you trying to accomplish?
- What makes it difficult or impossible today?
- Who experiences this problem?

### Current State

Now show *how* it's hard. Ground the reader in the existing system — how it works today and exactly where it falls short — before proposing to change it.

- How is this handled now — the current API, flow, or workaround?
- What concretely breaks, and where? Use examples, error messages, or numbers where you can.
- What are the paper cuts — the small, recurring frictions that add up — not just the outright failures?
- How often does this come up, and how costly is it when it does?

## Goals and Non-Goals

What are we optimizing for, and what is explicitly out of scope?

- What does a good solution need to do?
- What constraints does it operate under?
- What is explicitly *not* being addressed here?

## Proposal

What are we proposing, and what else did we weigh? List the options on equal footing with their pros and cons, recommended one first, so the reader can see the choice was made by weighing tradeoffs rather than asserted.

### Recommended: [name]

What the change is:

- Changes to the API (with code examples)
- How it integrates with existing features
- Expected behavior and usage patterns

**Pros:** what this option gets right, including against the goals above.

**Cons:** what it costs or gives up — every option, including this one, has some.

### Alternative: [name]

- What this option would look like
- **Pros:** ...
- **Cons:** ...
- Why we didn't recommend it

*If there's no genuine alternative — the design space is narrow or one approach clearly dominates — say so in a sentence rather than inventing a strawman. Still spell out the recommended option's own tradeoffs.*

## Developer Experience

Show what the developer experience looks like for the recommended option:

- Code examples showing typical usage
- Configuration or setup required
- Error messages and edge cases

## Consequences

What becomes easier or more difficult to do because of this change?

## Willingness to Implement

Are you willing to implement this if approved?

Yes / No / Maybe with guidance
```

## Writing the document

A few habits make these documents easier to review:

- **Earn the code.** Don't show an API, snippet, or interface until the reader understands the problem it solves — work through Problem, Current State, and Goals first. Code shown before the reader knows what is even going on reads as a solution to an unstated problem, and reviewers end up reverse-engineering your intent from the syntax. By the time the first snippet appears in the Proposal, the reader should already know why it needs to exist.
- **Lead with the rubric, not the verdict.** State the goals and constraints before the proposal, so the choice reads as reasoned rather than asserted.
- **Weigh options honestly.** List alternatives on equal footing with real pros and cons, recommended one first, so the recommendation is earned by the tradeoffs rather than propped up by strawmen. If there's no genuine alternative, say so plainly instead of inventing one.

## Accepted Designs

*No designs have been accepted yet.*
