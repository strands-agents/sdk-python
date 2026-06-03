# Agent Development Guide - TypeScript SDK

This document provides guidance for AI agents working on the Strands TypeScript SDK (`strands-ts/`). For human contributor guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Development Workflow

### 1. Environment Setup

See [CONTRIBUTING.md - TypeScript SDK](../CONTRIBUTING.md#typescript-sdk) for:

- Prerequisites (Node.js 20+, npm)
- Installation steps
- Verification commands

### 2. Making Changes

1. **Create feature branch**: `git checkout -b agent-tasks/{ISSUE_NUMBER}`
2. **Implement changes** following the patterns below
3. **Run quality checks** before committing (pre-commit hooks will run automatically)
4. **Commit with conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, etc.
5. **Push to remote**: `git push origin agent-tasks/{ISSUE_NUMBER}`
6. **Create pull request** following [PR.md](../dev-docs/PR.md) guidelines

### 3. Pull Request Guidelines

When creating pull requests, you **MUST** follow the guidelines in [PR.md](../dev-docs/PR.md) and use the [PR template](../.github/PULL_REQUEST_TEMPLATE.md). Key principles:

- **Focus on WHY**: Explain motivation and user impact, not implementation details
- **Document public API changes**: Show before/after code examples
- **Be concise**: Use prose over bullet lists; avoid exhaustive checklists
- **Target senior engineers**: Assume familiarity with the SDK
- **Exclude implementation details**: Leave these to code comments and diffs

See [PR.md](../dev-docs/PR.md) for the complete guidance and template.

### 4. Quality Gates

Pre-commit hooks automatically run:

- Build (via npm run build, required for workspace type resolution)
- Unit tests with coverage (via npm run test:coverage)
- WASM unit tests (via npm run test -w strands-wasm)
- Linting (via npm run lint)
- Format checking (via npm run format:check)
- Type checking (via npm run type-check)

All checks must pass before commit is allowed.

### 5. Testing Guidelines

When writing tests, you **MUST** follow the guidelines in [dev-docs/TESTING.md](../dev-docs/TESTING.md). Key topics covered:

- Test organization and file location
- Test batching strategy
- Object assertion best practices
- Test coverage requirements
- Multi-environment testing (Node.js and browser)

See [TESTING.md](../dev-docs/TESTING.md) for the complete testing reference.

## Coding Patterns and Best Practices

### Logging Style Guide

The SDK uses a structured logging format consistent with the Python SDK for better log parsing and searchability.

**Format**:

```typescript
// With context fields
logger.warn(`field1=<${value1}>, field2=<${value2}> | human readable message`)

// Without context fields
logger.warn('human readable message')

// Multiple statements in message (use pipe to separate)
logger.warn(`field=<${value}> | statement one | statement two`)
```

**Guidelines**:

1. **Context Fields** (when relevant):
   - Add context as `field=<value>` pairs at the beginning
   - Use commas to separate pairs
   - Enclose values in `<>` for readability (especially helpful for empty values: `field=<>`)
   - Use template literals for string interpolation

2. **Messages**:
   - Add human-readable messages after context fields
   - Use lowercase for consistency
   - Avoid punctuation (periods, exclamation points) to reduce clutter
   - Keep messages concise and focused on a single statement
   - If multiple statements are needed, separate them with pipe character (`|`)

**Examples**:

```typescript
// Good: Context fields with message
logger.warn(`stop_reason=<${stopReason}>, fallback=<${fallback}> | unknown stop reason, converting to camelCase`)
logger.warn(`event_type=<${eventType}> | unsupported bedrock event type`)

// Good: Simple message without context fields
logger.warn('cache points are not supported in openai system prompts, ignoring cache points')

// Good: Multiple statements separated by pipes
logger.warn(`request_id=<${id}> | processing request | starting validation`)

// Bad: Not using angle brackets for values
logger.warn(`stop_reason=${stopReason} | unknown stop reason`)

// Bad: Using punctuation
logger.warn(`event_type=<${eventType}> | Unsupported event type.`)
```

### Import Organization

Organize imports in this order:

```typescript
// 1. External dependencies
import { something } from 'external-package'

// 2. Internal modules (using relative paths)
import { Agent } from '../agent'
import { Tool } from '../tools'

// 3. Types (if separate)
import type { Options, Config } from '../types'
```

### File Organization Pattern

**For source files**:

```
strands-ts/src/
├── module.ts              # Source file
└── __tests__/
    └── module.test.ts     # Unit tests co-located
```

**Function ordering within files**:

- Functions MUST be ordered from most general to most specific (top-down reading)
- Public/exported functions MUST appear before private helper functions
- Main entry point functions MUST be at the top of the file
- Helper functions SHOULD follow in order of their usage

**For integration tests**:

```
strands-ts/test/integ/
└── feature.test.ts        # Tests public API
```

### TypeScript Type Safety

**Optional chaining for null safety**: Prefer optional chaining over verbose `typeof` checks when accessing potentially undefined properties:

```typescript
// Good: Optional chaining
return globalThis?.process?.env?.API_KEY

// Bad: Verbose typeof checks
if (typeof process !== 'undefined' && typeof process.env !== 'undefined') {
  return process.env.API_KEY
}
return undefined
```

**Strict requirements**:

- Always provide explicit return types
- Never use `any` type (enforced by ESLint)
- Use TypeScript strict mode features
- Leverage type inference where appropriate

### Class Field Naming Conventions

**Private fields**: Use underscore prefix for private class fields.

```typescript
// Good: Private fields with underscore prefix
export class Example {
  private readonly _config: Config
  private _state: State
}

// Bad: No underscore for private fields
export class Example {
  private readonly config: Config
}
```

#### Naming Conventions for New Features

When choosing names and constants that match an existing implementation in the Python SDK, use exactly the same literal used in the Python SDK. Wherever we can achieve compatibility, keep the previous convention.

#### Plugin Naming

Name plugins for what they do, not for the `Plugin` interface they implement.

```typescript
// Good
export class AgentSkills implements Plugin { ... }
export class DefaultModelRetryStrategy implements Plugin { ... }

// Bad
export class AgentSkillsPlugin implements Plugin { ... }
```

### Documentation Requirements

**TSDoc format** (required for all exported functions):

- All exported functions, classes, and interfaces must have TSDoc
- Include `@param` for all parameters
- Include `@returns` for return values
- Include `@example` only for exported classes (main SDK entry points like BedrockModel, Agent)
- Do NOT include `@example` for type definitions, interfaces, or internal types
- Interface properties MUST have single-line descriptions
- TSDoc validation enforced by ESLint

### Code Style Guidelines

**Formatting** (enforced by Prettier):

- No semicolons
- Single quotes
- Line length: 120 characters
- Tab width: 2 spaces
- Trailing commas in ES5 style

### Dependency Management

When adding or modifying dependencies, you **MUST** follow the guidelines in [dev-docs/DEPENDENCIES.md](../dev-docs/DEPENDENCIES.md). Key points:

- **`dependencies`**: Core SDK functionality that users don't interact with directly
- **`peerDependencies`**: Dependencies that cross API boundaries (users construct/pass instances)
- **`devDependencies`**: Build tools, testing frameworks, linters - not shipped to users

**Rule**: If a dependency crosses an API boundary, it **MUST** be a peer dependency.

## Things to Do

- Use relative imports for internal modules
- Co-locate unit tests with source under `__tests__` directories
- Follow nested describe pattern for test organization
- Write explicit return types for all functions
- Document all exported functions with TSDoc
- Use meaningful variable and function names
- Keep functions small and focused (single responsibility)
- Handle errors explicitly

## Things NOT to Do

- Use `any` type (enforced by ESLint)
- Put unit tests in separate `tests/` directory (use `strands-ts/src/**/__tests__/**`)
- Skip documentation for exported functions
- Use semicolons (Prettier will remove them)
- Commit without running pre-commit hooks
- Ignore linting errors
- Use implicit return types

## Development Commands

Quick reference:

```bash
npm test              # Run unit tests in Node.js
npm run test:browser  # Run unit tests in browser (Chromium via Playwright)
npm run test:all      # Run all tests in all environments
npm run test:integ    # Run integration tests
npm run test:coverage # Run tests with coverage report
npm run lint          # Check code quality
npm run format        # Auto-fix formatting
npm run type-check    # Verify TypeScript types
npm run build         # Compile TypeScript
```

## Agent-Specific Notes

### Writing code

- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST MATCH the style and formatting of surrounding code, even if it differs from standard style guides.
- Fix broken things immediately when you find them. Don't ask permission to fix bugs.

#### Code Comments

- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- Comments should be evergreen — never reference temporal context

### Integration with Other Files

- **CONTRIBUTING.md**: Contains testing/setup commands and human contribution guidelines
- **dev-docs/TESTING.md**: Comprehensive testing guidelines (MUST follow when writing tests)
- **dev-docs/PR.md**: Pull request guidelines and template
- **package.json**: Root workspace config that delegates to strands-ts
- **strands-ts/package.json**: SDK package config, dependencies, and npm scripts

## Additional Resources

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [Vitest Documentation](https://vitest.dev/)
- [TSDoc Reference](https://tsdoc.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Strands Agents Documentation](https://strandsagents.com/)
