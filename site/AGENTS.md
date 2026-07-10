# Agent Development Guide - Documentation Site

This document provides guidance for AI agents working on the Strands Agents documentation site (the `site/` directory). For human contributor guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Purpose and Scope
This directory contains the documentation site for Strands Agents, providing guides on how to develop with the SDK in both Python and TypeScript.

**AGENTS.md** contains agent-specific repository information including:
- Directory structure with summaries of what is included in each directory
- Development workflow instructions for agents to follow when developing features
- Coding patterns and testing patterns to follow when writing code
- Style guidelines, organizational patterns, and best practices

**For human contributors**: See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and contribution guidelines.

## Team Process Documents

When working on SDK features or documentation, familiarize yourself with these team processes:

* **[Feature Lifecycle Process](../team/FEATURE_LIFECYCLE.md)**: How features are added, versioned, deprecated, and graduated from experimental status
* **[API Bar Raising](../team/API_BAR_RAISING.md)**: Standards for API design quality
* **[Decisions](../team/DECISIONS.md)**: Key architectural and design decisions
* **[Tenets](../team/TENETS.md)**: Core principles guiding SDK development

These documents define the standards and processes that ensure consistency and quality across the Strands SDK.

## Documentation Skills and Voice References

Documentation authoring skills and shared reference material live under `.agents/`. The `.claude/skills`, `.claude/references`, `.kiro/skills`, and `.kiro/references` paths are symlinks into `.agents/`, so edits through any path hit the same files.

On Windows, these symlinks require Developer Mode or `git config core.symlinks true`. If they don't resolve on your machine, work directly in `.agents/` — every tool-specific path is just a pointer to the same files.

```
.agents/
├── skills/
│   ├── docs-writer/      # Draft or rewrite doc pages
│   ├── docs-reviewer/    # Review drafts before PR submission
│   ├── docs-audit/       # Assess published pages for quality
│   └── docs-planner/     # Prioritize the docs backlog
└── references/
    ├── voice-guide.md       # Five-layer voice stack
    ├── terminology.md       # Canonical terms (one concept, one term)
    ├── mdx-authoring.md     # Tabs, snippets, callouts
    └── code-verification.md # Verifying code examples against SDK source
```

| Skill | Purpose | Sample triggers |
|-------|---------|-----------------|
| `docs-writer`   | Draft or rewrite doc pages          | "write a doc", "draft a page", "rewrite the quickstart" |
| `docs-reviewer` | Review drafts before PR submission  | "review this draft", "is this ready to ship" |
| `docs-audit`    | Assess published pages for quality  | "audit this page", "check docs quality" |
| `docs-planner`  | Prioritize the docs backlog         | "plan docs work", "what docs need writing" |

When authoring or reviewing documentation pages, follow the voice guide and the terminology lock. When verifying code examples in docs, follow the tiered procedure in `.agents/references/code-verification.md`.

## Directory Structure

```
├── AGENTS.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SITE-ARCHITECTURE.md          # Detailed Astro/Starlight customizations
├── src/                          # Astro source files
│   ├── components/               # Custom Astro components
│   │   ├── overrides/            # Starlight component overrides
│   │   └── ...
│   ├── config/                   # Site configuration
│   ├── content/                  # Content collections
│   │   └── docs/                 # Documentation content (Markdown/MDX)
│   │       ├── api/
│   │       │   ├── python/
│   │       │   │   └── _generated/   # Symlink to .build/api-docs/python
│   │       │   └── typescript/
│   │       │       └── _generated/   # Symlink to .build/api-docs/typescript
│   │       ├── assets/
│   │       ├── community/
│   │       ├── contribute/
│   │       ├── examples/
│   │       ├── labs/
│   │       └── user-guide/
│   ├── layouts/                  # Custom layouts
│   ├── pages/                    # Astro pages
│   ├── plugins/                  # Remark/Rehype plugins
│   ├── styles/                   # Global styles
│   └── util/                     # Utility functions
├── astro.config.mjs              # Astro configuration
├── package.json                  # Node.js dependencies and scripts
├── tsconfig.json                 # TypeScript configuration
├── LICENSE
├── NOTICE
├── README.md
├── overrides/                    # Legacy MkDocs overrides (being migrated)
├── scripts/                      # Build and utility scripts
├── test/                         # Test files
└── test-snippets/                # TypeScript snippet test files
```
### Directory Purposes


**IMPORTANT**: After making changes that affect the directory structure (adding new directories, moving files, or adding significant new files), you MUST update this directory structure section to reflect the current state of the repository.

## Development Workflow for Agents

### 1. Environment Setup
#### Prerequisites

- Python 3.10+
- Node.js 20+, npm

#### Setup and Installation

```bash
npm install
```

#### Building and Previewing

Generate the static site:

```bash
npm run build
```

Run a local development server at http://localhost:4321/:

```bash
npm run dev
```

### 2. Making Changes

1. **Create feature branch**: `git checkout -b agent-tasks/{ISSUE_NUMBER}`
2. **Implement changes** following the patterns below
3. **Run quality checks** before committing (pre-commit hooks will run automatically)
4. **Commit with conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, etc.
5. **Push to remote**: `git push origin agent-tasks/{ISSUE_NUMBER}`

### 3. Quality Gates

Pre-commit hooks automatically run:
- Unit tests (via npm test)
- Format checking (via npm run format:check)
- Type checking (via npm run typecheck)

All checks must pass before commit is allowed.

## Coding Patterns and Best Practices

### Code Style Guidelines (for Typescript)

**Formatting** (enforced by Prettier):
- No semicolons
- Single quotes
- Line length: 120 characters
- Line length for doc snippet files under `src/content/docs/`: 90 characters
- Tab width: 2 spaces
- Trailing commas in ES5 style
- Template literal contents in doc snippets must also stay under 90 characters per line. Prettier does not enforce this automatically.

**Example**:
```typescript
export function example(name: string, options?: Options): Result {
  const config = {
    name,
    enabled: true,
    settings: {
      timeout: 5000,
      retries: 3,
    },
  }

  return processConfig(config)
}
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


## Agent-Specific Notes

### When Implementing Features

1. **Read task requirements** carefully from the GitHub issue
2. **Use existing patterns** as reference
3. **Run all checks** before committing (pre-commit hooks will enforce this)


### Integration with Other Files

- **CONTRIBUTING.md**: Contains testing/setup commands and human contribution guidelines
- **README.md**: Public-facing documentation, links to strandsagents.com
- **SITE-ARCHITECTURE.md**: Comprehensive documentation of Astro/Starlight customizations, components, and plugins
- **package.json**: Defines scripts for building, testing, and linting
- **src/config/navigation.yml**: Defines the navigation structure (loaded by `src/sidebar.ts` for Astro)

## Additional Resources

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [TSDoc Reference](https://tsdoc.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Strands Agents Documentation](https://strandsagents.com/)
- [TypeScript SDK](../strands-ts/)

# TypeScript Code Examples Guide

This guide explains how to add TypeScript code examples alongside Python examples in the Strands Agents documentation.

For detailed information about the Astro/Starlight CMS architecture, components, and customizations, see [SITE-ARCHITECTURE.md](SITE-ARCHITECTURE.md).

## Overview

The documentation supports showing both Python and TypeScript code examples side-by-side using:
- **Astro `<Tabs>/<Tab>` components** for language switching (auto-imported, no import statement needed)
- **Snippets** for external code file inclusion (still works via `src/plugins/remark-mkdocs-snippets.ts`)
- **TypeScript type checking** for code validation

### 1. Create TypeScript Code File

Create a `.ts` file alongside your `.md` file with snippet markers:

```typescript
// docs/user-guide/concepts/agents/agent-loop.ts
import { Agent } from '@strands-agents/sdk'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

// --8<-- [start:initialization]
// Initialize the agent with tools, model, and configuration
const agent = new Agent({
  tools: [notebook],
  systemPrompt: 'You are a helpful assistant.',
})
// --8<-- [end:initialization]

// --8<-- [start:processResult]
// Process user input
const result = await agent.invoke('Calculate 25 * 48')
// --8<-- [end:processResult]
```

### 2. Use Tabbed Content in Markdown

In your `.md` file, use `<Tabs>` and `<Tab>` components with snippet inclusion:

```markdown
<Tabs>
  <Tab label="Python">
    ```python
    from strands import Agent
    from strands_tools import calculator

    # Initialize the agent with tools, model, and configuration
    agent = Agent(
        tools=[calculator],
        system_prompt="You are a helpful assistant."
    )
    ```
  </Tab>
  <Tab label="TypeScript">
    ```typescript
    --8<-- "user-guide/concepts/agents/agent-loop.ts:initialization"
    ```
  </Tab>
</Tabs>
```

**Note**: `<Tabs>` and `<Tab>` are auto-imported via `astro-auto-import`, so no import statement is needed. Tabs with matching labels automatically sync across the page.

## Snippet Syntax

### Basic Snippet Inclusion

```markdown
--8<-- "path/to/file.ts:snippet_name"
```

### Snippet Markers in Code

Use HTML-style comments to mark snippet boundaries:

```typescript
// --8<-- [start:snippet_name]
// Your code here
// --8<-- [end:snippet_name]
```

**Note**: Leading spaces are automatically removed from included snippets, so indentation within the source file doesn't affect the final output. However, if the snippet file name is indented in the markdown, the content will be indented to that level as well. See [the documentation](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#dedent-subsections) for more information.

### Multiple Snippets in One File

```typescript
// --8<-- [start:initialization]
const agent = new Agent({ /* ... */ })
// --8<-- [end:initialization]

// --8<-- [start:usage]
const result = await agent.invoke('Hello')
// --8<-- [end:usage]
```

## Type Checking Integration

### Package.json Scripts

```json
{
  "scripts": {
    "test": "tsc --noEmit",
    "format": "prettier --write docs 'src/content/docs/**/*.ts'",
    "format:check": "prettier --check docs 'src/content/docs/**/*.ts'"
  }
}
```

## Best Practices

### 1. File Organization

```
docs/
├── user-guide/
│   └── concepts/
│       └── agents/
│           ├── agent-loop.md      # Documentation
│           └── agent-loop.ts      # TypeScript examples
```

### 2. Snippet Naming

Use descriptive snippet names that match the context:

```typescript
// --8<-- [start:basic_agent_creation]
// --8<-- [start:agent_with_tools]
// --8<-- [start:streaming_example]
```

### 3. Variable Scoping for Snippets

When multiple snippets in the same file use the same variable names, wrap snippets in functions to avoid TypeScript scoping conflicts. Place snippet markers **inside** the function so only the code is displayed in documentation:

```typescript
// ❌ Wrong: Snippet includes function definition
// --8<-- [start:example]
async function example() {
  const result = await agent.invoke('Hello')
  console.log(result)
}
// --8<-- [end:example]

// ✅ Correct: Function is for scoping only, snippet is just the code
async function example() {
  // --8<-- [start:example]
  const result = await agent.invoke('Hello')
  console.log(result)
  // --8<-- [end:example]
}
```

**Why:**
- TypeScript treats the entire file as a single scope with `isolatedModules: true`
- Multiple snippets with the same variable names cause redeclaration errors
- Functions provide scoping without cluttering the documentation with function definitions

### 4. Code Validation

- All TypeScript code should compile without errors
- Use `npm run test` to validate TypeScript
- Use `npm run format` to maintain consistent formatting

### 5. Fallback for Unsupported Features

For features not available in TypeScript, you can indicate this using custom frontmatter fields. See [SITE-ARCHITECTURE.md](SITE-ARCHITECTURE.md#custom-frontmatter-fields) for details on:
- `languages` - Indicate a feature is only available in specific SDK languages
- `community` - Mark pages as community-contributed
- `experimental` - Mark features as experimental

These render contextual banners at the top of pages automatically.

## Agent/LLM Instructions

When adding TypeScript examples to documentation:

1. **Create the TypeScript file** with the same base name as the markdown file
2. **Add snippet markers** around code sections you want to reference
3. **Use descriptive snippet names** that clearly indicate the code's purpose
4. **Validate TypeScript** by running `npm run typecheck:snippets` (for snippet files) and `npm run typecheck` (for other TypeScript)
5. **Update markdown** to use `<Tabs>/<Tab>` components with snippet inclusion
6. **Test locally** with `npm run dev` to ensure snippets render correctly at http://localhost:4321/

### Example Workflow

1. Edit `docs/path/to/example.ts`:
   ```typescript
   // --8<-- [start:new_feature]
   const feature = new Feature({ config: 'value' })
   // --8<-- [end:new_feature]
   ```

2. Update `docs/path/to/example.md`:
   ```markdown
   <Tabs>
     <Tab label="TypeScript">
       ```typescript
       --8<-- "path/to/example.ts:new_feature"
       ```
     </Tab>
   </Tabs>
   ```

3. Validate: `npm test`
4. Preview: `npm run dev`

## Benefits

- **Type Safety**: TypeScript compiler catches errors
- **DRY Principle**: Single source of truth for code examples
- **Consistency**: Automatic formatting and validation
- **Maintainability**: Changes to code automatically update documentation
- **IDE Support**: Full TypeScript language server support for code examples
