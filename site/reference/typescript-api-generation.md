# TypeScript API Reference Generation

Part of the site architecture reference. See [SITE-ARCHITECTURE.md](../SITE-ARCHITECTURE.md) for the overview. Paths are relative to `site/`.

The TypeScript API reference documentation is auto-generated from the SDK source code using [typedoc](https://typedoc.org/) with [typedoc-plugin-markdown](https://typedoc-plugin-markdown.org/).

## Generation Script (`scripts/api-generation-typescript.ts`)

**What it does:** Runs typedoc to generate markdown files, then post-processes them to add frontmatter.

**How to run:**
```bash
npm run sdk:generate:ts
# or
npx tsx scripts/api-generation-typescript.ts
```

**Input:** `.build/sdk-typescript/src` (cloned SDK repository)
**Output:** `.build/api-docs/typescript/{classes,interfaces,type-aliases,functions,namespaces}/*.md`

## TypeDoc Configuration (`typedoc.json`)

Key settings:
- `outputFileStrategy: "members"` - Creates separate files per class/interface/type/function
- `fileExtension: ".md"` - Outputs standard markdown format
- `basePath: ".build/sdk-typescript"` - Strips build path prefix from source links
- `hideBreadcrumbs: true`, `hidePageHeader: true` - Cleaner output for Starlight integration
- `excludeExternals: true` - Excludes re-exported symbols from external packages

## Post-Processing

The generation script performs these transformations after typedoc runs:

1. **Adds frontmatter** with title, slug, category, and `editUrl: false` (suppresses the "Edit this page" link since these files are generated):
   ```yaml
   ---
   title: "Agent"
   slug: docs/api/typescript/Agent
   category: classes
   editUrl: false
   ---
   ```
   For namespace members, the slug includes the namespace as a prefix separated by a colon:
   ```yaml
   ---
   title: "setupTracer"
   slug: docs/api/typescript/telemetry:setupTracer
   category: functions
   editUrl: false
   ---
   ```

2. **Fixes relative links** to match the flat slug structure (e.g., `../interfaces/AgentData.md` → `../AgentData.md`) and updates `.md` extensions to `.mdx`. For namespace members and namespace index pages, cross-member links are rewritten to absolute slug paths (e.g., `[TracerConfig](../interfaces/TracerConfig.md)` → `[TracerConfig](/api/typescript/telemetry:TracerConfig)`) to ensure correct resolution regardless of the page's own URL.

3. **Converts to MDX** — runs content through a `unified`/`remark-gfm` pipeline with `mdxToMarkdown()` serialization, which escapes characters that are valid in markdown but invalid in MDX (e.g. `{`, `}` outside code blocks). Content inside code fences is left untouched. Files are written as `.mdx` instead of `.md`. A targeted replacement also handles the literal string `<name>Data` that typedoc emits in prose to describe the naming pattern for data interfaces.

4. **Deletes the generated index.md** - We use our own custom index page instead

## Flat Slugs with Category Grouping

Unlike Python API docs which use hierarchical slugs based on module paths, TypeScript API docs use flat slugs:
- URL: `/docs/api/typescript/Agent/` (not `/docs/api/typescript/classes/Agent/`)
- The `category` frontmatter field is used for sidebar grouping

This keeps URLs clean while still organizing the sidebar by type (Classes, Interfaces, Type Aliases, Functions).

## Namespace Exports

When the SDK exports a namespace (e.g., `export * as telemetry from './telemetry/index.js'`), typedoc generates a nested directory structure under `namespaces/<ns>/`. The generation script handles this specially:

- The namespace index page (`namespaces/<ns>/index.md`) is kept and written as `namespaces/<ns>.mdx` with slug `api/typescript/<ns>` and category `namespaces`.
- Members of the namespace (classes, interfaces, functions, etc.) are flattened into the same top-level category directories as regular exports, but their slugs are prefixed with the namespace name using a colon separator: `docs/api/typescript/<ns>:<MemberName>`.
- All cross-member links within a namespace are rewritten to absolute slug paths to avoid broken relative links after flattening.

## Symlink Setup

The generated docs are accessed via a committed symlink:
```
src/content/docs/api/typescript/_generated -> ../../../../../.build/api-docs/typescript
```

The index page (`src/content/docs/api/typescript/index.mdx`) is a permanent file that imports the `TypeScriptApiList` component.

## Dynamic Sidebar (`src/dynamic-sidebar.ts`)

**What it does:** Builds a category-grouped sidebar structure from TypeScript API docs at runtime.

**How it works:**
1. Filters docs collection for `docs/api/typescript/*` pages
2. Groups docs by their `category` frontmatter field
3. Creates sidebar groups for Classes, Interfaces, Type Aliases, and Functions
4. Sorts entries alphabetically within each group

**Example structure:**
```
Namespaces
  └── telemetry
Classes
  ├── Agent
  ├── BedrockModel
  └── Tool
Interfaces
  ├── AgentConfig
  ├── TracerConfig       ← namespace member, slug: docs/api/typescript/telemetry:TracerConfig
  └── ToolSpec
Type Aliases
  ├── ContentBlock
  └── ToolChoice
Functions
  ├── configureLogging
  ├── setupTracer        ← namespace member, slug: docs/api/typescript/telemetry:setupTracer
  └── tool
```

## Index Page Component (`src/components/TypeScriptApiList.astro`)

**What it does:** Renders the API reference index page with a categorized list of all exports.

**How it works:** Uses the same `buildTypeScriptApiSidebar()` function as the route middleware to ensure consistency between the sidebar navigation and the index page listing.

## Content Collection Schema

The `category` field is defined in `src/content.config.ts`:
```typescript
extend: z.object({
  // ...
  category: z.string().optional(),
})
```

This allows the content collection to validate and expose the category for sidebar generation.
