# Architecture

## Content Location

`src/content/docs` is a symlink to `../../docs`. Starlight expects content in `src/content/docs/`, providing a consistent structure for documentation.

## Code Snippets from TypeScript Files

We use a custom remark plugin (`src/plugins/remark-mkdocs-snippets.ts`) to include TypeScript code in markdown docs. This keeps code examples type-checked and in sync with actual source files.

**Syntax in markdown:**
```typescript
--8<-- "user-guide/concepts/agents/hooks.ts:basic_example"
```

**Source file markers:**
```typescript
// --8<-- [start:basic_example]
const agent = new Agent({ tools: [myTool] })
// --8<-- [end:basic_example]
```

The plugin extracts the section between markers, dedents it, and inlines it into the code block. Paths resolve from `src/content/docs/`.

**Files:** `src/plugins/remark-mkdocs-snippets.ts`, `astro.config.mjs` (registers plugin)

## Mermaid Diagram Rendering

Starlight doesn't support Mermaid diagrams out of the box. We add client-side rendering via a component override.

**How it works:**

1. Override Starlight's `Head` component (`src/components/Head.astro`)
2. Script finds `<pre data-language="mermaid">` blocks on page load
3. Transforms them into `<pre class="mermaid">` elements
4. Mermaid.js (loaded from CDN) renders them as SVGs

**Why client-side?** Simpler than build-time rendering (no Puppeteer), and can match Starlight's dark/light theme.

**Files:** `src/components/Head.astro`, `astro.config.mjs` (registers override)
