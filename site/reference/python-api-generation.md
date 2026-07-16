# Python API Reference Generation

Part of the site architecture reference. See [SITE-ARCHITECTURE.md](../SITE-ARCHITECTURE.md) for the overview. Paths are relative to `site/`.

The Python API reference documentation is auto-generated from the SDK source code using pydoc-markdown.

## Generation Script (`scripts/api-generation-python.py`)

**What it does:** Parses Python source code from the SDK and generates MDX documentation files.

**How to run:**
```bash
uv run scripts/api-generation-python.py
```

**Input:** `.build/sdk-python/src` (cloned SDK repository)
**Output:** `.build/api-docs/python/*.mdx`

**Filtering:**
- Skips private modules (any module path containing `_` prefix)
- Skips explicitly excluded modules (e.g., `strands.agent` which just re-exports)

**Output format:** Each module becomes a flat MDX file named `strands.module.name.mdx` with frontmatter containing the title, slug, and `editUrl: false` (suppresses the "Edit this page" link on generated pages).

## Symlink Setup

The generated docs are accessed via a committed symlink:
```
src/content/docs/api/python/_generated -> ../../../../../.build/api-docs/python
```

This symlink is checked into git, so no manual setup is required. The generation script outputs to `.build/api-docs/python/`, and the symlink makes those files available to the content collection.

The index page (`src/content/docs/api/python/index.mdx`) is a permanent file (not generated) that imports the `PythonApiList` component.

## Dynamic Sidebar (`src/dynamic-sidebar.ts`)

**What it does:** Builds a hierarchical sidebar structure from Python API docs at runtime, and provides pagination utilities.

**How it works:**
1. Filters docs collection for `docs/api/python/*` pages
2. Parses module names from page titles (e.g., `strands.agent.agent`)
3. Builds a nested tree structure based on module path segments
4. Converts tree to Starlight sidebar entries with groups and links

**Pagination utilities:**
- `flattenSidebar()` - Converts nested sidebar structure to a flat list of links
- `getPrevNextLinks(sidebar, titlesByHref?)` - Finds the current page in the flattened sidebar and returns prev/next links. The optional `titlesByHref` map (href → title) overrides sidebar nav labels with actual page titles.

**Sorting:**
- Alphabetical A-Z within each level
- "Experimental" group always appears last
- Groups at depth ≥2 are collapsed by default

**Example transformation:**
```
strands.agent.agent      → Agent > Agent
strands.agent.base       → Agent > Base
strands.experimental.bidi.types.events → Experimental > Bidi > Types > Events
```

## Index Page Component (`src/components/PythonApiList.astro`)

**What it does:** Renders the API reference index page with a hierarchical list of all modules.

**How it works:** Uses the same `buildPythonApiSidebar()` function as the route middleware to ensure consistency between the sidebar navigation and the index page listing.

## Path Alias

Components can be imported using the `@components` alias:
```typescript
import PythonApiList from '@components/PythonApiList.astro'
```

This is configured in `tsconfig.json` under `compilerOptions.paths`.
