# Astro/Starlight CMS Customizations

This document explains the custom modifications made to the Astro/Starlight setup for the Strands Agents documentation site.

## Overview

We're using [Astro](https://astro.build/) with the [Starlight](https://starlight.astro.build/) documentation theme. However, we've made several customizations to preserve compatibility with our existing MkDocs-based documentation structure and navigation.

## Reference Modules

Deep-dive documentation for specific subsystems lives in focused modules under [`reference/`](reference/). Read them on demand:

| Module | Covers |
|--------|--------|
| [Python API Reference Generation](reference/python-api-generation.md) | pydoc-markdown pipeline, symlink setup, dynamic sidebar, index page |
| [TypeScript API Reference Generation](reference/typescript-api-generation.md) | typedoc pipeline, post-processing, flat slugs, namespace exports |
| [Custom Landing Page and Testimonials](reference/landing-page.md) | Landing layout, landing page assets, testimonials content collection |
| [URL Redirects](reference/url-redirects.md) | Old MkDocs URL redirects, `redirectFrom` frontmatter, `SLUG_RULES`, tests |
| [LLM-Friendly Documentation (llms.txt)](reference/llms-txt.md) | llms.txt endpoints, HTML-to-markdown rendering, `SITE_DOMAIN` |

Authoring-facing patterns (MDX components, snippet syntax, frontmatter fields, callouts) are documented in [.agents/references/mdx-authoring.md](../.agents/references/mdx-authoring.md).

## Key Customizations

### 1. Sidebar Generation (`src/sidebar.ts`)

**What it does:** Reads the navigation structure from `src/config/navigation.yml` and converts it to Starlight's sidebar format. It does not apply any collapse behavior — that is handled entirely by the route middleware.

**Why:** Starlight can auto-generate sidebars from the file structure, but we have a specific navigation layout defined in `navigation.yml` that we want to preserve. The config file also contains navbar and GitHub dropdown configuration.

**Collapse opt-in:** Add `collapsed: true` to any group in `navigation.yml` to make it collapsed by default. This value is passed through to the middleware, which is the sole owner of collapse decisions.

**Badges:** Badges (like "new", "community", "experimental") come from page frontmatter, not the navigation config. This allows page authors to control badges directly.

### 2. Route Middleware (`src/route-middleware.ts`)

**What it does:** Filters the sidebar at buildtime so each page only shows items from its top-level group, and applies collapse behavior via `applyCollapse()`. For API pages (Python and TypeScript), it dynamically generates sidebars from the docs collection and computes pagination links.

**Why:** Our sidebar is organized into top-level groups (User Guide, Community, Examples, etc.). Without this middleware, every page would show the entire sidebar. This middleware scopes the sidebar to the current section, providing a cleaner navigation experience.

**Python API sidebar:** When viewing pages under `docs/api/python/`, the middleware uses `buildPythonApiSidebar()` from `src/dynamic-sidebar.ts` to generate a nested sidebar structure based on module names (e.g., `strands.agent.agent` becomes `Agent > Agent`).

**TypeScript API sidebar:** When viewing pages under `docs/api/typescript/`, the middleware uses `buildTypeScriptApiSidebar()` to generate a category-grouped sidebar (Classes, Interfaces, Type Aliases, Functions).

**Pagination:** For API pages, the middleware also updates `starlightRoute.pagination` using `getPrevNextLinks()` from `src/dynamic-sidebar.ts`. This ensures the previous/next navigation links at the bottom of pages work correctly with the dynamically generated sidebar. Pagination labels use actual page titles (from the docs collection) rather than sidebar nav labels.

**Non-matching pages:** Pages that don't belong to any nav section (e.g., the landing page) now show an empty sidebar instead of the full sidebar.

**Pagination pruning for regular pages:** Starlight pre-computes prev/next links from the full sidebar before middleware runs. The middleware now prunes any links that fall outside the current nav section and overrides labels with actual page titles.

### 3. MkDocs Snippets Plugin (`src/plugins/remark-mkdocs-snippets.ts`)

**What it does:** Processes MkDocs-style code snippet references (`--8<--`) in markdown files.

**Why:** Our existing documentation uses MkDocs' snippet syntax to include code from external files. This plugin provides compatibility so we don't need to rewrite all our code examples.

Snippet syntax and authoring patterns are documented in [mdx-authoring.md](../.agents/references/mdx-authoring.md#snippet-inclusion).

### 4. Relative Link Resolution (`src/util/links.ts`, `src/components/PageLink.astro`)

**What it does:** Converts MkDocs-style relative file links to Astro slug-based URLs at render time.

**Why:** MkDocs uses relative links to files (e.g., `../tools/custom-tools.md`), while Astro uses slugs by default and doesn't validate internal links. Rather than rewriting all links to use slugs, we override the default `<a>` element to resolve relative paths automatically. This provides a better authoring experience—linking to files feels more natural than memorizing slug paths.

**How it works:**

1. `PageLink.astro` replaces the default anchor element via `astro-auto-import`
2. When rendering a link, it checks if the href is relative (not absolute, not anchor-only)
3. For relative links, it strips the site's base path from the current URL before resolving, then re-applies it to the result — this ensures correct behavior when the site is deployed under a sub-path
4. The resolved path is matched against the content collection to find the correct slug
5. If no match is found, a warning is logged during development

**Example resolution:**

From page `user-guide/concepts/agents/state.mdx`:
- `conversation-management.md` → `/user-guide/concepts/agents/conversation-management/`
- `../tools/custom-tools.md` → `/user-guide/concepts/tools/custom-tools/`
- `../tools/index.md` → `/user-guide/concepts/tools/`

**Slug generation:** The content collection uses a custom `generateId` function in `src/content.config.ts` that shares the same normalization logic (`normalizePathToSlug`) as link resolution. This ensures consistency between how pages are identified and how links resolve to them.

The collection base is `src/content` (not `src/content/docs`), so all doc slugs include a `docs/` prefix (e.g., `docs/user-guide/concepts/agents/state`). The `generateId` function strips this prefix from path-based slugs so that URLs remain clean (e.g., `/docs/user-guide/...`). Files with an explicit `slug` frontmatter field (such as generated API docs) use that value directly and must include the `docs/` prefix themselves.

### 5. API Reference Links (`@api` shorthand)

**What it does:** Provides a shorthand format for linking to API reference pages that's cleaner than relative paths.

**Syntax:**
```markdown
<!-- Python API -->
[@api/python/strands.agent.agent](link text)
[@api/python/strands.agent.agent#AgentResult](link text with anchor)

<!-- TypeScript API -->
[@api/typescript/Agent](link text)
[@api/typescript/Agent#constructor](link text with anchor)
```

**How it works:**

1. Links starting with `@api/` are detected by `isApiShorthand()` in `src/util/links.ts`
2. `resolveApiShorthand()` converts them to absolute paths (e.g., `/docs/api/python/strands.agent.agent/`)
3. `PageLink.astro` applies the site's base path for correct URL generation

**Why use this format:**
- Cleaner than relative paths with `../api-reference/python/...`
- Doesn't break when the linking page moves to a different directory
- Matches the actual URL structure of the generated API docs
- Validated against the content collection at build time

**Examples:**
```markdown
<!-- Instead of this (fragile, verbose): -->
[AgentResult](../api-reference/python/agent/agent_result.md#strands.agent.agent_result.AgentResult)

<!-- Use this (clean, stable): -->
[AgentResult](@api/python/strands.agent.agent_result#AgentResult)
```

## Configuration (`astro.config.mjs`)

The main config ties everything together:

```javascript
import { loadSidebarFromConfig } from "./src/sidebar.ts"
import remarkMkdocsSnippets from './src/plugins/remark-mkdocs-snippets.ts'
import AutoImport from 'astro-auto-import'

const sidebar = loadSidebarFromConfig(
  path.resolve('./src/config/navigation.yml'),
  path.resolve('./src/content')  // base is src/content, not src/content/docs
)

export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMkdocsSnippets],
  },
  integrations: [
    astroExpressiveCode({
      themes: ['github-light', 'github-dark'],
      // Follow Starlight's data-theme attribute instead of prefers-color-scheme
      themeCssSelector: (theme) => `[data-theme='${theme.type}']`,
    }),
    starlight({
      markdown: {
        // Ensures Starlight's rehype plugins run on API docs symlinked from .build/api-docs
        processedDirs: [path.resolve('.build/api-docs')],
      },
      sidebar: sidebar,
      routeMiddleware: './src/route-middleware.ts',
      // ...
    }),
    AutoImport({
      imports: [/* ... */],
      defaultComponents: {
        // Override anchor elements for relative link resolution
        a: './src/components/PageLink.astro'
      }
    })
  ],
})
```

Notable config details:
- `themeCssSelector` on Expressive Code makes code block themes follow Starlight's `[data-theme]` attribute rather than the browser's `prefers-color-scheme`, keeping syntax highlighting in sync with the site's theme toggle.
- `processedDirs` tells Starlight to run its rehype plugins (e.g. heading anchor links) on the real resolved paths of the API docs symlinks.

## Custom Components (`src/components/`)

For authoring guidance on using these components in MDX pages (auto-imports, when to use `<Syntax>` vs `<Tabs>`, frontmatter fields, sidebar badges), see [mdx-authoring.md](../.agents/references/mdx-authoring.md). This section documents the component implementations.

### `AutoSyncTabs`

A wrapper around Starlight's `Tabs` that auto-generates a `syncKey` from tab labels. Tabs with identical label sets automatically sync together across the page. Auto-imported globally as `Tabs` (and `Tab` maps to Starlight's `TabItem`) via [astro-auto-import](https://github.com/delucis/astro-auto-import).

### `Syntax`

Inline component for language-specific identifiers in prose. Renders the Python or TypeScript variant based on the global language toggle, avoiding the `"python_name (Python) or tsName (TypeScript)"` pattern. Auto-imported.

```mdx
Pass <Syntax py="context_manager" ts="contextManager" /> to configure...
```

Props:
- `py` (required): Python syntax variant
- `ts` (required): TypeScript syntax variant
- `plain` (default: `false`): Renders as plain text instead of `<code>`.

The component reads the same `localStorage` key as the language toggle and swaps live without page reload. For guidance on when to use `<Syntax>` versus `<Tabs>`, see [mdx-authoring.md](../.agents/references/mdx-authoring.md).

### `PageLink`

Replaces the default anchor element to enable MkDocs-style relative linking. Resolves relative hrefs against the current page's path and validates against the content collection. Logs warnings in development for broken links. Auto-imported as the default `a` element.

### Starlight Overrides (`src/components/overrides/`)

These override default Starlight components:

- **`Head.astro`**: Adds Mermaid diagram support and loads `SiteScripts` (Shortbread + WebSDK).
- **`Header.astro`**: Custom header with navigation tabs and theme-aware logos (see [Header Navigation](#header-navigation) below).
- **`Hero.astro`**: Suppresses the Starlight hero on `/blog/` paths. Blog pages pass a dummy `hero: { actions: [] }` to collapse Starlight's two-panel layout, and this override ensures that dummy hero has no visual output.
- **`MarkdownContent.astro`**: Injects the custom frontmatter banners (experimental, community, languages) at the top of page content.
- **`PageFrame.astro`**: Extends Starlight's default `PageFrame` to add a full-width site footer containing the `Copyright` component. The footer spans the content area (respecting sidebar offset) with `--sl-color-bg-nav` background to match the header.
- **`Sidebar.astro`** and **`SidebarSublist.astro`**: Custom sidebar navigation that mimics MkDocs Material theme's `navigation.sections` behavior.

#### Sidebar Navigation Style

The custom sidebar components provide a flatter navigation style.

**How it works:**
1. Top-level groups render as non-collapsible section headers (uppercase labels), unless `collapsed: true` is set in `navigation.yml`, in which case they render as a collapsible with a caret icon
2. Nested groups are collapsible with a caret icon
3. Group labels link to their first child page (clickable navigation)
4. Groups auto-expand when they contain the current page
5. Indentation only starts at depth 2+ (first level under section headers has no indent)

**Why:** Starlight's default sidebar shows all groups as collapsible accordions. This override provides a cleaner hierarchy where top-level sections are always visible, and nested groups can be both navigated to and expanded.

### Header Navigation

The custom header (`src/components/overrides/Header.astro`) replicates the navigation tabs from the MkDocs Material theme used on strandsagents.com.

**Features:**
- Navigation tabs displayed below the main header row on desktop
- Mobile dropdown menu next to the search bar for small screens
- GitHub repository dropdown (`src/components/GitHubDropdown.astro`) replacing the default social icons
- Theme-aware logos (`logo-header-light.svg` / `logo-header-dark.svg`)
- Active state detection using longest-match path logic

**Configuring Navigation Links:**

Edit `src/config/navbar.ts` to add, remove, or reorder navigation links:

```typescript
const rawNavLinks: NavLink[] = [
  { label: 'Home', href: '/' },
  {
    label: 'User Guide',
    href: '/user-guide/quickstart/overview/',
    basePath: '/user-guide/',  // Used for active state detection
  },
  {
    label: 'Contribute ❤️',
    href: 'https://github.com/strands-agents/harness-sdk/blob/main/CONTRIBUTING.md',
    external: true,  // Opens in new tab with arrow icon
  },
]
```

**GitHub dropdown:** `src/config/navbar.ts` also exports `githubSections` — an array of grouped repository links shown in the `GitHubDropdown` component (desktop) and the mobile nav menu. Edit this to add or remove repos/orgs.

**Active state logic:** The header uses `findCurrentNavSection()` from `src/route-middleware.ts` to determine which tab is active. It finds the nav link with the longest matching `basePath` (or `href` if no `basePath`) that the current URL starts with.

**Theme-aware logos:** The header renders both `logo-header-light.svg` and `logo-header-dark.svg`, using CSS to show the appropriate one based on the `[data-theme]` attribute. Logo files are in `src/assets/`.

### Internal Aside Components

Used by `MarkdownContent.astro` to render frontmatter banners:

- `ExperimentalAside.astro`
- `CommunityContributionAside.astro`
- `LanguageSupportAside.astro`

These are not meant to be imported directly in MDX files—use the frontmatter fields instead.

## Temporary Migration Files

The following files were created to support the MkDocs → Astro migration and should be deleted once migration is complete:

### Link Conversion Utilities

These files handle converting old MkDocs-style API reference links to the new `@api` shorthand format:

- `src/util/api-link-converter.ts` - Utility functions to detect and convert old API links
- `test/api-link-converter.test.ts` - Tests for the link converter

### Migration Scripts

These scripts assist with documentation maintenance:

- `scripts/update-quickstart.ts` - Quickstart-specific transformations
- `scripts/update-language-index.ts` - Updates language index pages
- `test/update-docs.test.ts` - Tests for API link conversion utilities

## Blog

The blog is a standalone section at `/blog/` with its own content collection, layouts, components, and routes — outside of Starlight's docs collection. It follows the same pattern as the custom landing page: reuses the Starlight header via `BlogLayout.astro` while opting out of the docs chrome (sidebar, table of contents, etc.).

### Content Collections

**Authors** (`src/content/authors.yaml`):
```yaml
- id: strands-team
  name: Strands Agents Team
  role: Core Team
  bio: The team behind the Strands Agents SDK.
```

Schema: `{ id, name, role, bio, avatar? }` — all strings. The `id` field is used as the reference key from blog post frontmatter. Stored as a single YAML file (array of author objects) rather than individual JSON files per author.

**Blog Posts** (`src/content/blog/*.mdx`):
```yaml
---
title: "Post Title"
date: 2026-02-20T00:00:00.000Z
description: "Short description for cards and meta tags."
authors: ["strands-team"]     # References author file IDs
tags: ["Open Source"]
draft: false                  # Excluded from production builds
coverImage: "/path/to/image"  # Optional
---
```

The `readingTime` field is injected automatically by the remark plugin (see below).

Both collections are registered in `src/content.config.ts` using glob loaders, following the same pattern as testimonials.

### Reading Time Remark Plugin (`src/plugins/remark-reading-time.ts`)

Extracts text from the markdown AST and injects a `readingTime` string (e.g., "3 min read") into `file.data.astro.frontmatter`. Registered in `astro.config.mjs` under `markdown.remarkPlugins`.

Dependencies: `reading-time`, `mdast-util-to-string`.

### Blog Utilities (`src/util/blog.ts`)

Helper functions used across all blog pages:

| Function | Purpose |
|----------|---------|
| `getPublishedPosts()` | All posts sorted by date desc, excludes drafts in prod |
| `getAllTags()` | Unique tags across all published posts |
| `getPostsByTag(tag)` | Posts filtered by tag |
| `getPostsByAuthor(authorId)` | Posts filtered by author ID |
| `resolveAuthors(ids)` | Looks up author collection entries by ID |
| `tagToSlug(tag)` / `slugToTag(slug)` | Bidirectional tag↔URL conversion |
| `formatDate(date)` | Human-readable date (e.g., "February 20, 2026") |

### Layouts

**`BlogLayout.astro`** — Base layout for all blog pages. Uses Starlight's `<StarlightPage>` component to get the full page shell (head, styles, theme, header) for free. Passes `hasSidebar={false}` and `template: 'splash'` to suppress sidebar and doc-page chrome. Passes `hero: { actions: [] }` to collapse Starlight's two-panel layout into a single content panel (suppressing the auto-generated `PageTitle`). Extra head tags (canonical URL, OG/Twitter meta, RSS autodiscovery) are injected via the `frontmatter.head` array. A named `<slot name="head" />` is forwarded for page-specific head content (e.g. JSON-LD). The `Hero` component override (`src/components/overrides/Hero.astro`) suppresses the hero on `/blog/` paths so the dummy hero value has no visual effect.

**`BlogPostLayout.astro`** — Wraps `BlogLayout` with article-specific chrome: title, date, reading time, description, author byline, tags, cover image. Injects JSON-LD Article schema via the head slot. OG image URL: `/blog/og/{slug}.png`.

### Components (`src/components/blog/`)

| Component | Purpose |
|-----------|---------|
| `BlogCard.astro` | Card for listing pages (cover, title, description, meta, tags). Glassmorphism styling matching landing page. |
| `BlogAuthorByline.astro` | Author avatar + name + role, links to `/blog/authors/[id]/` |
| `BlogTagList.astro` | Tag chips linking to `/blog/tags/[tagSlug]/` |
| `BlogPostGrid.astro` | Reusable card grid (auto-fill, 320px min, 1200px max). Resolves authors for all posts. |

### Pages

| Route | File | Description |
|-------|------|-------------|
| `/blog/` | `src/pages/blog/index.astro` | Index with tag filter bar + post grid |
| `/blog/[slug]` | `src/pages/blog/[slug].astro` | Individual post (via `getStaticPaths`) |
| `/blog/tags/[tag]/` | `src/pages/blog/tags/[tag].astro` | Posts filtered by tag |
| `/blog/authors/[author]/` | `src/pages/blog/authors/[author].astro` | Author page with bio + their posts |

### Navigation

Blog is added to the header nav in `src/config/navbar.ts`:
```typescript
{ label: 'Blog', href: '/blog/', basePath: '/blog/' }
```
Active state is handled by the existing `findCurrentNavSection()` longest-match logic.

### RSS Feeds

| Endpoint | File |
|----------|------|
| `/blog/feed.xml` | `src/pages/blog/feed.xml.ts` — Main feed (all posts) |
| `/blog/feed/[tag].xml` | `src/pages/blog/feed/[tag].xml.ts` — Per-tag feeds |

Uses `@astrojs/rss`. Currently includes description only (not full rendered content).

### AEO (Agentic Engine Optimization)

The blog extends the existing [llms.txt system](reference/llms-txt.md):

- **`/blog/[slug]/index.md`** — Raw markdown endpoint for each post (mirrors the `[...slug]/index.md.ts` pattern for docs). Uses `renderEntryToMarkdown()` with `basePath: /blog/${post.id}/`.
- **`/llms.txt`** — Extended with a `## Blog` section listing links to blog markdown endpoints.
- **`/llms-full.txt`** — Extended to render blog posts inline after docs content.
- **`src/util/render-to-markdown.ts`** — Generalized from `CollectionEntry<'docs'>` to `CollectionEntry<'docs'> | CollectionEntry<'blog'>` with an optional `basePath` parameter.

### OG Images

Build-time OG image generation at `/blog/og/[slug].png` using `astro-og-canvas`:
- 1200×630px images from post title + description
- Strands branding: dark background (#0E0E0E), Strands green (#00CC5F) left border

Implementation: `src/pages/blog/og/[slug].png.ts`

### robots.txt

`public/robots.txt` — Allows all crawlers including GPTBot, ClaudeBot, PerplexityBot. References sitemap.

## Dependency Version Pinning

### `astro-broken-links-checker`

This package is pinned to an exact version (`1.0.6`) rather than using a semver range. It's a low-popularity package, so we avoid automatic updates to prevent potentially pulling in malicious or breaking changes without an explicit review. Before upgrading, manually inspect the changelog and diff on the package's repository.

**Known bug:** The upstream plugin does not account for Astro's `base` path configuration, causing it to incorrectly flag all internal links as broken when the site is deployed under a sub-path. See [imazen/astro-broken-link-checker#16](https://github.com/imazen/astro-broken-link-checker/issues/16).

**Local fix:** Rather than waiting for an upstream fix, the plugin source has been inlined into `scripts/astro-broken-links-checker-index.js` and `scripts/astro-broken-links-checker-check-links.js`. The fix captures `config.base` in the `astro:config:setup` hook and strips the base prefix from internal links before resolving them against the `dist/` directory. `astro.config.mjs` imports from the local copy instead of the npm package.
