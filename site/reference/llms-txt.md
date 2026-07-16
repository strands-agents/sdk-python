# LLM-Friendly Documentation (`llms.txt`)

Part of the site architecture reference. See [SITE-ARCHITECTURE.md](../SITE-ARCHITECTURE.md) for the overview. Paths are relative to `site/`.

We provide machine-readable documentation following the [llms.txt specification](https://llmstxt.org/), optimized for both humans and AI agents.

## Why Custom Implementation

We evaluated existing Astro llms.txt plugins/integrations but found them lacking:
- They generated HTML or poorly formatted markdown with navigation clutter
- Links weren't properly resolved to our documentation structure
- No support for our custom components (tabs, code snippets, etc.)

Our implementation renders documentation through Astro's container API, applies custom HTML-to-markdown transformations, and generates clean output with correct links.

## Endpoints

- `/llms.txt` - Index with links to all docs organized by sidebar structure
- `/llms-full.txt` - Complete documentation content (excludes API reference)
- `/{slug}/index.md` - Any doc page in raw markdown format

## Implementation Files

| File | Purpose |
|------|---------|
| `src/pages/llms.txt.ts` | Generates index from sidebar structure |
| `src/pages/llms-full.txt.ts` | Renders all docs inline |
| `src/pages/[...slug]/index.md.ts` | Dynamic endpoint for individual pages |
| `src/util/render-to-markdown.ts` | Renders MDX entries via AstroContainer |
| `src/util/html-to-markdown.ts` | HTML→Markdown conversion with custom rules |

## HTML-to-Markdown Transformations

Uses [Turndown](https://github.com/mixmark-io/turndown) with custom rules:

- **Tables**: GFM plugin for proper markdown table syntax
- **Code blocks**: Handles both standard and Expressive Code syntax highlighting
- **Tab panels**: Wraps content with `(( tab "Label" ))` markers
- **Local links**: Rewrites to `/index.md` format for LLM consumption
- **Cleanup**: Removes screen-reader elements, empty anchors, tab navigation lists, scripts

## Link Handling

The `src/util/links.ts` module was extended:
- `toRawMarkdownUrl()` - Converts paths to index.md URLs, skips files with extensions
- `isLocalLink()` - Identifies links that should be converted (excludes .txt, external, anchors)
- `resolveHref()` - Special-cases `llms.txt` and `llms-full.txt` for proper resolution
- `getSiteOrigin()` - Returns the value of the `SITE_DOMAIN` environment variable (trailing slash stripped), or an empty string if unset. Used by `llms.txt` and `llms-full.txt` to produce absolute URLs when a domain is known.

## Absolute URLs via `SITE_DOMAIN`

By default, links in `llms.txt` and `llms-full.txt` are relative (path-only). Set the `SITE_DOMAIN` environment variable at build time to prefix all links with the full domain:

```bash
SITE_DOMAIN=https://strandsagents.com npm run build
```

Without `SITE_DOMAIN`, links remain relative (e.g. `/user-guide/quickstart/`). With it set, they become absolute (e.g. `https://strandsagents.com/user-guide/quickstart/`).
