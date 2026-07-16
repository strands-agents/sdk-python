# URL Redirects (Old MkDocs URLs to New CMS URLs)

Part of the site architecture reference. See [SITE-ARCHITECTURE.md](../SITE-ARCHITECTURE.md) for the overview. Paths are relative to `site/`.

The old MkDocs site used versioned URLs like `/latest/documentation/docs/<path>/` and `/1.x/documentation/docs/<path>/`. The new CMS uses clean paths like `/docs/<path>/`. Some pages also moved or were renamed. The redirect system handles both cases client-side via the 404 page.

## How It Works

1. **404 page** (`src/content/404.mdx`) renders `Redirect404.astro`, which runs a client-side script on every 404.
2. **`Redirect404.astro`** (`src/components/Redirect404.astro`) builds a `redirectFromMap` at build time (via `src/util/redirect.build.ts`) and passes it to the client-side script, which calls `resolveRedirectFromUrl()` with the current URL and map. If a target is found, `window.location.replace()` fires without adding a history entry.
3. **`src/util/redirect.ts`** contains the redirect logic:
   - `resolveRedirectFromUrl(url, redirectFromMap?)` — strips the version prefix (`/latest/`, `/1.x/`, `/1.5.x/`, etc.) and the `/documentation/` segment, then delegates to `resolveRedirect()`.
   - `resolveRedirect(slug, redirectFromMap?)` — checks `SLUG_RULES` first (highest priority), then falls back to `redirectFromMap` for frontmatter-based redirects.
4. **`src/util/redirect.build.ts`** — shared helper that calls `getCollection('docs')` and builds the `redirectFromMap` from all `redirectFrom` frontmatter arrays. Used by both `Redirect404.astro` and the sitemap coverage test.

## Page-Level Redirects (`redirectFrom` frontmatter)

Individual pages can declare old slugs that should redirect to them:

```yaml
---
title: My Page
redirectFrom:
  - docs/old/path/to/page
---
```

This is useful when a page moves to a new URL. The `redirectFrom` slugs are collected at build time into a map and passed to the client-side redirect script. They must also be registered in `test/known-routes.json` — the sitemap coverage test enforces this.

## Adding New Redirect Rules

Edit `SLUG_RULES` in `src/util/redirect.ts` for structural renames affecting many pages. For single-page moves, prefer `redirectFrom` frontmatter instead. Each `SLUG_RULES` entry has a `match` regex and a `to` string or function:

```typescript
// Static rename
{ match: exactly('docs/old/path'), to: 'docs/new/path' },

// Pattern-based rename (capture group 1 = everything after the prefix)
{ match: startsWith('docs/old-prefix'), to: (m) => `docs/new-prefix/${m[1]}` },
```

Helper builders from `src/utils/regex.ts`:
- `startsWith(prefix)` — matches slugs starting with `prefix/`, captures the rest in `m[1]`
- `exactly(s)` — matches the slug exactly

## Testing

- **`test/redirect.test.ts`** — unit tests for `resolveRedirect` and `resolveRedirectFromUrl` covering slug transforms, URL normalisation, trailing-slash preservation, and `redirectFromMap` priority rules.
- **`test/sitemap-coverage.test.ts`** — integration tests including:
  - Every `redirectFrom` slug declared in frontmatter has a corresponding entry in `test/known-routes.json` (fails with copy-paste-ready JSON if missing).
  - Every known route resolves to a valid CMS entry (uses `buildRedirectFromMap()` so frontmatter-based redirects are honoured).
  - Live sitemap coverage (controlled by `VERIFY_LIVE_SITEMAP=true`, skipped locally).

Run with:
```bash
npm test
```
