# Community Integration Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static community-packages tables with a searchable, filterable `/catalog` page backed by a YAML content collection, plus a scheduled stats-refresh workflow and a one-time PyPI/npm/GitHub backfill.

**Architecture:** A new `catalog` Astro content collection (one YAML file per integration, zod-validated) is the source of truth. A standalone `src/pages/catalog.astro` page renders all entries statically and filters them with a small vanilla client script (the same pattern as `src/pages/changelog/index.astro`). Popularity stats live in a bot-maintained JSON file joined at build time. Doc pages under `community/` remain the optional detail pages.

**Tech Stack:** Astro 7 + Starlight, zod (via `astro/zod`), vitest, vanilla TS client scripts, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-17-community-catalog-design.md`

## Global Constraints

- All work is inside `site/` except `.github/PULL_REQUEST_TEMPLATE/catalog-submission.md` and `.github/workflows/catalog-stats.yml`. Run npm commands from `site/`; run git commands from the repo root (all `git add` paths are repo-root-relative).
- Code style: Prettier, no semicolons, single quotes, 2-space indent, 120-char lines (site convention).
- No new runtime dependencies. No client-side frameworks. Filtering is vanilla script; no search library.
- `integrationType` enum (shared with docs frontmatter): `model-provider`, `tool`, `session-manager`, `memory-store`, `integration`, `plugin`, `agent-extension`, `intervention`.
- `sdk` enum: `agents`, `evals`; defaults to `agents`. The SDK filter facet renders only when ≥1 entry has `sdk: evals`.
- Trust badges enum: `verified` (extensible later). Language and category badges are derived, never declared.
- Comments are evergreen: WHAT/WHY only, never "changed from"/"previously".
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `ci:`).
- Structured logging in scripts: `field=<value> | lowercase message` (template literals in TS).
- Tests: vitest, files in `site/test/*.test.ts`. Astro collection access in tests works because `test/global-setup.ts` populates the data store. Run with `npm test -- <file>` from `site/`.
- Typecheck with `npm run typecheck` from `site/` before each commit.

---

### Task 1: Catalog content collection schema

**Files:**
- Modify: `site/src/content.config.ts` (add `catalogEntrySchema` + `catalog` collection; collections map is at the bottom, `~line 88`)
- Create: `site/src/content/catalog/strands-deepgram.yaml` (first real entry; serves as the fixture)
- Test: `site/test/catalog-collection.test.ts`

**Interfaces:**
- Produces: `catalogEntrySchema` (exported zod schema), `CatalogEntryData` (exported type), and a `catalog` collection queryable via `getCollection('catalog')`. Entry shape:

```ts
{
  name: string
  description: string
  integrationType: 'model-provider' | 'tool' | 'session-manager' | 'memory-store' | 'integration' | 'plugin' | 'agent-extension' | 'intervention'
  sdk: 'agents' | 'evals'                       // default 'agents'
  languages: {
    python?: { package: string; registry: string }      // registry: full PyPI URL
    typescript?: { package: string; registry: string }  // registry: full npm URL
  }
  github: string                                 // repo URL
  maintainer: string                             // GitHub username or org
  docsPage?: string                              // docs collection id, e.g. 'docs/community/tools/strands-deepgram'
  featured: boolean                              // default false
  badges: ('verified')[]                         // default []
  addedDate: Date                                // z.coerce.date()
}
```

- [ ] **Step 1: Write the failing test**

Create `site/test/catalog-collection.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { getCollection } from 'astro:content'
import { catalogEntrySchema } from '../src/content.config'

describe('catalog content collection', () => {
  it('loads catalog entries with validated data', async () => {
    const entries = await getCollection('catalog')
    expect(entries.length).toBeGreaterThan(0)

    const deepgram = entries.find((e) => e.id === 'strands-deepgram')
    expect(deepgram).toBeDefined()
    expect(deepgram!.data.name).toBe('strands-deepgram')
    expect(deepgram!.data.integrationType).toBe('tool')
    expect(deepgram!.data.sdk).toBe('agents')
    expect(deepgram!.data.languages.python?.package).toBe('strands-deepgram')
    expect(deepgram!.data.languages.typescript).toBeUndefined()
    expect(deepgram!.data.featured).toBe(false)
    expect(deepgram!.data.badges).toEqual([])
  })

  it('rejects an entry with no language blocks', () => {
    const result = catalogEntrySchema.safeParse({
      name: 'bad-entry',
      description: 'missing languages',
      integrationType: 'tool',
      languages: {},
      github: 'https://github.com/example/bad-entry',
      maintainer: 'example',
      addedDate: '2026-07-17',
    })
    expect(result.success).toBe(false)
  })

  it('rejects an unknown integrationType', () => {
    const result = catalogEntrySchema.safeParse({
      name: 'bad-type',
      description: 'bad type',
      integrationType: 'widget',
      languages: { python: { package: 'x', registry: 'https://pypi.org/project/x/' } },
      github: 'https://github.com/example/x',
      maintainer: 'example',
      addedDate: '2026-07-17',
    })
    expect(result.success).toBe(false)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npm test -- test/catalog-collection.test.ts`
Expected: FAIL — `catalogEntrySchema` is not exported / collection `catalog` does not exist.

- [ ] **Step 3: Add schema and collection to `content.config.ts`**

Add after `changelogFrontmatterSchema` (before `blogSchema`):

```ts
const catalogLanguageSchema = z.object({
  // Package name as published on the registry
  package: z.string(),
  // Full registry URL (PyPI project page or npm package page)
  registry: z.string().url(),
})

export const catalogEntrySchema = z
  .object({
    name: z.string(),
    description: z.string(),
    // Shared enum with the docs frontmatter integrationType below — keep in sync
    integrationType: z.enum([
      'model-provider',
      'tool',
      'session-manager',
      'memory-store',
      'integration',
      'plugin',
      'agent-extension',
      'intervention',
    ]),
    // Which SDK's ecosystem this belongs to. The catalog's SDK facet stays
    // hidden until at least one evals entry exists.
    sdk: z.enum(['agents', 'evals']).default('agents'),
    languages: z.object({
      python: catalogLanguageSchema.optional(),
      typescript: catalogLanguageSchema.optional(),
    }),
    github: z.string().url(),
    maintainer: z.string(),
    // Docs collection id of the detail page (e.g. 'docs/community/tools/strands-deepgram').
    // Optional: entries without one link out to their GitHub repo instead.
    docsPage: z.string().optional(),
    // Editorial fields — maintainer-granted only; submitters leave them unset.
    featured: z.boolean().default(false),
    badges: z.array(z.enum(['verified'])).default([]),
    // Drives the "New" badge on the catalog card.
    addedDate: z.coerce.date(),
  })
  .superRefine((d, ctx) => {
    if (!d.languages.python && !d.languages.typescript) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['languages'],
        message: 'at least one language block (python or typescript) is required',
      })
    }
  })
export type CatalogEntryData = z.infer<typeof catalogEntrySchema>
```

Add to the `collections` map (alongside `authors`, `blog`, …):

```ts
  catalog: defineCollection({
    loader: glob({
      base: 'src/content/catalog',
      pattern: '**/*.yaml',
    }),
    schema: catalogEntrySchema,
  }),
```

- [ ] **Step 4: Create the first entry**

Create `site/src/content/catalog/strands-deepgram.yaml`:

```yaml
name: strands-deepgram
description: Deepgram speech-to-text, text-to-speech, and audio intelligence
integrationType: tool
languages:
  python:
    package: strands-deepgram
    registry: https://pypi.org/project/strands-deepgram/
github: https://github.com/eraykeskinmac/strands-deepgram
maintainer: eraykeskinmac
docsPage: docs/community/tools/strands-deepgram
addedDate: 2026-07-17
```

- [ ] **Step 5: Run test to verify it passes**

Run: `rm -rf .astro && npm test -- test/catalog-collection.test.ts`
(The `rm -rf .astro` forces the global-setup dev-server sync to pick up the new collection.)
Expected: PASS (3 tests)

- [ ] **Step 6: Typecheck and commit**

Run: `npm run typecheck`
Expected: no errors.

```bash
git add site/src/content.config.ts site/src/content/catalog/strands-deepgram.yaml site/test/catalog-collection.test.ts
git commit -m "feat(site): add catalog content collection with zod schema"
```

---

### Task 2: Catalog entry processing util (badges, stats join, ordering)

**Files:**
- Create: `site/src/util/catalog.ts`
- Create: `site/src/data/catalog-stats.json` (empty seed: `{}`)
- Test: `site/test/catalog.test.ts`

**Interfaces:**
- Consumes: `CatalogEntryData` from `../src/content.config` (Task 1).
- Produces (all exported from `site/src/util/catalog.ts`):

```ts
export interface CatalogStats {
  stars?: number
  downloads?: { python?: number; typescript?: number }
  lastRelease?: string // ISO date
}
export type CatalogStatsFile = Record<string, CatalogStats> // keyed by entry id (filename without .yaml)

export interface CatalogCardModel {
  id: string
  name: string
  description: string
  integrationType: CatalogEntryData['integrationType']
  sdk: 'agents' | 'evals'
  languages: ('python' | 'typescript')[]
  href: string            // docs page path when docsPage set, else github URL
  external: boolean       // true when href is the github URL
  github: string
  registryLinks: { label: 'PyPI' | 'npm'; href: string }[]
  maintainer: string
  featured: boolean
  badges: string[]        // trust badges + 'new' when addedDate within NEW_BADGE_DAYS of buildDate
  stars?: number
  downloads?: number      // summed across languages
}

export const NEW_BADGE_DAYS = 30
export function toCardModel(
  id: string,
  data: CatalogEntryData,
  stats: CatalogStats | undefined,
  buildDate: Date
): CatalogCardModel
export function sortEntries(cards: CatalogCardModel[]): CatalogCardModel[] // featured first, then name A-Z
export function hasEvalsEntries(cards: CatalogCardModel[]): boolean
```

- [ ] **Step 1: Write the failing test**

Create `site/test/catalog.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { toCardModel, sortEntries, hasEvalsEntries, NEW_BADGE_DAYS } from '../src/util/catalog'
import type { CatalogEntryData } from '../src/content.config'

const BUILD_DATE = new Date('2026-07-17')

function entry(overrides: Partial<CatalogEntryData> = {}): CatalogEntryData {
  return {
    name: 'strands-example',
    description: 'Example integration',
    integrationType: 'tool',
    sdk: 'agents',
    languages: { python: { package: 'strands-example', registry: 'https://pypi.org/project/strands-example/' } },
    github: 'https://github.com/example/strands-example',
    maintainer: 'example',
    featured: false,
    badges: [],
    addedDate: new Date('2026-01-01'),
    ...overrides,
  }
}

describe('toCardModel', () => {
  it('links to the docs page when docsPage is set', () => {
    const card = toCardModel('strands-example', entry({ docsPage: 'docs/community/tools/strands-example' }), undefined, BUILD_DATE)
    expect(card.href).toBe('/docs/community/tools/strands-example/')
    expect(card.external).toBe(false)
  })

  it('links out to github when there is no docs page', () => {
    const card = toCardModel('strands-example', entry(), undefined, BUILD_DATE)
    expect(card.href).toBe('https://github.com/example/strands-example')
    expect(card.external).toBe(true)
  })

  it('derives language list and registry links', () => {
    const card = toCardModel(
      'strands-example',
      entry({
        languages: {
          python: { package: 'strands-example', registry: 'https://pypi.org/project/strands-example/' },
          typescript: { package: '@example/strands', registry: 'https://www.npmjs.com/package/@example/strands' },
        },
      }),
      undefined,
      BUILD_DATE
    )
    expect(card.languages).toEqual(['python', 'typescript'])
    expect(card.registryLinks).toEqual([
      { label: 'PyPI', href: 'https://pypi.org/project/strands-example/' },
      { label: 'npm', href: 'https://www.npmjs.com/package/@example/strands' },
    ])
  })

  it('adds the new badge when addedDate is within the window', () => {
    const recent = new Date(BUILD_DATE.getTime() - (NEW_BADGE_DAYS - 1) * 86_400_000)
    const card = toCardModel('strands-example', entry({ addedDate: recent, badges: ['verified'] }), undefined, BUILD_DATE)
    expect(card.badges).toEqual(['verified', 'new'])
  })

  it('omits the new badge when addedDate is outside the window', () => {
    const card = toCardModel('strands-example', entry(), undefined, BUILD_DATE)
    expect(card.badges).toEqual([])
  })

  it('joins stats and sums downloads, tolerating missing stats', () => {
    const withStats = toCardModel('strands-example', entry(), { stars: 42, downloads: { python: 100, typescript: 50 } }, BUILD_DATE)
    expect(withStats.stars).toBe(42)
    expect(withStats.downloads).toBe(150)

    const withoutStats = toCardModel('strands-example', entry(), undefined, BUILD_DATE)
    expect(withoutStats.stars).toBeUndefined()
    expect(withoutStats.downloads).toBeUndefined()
  })
})

describe('sortEntries', () => {
  it('puts featured entries first, then sorts by name', () => {
    const cards = [
      toCardModel('zeta', entry({ name: 'zeta' }), undefined, BUILD_DATE),
      toCardModel('alpha', entry({ name: 'alpha' }), undefined, BUILD_DATE),
      toCardModel('feat', entry({ name: 'feat', featured: true }), undefined, BUILD_DATE),
    ]
    expect(sortEntries(cards).map((c) => c.name)).toEqual(['feat', 'alpha', 'zeta'])
  })
})

describe('hasEvalsEntries', () => {
  it('detects evals entries', () => {
    const agents = toCardModel('a', entry(), undefined, BUILD_DATE)
    const evals = toCardModel('b', entry({ sdk: 'evals' }), undefined, BUILD_DATE)
    expect(hasEvalsEntries([agents])).toBe(false)
    expect(hasEvalsEntries([agents, evals])).toBe(true)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- test/catalog.test.ts`
Expected: FAIL — cannot resolve `../src/util/catalog`.

- [ ] **Step 3: Implement `site/src/util/catalog.ts`**

```ts
/**
 * Build-time processing for catalog entries: derives card view-models from
 * collection data, joins the bot-maintained stats file, and orders the grid.
 */

import type { CatalogEntryData } from '../content.config'

export interface CatalogStats {
  stars?: number
  downloads?: { python?: number; typescript?: number }
  lastRelease?: string
}

/** Shape of src/data/catalog-stats.json — keyed by entry id (filename without .yaml). */
export type CatalogStatsFile = Record<string, CatalogStats>

export interface CatalogCardModel {
  id: string
  name: string
  description: string
  integrationType: CatalogEntryData['integrationType']
  sdk: 'agents' | 'evals'
  languages: ('python' | 'typescript')[]
  href: string
  external: boolean
  github: string
  registryLinks: { label: 'PyPI' | 'npm'; href: string }[]
  maintainer: string
  featured: boolean
  badges: string[]
  stars?: number
  downloads?: number
}

/** Window (in days) during which an entry carries the derived "new" badge. */
export const NEW_BADGE_DAYS = 30

export function toCardModel(
  id: string,
  data: CatalogEntryData,
  stats: CatalogStats | undefined,
  buildDate: Date
): CatalogCardModel {
  const languages: ('python' | 'typescript')[] = []
  const registryLinks: CatalogCardModel['registryLinks'] = []
  if (data.languages.python) {
    languages.push('python')
    registryLinks.push({ label: 'PyPI', href: data.languages.python.registry })
  }
  if (data.languages.typescript) {
    languages.push('typescript')
    registryLinks.push({ label: 'npm', href: data.languages.typescript.registry })
  }

  const badges = [...data.badges]
  const ageDays = (buildDate.getTime() - data.addedDate.getTime()) / 86_400_000
  if (ageDays >= 0 && ageDays < NEW_BADGE_DAYS) badges.push('new')

  const docsHref = data.docsPage ? `/${data.docsPage}/` : undefined

  const pythonDownloads = stats?.downloads?.python ?? 0
  const typescriptDownloads = stats?.downloads?.typescript ?? 0
  const totalDownloads = pythonDownloads + typescriptDownloads

  return {
    id,
    name: data.name,
    description: data.description,
    integrationType: data.integrationType,
    sdk: data.sdk,
    languages,
    href: docsHref ?? data.github,
    external: !docsHref,
    github: data.github,
    registryLinks,
    maintainer: data.maintainer,
    featured: data.featured,
    badges,
    stars: stats?.stars,
    downloads: totalDownloads > 0 ? totalDownloads : undefined,
  }
}

/** Featured entries first, then alphabetical by name. */
export function sortEntries(cards: CatalogCardModel[]): CatalogCardModel[] {
  return [...cards].sort((a, b) => {
    if (a.featured !== b.featured) return a.featured ? -1 : 1
    return a.name.localeCompare(b.name)
  })
}

/** The SDK facet renders only when the catalog actually has evals entries. */
export function hasEvalsEntries(cards: CatalogCardModel[]): boolean {
  return cards.some((c) => c.sdk === 'evals')
}
```

- [ ] **Step 4: Create the empty stats seed**

Create `site/src/data/catalog-stats.json` containing exactly:

```json
{}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm test -- test/catalog.test.ts`
Expected: PASS (9 tests)

- [ ] **Step 6: Typecheck and commit**

Run: `npm run typecheck`

```bash
git add site/src/util/catalog.ts site/src/data/catalog-stats.json site/test/catalog.test.ts
git commit -m "feat(site): add catalog card model derivation and stats join"
```

---

### Task 3: Client-side filter matching logic

**Files:**
- Create: `site/src/util/catalog-filter.ts`
- Test: `site/test/catalog-filter.test.ts`

**Interfaces:**
- Consumes: nothing (pure functions over primitives so the page script can import it and vitest can test it without DOM).
- Produces:

```ts
export interface CatalogFilterState {
  search: string                 // raw search input
  types: Set<string>             // selected integrationType values; empty = all
  languages: Set<string>         // 'python' | 'typescript'; empty = all
  badges: Set<string>            // trust badges; empty = all
  sdk: string                    // 'agents' | 'evals'; always exactly one
}
export interface CardFilterData {
  search: string                 // lowercased name + description
  type: string
  languages: string[]
  badges: string[]
  sdk: string
}
export function matchesFilters(card: CardFilterData, state: CatalogFilterState): boolean
export function stateToQuery(state: CatalogFilterState): string    // '' or 'q=...&type=a,b&lang=...&badge=...&sdk=evals'
export function queryToState(query: string): CatalogFilterState    // inverse; unknown values dropped
```

- [ ] **Step 1: Write the failing test**

Create `site/test/catalog-filter.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { matchesFilters, stateToQuery, queryToState, type CatalogFilterState, type CardFilterData } from '../src/util/catalog-filter'

function state(overrides: Partial<CatalogFilterState> = {}): CatalogFilterState {
  return { search: '', types: new Set(), languages: new Set(), badges: new Set(), sdk: 'agents', ...overrides }
}

const card: CardFilterData = {
  search: 'strands-deepgram deepgram speech-to-text and audio intelligence',
  type: 'tool',
  languages: ['python'],
  badges: ['verified'],
  sdk: 'agents',
}

describe('matchesFilters', () => {
  it('matches everything with the default state', () => {
    expect(matchesFilters(card, state())).toBe(true)
  })

  it('matches case-insensitive substrings against name+description', () => {
    expect(matchesFilters(card, state({ search: 'Speech-To-Text' }))).toBe(true)
    expect(matchesFilters(card, state({ search: 'telemetry' }))).toBe(false)
  })

  it('filters by integration type (OR within the facet)', () => {
    expect(matchesFilters(card, state({ types: new Set(['tool', 'plugin']) }))).toBe(true)
    expect(matchesFilters(card, state({ types: new Set(['plugin']) }))).toBe(false)
  })

  it('filters by language', () => {
    expect(matchesFilters(card, state({ languages: new Set(['python']) }))).toBe(true)
    expect(matchesFilters(card, state({ languages: new Set(['typescript']) }))).toBe(false)
  })

  it('filters by badge', () => {
    expect(matchesFilters(card, state({ badges: new Set(['verified']) }))).toBe(true)
    expect(matchesFilters({ ...card, badges: [] }, state({ badges: new Set(['verified']) }))).toBe(false)
  })

  it('always filters by sdk', () => {
    expect(matchesFilters(card, state({ sdk: 'evals' }))).toBe(false)
  })

  it('ANDs across facets', () => {
    expect(matchesFilters(card, state({ search: 'deepgram', types: new Set(['tool']), languages: new Set(['python']) }))).toBe(true)
    expect(matchesFilters(card, state({ search: 'deepgram', types: new Set(['plugin']) }))).toBe(false)
  })
})

describe('URL state round-trip', () => {
  it('serializes only non-default state', () => {
    expect(stateToQuery(state())).toBe('')
    expect(stateToQuery(state({ search: 'sql', types: new Set(['tool']) }))).toBe('q=sql&type=tool')
    expect(stateToQuery(state({ sdk: 'evals' }))).toBe('sdk=evals')
  })

  it('round-trips through queryToState', () => {
    const s = state({ search: 'sql', types: new Set(['tool', 'plugin']), languages: new Set(['python']), badges: new Set(['verified']), sdk: 'evals' })
    const back = queryToState(stateToQuery(s))
    expect(back.search).toBe('sql')
    expect(back.types).toEqual(new Set(['tool', 'plugin']))
    expect(back.languages).toEqual(new Set(['python']))
    expect(back.badges).toEqual(new Set(['verified']))
    expect(back.sdk).toBe('evals')
  })

  it('drops unknown values when parsing', () => {
    const back = queryToState('type=widget&lang=rust&sdk=nope')
    expect(back.types.size).toBe(0)
    expect(back.languages.size).toBe(0)
    expect(back.sdk).toBe('agents')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- test/catalog-filter.test.ts`
Expected: FAIL — cannot resolve `../src/util/catalog-filter`.

- [ ] **Step 3: Implement `site/src/util/catalog-filter.ts`**

```ts
/**
 * Pure filter-matching and URL-state logic for the catalog page. Kept free of
 * DOM access so the page script can import it and vitest can test it directly.
 */

export interface CatalogFilterState {
  search: string
  types: Set<string>
  languages: Set<string>
  badges: Set<string>
  sdk: string
}

export interface CardFilterData {
  search: string
  type: string
  languages: string[]
  badges: string[]
  sdk: string
}

const KNOWN_TYPES = new Set([
  'model-provider',
  'tool',
  'session-manager',
  'memory-store',
  'integration',
  'plugin',
  'agent-extension',
  'intervention',
])
const KNOWN_LANGUAGES = new Set(['python', 'typescript'])
const KNOWN_BADGES = new Set(['verified'])
const KNOWN_SDKS = new Set(['agents', 'evals'])
const DEFAULT_SDK = 'agents'

/** Facets AND together; selections within a facet OR. Empty facet = no constraint. */
export function matchesFilters(card: CardFilterData, state: CatalogFilterState): boolean {
  if (card.sdk !== state.sdk) return false
  const q = state.search.trim().toLowerCase()
  if (q && !card.search.includes(q)) return false
  if (state.types.size > 0 && !state.types.has(card.type)) return false
  if (state.languages.size > 0 && !card.languages.some((l) => state.languages.has(l))) return false
  if (state.badges.size > 0 && !card.badges.some((b) => state.badges.has(b))) return false
  return true
}

/** Serialize non-default state to a query string ('' when everything is default). */
export function stateToQuery(state: CatalogFilterState): string {
  const params = new URLSearchParams()
  if (state.search.trim()) params.set('q', state.search.trim())
  if (state.types.size > 0) params.set('type', [...state.types].sort().join(','))
  if (state.languages.size > 0) params.set('lang', [...state.languages].sort().join(','))
  if (state.badges.size > 0) params.set('badge', [...state.badges].sort().join(','))
  if (state.sdk !== DEFAULT_SDK) params.set('sdk', state.sdk)
  return params.toString()
}

/** Parse a query string into filter state, silently dropping unknown values. */
export function queryToState(query: string): CatalogFilterState {
  const params = new URLSearchParams(query)
  const pick = (key: string, known: Set<string>) =>
    new Set((params.get(key) || '').split(',').filter((v) => known.has(v)))
  const sdkParam = params.get('sdk') || DEFAULT_SDK
  return {
    search: params.get('q') || '',
    types: pick('type', KNOWN_TYPES),
    languages: pick('lang', KNOWN_LANGUAGES),
    badges: pick('badge', KNOWN_BADGES),
    sdk: KNOWN_SDKS.has(sdkParam) ? sdkParam : DEFAULT_SDK,
  }
}
```

Note on `stateToQuery` output order: `URLSearchParams` preserves insertion order, so the serialized string is `q`, `type`, `lang`, `badge`, `sdk` — matching the test expectations.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- test/catalog-filter.test.ts`
Expected: PASS

- [ ] **Step 5: Typecheck and commit**

Run: `npm run typecheck`

```bash
git add site/src/util/catalog-filter.ts site/test/catalog-filter.test.ts
git commit -m "feat(site): add catalog filter matching and URL state logic"
```

---

### Task 4: Migrate all existing community doc pages to catalog entries

**Files:**
- Create: `site/src/content/catalog/*.yaml` — one per existing community integration page (~28 files, listed below)
- Test: extend `site/test/catalog-collection.test.ts`

**Interfaces:**
- Consumes: `catalog` collection (Task 1).
- Produces: a complete catalog matching today's community pages. Later tasks (page rendering) rely on this data existing.

- [ ] **Step 1: Extend the collection test with integrity checks**

Add to `site/test/catalog-collection.test.ts`:

```ts
  it('every docsPage points at a real docs collection entry', async () => {
    const [entries, docs] = await Promise.all([getCollection('catalog'), getCollection('docs')])
    const docIds = new Set(docs.map((d) => d.id))
    for (const e of entries) {
      if (e.data.docsPage) {
        expect(docIds.has(e.data.docsPage), `catalog entry ${e.id} docsPage ${e.data.docsPage} not found in docs`).toBe(true)
      }
    }
  })

  it('every community docs page with an integrationType has a catalog entry', async () => {
    const [entries, docs] = await Promise.all([getCollection('catalog'), getCollection('docs')])
    const cataloged = new Set(entries.map((e) => e.data.docsPage).filter(Boolean))
    const communityPages = docs.filter(
      (d) => d.id.startsWith('docs/community/') && d.data.community === true && d.data.integrationType
    )
    for (const page of communityPages) {
      expect(cataloged.has(page.id), `community page ${page.id} has no catalog entry`).toBe(true)
    }
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rm -rf .astro && npm test -- test/catalog-collection.test.ts`
Expected: FAIL — most community pages have no catalog entry yet.

- [ ] **Step 3: Generate one YAML entry per existing community page**

For each `.mdx` file under `site/src/content/docs/community/{tools,model-providers,session-managers,memory-stores,plugins,integrations,agent-extensions,interventions}/` that has `community: true` and an `integrationType` in its frontmatter (skip `overview.mdx` files — they have no `integrationType`), create `site/src/content/catalog/<file-stem>.yaml`.

Map frontmatter → entry fields:

| Entry field | Source |
|---|---|
| `name` | frontmatter `title` |
| `description` | frontmatter `description` |
| `integrationType` | frontmatter `integrationType` |
| `languages.python` | present if frontmatter `languages` includes `Python` or is absent; `package` = `project.pypi` URL path segment, `registry` = `project.pypi` |
| `languages.typescript` | present if frontmatter `languages` includes `TypeScript` or is absent; `registry` = `project.npm` if set |
| `github` | frontmatter `project.github` |
| `maintainer` | frontmatter `project.maintainer` |
| `docsPage` | the docs collection id, e.g. `docs/community/tools/strands-apify` |
| `addedDate` | `2026-01-01` for all migrated entries — they predate the catalog, and a launch-date value would give every migrated entry a spurious "new" badge (NEW_BADGE_DAYS = 30) |

Where a page lacks `project.github` or `project.maintainer` frontmatter, open the page body and use the repo URL it links to; if the page has no PyPI/npm link (e.g. AgentCore plugins that install via a shared package), point `registry` at the package's registry page named in the page's install command. Where the page is genuinely not a published package, still create the entry with the languages block derived from the install instructions.

Example — `site/src/content/catalog/strands-apify.yaml` (verify against the page's actual frontmatter before writing):

```yaml
name: strands-apify
description: Apify web scraping and automation
integrationType: tool
languages:
  python:
    package: strands-apify
    registry: https://pypi.org/project/strands-apify/
github: https://github.com/apify/strands-apify
maintainer: apify
docsPage: docs/community/tools/strands-apify
addedDate: 2026-01-01
```

Files to create (stems match the mdx file stems):
`strands-apify`, `strands-google`, `strands-hubspot`, `strands-perplexity`, `strands-spraay`, `strands-sql`, `strands-teams`, `strands-telegram-listener`, `strands-telegram`, `utcp`, `clova-studio`, `cohere`, `fireworksai`, `mlx`, `nebius-token-factory`, `nvidia-nim`, `ovhcloud-ai-endpoints`, `sglang`, `vllm`, `xai`, `agentcore-memory`, `strands-valkey-session-manager`, `agent-control`, `agentcore-payments`, `agentcore-tool-search`, `datadog-ai-guard`, `s3-vectors-memory`, `ag-ui`, `strands-code-agent`, `strands-agt`.

(`strands-deepgram` exists from Task 1 — update its `addedDate` to `2026-01-01` to match the other migrated entries.)

- [ ] **Step 4: Run test to verify it passes**

Run: `rm -rf .astro && npm test -- test/catalog-collection.test.ts`
Expected: PASS — both integrity checks green.

- [ ] **Step 5: Commit**

```bash
git add site/src/content/catalog/ site/test/catalog-collection.test.ts
git commit -m "feat(site): migrate existing community integrations to catalog entries"
```

---

### Task 5: CatalogCard and CatalogFilterBar components

**Files:**
- Create: `site/src/components/catalog/CatalogCard.astro`
- Create: `site/src/components/catalog/CatalogFilterBar.astro`
- Create: `site/src/styles/catalog.css`

**Interfaces:**
- Consumes: `CatalogCardModel` from `@util/catalog` (Task 2).
- Produces: `<CatalogCard card={CatalogCardModel} featured?: boolean />` renders an `<article class="cat-card" data-name data-type data-langs data-badges data-sdk data-search>`; `<CatalogFilterBar types={{value,label,count}[]} showSdkFacet={boolean} />` renders the search input + chip groups with `data-cat-search`, `data-cat-type`, `data-cat-lang`, `data-cat-badge`, `data-cat-sdk` hooks. No client script here — Task 6 wires behavior.

- [ ] **Step 1: Create `site/src/components/catalog/CatalogCard.astro`**

```astro
---
/**
 * A single integration card in the catalog grid. All filterable facets are
 * exposed as data-* attributes; the page script (catalog.astro) toggles
 * visibility, so the card itself stays a pure server-rendered component.
 */
import type { CatalogCardModel } from '../../util/catalog'

interface Props {
  card: CatalogCardModel
  featured?: boolean
}

const { card, featured = false } = Astro.props

const TYPE_LABELS: Record<string, string> = {
  'model-provider': 'Model Provider',
  tool: 'Tool',
  'session-manager': 'Session Manager',
  'memory-store': 'Memory Store',
  integration: 'Integration',
  plugin: 'Plugin',
  'agent-extension': 'Agent Extension',
  intervention: 'Intervention',
}

const LANG_LABELS: Record<string, string> = { python: 'Python', typescript: 'TypeScript' }

const numberFormat = new Intl.NumberFormat('en', { notation: 'compact' })
---

<article
  class:list={['cat-card', { 'cat-card--featured': featured }]}
  data-name={card.name}
  data-type={card.integrationType}
  data-langs={card.languages.join(' ')}
  data-badges={card.badges.join(' ')}
  data-sdk={card.sdk}
  data-search={`${card.name} ${card.description}`.toLowerCase()}
>
  <div class="cat-card-head">
    <a class="cat-card-title" href={card.href} rel={card.external ? 'noopener' : undefined}>
      {card.name}
      {card.external && <span class="cat-external" aria-label="external link">↗</span>}
    </a>
    <span class="cat-chip cat-chip--type">{TYPE_LABELS[card.integrationType]}</span>
  </div>
  <p class="cat-card-desc">{card.description}</p>
  <div class="cat-card-meta">
    {card.languages.map((lang) => <span class:list={['cat-badge', `cat-badge--${lang}`]}>{LANG_LABELS[lang]}</span>)}
    {card.badges.includes('verified') && <span class="cat-badge cat-badge--verified">Verified</span>}
    {card.badges.includes('new') && <span class="cat-badge cat-badge--new">New</span>}
  </div>
  <div class="cat-card-foot">
    <span class="cat-maintainer">by {card.maintainer}</span>
    <span class="cat-links">
      {card.stars !== undefined && <span class="cat-stat" title="GitHub stars">★ {numberFormat.format(card.stars)}</span>}
      {card.downloads !== undefined && (
        <span class="cat-stat" title="Downloads per week">⬇ {numberFormat.format(card.downloads)}</span>
      )}
      <a class="cat-link" href={card.github} rel="noopener" aria-label={`${card.name} on GitHub`}>GitHub</a>
      {card.registryLinks.map((link) => (
        <a class="cat-link" href={link.href} rel="noopener" aria-label={`${card.name} on ${link.label}`}>{link.label}</a>
      ))}
    </span>
  </div>
</article>
```

- [ ] **Step 2: Create `site/src/components/catalog/CatalogFilterBar.astro`**

```astro
---
/**
 * Search input + filter chips for the catalog. Renders static controls with
 * data-cat-* hooks; the page script owns all interactive behavior. The SDK
 * facet only renders when the catalog contains evals entries.
 */
interface FacetOption {
  value: string
  label: string
  count: number
}

interface Props {
  types: FacetOption[]
  showSdkFacet: boolean
}

const { types, showSdkFacet } = Astro.props

const LANGUAGES: FacetOption[] = [
  { value: 'python', label: 'Python', count: 0 },
  { value: 'typescript', label: 'TypeScript', count: 0 },
]
---

<form class="cat-filters" role="search" aria-label="Filter catalog" onsubmit="return false">
  <input
    class="cat-search"
    type="search"
    placeholder="Search integrations…"
    aria-label="Search integrations"
    data-cat-search
  />
  {
    showSdkFacet && (
      <div class="cat-facet" role="radiogroup" aria-label="SDK">
        <button class="cat-chip" type="button" role="radio" aria-checked="true" data-cat-sdk="agents">Agents</button>
        <button class="cat-chip" type="button" role="radio" aria-checked="false" data-cat-sdk="evals">Evals</button>
      </div>
    )
  }
  <div class="cat-facet" aria-label="Type">
    {types.map((t) => (
      <button class="cat-chip" type="button" data-cat-type={t.value} aria-pressed="false">
        {t.label}<span class="cat-count">{t.count}</span>
      </button>
    ))}
  </div>
  <div class="cat-facet" aria-label="Language">
    {LANGUAGES.map((l) => (
      <button class="cat-chip" type="button" data-cat-lang={l.value} aria-pressed="false">{l.label}</button>
    ))}
  </div>
  <div class="cat-facet" aria-label="Badges">
    <button class="cat-chip" type="button" data-cat-badge="verified" aria-pressed="false">Verified</button>
  </div>
</form>
```

- [ ] **Step 3: Create `site/src/styles/catalog.css`**

Follow the visual language of `src/styles/changelog.css` (check it for the site's CSS-variable usage — `--sl-color-*` custom properties, dark/light handled by Starlight). Required classes and behaviors:

```css
/* Catalog page: featured hero, filter bar, card grid.
   Uses Starlight theme variables so light/dark adapt automatically. */

.cat-page { max-width: 72rem; margin: 0 auto; padding: 0 1rem; }

.cat-featured { display: grid; grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr)); gap: 1rem; margin-block: 1.5rem; }

.cat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); gap: 1rem; margin-block: 1.5rem; }

.cat-card {
  display: flex; flex-direction: column; gap: 0.5rem;
  border: 1px solid var(--sl-color-gray-5); border-radius: 0.5rem; padding: 1rem;
  background: var(--sl-color-bg-nav);
}
.cat-card[hidden] { display: none; }
.cat-card--featured { border-color: var(--sl-color-text-accent); }

.cat-card-head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
.cat-card-title { font-weight: 600; color: var(--sl-color-white); text-decoration: none; }
.cat-card-title:hover { text-decoration: underline; }
.cat-external { font-size: 0.75em; }
.cat-card-desc { color: var(--sl-color-gray-2); font-size: 0.875rem; margin: 0; flex-grow: 1; }

.cat-card-meta { display: flex; flex-wrap: wrap; gap: 0.375rem; }
.cat-badge { font-size: 0.6875rem; padding: 0.125rem 0.5rem; border-radius: 999px; border: 1px solid var(--sl-color-gray-5); color: var(--sl-color-gray-2); }
.cat-badge--verified { border-color: var(--sl-color-text-accent); color: var(--sl-color-text-accent); }
.cat-badge--new { border-color: var(--sl-color-orange); color: var(--sl-color-orange); }

.cat-card-foot { display: flex; justify-content: space-between; align-items: center; font-size: 0.8125rem; color: var(--sl-color-gray-3); }
.cat-links { display: inline-flex; gap: 0.625rem; align-items: center; }
.cat-link { color: var(--sl-color-text-accent); text-decoration: none; }
.cat-stat { color: var(--sl-color-gray-3); }

.cat-filters { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin-block: 1rem; }
.cat-search { flex: 1 1 16rem; padding: 0.5rem 0.75rem; border: 1px solid var(--sl-color-gray-5); border-radius: 0.5rem; background: var(--sl-color-bg); color: var(--sl-color-white); }
.cat-facet { display: inline-flex; flex-wrap: wrap; gap: 0.375rem; }
.cat-chip {
  font-size: 0.8125rem; padding: 0.25rem 0.75rem; border-radius: 999px;
  border: 1px solid var(--sl-color-gray-5); background: transparent; color: var(--sl-color-gray-2); cursor: pointer;
}
.cat-chip[aria-pressed='true'], .cat-chip[aria-checked='true'] {
  border-color: var(--sl-color-text-accent); color: var(--sl-color-text-accent); font-weight: 600;
}
.cat-count { margin-left: 0.375rem; opacity: 0.7; }

.cat-section-title { margin-block: 2rem 0.5rem; }
.cat-empty { text-align: center; color: var(--sl-color-gray-3); margin-block: 3rem; }
.cat-submit { margin-block: 3rem; text-align: center; }
```

- [ ] **Step 4: Verify components compile**

Run: `npm run typecheck`
Expected: no errors. (Astro components are type-checked via the build; a full render check happens in Task 6.)

- [ ] **Step 5: Commit**

```bash
git add site/src/components/catalog/ site/src/styles/catalog.css
git commit -m "feat(site): add catalog card and filter bar components"
```

---

### Task 6: The /catalog page with client-side filtering

**Files:**
- Create: `site/src/pages/catalog.astro`
- Test: `site/test/catalog-page.test.ts` (build-output smoke test)

**Interfaces:**
- Consumes: `getCollection('catalog')`; `toCardModel`, `sortEntries`, `hasEvalsEntries` from `@util/catalog`; `matchesFilters`, `stateToQuery`, `queryToState` from `@util/catalog-filter`; `CatalogCard`, `CatalogFilterBar` components; `ChangelogLayout`-style wrapping via `StarlightPage`.
- Produces: the `/catalog/` route.

- [ ] **Step 1: Create `site/src/pages/catalog.astro`**

```astro
---
/**
 * Community integration catalog: searchable, filterable grid of community
 * packages. Entries come from the `catalog` content collection; popularity
 * stats from the bot-maintained src/data/catalog-stats.json. Everything is
 * rendered statically; the inline script only toggles card visibility.
 */
import { getCollection } from 'astro:content'
import StarlightPage from '@astrojs/starlight/components/StarlightPage.astro'
import CatalogCard from '../components/catalog/CatalogCard.astro'
import CatalogFilterBar from '../components/catalog/CatalogFilterBar.astro'
import { toCardModel, sortEntries, hasEvalsEntries, type CatalogStatsFile } from '../util/catalog'
import statsFile from '../data/catalog-stats.json'
import '../styles/catalog.css'

const stats = statsFile as CatalogStatsFile
const entries = await getCollection('catalog')
const buildDate = new Date()
const cards = sortEntries(entries.map((e) => toCardModel(e.id, e.data, stats[e.id], buildDate)))

const featured = cards.filter((c) => c.featured)
const showSdkFacet = hasEvalsEntries(cards)

const TYPE_LABELS: Record<string, string> = {
  'model-provider': 'Model Providers',
  tool: 'Tools',
  'session-manager': 'Session Managers',
  'memory-store': 'Memory Stores',
  integration: 'Integrations',
  plugin: 'Plugins',
  'agent-extension': 'Agent Extensions',
  intervention: 'Interventions',
}
const typeFacets = Object.entries(TYPE_LABELS)
  .map(([value, label]) => ({ value, label, count: cards.filter((c) => c.integrationType === value).length }))
  .filter((t) => t.count > 0)

const title = 'Community Catalog'
const description = 'Community-built tools, model providers, plugins, and integrations for Strands Agents.'
---

<StarlightPage frontmatter={{ title, description, template: 'splash', pagefind: false, hero: { actions: [] } }} hasSidebar={false}>
  <div class="cat-page">
    <p>
      Integrations built and maintained by the Strands community. Review a package before using it in
      production — quality and support vary by author.
    </p>

    {
      featured.length > 0 && (
        <>
          <h2 class="cat-section-title">Featured</h2>
          <div class="cat-featured">
            {featured.map((card) => (
              <CatalogCard card={card} featured />
            ))}
          </div>
        </>
      )
    }

    <h2 class="cat-section-title">All integrations</h2>
    <CatalogFilterBar types={typeFacets} showSdkFacet={showSdkFacet} />
    <p class="cat-note"><span data-cat-count>{cards.length}</span> integrations</p>
    <div class="cat-grid" data-cat-grid>
      {cards.map((card) => <CatalogCard card={card} />)}
    </div>
    <p class="cat-empty" data-cat-empty hidden>No integrations match the current filters.</p>

    <div class="cat-submit">
      <p>
        Built something for Strands? <a href="/docs/community/get-featured/">Add your integration to the catalog</a>.
      </p>
    </div>
  </div>
</StarlightPage>

<script>
  // Client-side faceted filtering over the statically-rendered cards. The
  // matching + URL-state logic lives in util/catalog-filter (unit-tested);
  // this script only reads the DOM and toggles [hidden].
  import { matchesFilters, stateToQuery, queryToState, type CardFilterData } from '../util/catalog-filter'

  const cards = Array.from(document.querySelectorAll<HTMLElement>('[data-cat-grid] .cat-card'))
  const searchInput = document.querySelector<HTMLInputElement>('[data-cat-search]')
  const typeBtns = Array.from(document.querySelectorAll<HTMLElement>('[data-cat-type]'))
  const langBtns = Array.from(document.querySelectorAll<HTMLElement>('[data-cat-lang]'))
  const badgeBtns = Array.from(document.querySelectorAll<HTMLElement>('[data-cat-badge]'))
  const sdkBtns = Array.from(document.querySelectorAll<HTMLElement>('[data-cat-sdk]'))
  const emptyEl = document.querySelector<HTMLElement>('[data-cat-empty]')
  const countEl = document.querySelector<HTMLElement>('[data-cat-count]')

  const state = queryToState(location.search)

  function cardData(el: HTMLElement): CardFilterData {
    return {
      search: el.dataset.search || '',
      type: el.dataset.type || '',
      languages: (el.dataset.langs || '').split(' ').filter(Boolean),
      badges: (el.dataset.badges || '').split(' ').filter(Boolean),
      sdk: el.dataset.sdk || 'agents',
    }
  }

  function apply() {
    let visible = 0
    for (const el of cards) {
      const show = matchesFilters(cardData(el), state)
      el.hidden = !show
      if (show) visible++
    }
    if (emptyEl) emptyEl.hidden = visible > 0
    if (countEl) countEl.textContent = String(visible)

    for (const btn of typeBtns) btn.setAttribute('aria-pressed', String(state.types.has(btn.dataset.catType || '')))
    for (const btn of langBtns) btn.setAttribute('aria-pressed', String(state.languages.has(btn.dataset.catLang || '')))
    for (const btn of badgeBtns) btn.setAttribute('aria-pressed', String(state.badges.has(btn.dataset.catBadge || '')))
    for (const btn of sdkBtns) btn.setAttribute('aria-checked', String(state.sdk === btn.dataset.catSdk))

    const query = stateToQuery(state)
    history.replaceState(null, '', query ? `?${query}` : location.pathname)
  }

  function toggle(set: Set<string>, value: string) {
    if (set.has(value)) set.delete(value)
    else set.add(value)
  }

  if (searchInput) {
    searchInput.value = state.search
    searchInput.addEventListener('input', () => {
      state.search = searchInput.value
      apply()
    })
  }
  for (const btn of typeBtns)
    btn.addEventListener('click', () => {
      toggle(state.types, btn.dataset.catType || '')
      apply()
    })
  for (const btn of langBtns)
    btn.addEventListener('click', () => {
      toggle(state.languages, btn.dataset.catLang || '')
      apply()
    })
  for (const btn of badgeBtns)
    btn.addEventListener('click', () => {
      toggle(state.badges, btn.dataset.catBadge || '')
      apply()
    })
  for (const btn of sdkBtns)
    btn.addEventListener('click', () => {
      state.sdk = btn.dataset.catSdk || 'agents'
      apply()
    })

  apply()
</script>
```

- [ ] **Step 2: Write the build smoke test**

Create `site/test/catalog-page.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { getCollection } from 'astro:content'
import { toCardModel, sortEntries } from '../src/util/catalog'

describe('catalog page data', () => {
  it('produces a card model for every collection entry', async () => {
    const entries = await getCollection('catalog')
    const cards = sortEntries(entries.map((e) => toCardModel(e.id, e.data, undefined, new Date())))
    expect(cards.length).toBe(entries.length)
    for (const card of cards) {
      expect(card.name.length).toBeGreaterThan(0)
      expect(card.href.length).toBeGreaterThan(0)
      expect(card.languages.length).toBeGreaterThan(0)
    }
  })
})
```

- [ ] **Step 3: Run tests and dev-server check**

Run: `npm test -- test/catalog-page.test.ts`
Expected: PASS

Run: `npm run dev` in the background, then fetch the page:
`curl -s http://localhost:4321/catalog/ | grep -c 'cat-card'`
Expected: a number ≥ the count of catalog entries (featured cards render twice). Also confirm `curl -s http://localhost:4321/catalog/ | grep 'data-cat-search'` outputs the search input. Stop the dev server.

- [ ] **Step 4: Full build**

Run: `npm run build`
Expected: build succeeds; the broken-links checker reports no new broken links.

- [ ] **Step 5: Commit**

```bash
git add site/src/pages/catalog.astro site/test/catalog-page.test.ts
git commit -m "feat(site): add /catalog page with client-side faceted filtering"
```

---

### Task 7: Navigation, redirect, and old-page retirement

**Files:**
- Modify: `site/src/config/navigation.yml` (navbar `Community` entry ~line 22; sidebar `Community` group ~line 294)
- Modify: `site/src/util/redirect.ts` (add rule to `SLUG_RULES`)
- Delete: `site/src/content/docs/community/community-packages.mdx`
- Modify: pages that link to `community-packages` (found via grep: `site/src/content/docs/user-guide/concepts/agents/session-management.mdx`, `site/src/content/docs/user-guide/concepts/model-providers/index.mdx`, `site/src/content/docs/contribute/index.mdx`, `site/src/content/docs/contribute/contributing/extensions.mdx`, `site/src/content/docs/community/get-featured.mdx`, `site/src/content/docs/community/interventions/overview.mdx`, `site/src/content/docs/community/memory-stores/overview.mdx`)
- Test: existing `site/test/redirect.test.ts` (add one case)

**Interfaces:**
- Consumes: the `/catalog` route (Task 6).
- Produces: `/docs/community/community-packages/` redirects to `/catalog/`; navbar Community points at `/catalog/`.

- [ ] **Step 1: Add the redirect test**

In `site/test/redirect.test.ts`, find the existing SLUG_RULES test block and add:

```ts
  it('redirects the retired community-packages page to the catalog', () => {
    expect(resolveRedirect('docs/community/community-packages')).toBe('catalog')
  })
```

(Match the existing test file's import and describe structure — add the case inside the existing `describe` that tests `resolveRedirect`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- test/redirect.test.ts`
Expected: FAIL — the new case returns `null`.

- [ ] **Step 3: Add the SLUG_RULE**

In `site/src/util/redirect.ts`, add to `SLUG_RULES`:

```ts
  // community-packages was replaced by the interactive catalog page
  {
    match: exactly('docs/community/community-packages'),
    to: 'catalog',
  },
```

- [ ] **Step 4: Update navigation.yml**

Navbar (line ~22): change the Community entry to:

```yaml
  - label: Community
    href: /catalog/
    basePath:
      - /catalog/
      - /docs/community/
      - /docs/labs/
      - /docs/contribute/
```

Sidebar Community group (line ~294): remove the `- docs/community/community-packages` line (keep `get-featured` and the rest).

- [ ] **Step 5: Delete the old page and fix inbound links**

Delete `site/src/content/docs/community/community-packages.mdx`.

For each file found by `grep -rln "community-packages" site/src/content/docs`, replace links to `community-packages` (`./community-packages.md`, `../../community/community-packages.md`, etc.) with `/catalog/`. Keep surrounding sentence wording intact; only swap the link target (relative `.md` links become the absolute `/catalog/` path since the target is no longer a docs page).

- [ ] **Step 6: Run tests and build**

Run: `npm test -- test/redirect.test.ts` → PASS
Run: `rm -rf .astro && npm test` → all suites PASS (sidebar test validates navigation.yml; content-collection test validates docs)
Run: `npm run build` → succeeds, broken-links checker green.

- [ ] **Step 7: Commit**

```bash
git add -A site/src
git commit -m "feat(site): route community nav to /catalog and retire community-packages page"
```

---

### Task 8: Disable the language toggle on single-language pages

**Files:**
- Modify: `site/src/components/LanguageToggle.astro`
- Test: manual dev-server verification (the toggle is presentation-only; the derivation logic is trivial frontmatter reading)

**Interfaces:**
- Consumes: `Astro.locals.starlightRoute.entry.data.languages` (docs frontmatter, already zod-validated).
- Produces: on pages whose `languages` frontmatter names exactly one language, the other language's button renders `disabled` with a tooltip.

- [ ] **Step 1: Compute page languages in the component frontmatter**

In `site/src/components/LanguageToggle.astro`, after the existing `currentPath` computation, add:

```ts
// On docs pages that declare a single supported language, the other button is
// disabled: there is nothing to switch to, and a silent no-op toggle reads as
// a bug. starlightRoute is absent on non-Starlight pages (landing/changelog),
// where the toggle stays fully enabled.
const routeData = (Astro.locals as { starlightRoute?: { entry?: { data?: { languages?: string | string[] } } } })
  .starlightRoute?.entry?.data
const pageLanguages = routeData?.languages
  ? (Array.isArray(routeData.languages) ? routeData.languages : [routeData.languages]).map((l) => l.toLowerCase())
  : undefined
const pythonDisabled = pageLanguages !== undefined && !pageLanguages.includes('python')
const typescriptDisabled = pageLanguages !== undefined && !pageLanguages.includes('typescript')
```

- [ ] **Step 2: Apply disabled state to the buttons**

On the Python button add:

```astro
    disabled={pythonDisabled}
    title={pythonDisabled ? 'This page is TypeScript-only' : undefined}
```

On the TypeScript button add:

```astro
    disabled={typescriptDisabled}
    title={typescriptDisabled ? 'This page is Python-only' : undefined}
```

Add disabled styling to the component's `<style>` block:

```css
  .lang-option:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .lang-option:disabled:hover {
    color: var(--sl-color-gray-3);
  }
```

- [ ] **Step 3: Guard the click handler**

In the inline script's click listener, change the existing guard `if (!btn) return;` to:

```js
      if (!btn || btn.disabled) return;
```

- [ ] **Step 4: Manual verification**

Run `npm run dev`. Load a Python-only community page, e.g. `http://localhost:4321/docs/community/tools/strands-deepgram/` — the TypeScript button must appear dimmed and unclickable, with the tooltip. Load `http://localhost:4321/docs/user-guide/quickstart/typescript/` (dual/absent languages) — both buttons active. Verify via dev-browser or curl + manual browser check. Stop the dev server.

- [ ] **Step 5: Typecheck, build, commit**

Run: `npm run typecheck` and `npm run build` → green.

```bash
git add site/src/components/LanguageToggle.astro
git commit -m "feat(site): disable language toggle on single-language pages"
```

---

### Task 9: Rewrite get-featured.mdx as the catalog submission guide + PR template

**Files:**
- Modify: `site/src/content/docs/community/get-featured.mdx` (full rewrite)
- Create: `.github/PULL_REQUEST_TEMPLATE/catalog-submission.md`

**Interfaces:**
- Consumes: the catalog entry schema (Task 1) — the guide documents its fields.
- Produces: the submission flow documentation the catalog page links to (`/docs/community/get-featured/`).

- [ ] **Step 1: Rewrite `get-featured.mdx`**

Replace the file content with:

````mdx
---
title: Add Your Integration to the Catalog
sidebar:
  label: "Get Featured"
---

Built something useful for Strands Agents? Add it to the [community catalog](/catalog/) so other developers can discover it. A catalog listing is one small YAML file — an on-site documentation page is optional.

## What we list

Reusable packages published to PyPI or npm that extend Strands Agents: model providers, tools, session managers, memory stores, plugins, integrations, agent extensions, and interventions. We don't list example agents or one-off projects.

:::tip[Starting from scratch?]
The [extension template](https://github.com/strands-agents/extension-template) gives you a ready-made project structure with testing, linting, and publishing already set up.
:::

## Add your entry

1. Fork [strands-agents/harness-sdk](https://github.com/strands-agents/harness-sdk)
2. Add one file: `site/src/content/catalog/<your-package>.yaml`
3. Open a PR using the **Catalog submission** template

Your entry:

```yaml
name: your-package-name
description: One sentence describing what it does
integrationType: tool        # tool | model-provider | session-manager | memory-store | integration | plugin | agent-extension | intervention
languages:
  python:                    # include the languages you publish for
    package: your-package-name
    registry: https://pypi.org/project/your-package-name/
  typescript:
    package: "@your-scope/your-package"
    registry: https://www.npmjs.com/package/@your-scope/your-package
github: https://github.com/your-org/your-repo
maintainer: your-github-username
addedDate: 2026-07-17        # the date you open the PR
```

| Field | Required | Notes |
|-------|----------|-------|
| `name` | ✅ | Display name in the catalog |
| `description` | ✅ | One sentence, shown on the card |
| `integrationType` | ✅ | One of the eight types above |
| `languages` | ✅ | At least one of `python` / `typescript`, each with `package` + `registry` |
| `github` | ✅ | Public repository URL |
| `maintainer` | ✅ | Your GitHub username or org |
| `addedDate` | ✅ | Date of submission (drives the "New" badge) |
| `docsPage` | Optional | Set only if you also add a docs page (below) |

Leave `featured` and `badges` unset — the Strands team grants those.

The site build validates your file against the schema, so a malformed entry fails CI with a clear error.

## Optional: add a documentation page

Without a docs page, your catalog card links to your GitHub repo — perfectly fine. If you want an on-site page with install/usage examples:

1. Add `site/src/content/docs/community/<type-directory>/<your-package>.mdx` (e.g. `community/tools/your-package.mdx`) with frontmatter:

```yaml
---
title: your-package-name
community: true
description: Same description as your catalog entry
integrationType: tool
languages: Python            # or TypeScript, or omit for both
sidebar:
  label: "display-name"
---
```

2. Keep the page a concise overview: installation, a working usage example, configuration. If your package supports both languages, use `<Tabs>` with `Python` / `TypeScript` labels so the page follows the site-wide language toggle.
3. Add the page under the Community section of `site/src/config/navigation.yml`.
4. Reference it from your catalog entry: `docsPage: docs/community/tools/your-package`

## Questions?

Ask in [Discord](https://discord.gg/strands) or open an issue at [strands-agents/harness-sdk](https://github.com/strands-agents/harness-sdk/issues).
````

- [ ] **Step 2: Create the PR template**

Create `.github/PULL_REQUEST_TEMPLATE/catalog-submission.md`:

```markdown
## Catalog submission

<!-- Adding your integration to the community catalog at strandsagents.com/catalog -->

**Package name:**
**Integration type:** <!-- tool | model-provider | session-manager | memory-store | integration | plugin | agent-extension | intervention -->

### Checklist

- [ ] The package is published to PyPI and/or npm, and the `registry` links in my entry resolve
- [ ] The GitHub repository is public and includes a license
- [ ] The description accurately states what the package does in one sentence
- [ ] The `integrationType` matches what the package actually is
- [ ] I am the package's maintainer, or I have the maintainer's consent to list it
- [ ] I left `featured` and `badges` unset (granted by the Strands team)
- [ ] (If adding a docs page) The usage example works against the current SDK release
```

- [ ] **Step 3: Build check and commit**

Run: `rm -rf .astro && npm test` → PASS (content-collection test picks up the rewritten page)
Run: `npm run build` → green (validates the mdx and its links).

```bash
git add site/src/content/docs/community/get-featured.mdx .github/PULL_REQUEST_TEMPLATE/catalog-submission.md
git commit -m "docs(site): rewrite get-featured as catalog submission guide with PR template"
```

---

### Task 10: Stats refresh script

**Files:**
- Create: `site/scripts/catalog/refresh-stats.ts`
- Test: `site/test/catalog-stats.test.ts`

**Interfaces:**
- Consumes: catalog YAML files (read from disk with `js-yaml` — the script runs outside Astro so it can't use `getCollection`); `CatalogStatsFile` shape from `@util/catalog` (Task 2).
- Produces: `npm run catalog:stats` (add to `package.json` scripts: `"catalog:stats": "tsx scripts/catalog/refresh-stats.ts"`), writing `site/src/data/catalog-stats.json`. Exported for tests: `buildStats(entries, fetchers)` — dependency-injected fetchers so tests never hit the network.

- [ ] **Step 1: Write the failing test**

Create `site/test/catalog-stats.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { buildStats, type StatsFetchers, type StatsEntry } from '../scripts/catalog/refresh-stats'

const entries: StatsEntry[] = [
  {
    id: 'strands-example',
    github: 'https://github.com/example/strands-example',
    python: 'strands-example',
    typescript: '@example/strands',
  },
  { id: 'strands-broken', github: 'https://github.com/example/strands-broken', python: 'strands-broken' },
]

function fetchers(overrides: Partial<StatsFetchers> = {}): StatsFetchers {
  return {
    githubRepo: async () => ({ stars: 42, lastRelease: '2026-07-01' }),
    pypiDownloads: async () => 100,
    npmDownloads: async () => 50,
    ...overrides,
  }
}

describe('buildStats', () => {
  it('aggregates stats per entry', async () => {
    const stats = await buildStats(entries, fetchers())
    expect(stats['strands-example']).toEqual({
      stars: 42,
      lastRelease: '2026-07-01',
      downloads: { python: 100, typescript: 50 },
    })
    expect(stats['strands-broken']).toEqual({ stars: 42, lastRelease: '2026-07-01', downloads: { python: 100 } })
  })

  it('skips failing sources without failing the run', async () => {
    const stats = await buildStats(
      entries,
      fetchers({
        githubRepo: async (repo) => {
          if (repo.includes('broken')) throw new Error('boom')
          return { stars: 42, lastRelease: '2026-07-01' }
        },
        pypiDownloads: async (pkg) => {
          if (pkg === 'strands-broken') throw new Error('boom')
          return 100
        },
      })
    )
    expect(stats['strands-example']?.stars).toBe(42)
    expect(stats['strands-broken']).toEqual({ downloads: {} })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- test/catalog-stats.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `site/scripts/catalog/refresh-stats.ts`**

```ts
/**
 * Refreshes site/src/data/catalog-stats.json with GitHub stars, release dates,
 * and registry download counts for every catalog entry. Run by the
 * catalog-stats.yml scheduled workflow; per-package failures are logged and
 * skipped so one broken upstream can't block the whole refresh.
 *
 * Usage: npm run catalog:stats   (requires GITHUB_TOKEN for the GitHub API)
 */

import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import yaml from 'js-yaml'

export interface StatsEntry {
  id: string
  github: string
  python?: string
  typescript?: string
}

export interface StatsFetchers {
  githubRepo(repoUrl: string): Promise<{ stars: number; lastRelease?: string }>
  pypiDownloads(pkg: string): Promise<number>
  npmDownloads(pkg: string): Promise<number>
}

export interface EntryStats {
  stars?: number
  lastRelease?: string
  downloads: { python?: number; typescript?: number }
}

export async function buildStats(
  entries: StatsEntry[],
  fetchers: StatsFetchers
): Promise<Record<string, EntryStats>> {
  const result: Record<string, EntryStats> = {}
  for (const entry of entries) {
    const stats: EntryStats = { downloads: {} }
    try {
      const repo = await fetchers.githubRepo(entry.github)
      stats.stars = repo.stars
      if (repo.lastRelease) stats.lastRelease = repo.lastRelease
    } catch (err) {
      console.warn(`entry=${entry.id}, source=github | fetch failed, skipping`, err)
    }
    if (entry.python) {
      try {
        stats.downloads.python = await fetchers.pypiDownloads(entry.python)
      } catch (err) {
        console.warn(`entry=${entry.id}, source=pypi | fetch failed, skipping`, err)
      }
    }
    if (entry.typescript) {
      try {
        stats.downloads.typescript = await fetchers.npmDownloads(entry.typescript)
      } catch (err) {
        console.warn(`entry=${entry.id}, source=npm | fetch failed, skipping`, err)
      }
    }
    result[entry.id] = stats
  }
  return result
}

// ── Live fetchers ─────────────────────────────────────────────────────────────

function githubApiHeaders(): Record<string, string> {
  const headers: Record<string, string> = { accept: 'application/vnd.github+json' }
  if (process.env.GITHUB_TOKEN) headers.authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  return headers
}

async function fetchJson(url: string, headers: Record<string, string> = {}): Promise<any> {
  const res = await fetch(url, { headers })
  if (!res.ok) throw new Error(`status=${res.status} url=${url}`)
  return res.json()
}

export const liveFetchers: StatsFetchers = {
  async githubRepo(repoUrl) {
    const slug = new URL(repoUrl).pathname.replace(/^\/|\/$/g, '')
    const repo = await fetchJson(`https://api.github.com/repos/${slug}`, githubApiHeaders())
    let lastRelease: string | undefined
    try {
      const release = await fetchJson(`https://api.github.com/repos/${slug}/releases/latest`, githubApiHeaders())
      lastRelease = (release.published_at as string | undefined)?.slice(0, 10)
    } catch {
      // Repos without releases 404 here; stars alone are still useful.
    }
    return { stars: repo.stargazers_count as number, lastRelease }
  },
  async pypiDownloads(pkg) {
    // pypistats.org: last-month downloads for the package.
    const data = await fetchJson(`https://pypistats.org/api/packages/${pkg}/recent`)
    return data.data.last_month as number
  },
  async npmDownloads(pkg) {
    const data = await fetchJson(`https://api.npmjs.org/downloads/point/last-month/${encodeURIComponent(pkg)}`)
    return data.downloads as number
  },
}

// ── CLI entry point ───────────────────────────────────────────────────────────

function loadEntries(catalogDir: string): StatsEntry[] {
  return readdirSync(catalogDir)
    .filter((f) => f.endsWith('.yaml'))
    .map((f) => {
      const data = yaml.load(readFileSync(path.join(catalogDir, f), 'utf-8')) as {
        github: string
        languages: { python?: { package: string }; typescript?: { package: string } }
      }
      return {
        id: f.replace(/\.yaml$/, ''),
        github: data.github,
        python: data.languages.python?.package,
        typescript: data.languages.typescript?.package,
      }
    })
}

const isDirectRun = process.argv[1]?.endsWith('refresh-stats.ts')
if (isDirectRun) {
  const catalogDir = path.resolve('src/content/catalog')
  const outPath = path.resolve('src/data/catalog-stats.json')
  const entries = loadEntries(catalogDir)
  console.log(`entries=${entries.length} | refreshing catalog stats`)
  const stats = await buildStats(entries, liveFetchers)
  writeFileSync(outPath, JSON.stringify(stats, null, 2) + '\n')
  console.log(`out=${outPath} | catalog stats written`)
}
```

Add to `site/package.json` scripts (after `"changelog:sync"`):

```json
    "catalog:stats": "tsx scripts/catalog/refresh-stats.ts",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- test/catalog-stats.test.ts`
Expected: PASS

- [ ] **Step 5: Exercise the live script once**

Run: `GITHUB_TOKEN=$(gh auth token) npm run catalog:stats`
Expected: `src/data/catalog-stats.json` populated with real numbers for the migrated entries (some may warn and skip — that's the designed behavior). Inspect the diff; the JSON must parse and match `CatalogStatsFile`. Then reload `/catalog/` in dev and confirm star/download stats render on cards.

- [ ] **Step 6: Typecheck and commit**

Run: `npm run typecheck`

```bash
git add site/scripts/catalog/ site/test/catalog-stats.test.ts site/package.json site/src/data/catalog-stats.json
git commit -m "feat(site): add catalog stats refresh script"
```

---

### Task 11: Scheduled stats workflow

**Files:**
- Create: `.github/workflows/catalog-stats.yml`

**Interfaces:**
- Consumes: `npm run catalog:stats` (Task 10).
- Produces: a weekly automated PR updating `site/src/data/catalog-stats.json`.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/catalog-stats.yml`, following the structure of `changelog-sync.yml` (same bot-fork PR pattern and token):

```yaml
name: "Catalog: Refresh Stats"

on:
  schedule:
    - cron: '23 6 * * 1'   # weekly, Monday morning UTC
  workflow_dispatch:

permissions:
  contents: read

jobs:
  refresh:
    # Upstream only: the bot fork mirrors main and lacks the bot token secret.
    if: github.repository == 'strands-agents/harness-sdk'
    runs-on: ubuntu-latest
    concurrency:
      group: catalog-stats
      cancel-in-progress: false
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-node@v7
        with:
          node-version-file: .node-version
      - run: npm ci --prefix site

      # Keep the bot fork current before cutting a PR branch from it
      # (same rationale as changelog-sync.yml).
      - name: Sync bot fork with upstream
        continue-on-error: true
        env:
          GH_TOKEN: ${{ secrets.CHANGELOG_BOT_TOKEN }}
        run: gh api repos/strands-agent/harness-sdk/merge-upstream -f branch=main

      - name: Refresh catalog stats
        working-directory: site
        env:
          GITHUB_TOKEN: ${{ secrets.CHANGELOG_BOT_TOKEN }}
        run: npm run catalog:stats

      - name: Open stats PR
        uses: peter-evans/create-pull-request@5f6978faf089d4d20b00c7766989d076bb2fc7f1 # v8.1.1
        with:
          token: ${{ secrets.CHANGELOG_BOT_TOKEN }}
          push-to-fork: strands-agent/harness-sdk
          add-paths: site/src/data/catalog-stats.json
          branch: catalog/stats-refresh
          title: "chore(site): refresh catalog stats"
          commit-message: "chore(site): refresh catalog stats"
          body: |
            Automated weekly refresh of community catalog popularity stats
            (GitHub stars, PyPI/npm downloads). No entry data is modified.
          delete-branch: true
```

- [ ] **Step 2: Validate the workflow syntax**

Run: `gh workflow list --repo strands-agents/harness-sdk >/dev/null 2>&1 && npx --yes @action-validator/cli .github/workflows/catalog-stats.yml || python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/catalog-stats.yml')); print('yaml ok')"`
Expected: `yaml ok` (or validator pass). The workflow itself can only be exercised on the upstream repo after merge via `workflow_dispatch`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/catalog-stats.yml
git commit -m "ci(site): add weekly catalog stats refresh workflow"
```

---

### Task 12: One-time backfill script

**Files:**
- Create: `site/scripts/catalog/backfill.ts`
- Test: `site/test/catalog-backfill.test.ts`

**Interfaces:**
- Consumes: catalog YAML dir (to skip already-cataloged packages); public PyPI/npm/GitHub search APIs via injected fetchers.
- Produces: `npm run catalog:backfill` (add script: `"catalog:backfill": "tsx scripts/catalog/backfill.ts"`) writing draft YAML files to `site/src/content/catalog/`. Exported for tests: `mergeCandidates(pypi, npm, github)` and `candidateToYaml(candidate)`.

- [ ] **Step 1: Write the failing test**

Create `site/test/catalog-backfill.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { mergeCandidates, candidateToYaml, inferType, type RegistryCandidate } from '../scripts/catalog/backfill'

describe('mergeCandidates', () => {
  it('merges pypi and npm packages that share a github repo into one candidate', () => {
    const pypi: RegistryCandidate[] = [
      { source: 'pypi', name: 'strands-widget', description: 'Widget tool for Strands', github: 'https://github.com/ex/strands-widget', registry: 'https://pypi.org/project/strands-widget/' },
    ]
    const npm: RegistryCandidate[] = [
      { source: 'npm', name: '@ex/strands-widget', description: 'Widget tool for Strands', github: 'https://github.com/ex/strands-widget', registry: 'https://www.npmjs.com/package/@ex/strands-widget' },
    ]
    const merged = mergeCandidates(pypi, npm, [])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.python?.package).toBe('strands-widget')
    expect(merged[0]!.typescript?.package).toBe('@ex/strands-widget')
  })

  it('keeps packages with different repos separate and excludes official org packages', () => {
    const pypi: RegistryCandidate[] = [
      { source: 'pypi', name: 'strands-a', description: 'a', github: 'https://github.com/ex/strands-a', registry: 'https://pypi.org/project/strands-a/' },
      { source: 'pypi', name: 'strands-agents-tools', description: 'official', github: 'https://github.com/strands-agents/tools', registry: 'https://pypi.org/project/strands-agents-tools/' },
    ]
    const merged = mergeCandidates(pypi, [], [])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.name).toBe('strands-a')
  })
})

describe('inferType', () => {
  it('classifies from keywords with tool as the fallback', () => {
    expect(inferType('strands-cohere', 'Cohere model provider for Strands')).toBe('model-provider')
    expect(inferType('strands-redis-session', 'Redis session manager')).toBe('session-manager')
    expect(inferType('strands-something', 'Does something')).toBe('tool')
  })
})

describe('candidateToYaml', () => {
  it('emits a valid entry with a REVIEW marker for uncertain types', () => {
    const yaml = candidateToYaml({
      name: 'strands-something',
      description: 'Does something',
      github: 'https://github.com/ex/strands-something',
      maintainer: 'ex',
      python: { package: 'strands-something', registry: 'https://pypi.org/project/strands-something/' },
      inferredType: 'tool',
      typeUncertain: true,
      addedDate: '2026-07-17',
    })
    expect(yaml).toContain('# REVIEW: integrationType inferred as fallback — verify')
    expect(yaml).toContain('integrationType: tool')
    expect(yaml).toContain('package: strands-something')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- test/catalog-backfill.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `site/scripts/catalog/backfill.ts`**

```ts
/**
 * ONE-TIME backfill: discovers community Strands integrations on PyPI, npm,
 * and GitHub and writes draft catalog YAML entries for human review. Not part
 * of any recurring pipeline — ongoing additions come through submission PRs.
 *
 * Usage: npm run catalog:backfill   (requires GITHUB_TOKEN)
 * Output: draft YAML files in src/content/catalog/ (existing files never overwritten).
 * Review every generated file before committing; prune spam and dead packages.
 */

import { existsSync, writeFileSync } from 'node:fs'
import path from 'node:path'

export interface RegistryCandidate {
  source: 'pypi' | 'npm' | 'github'
  name: string
  description: string
  github?: string
  registry?: string
  maintainer?: string
}

export interface MergedCandidate {
  name: string
  description: string
  github: string
  maintainer: string
  python?: { package: string; registry: string }
  typescript?: { package: string; registry: string }
  inferredType: string
  typeUncertain: boolean
  addedDate: string
}

// Official packages are documented in the SDK docs, not the community catalog.
const OFFICIAL_ORGS = ['strands-agents', 'strands-labs']

const TYPE_KEYWORDS: [string, RegExp][] = [
  ['model-provider', /model.provider|llm provider|inference/i],
  ['session-manager', /session.manager|session storage/i],
  ['memory-store', /memory.store|long.term memory/i],
  ['plugin', /\bplugin\b|lifecycle hook/i],
  ['agent-extension', /agent.extension|agent subclass/i],
  ['intervention', /\bintervention\b|guardrail/i],
  ['integration', /\bintegration\b|protocol|bridge|\bui\b/i],
]

export function inferType(name: string, description: string): string {
  const haystack = `${name} ${description}`
  for (const [type, pattern] of TYPE_KEYWORDS) {
    if (pattern.test(haystack)) return type
  }
  return 'tool'
}

function isOfficial(candidate: RegistryCandidate): boolean {
  const repoOrg = candidate.github ? new URL(candidate.github).pathname.split('/')[1] : undefined
  return repoOrg !== undefined && OFFICIAL_ORGS.includes(repoOrg)
}

/** Merge candidates across registries: same GitHub repo = same project. */
export function mergeCandidates(
  pypi: RegistryCandidate[],
  npm: RegistryCandidate[],
  github: RegistryCandidate[]
): MergedCandidate[] {
  const byRepo = new Map<string, MergedCandidate>()
  const today = new Date().toISOString().slice(0, 10)

  function upsert(c: RegistryCandidate): MergedCandidate | undefined {
    if (!c.github || isOfficial(c)) return undefined
    const key = c.github.replace(/\/$/, '').toLowerCase()
    let merged = byRepo.get(key)
    if (!merged) {
      const inferredType = inferType(c.name, c.description)
      merged = {
        // Prefer the repo name over registry-specific names (npm scopes etc.)
        name: key.split('/').pop() || c.name,
        description: c.description,
        github: c.github,
        maintainer: c.maintainer || new URL(c.github).pathname.split('/')[1] || 'unknown',
        inferredType,
        typeUncertain: inferredType === 'tool',
        addedDate: today,
      }
      byRepo.set(key, merged)
    }
    return merged
  }

  for (const c of pypi) {
    const m = upsert(c)
    if (m && c.registry) m.python = { package: c.name, registry: c.registry }
  }
  for (const c of npm) {
    const m = upsert(c)
    if (m && c.registry) m.typescript = { package: c.name, registry: c.registry }
  }
  for (const c of github) upsert(c)

  // A catalog entry needs at least one published package; repo-only hits
  // without a registry match are usually apps or examples, not packages.
  return [...byRepo.values()].filter((m) => m.python || m.typescript)
}

export function candidateToYaml(c: MergedCandidate): string {
  const lines: string[] = []
  if (c.typeUncertain) lines.push('# REVIEW: integrationType inferred as fallback — verify')
  lines.push(`name: ${c.name}`)
  lines.push(`description: ${JSON.stringify(c.description)}`)
  lines.push(`integrationType: ${c.inferredType}`)
  lines.push('languages:')
  if (c.python) {
    lines.push('  python:')
    lines.push(`    package: ${c.python.package}`)
    lines.push(`    registry: ${c.python.registry}`)
  }
  if (c.typescript) {
    lines.push('  typescript:')
    lines.push(`    package: ${JSON.stringify(c.typescript.package)}`)
    lines.push(`    registry: ${c.typescript.registry}`)
  }
  lines.push(`github: ${c.github}`)
  lines.push(`maintainer: ${c.maintainer}`)
  lines.push(`addedDate: ${c.addedDate}`)
  return lines.join('\n') + '\n'
}

// ── Discovery (live) ─────────────────────────────────────────────────────────

async function fetchJson(url: string, headers: Record<string, string> = {}): Promise<any> {
  const res = await fetch(url, { headers })
  if (!res.ok) throw new Error(`status=${res.status} url=${url}`)
  return res.json()
}

async function discoverPypi(): Promise<RegistryCandidate[]> {
  // PyPI's XML-RPC search is dead; use the simple JSON search on pypi.org via
  // the public search page API is unstable — query the JSON metadata of known
  // naming-convention candidates from the GitHub discovery pass instead, plus
  // packages whose name matches strands-*. libraries.io is an alternative if
  // this misses too much.
  const results: RegistryCandidate[] = []
  const search = await fetchJson('https://pypi.org/search/?q=strands&format=json').catch(() => null)
  const names: string[] =
    search?.results?.map((r: any) => r.name) ??
    // Fallback: GitHub code-search-derived names handled in main(); return empty here.
    []
  for (const name of names) {
    try {
      const meta = await fetchJson(`https://pypi.org/pypi/${name}/json`)
      const info = meta.info
      const repoUrl: string | undefined =
        info.project_urls?.Source || info.project_urls?.Repository || info.project_urls?.Homepage
      if (!repoUrl?.includes('github.com')) continue
      results.push({
        source: 'pypi',
        name: info.name,
        description: info.summary || '',
        github: repoUrl,
        registry: `https://pypi.org/project/${info.name}/`,
        maintainer: info.author || undefined,
      })
    } catch {
      // Package metadata unavailable — skip.
    }
  }
  return results
}

async function discoverNpm(): Promise<RegistryCandidate[]> {
  const data = await fetchJson('https://registry.npmjs.org/-/v1/search?text=strands%20agents&size=100')
  return (data.objects as any[])
    .map((o) => o.package)
    .filter((p) => /strands/i.test(p.name) || /strands/i.test(p.description || ''))
    .map((p) => ({
      source: 'npm' as const,
      name: p.name as string,
      description: (p.description as string) || '',
      github: (p.links?.repository as string | undefined)?.replace(/^git\+|\.git$/g, ''),
      registry: `https://www.npmjs.com/package/${p.name}`,
      maintainer: p.publisher?.username as string | undefined,
    }))
}

async function discoverGithub(): Promise<RegistryCandidate[]> {
  const headers: Record<string, string> = { accept: 'application/vnd.github+json' }
  if (process.env.GITHUB_TOKEN) headers.authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  const data = await fetchJson(
    'https://api.github.com/search/repositories?q=strands+agents+in:name,description,topics&per_page=100',
    headers
  )
  return (data.items as any[]).map((r) => ({
    source: 'github' as const,
    name: r.name as string,
    description: (r.description as string) || '',
    github: r.html_url as string,
    maintainer: r.owner?.login as string | undefined,
  }))
}

// ── CLI entry point ───────────────────────────────────────────────────────────

const isDirectRun = process.argv[1]?.endsWith('backfill.ts')
if (isDirectRun) {
  const catalogDir = path.resolve('src/content/catalog')
  const [pypi, npm, github] = await Promise.all([discoverPypi(), discoverNpm(), discoverGithub()])
  console.log(`pypi=${pypi.length}, npm=${npm.length}, github=${github.length} | candidates discovered`)
  const merged = mergeCandidates(pypi, npm, github)
  let written = 0
  let skipped = 0
  for (const candidate of merged) {
    const filePath = path.join(catalogDir, `${candidate.name}.yaml`)
    if (existsSync(filePath)) {
      skipped++
      continue
    }
    writeFileSync(filePath, candidateToYaml(candidate))
    written++
  }
  console.log(`written=${written}, skipped_existing=${skipped} | draft entries created — review before committing`)
}
```

Add to `site/package.json` scripts:

```json
    "catalog:backfill": "tsx scripts/catalog/backfill.ts",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- test/catalog-backfill.test.ts`
Expected: PASS

- [ ] **Step 5: Typecheck and commit the script (not its output)**

Run: `npm run typecheck`

```bash
git add site/scripts/catalog/backfill.ts site/test/catalog-backfill.test.ts site/package.json
git commit -m "feat(site): add one-time catalog backfill script"
```

---

### Task 13: Run the backfill and curate the launch catalog

**Files:**
- Create: `site/src/content/catalog/*.yaml` (whatever the curated backfill yields)

**Interfaces:**
- Consumes: `npm run catalog:backfill` (Task 12).
- Produces: the launch catalog content.

- [ ] **Step 1: Run discovery**

Run: `GITHUB_TOKEN=$(gh auth token) npm run catalog:backfill`
Expected: draft YAML files appear in `src/content/catalog/` (existing migrated entries are skipped, never overwritten). Note the PyPI search endpoint may not support `format=json` — if `pypi=0`, that's the known limitation; rely on the npm + GitHub passes, and additionally check the GitHub pass output for Python repos whose README shows a `pip install`, adding those entries by hand.

- [ ] **Step 2: Curate every draft**

For each new draft file:
- Delete it if the package is spam, abandoned (no release in >12 months AND trivial download counts), an example app rather than a reusable package, or unrelated to Strands.
- Resolve every `# REVIEW` marker: verify `integrationType` against the package README and remove the marker line.
- Verify the `registry` URLs resolve and `github` is public.
- Fix descriptions to one clear sentence.

- [ ] **Step 3: Validate and eyeball the page**

Run: `rm -rf .astro && npm test` → PASS (schema + integrity tests cover the new files)
Run: `npm run build` → green.
Run: `npm run dev` and review `/catalog/` — spot-check ~5 new cards' links.

- [ ] **Step 4: Refresh stats for the full catalog**

Run: `GITHUB_TOKEN=$(gh auth token) npm run catalog:stats`
Expected: stats JSON now covers backfilled entries.

- [ ] **Step 5: Commit**

```bash
git add site/src/content/catalog/ site/src/data/catalog-stats.json
git commit -m "feat(site): backfill community catalog from pypi, npm, and github"
```

---

### Task 14: Feature the launch picks and final end-to-end verification

**Files:**
- Modify: 3–5 `site/src/content/catalog/*.yaml` (set `featured: true`; pick well-maintained, representative entries across types — suggest `ag-ui`, `agentcore-memory`, `cohere`, plus the strongest backfilled entries; confirm picks with the user before committing)

**Interfaces:**
- Consumes: everything prior.
- Produces: the launch-ready catalog.

- [ ] **Step 1: Set featured flags**

Add `featured: true` to the agreed entries. ASK THE USER which entries to feature before setting the flags — this is an editorial call.

- [ ] **Step 2: Full verification pass**

- `rm -rf .astro && npm test` → all suites PASS
- `npm run typecheck` → green
- `npm run build` → green, no broken links
- `npm run dev`, then verify in a browser (dev-browser CLI):
  - `/catalog/` shows the featured hero row and the full grid
  - Search for "sql" narrows the grid; the URL gains `?q=sql`; reloading that URL restores the filter
  - Type/language/badge chips filter correctly and combine (AND across facets)
  - No SDK facet is visible (no evals entries yet)
  - A card without `docsPage` links to GitHub with the ↗ marker; a card with one links on-site
  - `/docs/community/community-packages/` redirects to `/catalog/`
  - On `/docs/community/tools/strands-deepgram/` the TypeScript toggle button is disabled with a tooltip
  - Stats render on cards that have them

- [ ] **Step 3: Commit**

```bash
git add site/src/content/catalog/
git commit -m "feat(site): feature launch integrations in community catalog"
```

---

## Self-Review Notes

- **Spec coverage:** data model (Task 1–2), catalog page + filtering (3, 5, 6), navigation/redirect (7), language toggle (8), submission flow (9), stats refresh (10–11), backfill (12–13), featured (14). Evals facet: schema in Task 1, hidden-until-populated logic in Tasks 2/5/6. All spec sections have tasks.
- **Type consistency:** `CatalogCardModel`, `CatalogStats`, `CatalogFilterState`, `CardFilterData` defined once (Tasks 2–3) and consumed by name in Tasks 5–6; `StatsEntry`/`StatsFetchers` (Task 10) used only within the stats script and test.
- **Known judgment points for implementers:** exact frontmatter of the ~30 migrated pages must be read from each file (Task 4 gives the mapping, not fabricated values); PyPI search API instability is called out in Tasks 12–13 with a fallback path.
